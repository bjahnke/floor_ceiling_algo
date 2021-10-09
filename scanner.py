"""
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from time import time
import httpx
import numpy as np
import requests
from matplotlib import pyplot as plt
import back_test_utils
import fc_data_gen
import tdargs
import typing as t
import pandas as pd
import trade_stats
from dotmap import DotMap
import tda_access
from back_test_utils import NoSwingsError
import yfinance as yf
from dataclasses import dataclass, field
import schedule
import time

account_info = tda_access.LocalClient.account_info()


@dataclass
class ScanOutInfo:
    _out_dir: str
    _png_dir: str
    _report_name: str
    _price_data_name: str

    def __post_init__(self):
        assert self._report_name[-5:] == '.xlsx'
        assert self._price_data_name[-4:] == '.csv'

    @property
    def report_fp(self):
        """report file path"""
        return fr'{self._out_dir}\{self._report_name}'

    @property
    def data_fp(self):
        return fr'{self._out_dir}\{self._price_data_name}'

    def png_fp(self, file_name):
        return fr'{self._png_dir}\{file_name}'


@dataclass
class Range:
    _min: t.Union[None, float] = field(default=None)
    _max: t.Union[None, float] = field(default=None)

    def __post_init__(self):
        Range._assert_min_le_max(self._min, self._max)

    @staticmethod
    def _assert_min_le_max(min_num, max_num):
        if None not in [min_num, max_num]:
            assert min_num <= max_num, 'min value must not be greater than max value'

    def is_in_range(self, num: float):
        return (
            (self._max is None or num <= self._max) and
            (self._min is None or num >= self._min)
        )


class StrNanInPriceError(Exception):
    """string nan in price history"""


failed = DotMap(_dynamic=False)
failed.no_candles = set()
failed.empty_data = set()
failed.no_swings = set()
failed.ticker_not_found = set()
failed.no_high_score = set()
failed.str_nan_in_price = set()

# TODO create scanner class with raw_scan_results as attribute
raw_scan_results = []
all_price_data: t.Union[pd.DataFrame, None] = None


def fc_scan_symbol(
        symbol: str,
        price_data: pd.DataFrame,
        scan_out_info: ScanOutInfo,
        bench_data: pd.DataFrame = None,
        freq_range: tdargs.FreqRangeArgs = tdargs.freqs.day.range(tdargs.periods.y5),
):
    """helper, scans one symbol, outputs data to files, returns scan results"""

    if bench_data is not None:
        price_data = fc_data_gen.create_relative_data(
            base_symbol=symbol,
            price_data=price_data,
            bench_data=bench_data,
            freq_range=freq_range
        )
    # relative_data = fc_data_gen.init_fc_data(
    #     base_symbol=symbol,
    #     bench_symbol=bench_symbol,
    #     equity=account_info.equity,
    #     freq_range=freq_range
    # )

    price_data, stats = fc_data_gen.init_fc_data(
        base_symbol=symbol,
        price_data=price_data,
        equity=account_info.equity
    )

    last_signal = price_data.signals.slices()[-1]
    cum_returns = last_signal.stats.cumulative_percent_returns
    cum_returns_total = price_data.signals.cumulative_returns()
    print(f'{symbol} all returns: {cum_returns_total[-1]}')
    print(f'{symbol} last signal returns: {cum_returns[-1]}')

    back_test_utils.graph_regime_fc(
        ticker=symbol,
        df=price_data,
        y='close',
        th=1.5,
        sl='sw_low',
        sh='sw_high',
        clg='ceiling',
        flr='floor',
        st=price_data['st_ma'],
        mt=price_data['mt_ma'],
        bs='regime_change',
        rg='regime_floorceiling',
        bo=200
    )
    plt.savefig(scan_out_info.png_fp(f'{symbol}.png'), bbox_inches='tight')
    # except tda_access.EmptyDataError:
    # except tda_access.TickerNotFoundError:
    # except NoSwingsError as err:
    # except fc_data_gen.FcLosesToBuyHoldError:
    scan_result = regime_scan(
        price_data=price_data,
        regime_floorceiling_col='regime_floorceiling',
        regime_change_col='regime_change',
        rebase_close_col='close',
        stock_close_col='b_close'
    )
    stats = stats.sort_values(by='risk_adjusted_returns').dropna(subset=['risk_adjusted_returns'])
    try:
        scan_result['st'] = stats.st.iloc[-1]
        scan_result['mt'] = stats.mt.iloc[-1]
    except IndexError:
        scan_result['st'] = np.NAN
        scan_result['mt'] = np.NAN

    scan_result['symbol'] = symbol
    scan_result['up_to_date'] = price_data.index[-1]

    return scan_result, price_data


def fc_scan_all(
        symbols: t.List[str],
        fetch_price_history: t.Callable[[str, tdargs.FreqRangeArgs], pd.DataFrame],
        scan_out_info: ScanOutInfo,
        freq_range: tdargs.FreqRangeArgs = tdargs.freqs.day.range(tdargs.periods.y5),
        bench_symbol: str = None,

        close_range: Range = Range(),
        volume_range: Range = Range(),
) -> t.List[t.Dict]:
    """
    scans all given symbols. builds overview report of scan results.
    Handles possible exceptions.
    :param bench_symbol:
    :param symbols:
    :param freq_range:
    :param scan_output_loc:
    :param fetch_price_history:
    :param close_range: filter out stocks by current close price
    :param volume_range: filter out stocks by current volume
    :return:
    """
    bench_data = None
    if bench_symbol is not None:
        bench_data = fetch_price_history(bench_symbol, freq_range)

    global all_price_data
    while len(symbols) > 0:
        symbol = symbols[0]

        try:
            data = fetch_price_history(symbol, freq_range)
            if (
                len(data.index) == 0 or
                not close_range.is_in_range(data.close[-1]) or
                not volume_range.is_in_range(data.volume[-1])
            ):
                print(f'{symbol} skipped due to price out of range')
                symbols.pop(0)
                continue

            scan_result, data = fc_scan_symbol(
                symbol=symbol,
                price_data=data,
                bench_data=bench_data,
                freq_range=freq_range,
                scan_out_info=scan_out_info
            )

            # TODO turn into class attribute when scanner becomes class
            data['symbol'] = symbol
            if all_price_data is None:
                all_price_data = data.copy()
            else:
                all_price_data = pd.concat([all_price_data, data])

        except tda_access.EmptyDataError:
            print(f'no data? {symbol}')
            failed.empty_data.add(symbol)
            symbols.pop(0)
        except tda_access.TickerNotFoundError:
            failed.empty_data.add(symbol)
            symbols.pop(0)
        except NoSwingsError as err:
            if symbol in err.args:
                failed.no_swings.add(symbols.pop(0))
            else:
                # only raise, since we already raised once
                # https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
                raise
        except StrNanInPriceError:
            failed.str_nan_in_price.add(symbols.pop(0))
        except fc_data_gen.FcLosesToBuyHoldError:
            failed.no_high_score.add(symbols.pop(0))
        except fc_data_gen.NoSignalsError:
            symbols.pop(0)
        except FileNotFoundError:
            # for some reason PRN can not be written to files, results in FileNotFoundError
            symbols.pop(0)
        else:
            raw_scan_results.append(scan_result)
            symbols.pop(0)
            print(f'{symbol}, {len(symbols)} left...')

    return raw_scan_results


def format_scan_results(scan_results_raw: t.List[t.Dict]) -> pd.DataFrame:
    # Instantiate market_regime df using pd.DataFrame. from_dict()
    market_regime = pd.DataFrame.from_dict(scan_results_raw)
    # Change the order of the columns
    if len(market_regime.index.to_list()) == 0:
        raise Exception("no data output")
    market_regime = market_regime[[
        'symbol',
        'signal_start',
        'signal',
        'st',
        'mt',
        'cum_absolute_returns',
        'score',
        'close',
        'position_size',
        'stop_loss_base',
    ]]
    # Sort columns by regime change date
    market_regime.sort_values(
        by=['signal_start'], ascending=False, inplace=True
    )
    return market_regime


def regime_scan(
    price_data: pd.DataFrame,
    regime_floorceiling_col: str,
    regime_change_col: str,
    rebase_close_col: str,
    stock_close_col: str,
) -> t.Dict:
    # Create a dataframe and dictionary list
    # Current regime
    regime = price_data[regime_floorceiling_col][-1]

    # # Find the latest regime change
    # regime_change_date = price_data[price_data[regime_change_col].diff() != 0].index[-1]
    #
    # cumulative_relative_returns = cumulative_percent_returns(
    #     price_data, regime_change_date, rebase_close_col
    # )
    # # adjust to positive for short side
    # cumulative_relative_returns *= price_data.signals.slices()[-1].signal[-1]

    # TODO accessor only usable for base price currently, need solution for applying to relative
    # cumulative_relative_returns = price_data.signals.cumulative_returns()

    signal_start_data = price_data.signals.slices()[-1].index[0]

    cumulative_absolute_returns = price_data.signals.cumulative_returns()[-1]

    position_size = price_data.signals.slices()[-1].eqty_risk_lot[-1]

    return {
        'signal': price_data.signal[-1],
        'signal_start': signal_start_data,
        # 'r_returns_last': cumulative_relative_returns,  # returns since last regime change (relative price)
        'cum_absolute_returns': cumulative_absolute_returns,  # returns since last regime change (base price)
        'score': price_data.score[-1],
        'close': price_data.b_close[-1],
        'position_size': position_size,
        'stop_loss_base': price_data.stop_loss_base[-1]
    }


def cumulative_percent_returns(price_data, regime_change_date, arg2):
    """cumulative percent returns"""
    prices_in_regime = price_data[regime_change_date:][arg2]
    log_returns = trade_stats.simple_returns(prices_in_regime)
    crr = trade_stats.cum_return_percent(log_returns)
    return crr[-1]


def main(
    symbols: t.List[str],
    fetch_price_history: t.Callable[[str, tdargs.FreqRangeArgs], pd.DataFrame],
    scan_out_info: ScanOutInfo,
    freq_range: tdargs.FreqRangeArgs = tdargs.freqs.day.range(tdargs.periods.y5),
    bench: str = None,
    close_range: Range = Range(),
    volume_range: Range = Range(),
):
    """wrapper simply for catching PermissionError if the output excel file is already open"""
    global raw_scan_results
    global all_price_data

    def post_process(raw_scan_res: t.List[t.Dict]):
        scan_out: pd.DataFrame = format_scan_results(raw_scan_res)
        # add columns post process for convenience
        scan_out['true_size'] = scan_out.position_size * scan_out.signal
        scan_out['trade_val'] = scan_out.true_size * scan_out.close
        scan_out['trade_risk'] = (
            (scan_out.signal * (scan_out.close - scan_out.stop_loss_base)) * scan_out.true_size
        )
        all_price_data.to_csv(scan_out_info.data_fp)
        scan_results_to_excel(scan_out, scan_out_info.report_fp)

    try:
        scan_results = fc_scan_all(
            bench_symbol=bench,
            symbols=symbols,
            freq_range=freq_range,
            fetch_price_history=fetch_price_history,
            close_range=close_range,
            volume_range=volume_range,
            scan_out_info=scan_out_info
        )
    except:
        # output existing results if any uncaught exception occurs
        if len(raw_scan_results) > 0:
            post_process(raw_scan_results)
        raise
    else:
        post_process(raw_scan_results)

    # reset globals upon completion so continuous scanner doesn't include prior runs in output data
    all_price_data = None
    raw_scan_results = []
    print(f'No candles: {failed.no_candles}')
    print(f'No data: {failed.empty_data}')
    print(f'No swings: {failed.no_swings}')
    print(f'No ticker: {failed.ticker_not_found}')
    print(f'No high score: {failed.no_high_score}')
    print(f'Str nan in price: {failed.str_nan_in_price}')
    failed.no_candles = set()
    failed.empty_data = set()
    failed.no_swings = set()
    failed.ticker_not_found = set()
    failed.no_high_score = set()
    failed.str_nan_in_price = set()

    print('done.')


def scan_results_to_excel(data: pd.DataFrame, file_path: str):
    """write data to file. if file opened prepend a number until successful"""
    i = 0
    prefix = ''
    while True:
        try:
            if i > 0:
                prefix = f'({i})'
            data.to_excel(f'{prefix}{file_path}')
            break
        except PermissionError:
            i += 1


def format_price_data(data: pd.DataFrame) -> pd.DataFrame:
    """format price data to what is expected by scanner/strategy code"""

    data.columns = map(lambda x: x.lower(), data.columns.to_list())

    data['b_high'] = data.high
    data['b_low'] = data.low
    data['b_close'] = data.close

    # convert date time to timezone unaware
    try:
        data.index = data.index.tz_convert(None)
    except AttributeError:
        pass

    return data[['open', 'high', 'low', 'close', 'volume', 'b_high', 'b_low', 'b_close']]


def yf_price_history(symbol, freq_range=None):
    try:
        price_data: pd.DataFrame = yf.Ticker(symbol).history(
            # period=datetime.now() - timedelta(days=58), interval='15m'
            period='1mo', interval='15m'
        )
    except json.decoder.JSONDecodeError:
        return pd.DataFrame()
    except requests.exceptions.ConnectionError:
        return pd.DataFrame()
    except AttributeError:
        return pd.DataFrame()

    if symbol == 'CCI30':
        price_data = pd.read_csv('cci30_OHLCV.csv')
        price_data.Date = pd.to_datetime(price_data.Date, infer_datetime_format=True)
        price_data.index = price_data.Date
        price_data = price_data.sort_index()

    price_data = price_data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    return format_price_data(price_data)


def get_sp500_symbols():
    SP500_URL = 'https://tda-api.readthedocs.io/en/latest/_static/sp500.txt'

    # Load S&P 500 composition from documentation
    # List[str]
    sp500 = httpx.get(
        SP500_URL,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    ).read().decode().split()
    return sp500


def test_signal(symbol):
    # price_data = yf_price_history('ADA-USD')
    price_data = yf_price_history(symbol)
    price_data, stats = fc_data_gen.init_fc_data(
        base_symbol='ADA-USD',
        price_data=price_data,
        equity=account_info.equity
    )
    return price_data


def scan_nasdaq():
    stocks = pd.read_excel(r'.\symbols\nasdaq.xlsx')
    main(
        # symbols=cb_usd_products.id.to_list(),
        symbols=stocks.Symbol.to_list(),
        fetch_price_history=yf_price_history,
        close_range=Range(_min=1, _max=30),
        # volume_range=Range(_min=100000)
    )


def continuous_scan(job: t.Callable, run_at_time: str = '01:00'):
    """"""
    schedule.every().day.at(run_at_time).do(job)
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == '__main__':
    main(
        symbols=[
                'IDXX',
                'RDM',
                'EW',
                'WAT',
                'SHW',
                'EXR',
                'ABMD',
                'MSCI'
            ],
        bench=None,
        freq_range=tdargs.freqs.m15.range(left_bound=datetime.utcnow() - timedelta(days=30)),
        fetch_price_history=tda_access.LocalClient.price_history,
        scan_out_info=ScanOutInfo(
            _out_dir=r'C:\Users\Brian\OneDrive\algo_data\csv',
            _png_dir=r'C:\Users\Brian\OneDrive\algo_data\png',
            _report_name='scan_out.xlsx',
            _price_data_name='price_data.csv'
        )
    )
    print('done')




