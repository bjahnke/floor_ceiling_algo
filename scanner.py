"""
"""
import cProfile
from pathlib import Path
from time import time
import httpx
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
account_info = tda_access.LocalClient.account_info()
import yfinance as yf


failed = DotMap(_dynamic=False)
failed.no_candles = set()
failed.empty_data = set()
failed.no_swings = set()
failed.ticker_not_found = set()
failed.no_high_score = set()


def fc_scan_symbol(
        symbol: str,
        price_data: pd.DataFrame,
        bench_data: pd.DataFrame = None,
        freq_range: tdargs.FreqRangeArgs = tdargs.freqs.day.range(tdargs.periods.y5),
        scan_output_loc: str = './scan_out'
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

    price_data = fc_data_gen.new_init_fc_data(
        base_symbol=symbol,
        price_data=price_data,
        equity=account_info.equity
    )

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

    Path(scan_output_loc).mkdir(parents=True, exist_ok=True)
    out_name = f'{scan_output_loc}/{symbol}'
    price_data.to_csv(f'{out_name}.csv')
    plt.savefig(f'{symbol}.png', bbox_inches='tight')
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
    scan_result['symbol'] = symbol
    scan_result['up_to_date'] = price_data.index[-1]

    return scan_result


def fc_scan_all(
        symbols: t.List[str],
        fetch_price_history: t.Callable[[str, tdargs.FreqRangeArgs], pd.DataFrame],
        freq_range: tdargs.FreqRangeArgs = tdargs.freqs.day.range(tdargs.periods.y5),
        bench_symbol: str = None,
        scan_output_loc: str = './scan_out'
):
    """
    scans all given symbols. builds overview report of scan results.
    Handles possible exceptions.
    :param bench_symbol:
    :param symbols:
    :param freq_range:
    :param scan_output_loc:
    :param fetch_price_history:
    :return:
    """
    list_dict = []
    bench_data = None
    if bench_symbol is not None:
        bench_data = fetch_price_history(bench_symbol, freq_range)

    while len(symbols) > 0:
        symbol = symbols[0]

        try:
            data = fetch_price_history(symbol, freq_range)
            scan_result = fc_scan_symbol(
                symbol=symbol,
                price_data=data,
                bench_data=bench_data,
                freq_range=freq_range
            )

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
        except fc_data_gen.FcLosesToBuyHoldError:
            failed.no_high_score.add(symbols.pop(0))
        else:
            list_dict.append(scan_result)
            symbols.pop(0)
            print(f'{symbol}, {len(symbols)} left...')

    # Instantiate market_regime df using pd.DataFrame. from_dict()
    market_regime = pd.DataFrame.from_dict(list_dict)
    # Change the order of the columns
    if len(market_regime.index.to_list()) == 0:
        raise Exception("no data output")
    market_regime = market_regime[[
        'symbol',
        'regime_change_date',
        'up_to_date',
        'regime',
        'signal',
        'relative_returns',
        'absolute_returns',
        'close',
        'position_size',
        'score',
        'stop_loss_base'
    ]]
    # Sort columns by regime change date
    market_regime.sort_values(
        by=['regime_change_date'], ascending=False, inplace=True
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
    # Find the latest regime change
    regime_change_date = price_data[price_data[regime_change_col].diff() != 0].index[-1]
    cumulative_relative_returns = cumulative_percent_returns(
        price_data, regime_change_date, rebase_close_col
    )

    cumulative_absolute_returns = cumulative_percent_returns(
        price_data, regime_change_date, stock_close_col
    )

    # TODO no position sizes to calculate on
    position_size = price_data.signals.slices[-1].eqty_risk_lot[-1]
    return {
        'regime': regime,
        'signal': price_data.signal[-1],
        'regime_change_date': regime_change_date,
        'relative_returns': cumulative_relative_returns,
        'absolute_returns': cumulative_absolute_returns,
        'close': price_data.b_close[-1],
        'position_size': position_size,
        'score': price_data.score[-1],
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
    freq_range: tdargs.FreqRangeArgs = tdargs.freqs.day.range(tdargs.periods.y5),
    bench: str = None,
):
    """wrapper simply for catching PermissionError if the output excel file is already open"""
    scan_results = fc_scan_all(
        bench_symbol=bench,
        symbols=symbols,
        freq_range=freq_range,
        fetch_price_history=fetch_price_history
    )
    scan_results['true_pos_size'] = scan_results.position_size * scan_results.signal
    # scan_results = fc_scan_all('SPX', ['ADP'])

    print(len(scan_results.index))
    while True:
        try:
            scan_results.to_excel('SPX_REGIME_SCAN.xlsx')
            break
        except PermissionError:
            inp = input('xlsx file is still open.\n\'n\' to quit:')
            if inp == 'n':
                break

    print('done.')

def yf_price_history(symbol, freq_range):
    data: pd.DataFrame = yf.Ticker(symbol).history(period='1y', interval='1h')
    if symbol == 'CCI30':
        data = pd.read_csv('cci30_OHLCV.csv')
        data.Date = pd.to_datetime(data.Date, infer_datetime_format=True)
        data.index = data.Date
        data = data.sort_index()

    data = data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close'
    })
    # the following columns are needed for compatibility with current
    # implementation of init_fc_data.
    data['b_high'] = data.high
    data['b_low'] = data.low
    data['b_close'] = data.close

    # convert date time to timezone unaware
    data.index = data.index.tz_convert(None)

    return data[['open', 'high', 'low', 'close', 'b_high', 'b_low', 'b_close']]


if __name__ == '__main__':
    # # TODO ADP
    # # TODO find method to get reliable updated source and with stocks
    SP500_URL = 'https://tda-api.readthedocs.io/en/latest/_static/sp500.txt'

    # Load S&P 500 composition from documentation
    # List[str]
    sp500 = httpx.get(
        SP500_URL,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    ).read().decode().split()
    # stocks = pd.read_excel('nasdaq.xlsx')


    start = time()
    main(
        symbols=['ADA-USD'],
        bench=None,
        freq_range=tdargs.freqs.day.range(tdargs.periods.y3),
        fetch_price_history=yf_price_history
    )
    print(f'Time Elapsed: {time()-start/60} minutes')
    # cProfile.run('main(symbols=[\'LB\'], bench=\'SPX\')', filename='output.prof')



