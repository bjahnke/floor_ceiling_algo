"""
"""
import cProfile
from time import time

import httpx
import fc_data_gen
import tdargs
import typing as t
import pandas as pd
import trade_stats
from dotmap import DotMap
import tda_access
from back_test_utils import NoSwingsError

account_info = tda_access.LocalClient.account_info()


def fc_scan_all(bench_symbol: str, symbols: t.List[str]):
    list_dict = []
    failed = DotMap(_dynamic=False)
    failed.no_candles = set()
    failed.empty_data = set()
    failed.no_swings = set()
    failed.ticker_not_found = set()
    failed.no_high_score = set()

    while len(symbols) > 0:
        symbol = symbols[0]
        try:
            relative_data = fc_data_gen.init_fc_data(
                base_symbol=symbol,
                bench_symbol=bench_symbol,
                equity=account_info.equity,
                freq_range=tdargs.freqs.day.range(tdargs.periods.y2)
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
            scan_result = regime_scan(
                price_data=relative_data,
                regime_floorceiling_col='regime_floorceiling',
                regime_change_col='regime_change',
                rebase_close_col='close',
                stock_close_col='b_close'
            )
            print(f'{symbol}, {len(symbols)} left...')
            scan_result['symbol'] = symbols.pop(0)
            scan_result['up_to_date'] = relative_data.index[-1]
            list_dict.append(scan_result)

    # Instantiate market_regime df using pd.DataFrame. from_dict()
    market_regime = pd.DataFrame.from_dict(list_dict)
    # Change the order of the columns
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
        'score'
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
        'score': price_data.score[-1]
    }


def cumulative_percent_returns(price_data, regime_change_date, arg2):
    """cumulative percent returns"""
    prices_in_regime = price_data[regime_change_date:][arg2]
    log_returns = trade_stats.simple_returns(prices_in_regime)
    crr = trade_stats.cum_return_percent(log_returns)
    return crr[-1]


def main(symbols: t.List[str], bench: str):

    scan_results = fc_scan_all(bench, symbols)
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


if __name__ == '__main__':
    # # TODO ADP
    # # TODO find method to get reliable updated source and with stocks
    # SP500_URL = 'https://tda-api.readthedocs.io/en/latest/_static/sp500.txt'
    #
    # # Load S&P 500 composition from documentation
    # # List[str]
    # sp500 = httpx.get(
    #     SP500_URL,
    #     headers={
    #         "User-Agent": "Mozilla/5.0"
    #     }
    # ).read().decode().split()
    stocks = pd.read_excel('nasdaq.xlsx')
    start = time()
    main(symbols=stocks.Symbol.to_list(), bench='SPX')
    print(f'Time Elapsed: {time()-start/60} minutes')
    # cProfile.run('main(symbols=[\'LB\'], bench=\'SPX\')', filename='output.prof')



