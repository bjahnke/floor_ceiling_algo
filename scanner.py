"""

"""
import httpx
import trade_df
import tdargs
import typing as t
import pandas as pd
import trade_stats
from dotmap import DotMap
import tda_access

account_info = tda_access.LocalClient.get_account_info()

def fc_scan_all(bench_symbol: str, symbols: t.List[str]):
    list_dict = []
    failed_fetch = DotMap(_dynamic=False)
    failed_fetch.no_candles = set()
    failed_fetch.no_datetime = set()
    failed_fetch.no_swings = set()

    request_counter = 0
    while len(symbols) > 0:
        symbol = symbols[0]
        try:
            relative_mdf = trade_df.RelativeMdf(
                base_symbol=symbol,
                bench_symbol=bench_symbol,
                freq_range=tdargs.freqs.day.range(tdargs.periods.y2)
            ).init_position_size(equity=account_info.equity)
        # except KeyError:
        #     failed_fetch.no_candles.add(symbol)
        # except AttributeError:
        #     failed_fetch.no_datetime.add(symbols.pop(0))
        except ValueError as err:
            if symbol in err.args:
                failed_fetch.no_swings.add(symbols.pop(0))
            else:
                # only raise, since we already raised once
                # https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
                raise
        else:
            scan_result = regime_scan(
                price_data=relative_mdf.data,
                regime_floorceiling_col='regime_floorceiling',
                regime_change_col='regime_change',
                rebase_close_col='close',
                stock_close_col='b_close'
            )
            print(f'{symbol}, {len(symbols)} left...')
            scan_result['symbol'] = symbols.pop(0)
            scan_result['up_to_date'] = relative_mdf.data.index[-1]
            list_dict.append(scan_result)

    # Instantiate market_regime df using pd.DataFrame. from_dict()
    market_regime = pd.DataFrame.from_dict(list_dict)
    # Change the order of the columns
    market_regime = market_regime[[
        'symbol',
        'regime_change_date',
        'up_to_date',
        'regime',
        'relative_returns',
        'absolute_returns',
        'close'
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
    price_data_cols = price_data.columns.to_list()
    signal_col = price_data_cols[7]
    position_size_col = 'eqty_risk_lot'


    # Create a dataframe and dictionary list
    # Current regime
    regime = price_data[regime_floorceiling_col][-1]
    # Find the latest regime change
    regime_change_date = price_data[price_data[regime_change_col].diff() != 0].index[-1]
    # Calculate cumulative returns for relative series
    rel_prices_in_regime = price_data[regime_change_date:][rebase_close_col]
    rel_log_returns = trade_stats.simple_returns(rel_prices_in_regime)
    crr = trade_stats.cum_return_percent(rel_log_returns)
    cumulative_relative_returns = crr[-1]

    base_prices_in_regime = price_data[regime_change_date:][stock_close_col]
    # Calculate cumulative returns for absolute series
    base_log_returns = trade_stats.simple_returns(base_prices_in_regime)
    car = trade_stats.cum_return_percent(base_log_returns)
    cumulative_absolute_returns = car[-1]

    position_size_date = price_data[price_data[position_size_col].diff() != 0].index[-2]
    position_size = price_data.loc[position_size_date][position_size_col]
    # Create a dictionary  of the columns required and append it to the list
    row = {
        'regime': regime,
        'regime_change_date': regime_change_date,
        'relative_returns': cumulative_relative_returns,
        'absolute_returns': cumulative_absolute_returns,
        'close': price_data.b_close[-1],
        'position_size': position_size
    }
    return row


def main():
    # TODO find method to get reliable updated source and with stocks
    SP500_URL = 'https://tda-api.readthedocs.io/en/latest/_static/sp500.txt'

    # Load S&P 500 composition from documentation
    # List[str]
    sp500 = httpx.get(
        SP500_URL,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    ).read().decode().split()
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    scan_results = fc_scan_all('SPX', sp500)
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
    main()
