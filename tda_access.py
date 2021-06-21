"""
this module contains functionality related to account information.
Pulls account data which can be used for creating orders or
analyzing order/balance/position history

TODO order history?
"""
from time import sleep, perf_counter

import tda
import selenium.webdriver
import pandas as pd
import tdargs
import json
from dataclasses import dataclass, field
import typing as t
from enum import Enum


def give_attribute(new_attr: str, value: t.Any) -> t.Callable:
    """
    gives function an attribute upon declaration
    :param new_attr: attribute name to add to function attributes
    :param value: value to initialize attribute to
    :return: decorator that will give the decorated function an attribute named
    after the input new_attr
    """
    def decorator(function: t.Callable) -> t.Callable:
        setattr(function, new_attr, value)
        return function
    return decorator


class EmptyDataError(Exception):
    """A dictionary with no data was received from request"""

class TickerNotFoundError(Exception):
    """error response from td api is Not Found"""

class Side(Enum):
    LONG = 1
    SHORT = -1
    CLOSE = 0


@dataclass
class Position:
    raw_position: dict
    symbol: str = field(init=False)
    value: int = field(init=False)
    qty: int = field(init=False)
    side: Side = field(init=False)

    def __post_init__(self):
        self.symbol = self.raw_position['instrument']['symbol']
        self.value = self.raw_position['marketValue']
        if self.raw_position['shortQuantity'] > 0:
            self.qty = self.raw_position['shortQuantity']
            self.side = Side.SHORT
        else:
            self.qty = self.raw_position['longQuantity']
            self.side = Side.LONG

@dataclass
class AccountInfo:
    acct_data_raw: t.Dict
    equity: float = field(init=False)
    liquid_funds: float = field(init=False)
    buy_power: float = field(init=False)
    positions: t.Dict[str, Position] = field(init=False)

    def __post_init__(self):
        cur_balance = self.acct_data_raw['securitiesAccount']['currentBalances']
        self.equity = cur_balance['equity']
        self.liquid_funds = cur_balance['moneyMarketFund'] + cur_balance['cashBalance']
        self.buy_power = cur_balance['buyingPower']
        self.positions = {
            pos['instrument']['symbol']: Position(pos['instrument'])
            for pos in self.acct_data_raw['securitiesAccount']['positions']
            if pos['instrument']['cusip'] != '9ZZZFD104'  # don't add position if it is money_market
        }

    def get_position(self, symbol: str) -> t.Union[Position, None]:
        return self.positions.get(symbol, Side.CLOSE)

    def get_symbols(self):
        return [symbol for symbol, _ in self.positions.items()]


# create td client
class LocalClient:
    tda_client = tda.auth.easy_client(
        api_key='UGLWOLA4LMXN684IG3MIMXMPDN1GBMNR',
        redirect_uri='https://localhost',
        token_path=r"C:\Users\Brian\Documents\_projects\_trading\credentials\token",
        webdriver_func=selenium.webdriver.Firefox,
    )

    @staticmethod
    def get_account_info() -> AccountInfo:

        resp = LocalClient.tda_client.get_account(
            # TODO Note: account id should remain private
            account_id=686081659,
            fields=[
                tda.client.Client.Account.Fields.ORDERS,
                tda.client.Client.Account.Fields.POSITIONS
            ]
        )
        # dump account data to txt for reference
        account_info_raw = resp.json()
        with open('account_data.json', 'w') as outfile:
            json.dump(account_info_raw, outfile, indent=4)

        return AccountInfo(account_info_raw)

    @staticmethod
    @give_attribute('request_sleeper', 3.7)
    def price_history(
            symbol: str,
            freq_range: tdargs.FreqRangeArgs,
    ) -> pd.DataFrame:
        """
        # TODO add type hints
        :param symbol:
        :param freq_range:
        :return:
        """
        func_self = LocalClient.price_history

        # get historical data, store as dataframe, convert datetime (ms) to y-m-d-etc
        while True:
            resp = LocalClient.tda_client.get_price_history(
                symbol,
                period_type=freq_range.range.period.type,
                period=freq_range.range.period.val,
                frequency_type=freq_range.freq.type,
                frequency=freq_range.freq.val,
                start_datetime=freq_range.range.start,
                end_datetime=freq_range.range.end,
            )
            history = resp.json()
            if history.get('candles', None) is not None:
                break
            elif history['error'] == 'Not Found':
                print(f'td api could not find symbol {symbol}')
                raise TickerNotFoundError(f'td api could not find symbol {symbol}')
            else:
                raise EmptyDataError(f'No data received for symbol {symbol}')
                # print(history['error'])
                # print(f'sleeping for {func_self.request_sleeper}...')
                # sleep(func_self.request_sleeper)
                # func_self.request_sleeper += .1

        df = pd.DataFrame(history['candles'])

        if history['empty'] is True:
            raise EmptyDataError(f'No data received for symbol {symbol}')

        # datetime given in ms, convert to readable date
        df.datetime = pd.to_datetime(df.datetime, unit='ms')

        # for truncating to date only (not hours/minutes/seconds)
        # df.datetime = df.datetime.dt.date

        # rename datetime to time for finplot compatibility
        df = df.rename(columns={'datetime': 'time'})
        df.index = df.time
        # drop columns other than those mentioned (maybe want to save volume)
        df.drop(df.columns.difference(['open', 'high', 'close', 'low']), 1, inplace=True)

        return df

