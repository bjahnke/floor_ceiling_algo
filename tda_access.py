"""
this module contains functionality related to account information.
Pulls account data which can be used for creating orders or
analyzing order/balance/position history

TODO order history?
"""
import datetime
from collections import namedtuple

from tda.orders.common import OrderType

import credentials
from time import sleep, perf_counter
import asyncio

import tda
import selenium.webdriver
import pandas as pd
import tdargs
from dataclasses import dataclass, field
import typing as t
from enum import Enum
import tda.orders.equities as toe
import tda.orders.common as toc
import json

OrderStatus = tda.client.Client.Order.Status

OrderData = namedtuple('OrderData', ['direction', 'quantity', 'stop_loss', 'status'])


_ACCOUNT_ID = credentials.ACCOUNT_ID


def parse_orders(orders: t.List[t.Dict]) -> t.Dict[int, t.Dict]:
    return {order['orderId']: order for order in orders}


def configure_stream(
        stream_client: tda.streaming.StreamClient,
        add_book_handler: t.Callable,
        book_subs,
        handlers: t.List[t.Callable],

):
    async def _initiate_stream(*symbols: str):
        await stream_client.login()
        await stream_client.quality_of_service(tda.streaming.StreamClient.QOSLevel.EXPRESS)

        for handler in handlers:
            add_book_handler(handler)

        await book_subs(symbols)

        while True:
            await stream_client.handle_message()

    return _initiate_stream


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


CLOSE_ORDER = {
    Side.LONG: lambda sym, qty: toe.equity_sell_market(sym, qty),
    Side.SHORT: lambda sym, qty: toe.equity_buy_to_cover_market(sym, qty),
}


OPEN_ORDER = {
    Side.LONG: lambda sym, qty: toe.equity_buy_market(sym, qty),
    Side.SHORT: lambda sym, qty: toe.equity_sell_short_market(sym, qty),
}


OPEN_STOP = {
    Side.LONG: lambda sym, qty, stop_price: (
        toe.equity_sell_market(sym, qty)
        .set_order_type(OrderType.STOP)
        .set_stop_price(stop_price)
    ),
    Side.SHORT: lambda sym, qty, stop_price: (
        toe.equity_buy_to_cover_market(sym, qty)
        .set_order_type(OrderType.STOP)
        .set_stop_price(stop_price)
    ),
}


@dataclass
class Position:
    raw_position: dict
    symbol: str = field(init=False)
    value: int = field(init=False)
    qty: int = field(init=False)
    _side: Side = field(init=False)

    def __post_init__(self):
        self.symbol = self.raw_position['instrument']['symbol']
        self.value = self.raw_position['marketValue']
        if self.raw_position['shortQuantity'] > 0:
            self.qty = self.raw_position['shortQuantity']
            self._side = Side.SHORT
        else:
            self.qty = self.raw_position['longQuantity']
            self._side = Side.LONG

    @property
    def side(self):
        return self._side

    def full_close(self):
        CLOSE_ORDER[self._side](self.symbol, self.qty)

    def open(self, quantity):
        OPEN_ORDER[self._side](self.symbol, quantity)


@dataclass
class AccountInfo:

    acct_data_raw: t.Dict
    equity: float = field(init=False)
    liquid_funds: float = field(init=False)
    buy_power: float = field(init=False)
    _positions: t.Dict[str, Position] = field(init=False)
    _pending_orders: t.Dict[int, t.Dict] = field(init=False)

    def __post_init__(self):
        cur_balance = self.acct_data_raw['securitiesAccount']['currentBalances']
        self.equity = cur_balance['equity']
        self.liquid_funds = cur_balance['moneyMarketFund'] + cur_balance['cashBalance']
        self.buy_power = cur_balance['buyingPower']
        self._positions = {
            pos['instrument']['symbol']: Position(pos)
            for pos in self.acct_data_raw['securitiesAccount']['positions']
            if pos['instrument']['cusip'] != '9ZZZFD104'  # don't add position if it is money_market
        }
        # self._pending_orders = self._parse_order_statuses()

    @property
    def positions(self):
        return self._positions

    @property
    def raw_orders(self):
        return self.acct_data_raw['securitiesAccount']['orderStrategies']

    # @property
    # def orders(self) -> t.Dict[int, t.Dict]:
    #     return self._pending_orders

    def get_position_info(self, symbol: str) -> t.Union[Position, None]:
        return self._positions.get(symbol, None)

    def get_symbols(self) -> t.List:
        return [symbol for symbol, _ in self._positions.items()]

    # def _parse_order_statuses(self) -> t.Dict[int, t.Dict]:
    #     """for convenient lookup of order status"""
    #     raw_orders = self.acct_data_raw['securitiesAccount']['orderStrategies']
    #     return parse_orders(raw_orders)


class _LocalClientMeta(type):
    _cached_account_info: t.Union[None, AccountInfo] = None
    _cached_orders: t.List[t.Dict] = None

    _TDA_CLIENT: tda.client.Client = tda.auth.easy_client(
        webdriver_func=selenium.webdriver.Firefox,
        **credentials.CLIENT_PARAMS
    )
    STREAM_CLIENT: tda.streaming.StreamClient = tda.streaming.StreamClient(
        client=_TDA_CLIENT,
        account_id=_ACCOUNT_ID
    )

    _stream_data = []

    def account_info(cls, cached=False) -> AccountInfo:
        if cached is False or cls._cached_account_info is None:
            resp = LocalClient._TDA_CLIENT.get_account(
                account_id=_ACCOUNT_ID,
                fields=[
                    tda.client.Client.Account.Fields.ORDERS,
                    tda.client.Client.Account.Fields.POSITIONS
                ]
            )
            # dump account data to txt for reference
            account_info_raw = resp.json()
            with open('account_data.json', 'w') as outfile:
                json.dump(account_info_raw, outfile, indent=4)
            cls._cached_account_info = AccountInfo(account_info_raw)
        return cls._cached_account_info

    def orders(cls, status: OrderStatus = None, cached=False):
        if cached is False or cls._cached_orders is None:
            cls._cached_orders = cls._TDA_CLIENT.get_orders_by_path(
                account_id=_ACCOUNT_ID,
                from_entered_datetime=datetime.datetime.utcnow() - datetime.timedelta(days=59),
                status=status
            ).json()
        return cls._cached_orders

    def orders_by_id(cls, status: OrderStatus = None, cached=False):
        return parse_orders(cls.orders(status=status, cached=cached))

    def get_order_data(cls, order_id, cached=False) -> OrderData:
        dir_to_enum = {
            'BUY': Side.LONG,
            'SELL_SHORT': Side.SHORT
        }
        order = cls.orders_by_id(cached=cached)[order_id]
        direction = dir_to_enum[order['orderLegCollection'][0]['instruction']]
        quantity = order['filledQuantity']
        return OrderData(direction=direction, quantity=quantity, stop_loss=None, status=order['status'])

    def place_order_spec(cls, order_spec) -> t.Tuple[int, str]:
        """place order with tda-api order spec, return order id"""
        cls._TDA_CLIENT.place_order(account_id=_ACCOUNT_ID, order_spec=order_spec)
        order_data = cls.orders()[0]
        return order_data['orderId'], order_data['status']

    def flush_orders(cls):
        for order in cls.orders():
            cls._TDA_CLIENT.cancel_order(order_id=order['orderId'], account_id=_ACCOUNT_ID)

    def init_listed_stream(cls):
        """
        stream price data of the given symbols every 500ms
        use this code to execute function: asyncio.run(LocalClient.initiate_stream(<enter symbols here>)
        """
        return configure_stream(
            stream_client=cls.STREAM_CLIENT,
            add_book_handler=cls.STREAM_CLIENT.add_listed_book_handler,
            handlers=[
                lambda msg: cls._stream_data.append(json.dumps(msg, indent=4))
            ],
            book_subs=cls.STREAM_CLIENT.listed_book_subs

        )

    def init_futures_stream(cls):
        return configure_stream(
            stream_client=cls.STREAM_CLIENT,
            add_book_handler=cls.STREAM_CLIENT.add_chart_futures_handler,
            handlers=[
                lambda msg: cls._stream_data.append(json.dumps(msg, indent=4)),
                lambda msg: print(json.dumps(msg, indent=4))
            ],
            book_subs=cls.STREAM_CLIENT.chart_futures_subs
        )


# create td client
class LocalClient(metaclass=_LocalClientMeta):
    cached_account_info: AccountInfo = None

    @classmethod
    def price_history(
            cls,
            symbol: str,
            freq_range: tdargs.FreqRangeArgs,
    ) -> pd.DataFrame:
        """
        # TODO add type hints
        :param symbol:
        :param freq_range:
        :return:
        """
        # get historical data, store as dataframe, convert datetime (ms) to y-m-d-etc
        while True:
            resp = cls._TDA_CLIENT.get_price_history(
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
        df = df[['open', 'high', 'close', 'low']]

        return df


# class LocalStreamClient(metaclass=_LocalClientMeta):
#     """class for streaming related activities"""
#     _STREAM_CLIENT: tda.streaming.StreamClient = tda.streaming.StreamClient(
#         client=_TDA_CLIENT,
#         account_id=_ACCOUNT_ID
#     )


if __name__ == '__main__':
    stream_func = LocalClient.init_futures_stream()
    asyncio.run(stream_func('/MESU21'))
    print('hi')

