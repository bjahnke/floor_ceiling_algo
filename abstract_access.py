import json
from abc import ABC, abstractmethod
import typing as t
from datetime import datetime, timedelta
from enum import Enum, auto

from dataclasses import dataclass
from strategy_utils import Side
import pandas as pd
import pd_accessors

@dataclass
class Condition:
    case: t.Callable[[t.Any], bool]
    result: t.Any


Condition(
    case=lambda x: x > 0,
    result='buy'
)

Condition(
    case=lambda x: x < 0,
    result='buy'
)


class AbstractBrokerClient(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def account_info(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def price_history(self, symbol, freq_range):
        raise NotImplementedError

    @abstractmethod
    def place_order_spec(self, order_spec):
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id):
        raise NotImplementedError

    @abstractmethod
    def get_order_data(self):
        raise NotImplementedError

    @abstractmethod
    def init_position(self, symbol, quantity, side):
        raise NotImplementedError


class AbstractBrokerAccount(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def positions(self) -> t.Dict:
        raise NotImplementedError

    @abstractmethod
    def get_symbols(self):
        raise NotImplementedError


class AbstractOrders(ABC):
    @abstractmethod
    def _long_open(self, ):
        pass

    @abstractmethod
    def _long_close(self):
        pass

    @abstractmethod
    def _short_open(self):
        pass

    @abstractmethod
    def _short_close(self):
        pass


class ReSize(Enum):
    INC = auto()
    DEC = auto()


class AbstractPosition(ABC):
    _raw_position: t.Dict
    _symbol: str
    _qty: float
    _side: Side
    _stop_value = float

    def __init__(self, symbol, qty, side, raw_position=None, stop_value=None, data_row=None):
        self._symbol = symbol
        self._qty = qty
        self._side = Side(side)
        self._raw_position = raw_position
        self._stop_value = stop_value
        self._data_row = data_row
        self._stop_type = None

    @property
    def side(self):
        return self._side

    @property
    def qty(self):
        return self._qty

    def set_size(self, new_qty: int):
        """
        if new_qty != current quantity,
        execute an order for this position to attain
        the desired size
        :return:
        """
        assert new_qty >= 0, 'input quantity must be >= 0'
        order_spec = None
        size_delta = new_qty - self._qty
        if size_delta != 0:
            self._qty = new_qty
            trade_qty = abs(size_delta)
            if size_delta > 0:
                """increase position size"""
                order_spec = self._open(quantity=trade_qty)
            else:
                """decrease position size"""
                order_spec = self._close(quantity=trade_qty)
        return order_spec

    def init_stop_loss(self, stop_type) -> t.Union[t.Callable]:
        order = None
        if self._qty > 0:
            self._stop_type = stop_type
            order = self._stop_order()
        return order

    @abstractmethod
    def _stop_order(self) -> t.Callable:
        raise NotImplementedError

    @abstractmethod
    def _open(self, quantity) -> t.Union[t.Callable, None]:
        raise NotImplementedError

    @abstractmethod
    def _close(self, quantity) -> t.Union[t.Callable, None]:
        raise NotImplementedError

    def open_order(self) -> t.Union[t.Callable, None]:
        return self._open(self.qty) if self._qty > 0 else None

    def full_close(self):
        """fully close the position"""
        self._close(quantity=self._qty)


OHLC_VALUES = t.Tuple[float, float, float, float]


class CsvPermissionError(Exception):
    """PermissionError raised when attempting to read/write from price history csv"""


class LiveQuotePermissionError(Exception):
    """PermissionError raised when attempting to read/write from price history csv"""


class Bar:
    def __init__(self):
        self._open = None
        self._high = None
        self._low = None
        self._close = None

    def init_new(self, o, h, l, c):
        self._open = o
        self._high = h
        self._low = l
        self._close = c

    def update(self, o, h, l, c):
        try:
            self._low = min(self._low, l)
            self._high = max(self._high, h)
            self._close = c
        except TypeError:
            self.init_new(o, h, l, c)

    def get_ohlc_values(self) -> OHLC_VALUES:
        return self._open, self._high, self._low, self._close

    def get_ohlc(self):
        return {
            'open': self._open,
            'high': self._high,
            'low': self._low,
            'close': self._close
        }


class AbstractStreamParser(ABC):
    _prices: OHLC_VALUES

    def __init__(
        self,
        symbol,
        live_quote_file_path=None,
        price_history_file_path=None,
        interval: int = 1
    ):
        self._live_quote_file_path = live_quote_file_path
        self._price_history_file_path = price_history_file_path

        self._symbol = symbol
        self._interval = interval
        self._bar_data = Bar()
        self._price_data = pd.DataFrame().price_data.init()
        self._prices = None
        self._prev_sequence = None

    @abstractmethod
    def retrieve_ohlc(self, data: dict) -> OHLC_VALUES:
        """get prices from ticker stream"""
        raise NotImplementedError

    def update_ohlc(self, data: dict):
        """
        given data for a specific symbol, update the ohlc values
        for given interval
        :param data:
        :return:
        """
        self._prices = self.retrieve_ohlc(data)
        tm = datetime.utcnow()
        if tm.minute % self._interval == 0 and tm.second < 5:
            self._add_new_row()
            update_method = self._bar_data.init_new
        else:
            update_method = self._bar_data.update

        update_method(*self._prices)

    def _add_new_row(self):
        """add new bar to price data"""
        now = datetime.utcnow()
        lag = timedelta(seconds=now.second, microseconds=now.microsecond)
        tm = now - lag  # round out ms avoids appending many rows within the same second
        if tm not in self._price_data.index:
            print(f'{self._symbol} {tm} (lag): {lag}')
            row_data = (self._symbol, ) + self._bar_data.get_ohlc_values()
            new_row = pd.DataFrame(
                [row_data],
                columns=self._price_data.columns.to_list(),
                index=[tm]
            )
            new_row.to_csv(self._price_history_file_path, mode='a', header=False)
            self._price_data = pd.concat([self._price_data, new_row])

    def get_ohlc(self) -> t.Dict:
        res = self._bar_data.get_ohlc()
        res['symbol'] = self._symbol
        return res


class AbstractTickerStream:
    stream_parser: t.Type[AbstractStreamParser]

    def __init__(
        self,
        stream,
        stream_parser,
        quote_file_path: str,
        history_file_path: str,
        interval: int = 1
    ):
        self._stream = stream
        self._stream_parser = stream_parser
        self._quote_file_path = quote_file_path
        self._history_file_path = history_file_path
        self._interval = interval
        self._stream_parsers = {}
        self._live_data_out = {}
        self._price_data = pd.DataFrame().price_data.init()
        self._price_data.to_csv(self._history_file_path)

        self._permission_error_count = 0

    @abstractmethod
    def run_stream(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_symbol(msg) -> str:
        raise NotImplementedError

    def handle_stream(self, msg):
        """handles the messages, translates to ohlc values, outputs to json and csv"""
        # start_time = time()
        try:
            # TODO get symbol via interface
            symbol = self.__class__.get_symbol(msg)
            if symbol not in self._stream_parsers:
                self._stream_parsers[symbol] = self._stream_parser(
                    symbol,
                    interval=self._interval,
                )
        except KeyError:
            pass
        else:
            self._stream_parsers[symbol].update_ohlc(msg)
            current_ohlc = self._stream_parsers[symbol].get_ohlc()
            self._live_data_out[symbol] = current_ohlc
            try:
                with open(self._quote_file_path, 'w') as live_quote_file:
                    json.dump(self._live_data_out, live_quote_file, indent=4)
            except PermissionError:
                print(f'Warning JSON permission error')








