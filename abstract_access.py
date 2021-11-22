import json
from abc import ABC, abstractmethod
import typing as t
from datetime import datetime, timedelta
from enum import Enum, auto

from dataclasses import dataclass
from strategy_utils import Side
import pandas as pd
import pd_accessors
import asyncio

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
DATA_FETCH_FUNCTION = t.Callable[[str, int, int, str], t.Tuple[pd.DataFrame, t.Any]]


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


class StreamState(Enum):
    INITIAL = auto()
    WAIT = auto()
    FILL_GAP = auto()
    NORMAL_UPDATE = auto()
    ADD_NEW_BAR = auto()


class AbstractStreamParser(ABC):
    _quoted_prices: OHLC_VALUES
    _price_data: pd.DataFrame
    _fetch_price_data: DATA_FETCH_FUNCTION

    def __init__(
        self,
        symbol,
        fetch_price_data: DATA_FETCH_FUNCTION,
        live_quote_file_path=None,
        history_path='',
        interval: int = 1,
    ):
        self._stream_state = StreamState.INITIAL
        self._stream_init_time = None
        self._target_fetch_time = None
        self._live_quote_file_path = live_quote_file_path
        self._symbol = symbol
        self._fetch_price_data = fetch_price_data
        self._history_file_path = self.__class__._init_price_history_path(history_path, symbol)
        self._interval = interval
        self._bar_data = Bar()
        self._price_data = None
        self._quoted_prices = None
        self._prev_sequence = None

        self._state_table = {
            StreamState.INITIAL: self._init_stream_data,
            StreamState.FILL_GAP: self._allow_fill_data_gap,
            StreamState.NORMAL_UPDATE: self._do_nothing
        }

    @abstractmethod
    def retrieve_ohlc(self, data: dict) -> OHLC_VALUES:
        """get prices from ticker stream"""
        raise NotImplementedError

    @property
    def history_file_path(self):
        return

    def fetch_price_data(self) -> t.Tuple[pd.DataFrame, t.Any]:
        """x days worth of minute data by the give interval"""
        x = 10
        return self._fetch_price_data(self._symbol, self._interval, x, 'm')

    def update_ohlc_state(self, data: t.Dict):
        self._quoted_prices = self.retrieve_ohlc(data)
        time_stamp = datetime.utcnow()

        self._stream_state = self._state_table[self._stream_state](time_stamp)

        self.update_ohlc(self._quoted_prices, time_stamp)

    def _init_stream_data(self, time_stamp) -> StreamState:
        """
        make an initial price history call,
        calculate the amount of time needed for the
        stream to run to close the data gap between
        price history end and stream start
        """
        self._stream_init_time = time_stamp
        self._price_data, delay = self.fetch_price_data()
        next_bar_time = (
            self._stream_init_time.replace(second=0, microsecond=0) +
            timedelta(minutes=self._stream_init_time.minute % self._interval)
        )
        self._target_fetch_time = next_bar_time + delay
        return StreamState.FILL_GAP

    def _allow_fill_data_gap(self, time_stamp) -> StreamState:
        """
        wait for current time to exceed target time,
        get price history again. Now there is no longer a data gap
        """
        next_state = StreamState.FILL_GAP
        if time_stamp > self._target_fetch_time:
            self._price_data, _ = self.fetch_price_data()
            self._price_data.to_csv(self._history_file_path)
            next_state = StreamState.NORMAL_UPDATE
        return next_state

    def _do_nothing(self, time_stamp):
        return self._stream_state

    def update_ohlc(self, quoted_prices: OHLC_VALUES, time_stamp):
        """
        given data for a specific symbol, update the ohlc values
        for given interval
        :param time_stamp:
        :param data:
        :return:
        """
        if time_stamp.minute % self._interval == 0:
            if self._stream_state == StreamState.NORMAL_UPDATE:
                self._add_new_row(time_stamp)
            update_method = self._bar_data.init_new
        else:
            update_method = self._bar_data.update

        update_method(*quoted_prices)

    def _add_new_row(self, time_stamp):
        """add new bar to price data"""
        lag = timedelta(seconds=time_stamp.second, microseconds=time_stamp.microsecond)
        tm = time_stamp - lag  # round out ms avoids appending many rows within the same second
        if tm not in self._price_data.index:
            print(f'{self._symbol} {tm} (lag): {lag}')
            row_data = (self._symbol, ) + self._bar_data.get_ohlc_values()
            self._price_data.loc[tm] = row_data
            self._price_data.reset_index().to_feather(self._history_file_path)

    def get_ohlc(self) -> t.Dict:
        res = self._bar_data.get_ohlc()
        res['symbol'] = self._symbol
        return res

    @staticmethod
    def _init_price_history_path(price_history_path, symbol):
        full_path = f'{symbol}.ftr'
        if len(price_history_path) > 0:
            full_path = f'{price_history_path}\\{full_path}'
        return full_path


class AbstractTickerStream:
    _stream_parser: t.Type[AbstractStreamParser]
    _price_data: t.Union[pd.DataFrame]

    def __init__(
        self,
        stream,
        stream_parser,
        quote_file_path: str,
        history_path: str,
        fetch_price_data: DATA_FETCH_FUNCTION,
        interval: int = 1
    ):
        self._stream = stream
        self._stream_parser = stream_parser
        self._quote_file_path = quote_file_path
        self._history_path = history_path
        self._interval = interval
        self._stream_parsers = {}
        self._live_data_out = {}
        self._fetch_price_data = fetch_price_data
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
                    fetch_price_data=self._fetch_price_data,
                    history_path=self._history_path
                )
        except KeyError:
            pass
        else:
            self._stream_parsers[symbol].update_ohlc_state(msg)
            current_ohlc = self._stream_parsers[symbol].get_ohlc()
            self._live_data_out[symbol] = current_ohlc
            try:
                with open(self._quote_file_path, 'w') as live_quote_file:
                    json.dump(self._live_data_out, live_quote_file, indent=4)
            except PermissionError:
                print(f'Warning JSON permission error')







