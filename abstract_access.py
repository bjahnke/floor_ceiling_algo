from abc import ABC, abstractmethod
import typing as t
from enum import Enum, auto

from better_abc import abstract_attribute, ABCMeta
from dataclasses import dataclass
from strategy_utils import Side

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

    def __init__(self, symbol, qty, side, raw_position=None, stop_value=None):
        self._symbol = symbol
        self._qty = qty
        self._side = Side(side)
        self._raw_position = raw_position
        self._stop_value = stop_value
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
        self._qty = new_qty
        size_delta = new_qty - self._qty
        if size_delta != 0:
            trade_qty = abs(size_delta)
            if size_delta > 0:
                """increase position size"""
                order_spec = self._open(quantity=trade_qty)
            else:
                """decrease position size"""
                order_spec = self._close(quantity=trade_qty)
        return order_spec

    def init_stop_loss(self, stop_type) -> t.Union[t.Callable]:
        self._stop_type = stop_type
        return self._stop_order()

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
        return self._open(self.qty)

    def full_close(self):
        """fully close the position"""
        self._close(quantity=self._qty)











