from abc import ABC, abstractmethod
import typing as t


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











