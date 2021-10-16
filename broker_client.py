from abc import ABC, abstractmethod


class BrokerClient(ABC):

    @abstractmethod
    def account_info(self):
        raise NotImplementedError

    @abstractmethod
    def price_history(self, symbol, freq_range):
        raise NotImplementedError

    @abstractmethod
    def place_order_spec(self):
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self):
        raise NotImplementedError

    @abstractmethod
    def get_order_data(self):
        raise NotImplementedError




