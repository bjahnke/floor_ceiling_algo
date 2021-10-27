import pandas as pd
from w3rw.cex.coinbase_pro.messenger import Auth, Messenger
from w3rw.cex.coinbase_pro.client import get_client
from abstract_access import AbstractBrokerClient
from w3rw.cex.coinbase_pro.socket import Stream, Token


class CbproStream:
    def __init__(self, key: str, secret: str, passphrase: str):
        self._stream = Stream(
            auth=Token(
                key=key,
                secret=secret,
                passphrase=passphrase
            )
        )

    def init_stream(self, symbols):
        pass

    @property
    def stream(self):
        return self._stream


# class CbproClient(BrokerClient):
class CbproClient:

    def __init__(self, key: str, secret: str, passphrase: str):
        # super().__init__()
        self.__key = key
        self.__secret = secret
        self.__passphrase = passphrase
        self._stream = None
        self._client = get_client(key=key, secret=secret, passphrase=passphrase)

    @property
    def client(self):
        return self._client

    @property
    def usd_products(self) -> pd.DataFrame:
        all_products = pd.DataFrame(self._client.product.list())
        return all_products[all_products.quote_currency == 'USD']

    def account_info(self):
        raise NotImplementedError

    def price_history(self, symbol: int, start: str, end: str, granularity: int) -> pd.DataFrame:
        assert granularity in [60, 300, 900, 3600, 21600, 86400]
        history_params = {
            'start': start,
            'end': end,
            'granularity': granularity
        }

        data = self._client.product.candles(symbol, history_params)

        data = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        data.time = pd.to_datetime(data.time, unit='s')
        data.index = data.time
        data['b_high'] = data.high
        data['b_low'] = data.low
        data['b_close'] = data.close
        data = data.sort_index()
        return data[['b_high', 'b_low', 'b_close', 'open', 'high', 'close', 'low', 'volume']]

    def place_order_spec(self, order_spec):
        raise NotImplementedError

    def cancel_order(self, order_id):
        raise NotImplementedError

    def get_order_data(self):
        raise NotImplementedError

    def init_stream(self):
        self._stream = CbproStream(
            key=self.__key,
            secret=self.__secret,
            passphrase=self.__passphrase
        )
        return self._stream.stream






