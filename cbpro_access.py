import pandas as pd
import typing as t
from coinbase_pro.client import get_client
from coinbase_pro.socket import get_stream
from abstract_access import AbstractStreamParser, AbstractTickerStream


class CbproStream:
    def __init__(self, key: str, secret: str, passphrase: str):
        self._stream = get_stream({
            'key': key,
            'secret': secret,
            'passphrase': passphrase,
            'authority': 'wss://ws-feed.pro.coinbase.com'
        })

    def init_stream(self, symbols):
        pass

    @property
    def stream(self):
        return self._stream


# class CbproClient(BrokerClient):
class CbproClient:

    def __init__(self, key: str, secret: str, passphrase: str):
        self.__key = key
        self.__secret = secret
        self.__passphrase = passphrase
        self._stream = None
        self._client = get_client({
            'key': key,
            'secret': secret,
            'passphrase': passphrase,
            'authority': 'wss://ws-feed.pro.coinbase.com'
        })

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
        assert granularity in {60, 300, 900, 3600, 21600, 86400}
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


def cbpro_init_stream(symbols: t.List[str]):
    stream = CbproStream(
        key='b4cddf747a1bdeaea5d3b0b0624b7826',
        secret='B/CwVGFOxQldQc6JQDCrFCDdNMKmstasi3j/i7PEomwa4MwQ3gYMOeHzpkej9DSF/wpk58p5Z3zG/sW6WbnMJg==',
        passphrase='ppc7rad3atc'
    ).stream

    stream.connect()
    stream.send({
        'type': 'subscribe',
        'product_ids': symbols,
        'channels': ['ticker']
    })

    # receive response data before collecting stream data
    sub_data = stream.receive()

    return stream


class CbProTickerStream(AbstractTickerStream):
    def run_stream(self):
        while True:
            msg = self._stream.receive()
            self.handle_stream(msg)

    @staticmethod
    def get_symbol(msg):
        return msg['product_id']


class CbProStreamParse(AbstractStreamParser):
    def retrieve_ohlc(self, data: dict):
        """get price from ticker stream"""
        current_sequence = data['sequence']
        if (
            self._prev_sequence is not None
            and current_sequence <= self._prev_sequence
        ):
            return self._prices

        self._prev_sequence = current_sequence
        return (float(data['price']), ) * 4


if __name__ == '__main__':
    print('running stream')
    cbpro_stream = cbpro_init_stream(['ADA-USD', 'ETH-USD'])
    CbProTickerStream(
        stream=cbpro_stream,
        stream_parser=CbProStreamParse,
        quote_file_path='live_quotes.json',
        history_file_path='live_data.csv',
        interval=1
    ).run_stream()
