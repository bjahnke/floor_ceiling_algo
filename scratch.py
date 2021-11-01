import pandas as pd

from cbpro_access import CbproStream
from time import sleep, time, strftime
from datetime import datetime, timedelta
from pprint import pprint
import typing as t
import json


def init_stream(symbols: t.List[str]):
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


def ticker_stream(symbols: t.List[str], file_path: str, interval: int = 1):
    stream = init_stream(symbols)
    start_time = time()
    stream_parsers = dict()
    live_data_out = dict()
    while True:
        data = stream.receive()
        try:
            symbol = data['product_id']
            if stream_parsers.get(symbol, None) is None:
                stream_parsers[symbol] = TickerStreamParse(symbol)
        except KeyError:
            pass
        else:
            stream_parsers[symbol].update_ohlc(data)
            current_ohlc = stream_parsers[symbol].get_ohlc()
            live_data_out[symbol] = current_ohlc
            try:
                with open(file_path, 'w') as live_quote_file:
                    json.dump(live_data_out, live_quote_file)
            except PermissionError:
                pass


class TickerStreamParse:

    def __init__(self, symbol, interval: int = 1):
        self._symbol = symbol
        self._prev_sequence = None
        self._price = None
        self._open = None
        self._high = None
        self._low = None
        self._close = None
        self._interval = interval
        self._price_data = pd.DataFrame(columns=['open', 'high', 'low', 'close'])

    def get_price(self, data: dict):
        """get price from ticker stream"""
        current_sequence = data['sequence']
        if self._prev_sequence is None or current_sequence > self._prev_sequence:
            self._prev_sequence = current_sequence
            price = float(data['price'])
        else:
            price = self._price

        return price

    def update_ohlc(self, data: dict):
        self._price = self.get_price(data)
        y, mo, d, h, m, s = map(int, strftime("%Y %m %d %H %M %S").split())
        if m % self._interval == 0 and s == 0:
            self._add_new_row()
            self._open = None
            self._high = None
            self._low = None
            self._close = None

        self._open = self._price if self._open is None else self._open
        self._low = self._price if self._low is None else min(self._low, self._price)
        self._high = self._price if self._high is None else max(self._high, self._price)
        self._close = self._price

    def _add_new_row(self):
        """add new bar to price data"""
        tm = datetime.utcnow()
        tm = tm - timedelta(seconds=tm.second % 1, microseconds=tm.microsecond)
        if tm not in self._price_data.index:
            new_row = pd.DataFrame(
                [[self._open, self._high, self._low, self._close]],
                columns=self._price_data.columns.to_list(),
                index=[tm]
            )
            self._price_data = pd.concat([self._price_data, new_row])

    def get_ohlc(self):
        return {
            'symbol': self._symbol,
            'open': self._open,
            'high': self._high,
            'low': self._low,
            'close': self._close
        }


def crypto_stream(symbols: t.List[str], interval: int = None):
   pass


if __name__ == '__main__':
    print('running stream')
    ticker_stream(['ADA-USD', 'ETH-USD'], file_path=r'live_quotes.json', interval=15)
    # print(datetime.utcnow() - timedelta(minutes=15))
