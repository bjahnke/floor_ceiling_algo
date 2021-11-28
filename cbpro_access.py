from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from time import time

import pandas as pd
import typing as t

from strategy_utils import Side

import abstract_access
import yfinance_translate as yft
from coinbase_pro.client import get_client, get_messenger
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


_INTERVAL = {
    '1m': 60,
    '5m': 300,
    '15m': 900,
    '1h': 3600,
    '6h': 21600,
    '1d': 86400,
}


class Position(abstract_access.AbstractPosition):
    """
    Note: no trailing stops yet for cb pro. trailing stop will trip per the
    strategy parameters instead
    """
    @classmethod
    def init_from_raw_data(cls, raw_position_data: t.Dict):
        base_currency = 'USD'
        symbol = raw_position_data['currency']
        pair_id = f'{symbol}-{base_currency}'
        return cls(
            symbol=pair_id,
            qty=raw_position_data['balance'],
            side=Side.LONG,  # crypto is long only for now
            raw_position=raw_position_data
        )

    def _stop_order(self):
        """stop losses managed locally"""
        return None
        # return {
        #     # 'profile_id': 'default', # no profile id uses default
        #     'product_id': self._symbol,
        #     'type': 'stop',
        #     'side': 'sell',
        #     'stop': 'loss',
        #     'time_in_force': 'GTC',
        #     'size': self._qty,
        #     'stop_price': self._stop_value
        # }

    def _open(self, quantity):
        return {
            # 'profile_id': 'default', # no profile id uses default
            'product_id': self._symbol,
            'type': 'market',
            'side': 'buy',
            'time_in_force': 'GTC',
            'size': quantity
        }

    def _close(self, quantity):
        return {
            # 'profile_id': 'default', # no profile id uses default
            'product_id': self._symbol,
            'type': 'market',
            'side': 'sell',
            'time_in_force': 'GTC',
            'size': quantity
        }


# class CbproAccount(abstract_access.AbstractBrokerAccount):
class AccountInfo:
    def __init__(self, positions: t.Dict, equity):
        self._positions = positions
        self._equity = equity
    
    @property
    def positions(self) -> t.Dict[str, Position]:
        return self._positions

    @property
    def equtiy(self):
        return self._equity

    def get_symbols(self) -> t.List:
        return list(self._positions.keys())


class _OrderStatus(Enum):
    """
    pending - Pending transactions (e.g. a send or a buy)
    completed - Completed transactions (e.g. a send or a buy)
    failed - Failed transactions (e.g. failed buy)
    expired - Conditional transaction expired due to external factors
    canceled - Transaction was canceled
    waiting_for_signature - Vault withdrawal is waiting for approval
    waiting_for_clearing - Vault withdrawal is waiting to be cleared
    """
    FILLED = 'completed'
    REJECTED = 'failed'
    ORDER_PENDING = 'pending'


@dataclass
class DuckStatus:
    status: _OrderStatus


# class CbproClient(BrokerClient):
class CbproClient:
    OrderStatus = _OrderStatus

    def __init__(self, key: str, secret: str, passphrase: str):
        self.__key = key
        self.__secret = secret
        self.__passphrase = passphrase
        self._stream = None
        self._client = get_client({
            'key': key,
            'secret': secret,
            'passphrase': passphrase,
            'authority': 'https://api.pro.coinbase.com'
        })

        self._alt_messenger = get_messenger({
            'key': key,
            'secret': secret,
            'passphrase': passphrase,
            'authority': 'https://api.coinbase.com'
        })

    @property
    def client(self):
        return self._client

    @property
    def usd_products(self) -> pd.DataFrame:
        all_products = pd.DataFrame(self._client.product.list())
        return all_products[all_products.quote_currency == 'USD']

    def account_info(self, *_, **__):
        """create account info object from most recent api call"""
        base_currency = 'USD'
        active_positions = {}
        equity = 0
        for raw_product_info in self._client.account.list():
            balance = self._product_balance(raw_product_info)
            if balance > 0:
                if raw_product_info['currency'] != base_currency:
                    active_positions[raw_product_info['currency']] = Position.init_from_raw_data(raw_product_info)
                    equity += self._product_balance(raw_product_info)
        equity += self.usd_balance()
        return AccountInfo(active_positions, equity)

    def _accounts(self):
        """retrieves position info for this account for all products on coinbase"""
        return self._client.account.list()

    def _profiles(self):
        return self._client.profile.list()

    def delayed_balance(self) -> float:
        """
        get the total account value in usd via current exchange rates.
        execution time scalable with number of active positions due to max 2
        api calls. But exchange rate data is delayed by an unknown factor
        of time
        """
        # gets current exchange rate of all products in USD (slightly delayed)
        exchange_rates = self._alt_messenger.get('/v2/exchange-rates').json()['data']['rates']
        balance = 0
        for account in self._accounts():
            product_balance = float(account['balance'])
            currency_name = account['currency']
            if currency_name in exchange_rates:
                balance += product_balance / float(exchange_rates[currency_name])

        balance += self.usd_balance()

        return balance

    def current_balance(self) -> float:
        """
        Get current balance in usd. Api calls = 1 + number of open positions:
        Execution time will be slower if number of active positions is greater than 10/15
        do to api throttling
        :return:
        """
        balance = 0
        for product_info in self._client.account.list():
            balance += self._product_balance(product_info)
        balance += self.usd_balance()
        return balance

    def _product_balance(self, raw_product_info) -> float:
        base = 'USD'
        balance = 0
        product_balance = float(raw_product_info['balance'])
        if product_balance > 0:
            currency_name = raw_product_info['currency']
            pair_id = f'{currency_name}-{base}'
            ticker_data = self._client.product.ticker(pair_id)
            if 'ask' in ticker_data:
                balance += product_balance * float(ticker_data['ask'])
        return balance

    def usd_balance(self) -> float:
        """get the available usd balance for this account"""
        return float(self._client.account.get('ccdf8ad5-cc77-4bad-bc15-6201791d21bf')['available'])

    def price_history(self, symbol: str, interval: int, num_bars: int = None, interval_type='m') -> pd.DataFrame:
        history_params = {
            'granularity': _INTERVAL[f'{interval}{interval_type}'],
            'start': '',
            'end': ''
        }
        if num_bars is not None:
            history_params['start'] = datetime.fromtimestamp(time() - num_bars * interval)
            history_params['end'] = datetime.fromtimestamp(time())
        data = self._client.product.candles(symbol, history_params)

        data = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        data.index = data.time
        data.index = pd.to_datetime(data.index, unit='s')
        data['symbol'] = symbol
        data = data.sort_index()
        return data[['symbol', 'open', 'high', 'close', 'low', 'volume']]

    def extended_price_history(self, symbol: str, interval: int, num_bars: int = None, interval_type='m'):
        """get price history exceeding 300 bar limit imposed by cbpro api"""
        remaining_data = self.price_history(symbol, interval=interval)
        if num_bars >= 300:
            delayed_data, delay = yft.yf_price_history_stream(symbol, interval, num_bars, interval_type=interval_type)
            bars = 1
            while True:
                bars += 1
                start_time = timedelta(seconds=bars * interval)
                if start_time > delay:
                    break

            return pd.concat([delayed_data, remaining_data])

    def place_order_spec(self, order_spec) -> t.Union[t.Tuple[str, _OrderStatus], t.Tuple[None, None]]:
        # TODO return order id
        order_id = None
        status = None
        if order_spec is not None:
            order_resp = self._client.order.post(order_spec)
            order_id = order_resp['id']
            status = _OrderStatus(order_resp['order_resp'])
        return order_id, status

    def cancel_order(self, order_id):
        raise NotImplementedError

    def get_order_data(self, order_id):
        resp = self._client.order.get(order_id)
        return DuckStatus(status=_OrderStatus(resp['Status']))

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
    subscribe_msg = {
        'type': 'subscribe',
        'product_ids': symbols,
        'channels': ['ticker']
    }

    return stream, subscribe_msg


class CbProTickerStream(AbstractTickerStream):
    """
    TODO
        - remove delay fill code
        - give symbol: price_data as input
    """
    def run_stream(self, symbols, stream_generator: t.Callable[[t.List[str]], t.Any], get_data_delays):
        """
        The main loop distributes messages to processes via
        pipes based on the symbol in the message.
        :param get_data_delays:
        :param stream_generator: function returning a stream client
        :param symbols: symbols to stream
        :return:
        """
        symbol_delays = get_data_delays(symbols, self._interval, 5)
        self._init_stream_parsers(symbol_delays)
        msg_queue = self._init_processes(symbols)
        stream, subscribe_msg = stream_generator(symbols)
        last_subscribe = time() - 5
        send_subscribe = True
        while True:
            # if send_subscribe:
            #     last_subscribe = time()
            #     stream.send(subscribe_msg)
            #
            # send_subscribe = time() - last_subscribe > 3

            msg = stream.receive()
            symbol = self.get_symbol(msg)
            if symbol is not None:
                print(msg)
                msg_queue.put(msg)

    @staticmethod
    def get_symbol(msg):
        return msg.get('product_id', None)


class CbProStreamParse(AbstractStreamParser):
    def retrieve_ohlc(self, data: dict):
        """get price from ticker stream"""
        current_sequence = data['sequence']
        if (
            self._prev_sequence is not None
            and current_sequence <= self._prev_sequence
        ):
            return self._quoted_prices

        self._prev_sequence = current_sequence
        return (float(data['price']), ) * 4


# def get_data(symbol):
#     return yft.yf_price_history_stream(symbol, interval=15, days=50)[0]


# async def write_data(values, columns, out_symbol):
#     async with aiofiles.open(f'{out_symbol}.csv', mode='w', encoding='utf-8', newline='') as afp:
#         writer = aiocsv.AsyncWriter(afp)
#         await writer.writerow(columns)
#         await writer.writerows(values)


# def f(args):
#     print(args[0], time()-args[1])


if __name__ == '__main__':
    client = CbproClient(
        key='b4cddf747a1bdeaea5d3b0b0624b7826',
        secret='B/CwVGFOxQldQc6JQDCrFCDdNMKmstasi3j/i7PEomwa4MwQ3gYMOeHzpkej9DSF/wpk58p5Z3zG/sW6WbnMJg==',
        passphrase='ppc7rad3atc'
    )
    new_position = Position(
        'ADA-USD',
        qty=1,
        side=Side.LONG,
        stop_value=0.5
    )
    order = new_position.init_stop_loss(None)
    resp = client.place_order_spec(order)
    print('done')
    # daily_scan = pd.read_excel(r'C:\Users\Brian\OneDrive\algo_data\csv\cbpro_scan_out.xlsx')
    # in_symbols = daily_scan.symbol[daily_scan.score > 1].to_list()[:4]
    # start_time = time()
    #
    # print('running stream')
    # # cbpro_stream = cbpro_init_stream(in_symbols)
    #
    # ticker_stream = CbProTickerStream(
    #     stream=None,
    #     stream_parser=CbProStreamParse,
    #     fetch_price_data=yft.yf_price_history_stream,
    #     quote_file_path='live_quotes.json',
    #     history_path=r'C:\Users\bjahn\PycharmProjects\algo_data',
    #     interval=1
    # )
    # ticker_stream.run_stream(['ADA-USD', 'ETH-USD'], cbpro_init_stream, yft.yf_get_delays)
