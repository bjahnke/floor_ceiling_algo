from datetime import datetime
from enum import Enum, auto

import pandas as pd
from copy import copy

import tda.client
from pandas import Timestamp, Timedelta

import fc_data_gen
import tda_access
import typing as t
import tdargs
from collections import namedtuple
from tda_access import AccountInfo

OrderStatus = tda.client.Client.Order.Status

_TradeStates = namedtuple('TradeStates', 'managed not_managed')
_OrderData = namedtuple('_Signal', ['direction', 'quantity', 'stop_loss'])


# =======
# lib function
# =======


def get_minimum_freq(date_times: t.Iterable[Timestamp]) -> Timedelta:
    """
    get the minimum frequency across a series of timestamps.
    Used to determine frequency of a series while taking into
    account larger than normal differences in bar times due to
    weekends and holiday
    :param date_times:
    """
    minimum = datetime.today() - date_times[-10]
    for i, date in enumerate(date_times):
        if date == date_times[-1]:
            break
        current_diff = date_times[i + 1] - date
        minimum = min(minimum, current_diff)

    return minimum





class Input:
    BUY = 1
    SELL = -1
    CLOSE = pd.NA


class SymbolData:
    _name: str
    _data: pd.DataFrame
    _update_price_history: t.Callable[[], pd.DataFrame]
    _bar_freq: Timedelta

    def __init__(
        self,
        base_symbol: str,
        bench_symbol: str,
        freq_range=tdargs.freqs.day.range(tdargs.periods.y2)
    ):
        self._name = base_symbol
        self._bench_symbol = bench_symbol
        self._freq_range = freq_range
        self._data = fc_data_gen.init_fc_data(
            base_symbol=self._name,
            bench_symbol=self._bench_symbol,
            equity=tda_access.LocalClient.account_info.equity,
            freq_range=self._freq_range
        )
        self._bar_freq = get_minimum_freq(self._data.index)

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._data

    @property
    def bar_freq(self):
        return self._bar_freq

    def update_ready(self):
        """check if new bar is ready to be retrieved, prevents redundant API calls"""
        return (datetime.now() - self._data.index[-1]) > self._bar_freq

    def update_data(self) -> bool:
        """attempt to get new price history, update strategy"""
        current_data = self._data.index[-1]
        new_data = False
        if self.update_ready:
            self._data = fc_data_gen.init_fc_data(
                base_symbol=self._name,
                bench_symbol=self._bench_symbol,
                equity=tda_access.LocalClient.account_info.equity,
                freq_range=self._freq_range
            )
            if self._data.index[-1] != current_data:
                new_data = True
            else:
                print('price history called, but no new data')
        return new_data

    def parse_signal(self) -> _OrderData:
        """get order data from current bar"""
        current_bar = self._data.iloc[-1]
        return _OrderData(
            direction=signal,
            # for eqty_risk_lot, short position lot will be negative if valid.
            # so multiply by signal to get the true lot
            quantity=position_size * current_bar.signal,
            stop_loss=self._data.stop_loss
        )


class AccountManager:
    managed: t.List[SymbolData]
    """
    TODO:
        - dump MDF creation parameters for all symbols to json file
        - dump price history data to excel file
        - when loading active positions from account, load MDF params and excel data
    """

    def __init__(self, *signal_data: SymbolData):
        self._account_info = tda_access.LocalClient.account_info
        self.signal_data = signal_data
        trade_states = init_states(self._account_info.get_symbols(), self.signal_data)
        # symbols AccountManager is actively trading
        self.managed = trade_states.managed
        # symbols we have active positions in but are not being managed by AccountManager for this run-time
        self.not_managed = trade_states.not_managed
        # self.reload_meta_dfs()
        self.run_manager()

    def update_managed_symbols(self):
        """
        :return:
        """
        updated_symbols = []
        for symbol_data in self.managed:
            symbol_data.update_data()

    def run_manager(self):
        """
        TODO:
            1.) update date for staged and active symbols
            3.) for each staged symbol that updated:
                - (create order if force trade flag is true?) else:
                - get signal and abs(position size) from most recent row
                - if position size is not NaN, create order (1=buy, -1=sell) with position size
                    -
        :return:
        """
        while True:
            for symbol, data in self.managed.items():
                position_info = self._account_info.positions.get(symbol, None)
                current_signal = data.signals.current
                # TODO if order is pending for this symbol?
                #   pending orders visible in account info?
                if position_info is not None:
                    if current_signal != position_info.side:
                        # TODO probably close position, even if somehow -1 to 1 or vice versa
                        pass
                    pass
                elif current_signal in [Input.BUY, Input.SELL]:
                    # TODO attempt to open a position

                    pass


def init_states(active_symbols: t.List[str], signal_data: t.Iterable[SymbolData]) -> _TradeStates:
    """
    symbols not passed to AccountManager will not be used by the algorithm.
    Only symbols with data passed in to create AccountManager instance are
    actively managed by AccountManager.
    :param active_symbols: symbols with active positions on this account
    :param signal_data: input signal data that we still want to trade
    :return:
    """
    active_symbols_local = copy(active_symbols)
    managed = []

    for data in signal_data:
        managed.append(data)
        active_symbols_local.remove(data.name)
    # positions remaining in active symbols will not be touch by the algo
    return _TradeStates(managed=managed, not_managed=active_symbols_local)


class SymbolState(Enum):
    REST = auto()
    ORDER_PENDING = auto()
    FILLED = auto()
    ERROR = auto()


class SymbolManager:
    symbol_data: SymbolData
    account_data: tda_access.AccountInfo
    trade_state: SymbolState
    order_id: t.Union[int, None]

    def __init__(self, symbol_data: SymbolData):
        self.symbol_data = symbol_data
        self.account_data = tda_access.LocalClient.account_info
        self.trade_state = self._init_trade_state()
        self.order_id = None

    @property
    def entry_bar(self):
        """
        get entry time via order id,
        get bar
        """
        return

    def update_trade_state(self):
        """update trade state"""
        state_lookup = {
            SymbolState.FILLED: self.filled,
            SymbolState.REST: self.rest,
            SymbolState.ORDER_PENDING: self.order_pending,
            SymbolState.ERROR: self.error
        }
        state_lookup[self.trade_state]()

    def _init_trade_state(self) -> SymbolState:
        """initialize current trade state of this symbol"""
        # TODO how unlikely is it that order is pending during initialization?
        state = SymbolState.REST
        if self.account_data.positions.get(self.symbol_data.name, default=None) is not None:
            state = SymbolState.FILLED
        return state

    def filled(self):
        if self.symbol_data.update_data() is True:
            order_data = self.symbol_data.parse_signal()
            position = tda_access.LocalClient.account_info.positions.get(self.symbol_data.name, None)
            if position is None:
                # if no position found for this symbol,
                # stop loss was triggered or position closed externally
                self.trade_state = SymbolState.REST
            elif tda_access.Side(order_data.direction) != position.side:
                position.full_close()
                self.trade_state = SymbolState.REST
            else:
                """remain in current state"""

    def rest(self):
        if self.symbol_data.update_data():
            order_data = self.symbol_data.parse_signal()
            order_lambda = tda_access.OPEN_ORDER.get(
                tda_access.Side(order_data.direction),
                None
            )
            # no order template corresponding to the current signal val means trade signal not given
            if order_lambda is not None:
                self.order_id = tda_access.LocalClient.place_order_spec(
                    order_lambda(self.symbol_data.name, order_data.quantity)
                )
                # TODO implement stop_loss
                # stop_loss_lambda
                # stop_loss_id = tda_access.LocalClient.place_order_spec(
                #     order_lambda(self.)
                # )

                self.order_pending(tda_access.LocalClient.cached_account_info)
            else:
                """no signal, remain in current state"""

    def order_pending(self, account_info=None):
        """resolve the status of the current order (self.order_id is id of the current order)"""
        if account_info is None:
            order_status = tda_access.LocalClient.account_info.orders[self.order_id]
        else:
            order_status = account_info.orders[self.order_id]

        if order_status == OrderStatus.FILLED:
            self.trade_state = SymbolState.FILLED
        elif order_status == OrderStatus.REJECTED:
            self.trade_state = SymbolState.ERROR
        else:
            self.trade_state = SymbolState.ORDER_PENDING

    def error(self):
        """do nothing, remain in error state"""


if __name__ == '__main__':
    data = SymbolData('GPRO', 'SPX', tdargs.freqs.day.range(tdargs.periods.y2))
    current_bar_data = data.data.iloc[-1]
    signal = current_bar_data.signal
    stop_loss = current_bar_data.stop_loss
    position_size = current_bar_data.eqty_risk_lot

    print('done.')

