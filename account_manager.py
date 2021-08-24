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


# =======
# lib function
# =======


def get_minimum_freq(date_times: pd.Index) -> Timedelta:
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
    _data: t.Union[None, pd.DataFrame]
    _update_price_history: t.Callable[[], pd.DataFrame]
    _bar_freq: t.Union[None, Timedelta]

    def __init__(
        self,
        base_symbol: str,
        bench_symbol: str,
        freq_range=tdargs.freqs.day.range(tdargs.periods.y2)
    ):
        self._name = base_symbol
        self._bench_symbol = bench_symbol
        self._freq_range = freq_range
        self._data = None
        self._bar_freq = None

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
        new_data = False
        if self._data is None:
            self._data = self.fetch_data()
            self._bar_freq = get_minimum_freq(self._data.index)
            new_data = True
        elif self.update_ready:
            current_data = self._data.index[-1]
            self._data = self.fetch_data()
            if self._data.index[-1] != current_data:
                new_data = True
            else:
                print('price history called, but no new data')
        return new_data

    def fetch_data(self):
        return fc_data_gen.init_fc_data(
            base_symbol=self._name,
            bench_symbol=self._bench_symbol,
            equity=tda_access.LocalClient.account_info().equity,
            freq_range=self._freq_range
        )

    def parse_signal(self) -> tda_access.OrderData:
        """get order data from current bar"""
        current_bar = self._data.iloc[-1]
        return tda_access.OrderData(
            direction=tda_access.Side(current_bar.signal),
            # for eqty_risk_lot, short position lot will be negative if valid.
            # so multiply by signal to get the true lot
            quantity=current_bar.eqty_risk_lot * current_bar.signal,
            # TODO usage of base price stop loss is temporary until
            #   relative stop loss is implemented with data streaming
            stop_loss=current_bar.stop_loss_base
        )


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
        self.account_data = tda_access.LocalClient.account_info(cached=True)
        self.trade_state = self._init_trade_state()
        self.order_id = None
        self.stop_order_id = None

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
        self.trade_state = state_lookup[self.trade_state]()

    def _init_trade_state(self) -> SymbolState:
        """initialize current trade state of this symbol"""
        # TODO how unlikely is it that order is pending during initialization?
        state = SymbolState.REST
        if self.account_data.positions.get(self.symbol_data.name, None) is not None:
            state = SymbolState.FILLED
        return state

    def filled(self):
        new_trade_state = self.trade_state
        if self.symbol_data.update_data() is True:
            order_data = self.symbol_data.parse_signal()
            position = tda_access.LocalClient.account_info().positions.get(self.symbol_data.name, None)
            if position is None:
                # if no position found for this symbol,
                # stop loss was triggered or position closed externally
                new_trade_state = SymbolState.REST
            elif tda_access.Side(order_data.direction) != position.side:
                position.full_close()
                self.order_id = None
                self.stop_order_id = None
                new_trade_state = SymbolState.REST
            else:
                new_trade_state = SymbolState.FILLED
        return new_trade_state

    def rest(self):
        new_trade_state = self.trade_state
        if self.symbol_data.update_data():
            order_data = self.symbol_data.parse_signal()
            order_lambda = tda_access.OPEN_ORDER.get(
                order_data.direction,
                None
            )
            # no order template corresponding to the current signal val means trade signal not given
            if order_lambda is not None:
                self.order_id, order_status = tda_access.LocalClient.place_order_spec(
                    order_lambda(self.symbol_data.name, order_data.quantity)
                )
                stop_loss_lambda = tda_access.OPEN_STOP[tda_access.Side(order_data.direction)]
                self.stop_order_id = tda_access.LocalClient.place_order_spec(
                    stop_loss_lambda(
                        sym=self.symbol_data.name,
                        qty=order_data.quantity,
                        stop_price=order_data.stop_loss
                    )
                )
                # if order already filled, we can used the cached account info to place stop loss order,
                # saves us an additional API call
                new_trade_state = self.order_pending(
                    tda_access.LocalClient.get_order_data(order_id=self.order_id, cached=True)
                )
            else:
                new_trade_state = SymbolState.REST
        return new_trade_state

    def order_pending(self, order_info: tda_access.OrderData = None):
        # sourcery skip: lift-return-into-if
        """resolve the status of the current order (self.order_id is id of the current order)"""
        if order_info is None:
            order_info = tda_access.LocalClient.get_order_data(order_id=self.order_id)

        if order_info.status == OrderStatus.FILLED:
            new_trade_state = SymbolState.FILLED
        elif order_info.status == OrderStatus.REJECTED:
            new_trade_state = SymbolState.ERROR
        else:
            new_trade_state = SymbolState.ORDER_PENDING
        return new_trade_state

    def error(self):
        """do nothing, remain in error state"""


class AccountManager:
    managed: t.List[SymbolManager]
    """
    TODO:
        - dump MDF creation parameters for all symbols to json file
        - dump price history data to excel file
        - when loading active positions from account, load MDF params and excel data
    """

    def __init__(self, *signal_data: SymbolData):
        self._account_info = tda_access.LocalClient.account_info()
        self.signal_data = signal_data
        trade_states = init_states(self._account_info.get_symbols(), self.signal_data)
        # symbols AccountManager is actively trading
        self.managed = trade_states.managed
        # symbols we have active positions in but are not being managed by AccountManager for this run-time
        self.not_managed = trade_states.not_managed
        # self.reload_meta_dfs()
        self.run_manager()

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
            for symbol_manager in self.managed:
                symbol_manager.update_trade_state()


def init_states(active_symbols: t.List[str], symbol_data: t.Iterable[SymbolData]) -> _TradeStates:
    """
    symbols not passed to AccountManager will not be used by the algorithm.
    Only symbols with data passed in to create AccountManager instance are
    actively managed by AccountManager.
    :param active_symbols: symbols with active positions on this account
    :param symbol_data: input signal data that we still want to trade
    :return:
    """
    active_symbols_local = copy(active_symbols)
    managed = []

    for data in symbol_data:
        managed.append(SymbolManager(data))
        if data.name in active_symbols_local:
            active_symbols_local.remove(data.name)
    # positions remaining in active symbols will not be touch by the algo
    return _TradeStates(managed=managed, not_managed=active_symbols_local)


if __name__ == '__main__':
    da = SymbolData('GPRO', 'SPX', tdargs.freqs.day.range(tdargs.periods.y2))
    current_bar_data = da.data.iloc[-1]
    signal = current_bar_data.signal
    stop_loss = current_bar_data.stop_loss
    position_size = current_bar_data.eqty_risk_lot

    print('done.')

