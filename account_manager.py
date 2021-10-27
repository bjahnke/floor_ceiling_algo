from __future__ import annotations
import pickle
from datetime import datetime, timedelta
from enum import Enum, auto
from time import perf_counter

import httpx
from matplotlib import pyplot as plt

import back_test_utils
import pandas as pd
from copy import copy

import tda.client
from pandas import Timedelta

import fc_data_gen
import pd_accessors
import tda_access
import typing as t
import tdargs
from collections import namedtuple

from strategy_utils import SignalStatus
from tda_access import Side
from scanner import yf_price_history

OrderStatus = tda.client.Client.Order.Status

_TradeStates = namedtuple('TradeStates', 'managed not_managed')


# =======
# lib function
# =======


class Input:
    BUY = 1
    SELL = -1
    CLOSE = pd.NA


class SymbolData:
    """"""
    _name: str
    _bench_data: t.Union[None, pd.DataFrame]
    _update_price_history: t.Callable[[], pd.DataFrame]
    _bar_freq: t.Union[None, Timedelta]
    _valid_signals: t.List[SignalStatus]

    def __init__(
        self,
        base_symbol: str,
        short_ma: int,
        mid_ma: int,
        broker_client,
        bench_symbol: t.Union[str, None] = None,
        freq_range=tdargs.freqs.day.range(tdargs.periods.y2),
        enter_on_fresh_signal=False
    ):
        self._broker_client = broker_client
        self._name = base_symbol
        self._bench_symbol = bench_symbol
        self._freq_range = freq_range
        self._bench_data = None
        self._bar_freq = None
        self._ENTER_ON_FRESH_SIGNAL = enter_on_fresh_signal
        self._short_ma = short_ma
        self._mid_ma = mid_ma
        self.error_log = []
        self._account_data = None
        self._cached_data = None
        self._valid_signals = self._init_valid_signals()

    def _init_valid_signals(self):
        valid_signals = [
            SignalStatus.SHORT,
            SignalStatus.LONG,
            SignalStatus.NEW_SHORT,
            SignalStatus.NEW_LONG,
            SignalStatus.NEW_CLOSE,
            SignalStatus.CLOSE
        ]
        if self._ENTER_ON_FRESH_SIGNAL:
            valid_signals = [
                SignalStatus.NEW_SHORT,
                SignalStatus.NEW_LONG,
                SignalStatus.NEW_CLOSE,
                SignalStatus.CLOSE
            ]
        return valid_signals

    @property
    def broker_client(self):
        return self._broker_client

    @property
    def account_data(self):
        return self._account_data

    @account_data.setter
    def account_data(self, value):
        self._account_data = value

    @property
    def name(self):
        return self._name

    def get_price_data(self):
        # new_data = yf_price_history(symbol=self._name)
        try:
            new_data = self._broker_client.price_history(
                symbol=self._name,
                freq_range=tdargs.freqs.m15.range(left_bound=datetime.utcnow() - timedelta(days=30))
            )
            self._cached_data = new_data
            # print(f'{self._name}^ {perf_counter()}')
        except (tda_access.EmptyDataError, tda_access.FaultReceivedError):
            # new_data = self._cached_data
            new_data = None

        if self._bench_symbol is not None:
            # TODO update to use tda price history
            self._bench_data = yf_price_history(symbol=self._bench_symbol)

        return new_data

    def get_current_signal(self) -> tda_access.OrderData:
        # new_data = yf_price_history(symbol=self._name)
        new_data = self.get_price_data()
        if new_data is None:
            return tda_access.OrderData.no_signal()

        try:
            analyzed_data, stats = fc_data_gen.init_fc_data(
                base_symbol=self._name,
                price_data=new_data,
                equity=self._account_data.equity,
                st_list=self._short_ma,
                mt_list=self._mid_ma,
                # TODO pass in broker to symbol manager. req account_info().equity in AbstractClient.AccountInfo
            )
        except (ValueError, TypeError):
            raise
        except Exception as ex:
            if str(ex) not in self.error_log:
                print(str(ex))
                self.error_log.append(str(ex))
            order_data = tda_access.OrderData.no_signal()
        else:
            self._cached_data = analyzed_data
            if analyzed_data.signals.status in self._valid_signals:
                order_data = tda_access.OrderData(
                    name=self._name,
                    direction=Side(analyzed_data.signals.current),
                    # quantity=current_bar.eqty_risk_lot * current_bar.signal,
                    quantity=1,
                    stop_loss=analyzed_data.stop_loss_base.iloc[-1]
                )
            else:
                order_data = tda_access.OrderData.no_signal()

            # back_test_utils.graph_regime_fc(
            #     ticker=self._name,
            #     df=analyzed_data,
            #     y='close',
            #     th=1.5,
            #     sl='sw_low',
            #     sh='sw_high',
            #     clg='ceiling',
            #     flr='floor',
            #     st=analyzed_data['st_ma'],
            #     mt=analyzed_data['mt_ma'],
            #     bs='regime_change',
            #     rg='regime_floorceiling',
            #     bo=200
            # )
            # try:
            #     plt.savefig(rf'C:\Users\temp\OneDrive\algo_data\png\live_trade\{self._name}.png', bbox_inches='tight')
            # except Exception as e:
            #     print(e)

        return order_data


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
    _status: str

    def __init__(self, symbol_data: SymbolData):
        self.symbol_data = symbol_data
        self._broker_client = symbol_data.broker_client
        # TODO pass in broker to symbol manager. req abstract AccountInfo param cached: bool (or **kwarg)
        self.account_data = self._broker_client.account_info(cached=True)
        self.trade_state = self._init_trade_state()
        self.order_id = None
        self.stop_order_id = None
        self._status = 'OKAY'
        self._current_signal = None

        self._STATE_LOOKUP = {

            SymbolState.REST: self.rest,
            SymbolState.FILLED: self.filled,
            SymbolState.ORDER_PENDING: self.order_pending,
            SymbolState.ERROR: self.error
        }

    @property
    def entry_bar(self):
        """
        get entry time via order id,
        get bar
        """
        return

    def update_trade_state(self):
        """update trade state"""
        initial_trade_state = self.trade_state
        self.trade_state = self._STATE_LOOKUP[initial_trade_state]()
        if self.trade_state != initial_trade_state:
            print(f'{self.symbol_data.name}: {initial_trade_state} -> {self.trade_state}')

    def _init_trade_state(self) -> SymbolState:
        """initialize current trade state of this symbol"""
        # TODO how unlikely is it that order is pending during initialization?
        state = SymbolState.REST

        # if there is an open position, it has passed the rest state already
        if (position_data := self.account_data.positions.get(self.symbol_data.name, None)) is not None:
            # order_data = tda_access.OrderData(
            #     name=self.symbol_data.name,
            #     direction=position_data.side,
            #     quantity=position_data.qty,
            # )
            state = SymbolState.FILLED
        return state

    def filled(self) -> SymbolState:
        new_trade_state = SymbolState.FILLED
        current_signal = self.symbol_data.get_current_signal()
        new_order = current_signal.close_order_spec

        position = self._broker_client.account_info().positions.get(self.symbol_data.name, None)
        if position is None:
            # if no position found for this symbol,
            # stop loss was triggered or position closed externally
            new_trade_state = SymbolState.REST
        # TODO elif new_order is not None and current_signal.direction != position.side:
        elif new_order is not None and current_signal.direction != position.side:
            self.order_id, order_status = self._broker_client.place_order_spec(position.full_close())
            # TODO retrieve stop order id from TDA order log if hard reset occurs
            self._broker_client.cancel_order(self.stop_order_id)
            self.order_id = None
            self.stop_order_id = None
            self._current_signal = None
            new_trade_state = SymbolState.REST

        return new_trade_state

    def rest(self) -> SymbolState:
        new_trade_state = SymbolState.REST
        self._current_signal = self.symbol_data.get_current_signal()
        order_spec = self._current_signal.open_order_spec

        # no order template corresponding to the current signal val means trade signal not given
        if order_spec is not None:
            self.order_id, order_status = self._broker_client.place_order_spec(order_spec)
            self._current_signal.status = OrderStatus(order_status)
            new_trade_state = SymbolState.ORDER_PENDING
        return new_trade_state

    def order_pending(self) -> SymbolState:
        # sourcery skip: lift-return-into-if
        """
        resolve the status of the current order
        set stop loss if status is filled
        """
        order_data = self._broker_client.get_order_data(self.order_id)
        if order_data.status == OrderStatus.FILLED:
            # must wait for open order to fill before setting stop,
            # otherwise it will cancel the initial order
            self.stop_order_id = self._broker_client.place_order_spec(
                self._current_signal.stop_order_spec
            )
            new_trade_state = SymbolState.FILLED
        elif order_data.status == OrderStatus.REJECTED:
            new_trade_state = SymbolState.ERROR
        else:
            new_trade_state = SymbolState.ORDER_PENDING
        return new_trade_state

    def error(self) -> SymbolState:
        """
        TODO: wait until current signal is None, then set state to REST
        """
        
        return SymbolState.ERROR


class AccountManager:
    managed: t.List[SymbolManager]

    FILE_PATH = '.\\pkl_data\\account_manager.pkl'
    """
    TODO:
        - dump MDF creation parameters for all symbols to json file
        - dump price history data to excel file
        - when loading active positions from account, load MDF params and excel data
    """

    def __init__(self, broker_client, *signal_data: SymbolData):
        # TODO abstract out LocalClient: req AbstractClass containing account_info()
        self._broker_client = broker_client
        self._account_info = self._broker_client.account_info()
        self.signal_data = signal_data
        trade_states = init_states(self._account_info.get_symbols(), self.signal_data)
        # symbols AccountManager is actively trading
        self.managed = trade_states.managed
        for symbol_state in self.managed:
            symbol_state.symbol_data.account_data = self._account_info
        # symbols we have active positions in but are not being managed by AccountManager for this run-time
        self.not_managed = trade_states.not_managed
        # self.reload_meta_dfs()

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
        timeouts = []
        while True:
            for symbol_manager in self.managed:
                try:
                    symbol_manager.update_trade_state()
                    self.to_pickle()
                except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ReadError) as e:
                    print(e)
                    if str(e) not in timeouts:
                        timeouts.append(e)

    def to_pickle(self):
        with open(self.__class__.FILE_PATH, 'wb') as file_handler:
            pickle.dump(self, file_handler)

    @classmethod
    def load_from_pickle(cls) -> AccountManager:
        with open(cls.FILE_PATH, 'rb') as file_handler:
            account_manager = pickle.load(file_handler)
        return account_manager


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
    print('done.')

