from __future__ import annotations
import pickle
from enum import Enum, auto

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
    _data: t.Union[None, pd.DataFrame]
    _bench_data: t.Union[None, pd.DataFrame]
    _update_price_history: t.Callable[[], pd.DataFrame]
    _bar_freq: t.Union[None, Timedelta]
    _fc_kwargs: dict

    def __init__(
        self,
        base_symbol: str,
        fetch_data_function: t.Callable,
        short_ma: int,
        mid_ma: int,
        bench_symbol: t.Union[str, None] = None,
        brokerage_client=None,
        freq_range=tdargs.freqs.day.range(tdargs.periods.y2),
        market_type: t.Optional[tda.client.Client.Markets] = tda.client.Client.Markets.EQUITY,
        enter_on_fresh_signal=False
    ):
        self._name = base_symbol
        self._bench_symbol = bench_symbol
        self._freq_range = freq_range
        self._fetch_data = fetch_data_function
        self._data = None
        self._bench_data = None
        self._bar_freq = None
        self.MARKET = market_type
        self._ENTER_ON_FRESH_SIGNAL = enter_on_fresh_signal
        self._short_ma = short_ma
        self._mid_ma = mid_ma

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._data

    def update_ready(self):
        """check if new bar is ready to be retrieved, prevents redundant API calls"""
        return self._data.update_check.is_ready(self.MARKET, self._bar_freq)

    def get_current_signal(self) -> tda_access.OrderData:
        new_data = yf_price_history(symbol=self._name)
        if self._bench_symbol is not None:
            self._bench_data = yf_price_history(symbol=self._bench_symbol)
        try:
            analyzed_data, stats = fc_data_gen.init_fc_data(
                base_symbol=self._name,
                price_data=new_data,
                equity=None,
                st_list=self._short_ma,
                mt_list=self._mid_ma,
                # TODO pass in broker to symbol manager. req account_info().equity in AbstractClient.AccountInfo
            )
        except:
            order_data = tda_access.OrderData(
                name=self._name,
                direction=Side.CLOSE,
                quantity=0
            )
        else:
            # current bar is the last closed bar which is prior to the current bar
            current_bar = analyzed_data.iloc[-2]
            current_signal = Side(current_bar.signal)
            order_data = tda_access.OrderData(
                name=self._name,
                direction=current_signal,
                quantity=current_bar.eqty_risk_lot * current_bar.signal,
                stop_loss=current_bar.stop_loss_base
            )
            # TODO this code should probably be in SymbolManager somehow
            #   possibly need to merge the 2 classes
            if self._ENTER_ON_FRESH_SIGNAL:
                prior_bar_signal = Side(analyzed_data.iloc[-3].signal)
                if Side.CLOSE != current_signal == prior_bar_signal:
                    order_data = tda_access.OrderData(
                        name=self._name,
                        direction=current_signal,
                        quantity=0
                    )
        return order_data

    def clear_stored_fc_args(self):
        """"""
        self._fc_kwargs = dict()


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
        # TODO pass in broker to symbol manager. req abstract AccountInfo param cached: bool (or **kwarg)
        self.account_data = tda_access.LocalClient.account_info(cached=True)
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
        self.trade_state = self._STATE_LOOKUP[self.trade_state]()

    def _init_trade_state(self) -> SymbolState:
        """initialize current trade state of this symbol"""
        # TODO how unlikely is it that order is pending during initialization?
        state = SymbolState.REST
        if self.account_data.positions.get(self.symbol_data.name, None) is not None:
            state = SymbolState.FILLED
        return state

    def filled(self) -> SymbolState:
        new_trade_state = SymbolState.FILLED
        current_signal = self.symbol_data.get_current_signal()
        position = tda_access.LocalClient.account_info().positions.get(self.symbol_data.name, None)
        if position is None:
            # if no position found for this symbol,
            # stop loss was triggered or position closed externally
            new_trade_state = SymbolState.REST
        elif current_signal.direction != position.side:
            self.order_id, order_status = tda_access.LocalClient.place_order_spec(position.full_close())
            self.order_id = None
            self.stop_order_id = None
            self._current_signal = None
            self.symbol_data.clear_stored_fc_args()
            new_trade_state = SymbolState.REST

        return new_trade_state

    def rest(self) -> SymbolState:
        new_trade_state = SymbolState.REST
        self._current_signal = self.symbol_data.get_current_signal()
        order_spec = self._current_signal.open_order_spec

        # no order template corresponding to the current signal val means trade signal not given
        if order_spec is not None:
            self.order_id, order_status = tda_access.LocalClient.place_order_spec(order_spec)
            self._current_signal.status = OrderStatus(order_status)
            new_trade_state = SymbolState.ORDER_PENDING
        return new_trade_state

    def order_pending(self) -> SymbolState:
        # sourcery skip: lift-return-into-if
        """
        resolve the status of the current order
        set stop loss if status is filled
        """
        if self._current_signal.status == OrderStatus.FILLED:
            # must wait for open order to fill before setting stop,
            # otherwise it will cancel the initial order
            self.stop_order_id = tda_access.LocalClient.place_order_spec(
                self._current_signal.close_order_spec
            )
            new_trade_state = SymbolState.FILLED
        elif self._current_signal.status == OrderStatus.REJECTED:
            new_trade_state = SymbolState.ERROR
        else:
            new_trade_state = SymbolState.ORDER_PENDING
        return new_trade_state

    def error(self) -> SymbolState:
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

    def __init__(self, *signal_data: SymbolData):
        # TODO abstract out LocalClient: req AbstractClass containing account_info()
        self._account_info = tda_access.LocalClient.account_info()
        self.signal_data = signal_data
        trade_states = init_states(self._account_info.get_symbols(), self.signal_data)
        # symbols AccountManager is actively trading
        self.managed = trade_states.managed
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
        while True:
            for symbol_manager in self.managed:
                symbol_manager.update_trade_state()
                self.to_pickle()

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
    da = SymbolData('GPRO', 'SPX', tdargs.freqs.day.range(tdargs.periods.y2))
    current_bar_data = da.data.iloc[-1]
    signal = current_bar_data.signal
    stop_loss = current_bar_data.stop_loss
    position_size = current_bar_data.eqty_risk_lot

    print('done.')

