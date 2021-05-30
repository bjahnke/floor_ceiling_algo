import tda
from copy import copy
import trade_df
import tda_access
import typing as t
import tdargs
from collections import namedtuple
from enum import Enum

_TradeStates = namedtuple('TradeStates', 'managed not_managed')

# =======
# lib function
# =======

class TradeState(Enum):
    SELL = -1
    CLOSE = 0
    BUY = 1


def enter_long_position(symbol: str, quantity: int):
    """
    TODO:
        - Confirm order success
        - Set stop loss
    :param symbol:
    :param quantity:
    :return:
    """
    tda.orders.equities.equity_buy_market(symbol, quantity)


def exit_long_position(symbol: str, quantity: int):
    tda.orders.equities.equity_sell_market(symbol, quantity)


def enter_short_position(symbol: str, quantity: int):
    tda.orders.equities.equity_sell_short_market(symbol, quantity)


def exit_short_position(symbol: str, quantity: int):
    tda.orders.equities.equity_buy_to_cover_market(symbol, quantity)


TS = TradeState


class AccountManager:
    """
    TODO:
        - dump MDF creation parameters for all symbols to json file
        - dump price history data to excel file
        - when loading active positions from account, load MDF params and excel data
    """
    account_info: tda_access.AccountInfo

    def __init__(self, signal_data: t.List[trade_df.RelativeMdf]):
        self.account_info = tda_access.LocalClient.get_account_info()
        self.signal_data = {stock_signals.symbol: stock_signals for stock_signals in signal_data}
        trade_states = init_states(self.account_info.get_symbols(), self.signal_data)
        # symbols AccountManager is actively trading
        self.managed = trade_states.managed
        # symbols we have active positions in but are not being managed by AccountManager for this run-time
        self.not_managed = trade_states.not_managed
        # self.reload_meta_dfs()
        # TODO is it pointless to use enum?
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
        for symbol, symbol_manager in self.managed.items():
            symbol_manager.update_state()


def init_states(active_symbols: t.List[str], signal_data: t.Dict[str, trade_df.RelativeMdf]) -> _TradeStates:
    """
    Only symbols with data passed in to create AccountManager instance are
    actively managed by AccountManager.
    :param active_symbols: symbols with active positions on this account
    :param signal_data: input signal data that we still want to trade
    :return:
    """
    active_symbols_local = copy(active_symbols)
    managed = {}

    for symbol, data in signal_data.items():
        managed[symbol] = SymbolManager(symbol, signal_data[symbol])
        active_symbols_local.remove(symbol)

    return _TradeStates(managed=managed, not_managed=active_symbols_local)


class SymbolManager:
    trade_states = {
        (TS.CLOSE, TS.BUY): enter_long_position,
        (TS.CLOSE, TS.SELL): enter_short_position,
        (TS.BUY, TS.CLOSE): exit_long_position,
        (TS.SELL, TS.CLOSE): exit_short_position,
        (TS.BUY, TS.BUY): lambda *args: None,
        (TS.SELL, TS.SELL): lambda *args: None,
        (TS.CLOSE, TS.CLOSE): lambda *args: None,
        (TS.BUY, TS.SELL): lambda *args: None,
        (TS.SELL, TS.BUY): lambda *args: None,
    }

    def __init__(self, symbol: str, symbol_mdf: trade_df.RelativeMdf):
        # TODO closing has to be based off
        self.symbol = symbol
        self.mdf = symbol_mdf

    def update_data(self):
        pass

    def get_state(self) -> t.Tuple[TradeState, TradeState]:
        int_state = self.mdf.get_signal()
        return TradeState(int_state[0]), TradeState(int_state[1]),

    def update_state(self):
        """
        TODO
            - get current state
            - pass current state into trade_states
        :return:
        """
        SymbolManager.trade_states[self.get_state()]
        pass
