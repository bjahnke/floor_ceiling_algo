import numpy as np
import tda
import pandas as pd
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

class TradeState:
    PENDING = -2
    SELL = -1
    CLOSE = 0
    BUY = 1

class Input:
    BUY = 1
    SELL = -1
    CLOSE = pd.NA

TS = TradeState


class AccountManager:
    """
    TODO:
        - dump MDF creation parameters for all symbols to json file
        - dump price history data to excel file
        - when loading active positions from account, load MDF params and excel data
    """

    def __init__(self, **signal_data: pd.DataFrame):
        self._account_info = tda_access.LocalClient.account_info
        self.signal_data = signal_data
        trade_states = init_states(self._account_info.get_symbols(), self.signal_data)
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


def init_states(active_symbols: t.List[str], signal_data: t.Dict[str, pd.DataFrame]) -> _TradeStates:
    """
    symbols not passed to AccountManager will not be used by the algorithm.
    Only symbols with data passed in to create AccountManager instance are
    actively managed by AccountManager.
    :param active_symbols: symbols with active positions on this account
    :param signal_data: input signal data that we still want to trade
    :return:
    """
    active_symbols_local = copy(active_symbols)
    managed = {}

    for symbol, data in signal_data.items():
        managed[symbol] = data
        active_symbols_local.remove(symbol)
    # whatever is left in active_symbols is not managed by the algo
    return _TradeStates(managed=managed, not_managed=active_symbols_local)


class SymbolManager:
    """not sure if this class is necessary"""
    pass
