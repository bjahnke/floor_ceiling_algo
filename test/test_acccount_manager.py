"""
account_manager.py tests
"""
from typing import List

from pandas import Timestamp, Timedelta

import account_manager as am
import tda_access
from datetime import datetime

import tdargs


def test_manager_init():
    """
    Upon initialization of the account manager, it will retreive
    data from the given account.
    """
    acct_manager = am.AccountManager()
    assert type(acct_manager.account_info) == am.tda_access.AccountInfo
    """
    All positions retrieved from the account will be stored as 'active'
    trade calculations will be performed on the 
    """


def test_account_add_remove():
    """
    Symbols can be staged to or un-staged from the account,
    indicating to the algorithm.
    Staged: algorithm will look for opportunities to make trades
        with this symbol.
    Un-staged: algorithm will cease to look for opportunities to
        trade the symbol after the current position is closed.
    """
    acct_info = tda_access.LocalClient.account_info
    acct_info.stage("")


def scratch_update_check():
    ph = tda_access.LocalClient.price_history(
        "GPRO", freq_range=tdargs.freqs.day.range(tdargs.periods.y2)
    )

    minimum = datetime.today() - ph.index[-10]
    maximum = datetime.today() - datetime.today()
    for i, date in enumerate(ph.index):
        if date == ph.index[-1]:
            break

        current_diff = ph.index[i + 1] - date

        minimum = min(minimum, current_diff)
        maximum = max(maximum, current_diff)

    current_bar = ph.index[-1]
    prev_bar = ph.index[-2]

    check = (datetime.now() - prev_bar) > minimum

    print("Done.")


def test_get_minimum_freq():
    ph = tda_access.LocalClient.price_history(
        "GPRO", freq_range=tdargs.freqs.day.range(tdargs.periods.y2)
    )
    freq = am.get_minimum_freq(ph.index)

    current_bar = ph.index[-1]
    assert (datetime.now() - current_bar) <= freq

    prev_bar = ph.index[-2]
    assert (datetime.now() - prev_bar) > freq


if __name__ == "__main__":
    test_get_minimum_freq()
