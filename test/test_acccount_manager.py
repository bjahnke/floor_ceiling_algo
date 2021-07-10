"""
account_manager.py tests
"""
import account_manager as am

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
    acct_info = tda_access.LocalClient.get_account_info()
    acct_info.stage('')