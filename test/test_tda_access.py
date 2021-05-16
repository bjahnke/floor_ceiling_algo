"""
test tda access functionality
"""

import tda_access


def test_account_info_args():
    acct_info = tda_access.LocalClient.get_account_info()
    cur_balance = acct_info.acct_data_raw['securitiesAccount']['currentBalances']

    assert acct_info.equity == cur_balance['equity']
    assert acct_info.liquid_funds == cur_balance['moneyMarketFund'] + cur_balance['cashBalance']

def test_account_add_remove():
    """Symbols can be added and removed from account"""



if __name__ == '__main__':
    test_account_info_args()
