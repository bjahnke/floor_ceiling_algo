"""
test tda access functionality
"""
import pprint

from tda.orders.equities import equity_buy_market, equity_sell_market
from tda.client import Client
import tda_access as ta
import pandas as pd


COS = Client.Order.Status


def test_account_info_args():
    acct_info = ta.LocalClient.account_info
    cur_balance = acct_info.acct_data_raw['securitiesAccount']['currentBalances']
    assert acct_info.equity == cur_balance['equity']
    assert acct_info.liquid_funds == cur_balance['moneyMarketFund'] + cur_balance['cashBalance']


def test_order():
    buy_order = equity_buy_market('GPRO', 1)
    ta.LocalClient.submit_order(buy_order)
    print('done.')


if __name__ == '__main__':
    # acct_orders = ta.LocalClient.account_info.acct_data_raw['securitiesAccount']['orderStrategies']
    query_orders = ta.LocalClient.orders()
    print('a')

