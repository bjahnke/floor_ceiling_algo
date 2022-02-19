"""
test tda access functionality
"""
import pprint
import typing as t
from datetime import datetime, timedelta

from tda.orders.equities import equity_buy_market, equity_sell_market
from tda.client import Client
import tda_access as ta
import pandas as pd
import tdargs

COS = Client.Order.Status


def test_account_info_args():
    acct_info = ta.LocalClient.account_info
    cur_balance = acct_info.acct_data_raw["securitiesAccount"]["currentBalances"]
    assert acct_info.equity == cur_balance["equity"]
    assert (
        acct_info.liquid_funds
        == cur_balance["moneyMarketFund"] + cur_balance["cashBalance"]
    )


def test_order():
    buy_order = equity_buy_market("GPRO", 1)
    ta.LocalClient.submit_order(buy_order)
    print("done.")


def test_get_order_id():
    res_order_id = ta.LocalClient.place_order_spec(equity_buy_market("GPRO", 1))

    orders: t.List[t.Dict] = ta.LocalClient.orders()
    expected_order_id = orders[0]["orderId"]

    assert res_order_id == expected_order_id


def check_order_details():
    """use with debugger to check composition of order dictionary"""
    orders: t.List[t.Dict] = ta.LocalClient.orders()
    order_details = orders[0]["orderLegCollection"][0]
    direction = order_details["instruction"]
    qty = order_details["quantity"]
    symbol = order_details["instrument"]["symbol"]
    order_id = orders[0]["orderId"]

    print("done")


def test_market_hours():
    res = ta.LocalClient.TDA_CLIENT.get_hours_for_single_market(
        Client.Markets.EQUITY, datetime.now()
    )
    res = res.json()
    data = res["equity"]["EQ"]
    is_open = data["isOpen"]
    market_start = data["sessionHours"]["regularMarket"][0]["start"][:-6]
    market_end = data["sessionHours"]["regularMarket"][0]["end"][:-6]
    mstart_datetime = datetime.strptime(market_start, "%Y-%m-%dT%H:%M:%S")
    print("done")


def test_pickle():
    r1 = pd.read_pickle("new_pickle.pkl")
    res = ta.LocalClient.price_history("AFL", tdargs.freqs.day.range(tdargs.periods.y5))
    res.to_pickle("new_pickle")


if __name__ == "__main__":
    test_pickle()
