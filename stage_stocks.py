from typing import List

import tda_access
from account_manager import SymbolData, AccountManager
from scanner import yf_price_history
import pandas as pd
import pd_accessors
import cbpro_access

daily_scan = pd.read_excel(r'C:\Users\bjahn\OneDrive\algo_data\csv\scan_out_15m_200d_hp.xlsx')


def symbol_data_factory(*symbols: str) -> List[SymbolData]:
    res = []
    for symbol in symbols:
        scan_data: pd.DataFrame = daily_scan.scan_data.by_symbol(symbol)
        res.append(
            SymbolData(
                base_symbol=symbol,
                broker_client=tda_access.LocalClient,
                short_ma=int(scan_data.st.iloc[-1]),
                mid_ma=int(scan_data.mt.iloc[-1]),
                enter_on_fresh_signal=True
            )
        )
    return res


def main(min_score: float):
    try:
        account_manager = AccountManager.load_from_pickle()
    except FileNotFoundError:
        symbol_watchlist: pd.Series = daily_scan.symbol[daily_scan.score >= min_score]
        account_manager = AccountManager(
            tda_access.LocalClient,
            *symbol_data_factory(*symbol_watchlist.to_list())
        )
    account_manager.run_manager()


if __name__ == '__main__':
    # print('start')
    # main(min_score=2.5)
    local_client = cbpro_access.CbproClient(
        key='309c01c1e6a9b92e3fd2fb8d933c3c17',
        secret='WXSGBBCq6QSg+moUTEmK7A0jQQX/jmTgDbFosjCj2qLTFjHfeaNLl+cnNmHcpwsU4vVgB2kLpdYZ9NIbFtgaxg==',
        passphrase='mfi7arvg829'
    )
    products = local_client.usd_products
    data = local_client.price_history('GRT-USD', start='2021-9-17', end='2021-10-17', granularity=21600)
    print('done')


