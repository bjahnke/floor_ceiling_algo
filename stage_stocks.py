from typing import List

import tda_access
from account_manager import SymbolData, AccountManager
from scanner import yf_price_history
import pandas as pd

# import pd_accessors
# import cbpro_access

daily_scan = pd.read_excel(r"C:\Users\bjahn\OneDrive\algo_data\csv\scan_out.xlsx")


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
                enter_on_fresh_signal=True,
            )
        )
    return res


def main(min_score: float, load_pickle=True):
    account_manager = None
    if load_pickle:
        try:
            account_manager = AccountManager.load_from_pickle()
        except FileNotFoundError:
            pass
    if account_manager is None:
        symbol_watchlist: pd.Series = daily_scan.symbol[daily_scan.score > min_score]
        account_manager = AccountManager(
            tda_access.LocalClient, *symbol_data_factory(*symbol_watchlist.to_list())
        )
    account_manager.run_manager()


if __name__ == "__main__":
    print("start")
    main(min_score=1, load_pickle=False)
