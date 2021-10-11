from typing import List

from account_manager import SymbolData, AccountManager
from scanner import yf_price_history
import pandas as pd
import pd_accessors

daily_scan = pd.read_excel(r'C:\Users\temp\OneDrive\algo_data\csv\scan_out_hp.xlsx')


def symbol_data_factory(*symbols: str) -> List[SymbolData]:
    res = []
    for symbol in symbols:
        scan_data = daily_scan.scan_data.by_symbol(symbol)
        res.append(
            SymbolData(
                symbol,
                yf_price_history,
                short_ma=scan_data.st,
                mid_ma=scan_data.mt,
                enter_on_fresh_signal=True
            )
        )
    return res


def main():
    try:
        account_manager = AccountManager.load_from_pickle()
    except FileNotFoundError:
        account_manager = AccountManager(
            *symbol_data_factory(*[
                'IDXX',
                'RMD',
                'MSCI',
                'EW',
                'ZTS',
                'TTWO',
                'XYL',
                'EXR',
                'PSX',
                'WAT',
                'DVA',
                'SHW',
                'UHS',
                'HCA',
                'AON',
                'ADM',
                'KEYS',
                'ULTA',
                'NEE',
                'WLTW',
                'CERN',
                'COG',
                'ORCL',
            ])
        )
    account_manager.run_manager()


if __name__ == '__main__':
    main()


