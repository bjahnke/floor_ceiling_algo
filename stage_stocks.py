from account_manager import SymbolData, AccountManager
from scanner import yf_price_history
import pandas as pd
import pd_accessors

daily_scan = pd.read_excel(r'C:\Users\temp\OneDrive\algo_data\csv\scan_out_hp.xlsx')


def symbol_data_factory(symbol: str) -> SymbolData:
    scan_data = daily_scan.scan_data.by_symbol(symbol)
    return SymbolData(
        symbol,
        yf_price_history,
        short_ma=scan_data.st,
        mid_ma=scan_data.ma,
        enter_on_fresh_signal=True
    )


def main():
    try:
        account_manager = AccountManager.load_from_pickle()
    except FileNotFoundError:
        account_manager = AccountManager(
            symbol_data_factory('IDXX'),
            symbol_data_factory('RMD'),
            symbol_data_factory('EW'),
            symbol_data_factory('WAT'),
            symbol_data_factory('SHW'),
            symbol_data_factory('EXR'),
            symbol_data_factory('ABMD'),
            symbol_data_factory('MSCI'),
        )
    account_manager.run_manager()


if __name__ == '__main__':
    main()


