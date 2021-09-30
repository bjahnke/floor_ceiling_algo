from account_manager import SymbolData, AccountManager
from scanner import yf_price_history
import pickle
import typing as t


def main():
    try:
        account_manager = AccountManager.load_from_pickle()
    except FileNotFoundError:
        account_manager = AccountManager(
            SymbolData('BCTX', yf_price_history, enter_on_fresh_signal=True),
            SymbolData('CRVS', yf_price_history, enter_on_fresh_signal=True),
            SymbolData('EEIQ', yf_price_history, enter_on_fresh_signal=True),
            SymbolData('LZ', yf_price_history, enter_on_fresh_signal=True),
            SymbolData('EAR', yf_price_history, enter_on_fresh_signal=True),
            SymbolData('NCTY', yf_price_history, enter_on_fresh_signal=True),
            SymbolData('XYL', yf_price_history, enter_on_fresh_signal=True),
        )
    account_manager.run_manager()


if __name__ == '__main__':
    main()


