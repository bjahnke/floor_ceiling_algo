from account_manager import SymbolData, AccountManager
from scanner import yf_price_history
import pickle
import typing as t


def main():
    try:
        account_manager = AccountManager.load_from_pickle()
    except FileNotFoundError:
        account_manager = AccountManager(
            SymbolData('BCTX', yf_price_history),
            SymbolData('CRVS', yf_price_history),
            SymbolData('EEIQ', yf_price_history),
            SymbolData('LZ', yf_price_history),
            SymbolData('EAR', yf_price_history),
            SymbolData('NCTY', yf_price_history),
        )
    account_manager.run_manager()


if __name__ == '__main__':
    main()


