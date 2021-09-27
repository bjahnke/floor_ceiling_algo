from account_manager import SymbolData, AccountManager
from scanner import yf_price_history
import pickle
import typing as t


def main():
    try:
        account_manager = AccountManager.load_from_pickle()
    except FileNotFoundError:
        account_manager = AccountManager(
            SymbolData('TCACU', yf_price_history),
            SymbolData('BCTX', yf_price_history),
            SymbolData('CRVS', yf_price_history),
            SymbolData('HIIU', yf_price_history),
        )
    account_manager.run_manager()


if __name__ == '__main__':
    main()


