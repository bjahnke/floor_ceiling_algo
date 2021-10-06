from account_manager import SymbolData, AccountManager
from scanner import yf_price_history


def main():
    try:
        account_manager = AccountManager.load_from_pickle()
    except FileNotFoundError:

        account_manager = AccountManager(
            SymbolData('IDXX', yf_price_history, enter_on_fresh_signal=True),
            SymbolData('RMD', yf_price_history, enter_on_fresh_signal=True),
            SymbolData('EW', yf_price_history, enter_on_fresh_signal=True),
            SymbolData('WAT', yf_price_history, enter_on_fresh_signal=True),
            SymbolData('SHW', yf_price_history, enter_on_fresh_signal=True),
            SymbolData('EXR', yf_price_history, enter_on_fresh_signal=True),
            SymbolData('ABMD', yf_price_history, enter_on_fresh_signal=True),
            SymbolData('MSCI', yf_price_history, enter_on_fresh_signal=True),
        )
    account_manager.run_manager()


if __name__ == '__main__':
    main()


