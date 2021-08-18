import tda_access
import trade_df
from account_manager import SymbolData, AccountManager

if __name__ == '__main__':
    AccountManager(
        SymbolData('AFL', 'SPX')
    )
