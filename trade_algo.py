import tda_access
import trade_df


def temp_trade_algo(account_info: tda_access.AccountInfo, signal_data: trade_df.RelativeMdf):
    current_position = account_info.get_position(signal_data.symbol)




