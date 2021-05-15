"""
Classes to ease organization and modification of price history data

TD (or api) only seems to like period for daily bars and slice
for inter-day bars.

TODO Consider the following for implementation
    - *** assert _TimePeriodArgs when frequency is day or greater
    - * inter day data is limited. Request exceeding the limit will only return the
    allowed amount (without any warning). Inform or assert when this happens
    - (maybe)Uniform input so user doesn't have to worry about this silly quirk
    - raise exception on failed request, provide meaningful error
"""
import abc
import back_test_utils as btu
import tdargs
import pandas as pd
import typing as t

class DfMetaData(abc.ABC):
    """
    Metadata wrapper base class for price history Data Frames.
    Storage for useful information relating to how the
    Dataframe was created.
    """
    def __init__(
        self,
        symbol: str,
        freq_range: tdargs.FreqRangeArgs,
    ):
        """
        :param symbol:
        :param freq_range:
        """
        self.symbol = symbol
        self.FREQ_RANGE = freq_range


class PriceMdf:
    """
    Pulls EOD price data from broker based on given input args. Price data
    is stored as dataframe attribute. Input args are stored as attributes
    to provide some meta data about the Data Frame.
    """
    def __init__(
        self,
        symbol: str,
        freq_range: tdargs.FreqRangeArgs,
    ):
        """
        :param symbol: ticker symbol to pull price history on
        :param freq_range:
        """
        self.symbol = symbol
        self.prefix = 'b'
        self.FREQ_RANGE = freq_range
        self.data = btu.LocalClient.price_history(
            symbol=symbol,
            freq_range=freq_range,
        )

    def init_swings(self, argrelwindow: int = 20):
        """
        :param argrelwindow: rolling window to obtain local swings
        :return:
        """
        self.data = btu.swings(
            df=self.data,
            high='high',
            low='low',
            argrelwindow=argrelwindow,
            prefix='sw'
        )

    def init_fc_regime(
        self,
        argrelwindow: int = 20,
        threshold: int = 1.5,
        stdev_window: int = 63
    ):
        """
        :param argrelwindow: rolling window to obtain local swings
        :param threshold: minimum deviation from mean required to print regime
        :param stdev_window: rolling window to calculate standard deviation
        :return:
        """
        self.init_swings(argrelwindow=argrelwindow)
        self.data = btu.regime_fc(
            df=self.data,
            close='close',
            swing_low='sw_low',
            swing_high='sw_high',
            threshold=threshold,
            stdev_window=stdev_window
        )

    def rolling_stdev(self, window=63, min_periods=1, decimals=3, price_col=None):
        stdev = btu.rolling_stdev(
            self.data.close,
            window=window,
            min_periods=min_periods,
            decimals=decimals,
        )
        # TODO create stdev with data index?

    def merge_copy(self) -> pd.DataFrame:
        """
        A copy of self.data with symbol attached to original column names.
        Used for merging with other PriceDf data
        :return:
        """
        return merge_copy(data=self.data, prefix=self.prefix)

    def revert_copy(self, merge_copy_df: pd.DataFrame) -> pd.DataFrame:
        """
        revert column names by removing any prefix that may have been added
        :param merge_copy_df:
        :return:
        """
        return revert_copy(data=merge_copy_df, prefix=self.symbol)

    def merge_reshape(self, other_ph: pd.DataFrame) -> pd.DataFrame:
        """
        :param other_ph:
        :return:
        """
        data_copy = self.merge_copy()
        data_copy_columns = list(data_copy.columns)
        reshaped = other_ph.join(data_copy, how='left')
        reshaped = reshaped[data_copy_columns]
        reshaped = self.revert_copy(reshaped)
        return reshaped


class RelativeMdf(PriceMdf):
    def __init__(
        self,
        base_symbol: str,
        bench_symbol: str,
        freq_range: tdargs.FreqRangeArgs,
        forex_symbol: t.Optional[str] = None,
    ):
        """
        :param bench_symbol: benchmark symbol to grab EOD data for
        :param forex_symbol: forex symbol to grab EOD data for (used if company is foreign)
        :return:
        """
        super().__init__(
            symbol=base_symbol,
            freq_range=freq_range
        )
        relative_data = init_relative(
            self.data,
            bench_symbol=bench_symbol,
            forex_symbol=forex_symbol,
            freq_range=freq_range
        )
        base_cpy = self.merge_copy()
        self.data = relative_data.join(base_cpy)
        self.data = self.data[
            ['open', 'high', 'low', 'close', 'b_close']
        ]
        self.prefix = 'r'

    def init_fc_position_size(
        self,
        tcs: float = 0.0025,
        percentile: float = 0.05,
        minperiods: int = 50,
        window: int = 200,
        limit: int = 5,
        argrelwindow: int = 20,
        threshold: int = 1.5,
        stdev_window: int = 63,
        st_list: range = range(10, 101, 10),
        mt_list: range = range(50, 201, 20),
    ):
        """
        :param tcs: TODO?
        :param percentile: TODO?
        :param minperiods: TODO?
        :param window: TODO?
        :param limit: TODO?
        :param argrelwindow: rolling window to obtain local swings
        :param threshold: minimum deviation from mean required to print regime
        :param stdev_window: rolling window to calculate standard deviation
        :param st_list: TODO generate permutation here or initialize signal
                            > generator in place and score the return of this function
        :param mt_list:
        :return:
        """
        self.init_fc_regime(
            argrelwindow=argrelwindow,
            threshold=threshold,
            stdev_window=stdev_window
        )
        self.data = btu.fc_position_size(
            fc_data=self.data,
            symbol=self.symbol,
            base_close='b_close',
            relative_close='close',
            st_list=st_list,
            mt_list=mt_list,
            tcs=tcs,
            percentile=percentile,
            minperiods=minperiods,
            window=window,
            limit=limit,
            best_risk_adjusted_returns=0
        )[0]

    def init_position_sizes(self):
        pass

def merge_copy(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    A copy of data with prefix appended to column names.
    Used for merging with other PriceDf data
    :return:
    """
    new_names = {column: f'{prefix}_{column}' for column in data.columns}
    renamed_data = data.rename(columns=new_names)
    return renamed_data


def revert_copy(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    revert_names = {col: col.replace(f'{prefix}_', '') for col in data.columns}
    reverted_data = data.rename(columns=revert_names)
    return reverted_data


def init_relative(
    base_price: pd.DataFrame,
    bench_symbol: str,
    freq_range: tdargs.FreqRangeArgs,
    forex_symbol: t.Optional[str] = None
) -> pd.DataFrame:
    """
    :param base_price:
    :param bench_symbol: benchmark symbol to grab EOD data for
    :param freq_range:
    :param forex_symbol: forex symbol to grab EOD data for (used if company is foreign)
    :return:
    """
    bench_mdf = PriceMdf(
        symbol=bench_symbol,
        freq_range=freq_range
    )
    reshaped_forex = None
    if forex_symbol is not None:
        forex_mdf = PriceMdf(
            symbol=forex_symbol,
            freq_range=freq_range
        )
        reshaped_forex = forex_mdf.merge_reshape(base_price)

    reshaped_bench = bench_mdf.merge_reshape(base_price)
    relative_data = btu.relative_series(
        base_price,
        reshaped_bench,
        reshaped_forex
    )
    return relative_data


def position_size(
    equity,  # K
    data: pd.DataFrame,
    constant_risk: float,
    constant_weight: float,
    signal_col: str,
    stop_loss_col: str,

):
    # K = 1000000
    # constant_risk = 0.25 / 100
    # constant_weight = 3 / 100

    # signal = data[data_cols[7]]
    signal = data[signal_col]
    signal[pd.isnull(signal)] = 0

    position = data[signal_col].shift(1)
    position[pd.isnull(position)] = 0
    stop_loss = data[stop_loss_col]

    # Calculate the daily Close chg_1d
    close_1d = data['b_close'].diff().fillna(0)

    # Define posSizer weight
    data['eqty_risk'] = btu.equity_at_risk(
        px_adj=data['close'], stop_loss=stop_loss, risk=constant_risk)

    # Instantiation of equity curves
    data['equity_at_risk'] = equity
    data['equal_weight'] = equity

    # Instantiate position sizes
    data['eqty_risk_lot'] = 0
    data['equal_weight_lot'] = 0
    # Instantiation of round_lot for posSizer
    eqty_risk_lot = 0
    equal_weight_lot = 0

    for i in range(len(data)):

        data['equity_at_risk'].iat[i] = (
            data['equity_at_risk'].iat[i-1] + close_1d.iat[i] * eqty_risk_lot * abs(position.iat[i])
        )

        data['equal_weight'].iat[i] = (
            data['equal_weight'].iat[i-1] + close_1d.iat[i] * equal_weight_lot * position.iat[i]
        )

        if (signal.iat[i-1] == 0) & (signal.iat[i] != 0):
            eqty_risk_lot = btu.round_lot(
                weight=data['eqty_risk'].iat[i],
                capital=data['equity_at_risk'].iat[i],
                fx_rate=1,
                price_local=data['b_close'].iat[i],
                roundlot=100
            )
            data.eqty_risk_lot.iat[i] = eqty_risk_lot

            equal_weight_lot = btu.round_lot(
                weight=constant_weight,
                capital=data['equal_weight'].iat[i],
                fx_rate=1,
                price_local=data['b_close'].iat[i],
                roundlot=100
            )
            data.equal_weight_lot.iat[i] = equal_weight_lot

        else:
            pass


if __name__ == '__main__':
    mdf = RelativeMdf(
        base_symbol='GME',
        bench_symbol='SPX',
        freq_range=tdargs.freqs.day.range(tdargs.periods.y2)
    )
    mdf.init_fc_position_size()
    columns = list(mdf.data.columns)
    position_size(
        equity=1000000,
        constant_risk=0.25/100,
        constant_weight=3/100,
        signal_col=columns[7],
        stop_loss_col=columns[8],
        data=mdf.data
    )
    mdf.data.to_excel('out.xlsx')
    print(mdf.data)
