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
from __future__ import annotations
import abc
from copy import copy

import back_test_utils as btu
import tdargs
import pandas as pd
import typing as t
import matplotlib.pyplot as plt
from dotmap import DotMap
from datetime import datetime

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
        self.param_log = DotMap({
            self.symbol: {
                'freq_range': self.symbol
            }
        })


class PriceMdf:
    """
    Pulls EOD price data from broker based on given input args. Price data
    is stored as dataframe attribute. Input args are stored as attributes
    to provide some meta data about the Data Frame.
    """
    prefix: str
    data: pd.DataFrame
    FREQ_RANGE: tdargs.FreqRangeArgs

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
        try:
            self.data = btu.swings(
                df=self.data,
                high='high',
                low='low',
                argrelwindow=argrelwindow,
                prefix='sw'
            )
        except ValueError as err:
            # pass along the symbol that swing failed to calculate
            err.args += (self.symbol,)
            raise err
        return self

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
        return self

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
    """
    TODO
        - add @classmethod for loading data from pickle
        - Need to know what crossover i was using??
    """
    # optimize api call limit by only getting bench data once
    bench_data = {}

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
        self.param_log = DotMap()
        self.param_log.init_relative.base_symbol = base_symbol
        self.param_log.init_relative.bench_symbol = bench_symbol
        self.param_log.init_relative.forex_symbol = forex_symbol
        self.param_log.init_relative.freq_range = freq_range

        super().__init__(
            symbol=base_symbol,
            freq_range=freq_range
        )
        bench_mdf = RelativeMdf.bench_data.get(bench_symbol, None)
        if bench_mdf is None:
            bench_mdf = PriceMdf(
                symbol=bench_symbol,
                freq_range=freq_range
            )
            RelativeMdf.bench_data[bench_symbol] = bench_mdf

        relative_data = init_relative(
            self.data,
            bench_mdf=bench_mdf,
            forex_symbol=forex_symbol,
            freq_range=freq_range
        )

        base_cpy = self.merge_copy()
        self.data = relative_data.join(base_cpy)
        self.data = self.data[
            ['open', 'high', 'low', 'close', 'b_close']
        ]
        self.prefix = 'r'

    def init_fc_signal_stoploss(
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
        try:
            result = btu.init_fc_signal_stoploss(
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
        except AttributeError:
            raise

        my_cols = self.data.columns.to_list()
        res_cols = result.columns.to_list()
        self.data = result.join(
            self.data[['regime_change']]
        )
        return self

    def init_position_size(
        self,
        equity: float,
        round_lot: int = 1,
        constant_risk: float = 0.25/100,
        constant_weight: float = 3/100,
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
        # store all input params
        params = copy(locals())
        for key, value in params.items():
            self.param_log.init_position_size[key] = value

        self.init_fc_signal_stoploss(
            tcs=tcs,
            percentile=percentile,
            minperiods=minperiods,
            window=window,
            limit=limit,
            argrelwindow=argrelwindow,
            threshold=threshold,
            stdev_window=stdev_window,
            st_list=st_list,
            mt_list=mt_list,
        )
        cols = self.data.columns.to_list()
        self.data = init_position_size(
            data=self.data,
            equity=equity,
            round_lot=round_lot,
            constant_risk=constant_risk,
            constant_weight=constant_weight,
            signal_col=cols[7],  # TODO rename column 'signal', store signal info in meta data
            stop_loss_col=cols[8],  # TODO rename column 'stop_loss', store signal info in meta data
        )

        return self

    def get_updated_data(self) -> t.Union[RelativeMdf, None]:
        try:
            new_data = RelativeMdf(
                **self.param_log.init_relative
            )
            new_data.init_position_size(
                **self.param_log.init_position_size
            )
        # TODO what exception would throw first?
        except:
            new_data = None
        return new_data

    def ready_for_next_bar(self):
        return (self.data.index[-1] - self.data.index[-2]) - (datetime.now() - self.data.index[-1]) <= 0

    def get_signal(self) -> (int, int):
        """return last 2 rows of signal column"""
        # TODO signal and stop loss column needs to be given generic name
        cols = self.data.columns.to_list()
        signal_col = self.data.cols[7]
        return signal_col[-1], signal_col[-2]


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
    bench_mdf: PriceMdf,
    freq_range: tdargs.FreqRangeArgs,
    forex_symbol: t.Optional[str] = None
) -> pd.DataFrame:
    """
    :param base_price:
    :param bench_mdf: benchmark symbol to grab EOD data for
    :param freq_range:
    :param forex_symbol: forex symbol to grab EOD data for (used if company is foreign)
    :return:
    """

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
    if pd.isna(relative_data.close[-1]):
        relative_data = relative_data[:-1]
    return relative_data


def init_position_size(
    data: pd.DataFrame,
    equity: float,  # K
    constant_risk: float,
    constant_weight: float,
    signal_col: str,
    stop_loss_col: str,
    round_lot: int,
) -> pd.DataFrame:
    """
    :param round_lot:
    :param equity: total value of account
    :param data:
    :param constant_risk:
    :param constant_weight:
    :param signal_col:
    :param stop_loss_col:
    :return:
    """
    data_cpy = data.copy()
    # K = 1000000
    # constant_risk = 0.25 / 100
    # constant_weight = 3 / 100

    # signal = data[data_cols[7]]
    signal = data_cpy[signal_col]
    signal[pd.isnull(signal)] = 0

    position = data_cpy[signal_col].shift(1)
    position[pd.isnull(position)] = 0
    stop_loss = data_cpy[stop_loss_col]

    # Calculate the daily Close chg_1d
    close_1d = data_cpy['b_close'].diff().fillna(0)

    # Define posSizer weight
    data_cpy['eqty_risk'] = btu.equity_at_risk(
        px_adj=data_cpy['close'], stop_loss=stop_loss, risk=constant_risk)

    # Instantiation of equity curves
    data_cpy['equity_at_risk'] = equity
    data_cpy['equal_weight'] = equity

    # Instantiate position sizes
    data_cpy['eqty_risk_lot'] = 0
    data_cpy['equal_weight_lot'] = 0
    # Instantiation of round_lot for posSizer
    eqty_risk_lot = 0
    equal_weight_lot = 0

    for i in range(len(data_cpy)):
        # abs because sign of eqty_risk_lot determines long short? TODO but why not abs the risk lot instead?
        EAR_calc = data_cpy['equity_at_risk'].iat[i - 1] + close_1d.iat[i] * eqty_risk_lot * abs(position.iat[i])
        data_cpy['equity_at_risk'].iat[i] = EAR_calc

        EW_calc = data_cpy['equal_weight'].iat[i - 1] + close_1d.iat[i] * equal_weight_lot * position.iat[i]
        data_cpy['equal_weight'].iat[i] = EW_calc

        if (signal.iat[i-1] == 0) & (signal.iat[i] != 0):
            eqty_risk_lot = btu.round_lot(
                weight=data_cpy['eqty_risk'].iat[i],
                capital=data_cpy['equity_at_risk'].iat[i],
                fx_rate=1,
                price_local=data_cpy['b_close'].iat[i],
                roundlot=round_lot
            )
            data_cpy.eqty_risk_lot.iat[i] = eqty_risk_lot

            equal_weight_lot = btu.round_lot(
                weight=constant_weight,
                capital=data_cpy['equal_weight'].iat[i],
                fx_rate=1,
                price_local=data_cpy['b_close'].iat[i],
                roundlot=round_lot
            )
            data_cpy.equal_weight_lot.iat[i] = equal_weight_lot
        else:
            pass
    return data_cpy


if __name__ == '__main__':
    mdf = RelativeMdf(
        base_symbol='GME',
        bench_symbol='SPX',
        freq_range=tdargs.freqs.day.range(tdargs.periods.y2)
    )
    mdf.init_fc_signal_stoploss()
    columns = list(mdf.data.columns)
    init_position_size(
        equity=1000000,
        constant_risk=0.25/100,
        constant_weight=3/100,
        signal_col=columns[7],
        stop_loss_col=columns[8],
        data=mdf.data
    )
    mdf.data.s90150 = mdf.data.s90150.replace(0, pd.NA)
    try:
        mdf.data.to_excel('out.xlsx')
    except PermissionError:
        pass
    print(mdf.data)

    print(mdf.data.s90150)
    # mdf.data.sl90150.loc[mdf.data.s90150 == pd.NA] = pd.NA
    mdf.data[['close', 'b_close', 's90150', 'sl90150']].plot(
        secondary_y=['s90150'],
        style=['k:', 'k', 'm', 'r']
    )
    plt.show()

