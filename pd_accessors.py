import pandas as pd
import numpy as np
from abc import ABCMeta
import typing as t

from pandas import Timedelta

from strategy_utils import Side
import trade_stats
import tda_access
from datetime import datetime, timedelta


def get_minimum_freq(date_times: pd.Index) -> Timedelta:
    """
    get the minimum frequency across a series of timestamps.
    Used to determine frequency of a series while taking into
    account larger than normal differences in bar times due to
    weekends and holiday
    :param date_times:
    """
    minimum = datetime.today() - date_times[-10]
    for i, date in enumerate(date_times):
        if date == date_times[-1]:
            break
        current_diff = date_times[i + 1] - date
        minimum = min(minimum, current_diff)

    return minimum


def _validate(df: pd.DataFrame, mandatory_cols):
    """"""
    col_not_found = [col for col in mandatory_cols if col not in df.columns.to_list()]
    if col_not_found:
        raise AttributeError(f'{col_not_found} expected in DataFrame')


class DfAccessorBase(metaclass=ABCMeta):
    mandatory_cols: t.List[str]

    def __init__(self, df: pd.DataFrame):
        self._validate(df)
        self._obj = df

    @classmethod
    def _validate(cls, df: pd.DataFrame):
        _validate(df, cls.mandatory_cols)


@pd.api.extensions.register_series_accessor('price_opr')
class PriceOperators:
    def __init__(self, series: pd.Series):
        self._obj = series

    def sma(self, ma_per: int, min_per: int, decimals: int):
        return round(
            self._obj
            .rolling(window=ma_per, min_periods=int(round(ma_per * min_per, 0)))
            .mean(),
            decimals
        )

    def sma_vector(
        self,
        sma_pairs: t.List[t.Tuple[int, int]],
        min_per: int = 1,
        decimals: int = 5,
        concat: bool = True
    ) -> t.Union[pd.DataFrame, t.List[pd.Series]]:
        """"""
        deltas = []
        for pair in sma_pairs:
            r_st_ma, r_mt_ma = [
                self._obj.price_opr.sma(ma_per=val, min_per=min_per, decimals=decimals)
                for val in pair
            ]
            name = f's_{pair[0]}_{pair[1]}'
            stmt_delta = np.sign(r_st_ma - r_mt_ma).fillna(0).rename(name)
            deltas.append(stmt_delta)

        if concat:
            deltas = pd.concat(deltas, axis=1)

        return deltas

    def sma_signals_vector(
        self,
        sma_pairs: t.List[t.Tuple[int, int]],
        min_per: int = 1,
        decimals: int = 5,
    ):
        sma_vector = self._obj.price_opr.sma_vector(sma_pairs=sma_pairs, min_per=min_per, decimals=decimals)
        regime_df = pd.DataFrame(self._obj).values
        signals_vector = np.sign(sma_vector * regime_df)
        signals_vector[signals_vector != 1] = np.nan
        return signals_vector * regime_df


@pd.api.extensions.register_dataframe_accessor('swings')
class Swings(DfAccessorBase):
    mandatory_cols = [
        'high',
        'low',
    ]

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)


@pd.api.extensions.register_dataframe_accessor('fc')
class FcData(DfAccessorBase):
    mandatory_cols = [
        'close',
        'b_close',
        'r_return_1d',
        'return_1d',
        'regime_floorceiling',
        'sw_low',
        'sw_high',
        'signal',
        'stop_loss',
        'score',
        'trades',
        'r_perf',
        'csr',
        'geo_GE',
        'sqn',
        'regime_change',
        'equal_weight_lot',
        'eqty_risk',
        'equity_at_risk',
        'equal_weight',
    ]

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    @property
    def signal_starts(self):
        """get all rows where a signal is generated"""
        return self._obj[self._obj.signal.shift(-1).isnull() & self._obj.signal.notnull()]

    @property
    def position_sizes(self):
        """"""
        return None

@pd.api.extensions.register_dataframe_accessor('signals')
class Signals(DfAccessorBase):
    mandatory_cols = [
        'signal'
    ]

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    @property
    def starts(self) -> pd.DataFrame:
        """transition from nan to non-nan means signal has started"""
        return self._obj[self._obj.signal.shift(1).isnull() & self._obj.signal.notnull()]

    @property
    def ends(self) -> pd.DataFrame:
        """transition from non-nan to nan means signal has ended"""
        return self._obj[self._obj.signal.shift(-1).isnull() & self._obj.signal.notnull()]

    def slices(
        self,
        side: Side = None,
        concat: bool = False
    ) -> t.Union[t.List[pd.DataFrame], pd.DataFrame]:
        """

        :param side: filters for trades in the same direction as the given value
        :param concat: for convenience when writing long queries. Will concat slices
                       into a single DataFrame if true
        :return: if concat = False, a list of signals. if concat = True, a single dataframe
                 containing all signals
        """
        # TODO loop iter on DataFrame bad but there should only be small amount of signals
        starts = self.starts
        ends = self.ends
        res = []
        for i, end_date in enumerate(ends.index.to_list()):
            trade_slice = self._obj.loc[starts.index[i]: end_date]
            if (
                side is None or
                Side(trade_slice.signal[0]) == side
            ):
                res.append(trade_slice)

        if concat:
            res = pd.concat(res)

        return res

    @property
    def all(self) -> pd.DataFrame:
        """dataframe of all signals (trades)"""
        return pd.concat(self.slices())

    @property
    def current(self):
        """the current signal of the most recent bar. [-1, 0, 1, nan]"""
        return self._obj.signal[-1]

    @property
    def prev(self):
        return self._obj.signal[-2]

    def cumulative_returns(self, side: Side = None) -> pd.Series:
        """cumulative_returns for all signals"""
        slice_returns = [
            signal_data.stats.daily_log_returns * signal_data.signal[-1]
            for signal_data in self.slices(side)
        ]
        res = 0
        if len(slice_returns) > 1:
            daily_log_returns = pd.concat(slice_returns)
        else:
            daily_log_returns = slice_returns[0]

        if slice_returns:
            res = trade_stats.cum_return_percent(daily_log_returns)

        return res


@pd.api.extensions.register_dataframe_accessor('stats')
class Stats(DfAccessorBase):
    mandatory_cols = ['close', 'b_close']

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    @property
    def cumulative_percent_returns(self):
        """"""
        return trade_stats.cum_return_percent(self.daily_log_returns)

    @property
    def daily_log_returns(self):
        return trade_stats.simple_returns(self._obj.b_close)


@pd.api.extensions.register_dataframe_accessor('lots')
class Lots(DfAccessorBase):
    mandatory_cols = [
        'eqty_risk_lot',
        'signal'
    ]

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def true_lots(self):
        """
        changes negative short lots to positive
        positive short lots to negative (account too small for min position size)
        negative long lots remain negative (account too small)
        positive long lots remain positive
        :return:
        """
        return self._obj.eqty_risk_lot * self._obj.signal


@pd.api.extensions.register_dataframe_accessor('update_check')
class UpdateCheck:
    def __init__(self, df: pd.DataFrame):
        self._obj = df

    def is_ready(self, market_type: tda_access.tda.client.Client.Markets, data_freq: t.Optional[timedelta] = None):
        """check if new bar is ready to be retrieved, prevents redundant API calls"""
        if data_freq is None:
            data_freq = get_minimum_freq(self._opj.index)
        ready = False
        if (
            tda_access.LocalClient.market_is_open(market_type) or
            # Allow for extra time to get data because, theoretically, market
            # will be closed when the last bar closes
            tda_access.LocalClient.market_was_open(market_type, time_ago=data_freq)
        ):
            ready = (datetime.now() - self._data.index[-1]) > data_freq
        return ready