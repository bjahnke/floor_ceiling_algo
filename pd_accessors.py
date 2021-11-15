import pandas as pd
import numpy as np
from abc import ABCMeta
import typing as t

from pandas import Timedelta

from strategy_utils import Side, SignalStatus
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

    @property
    def status(self) -> SignalStatus:
        """gives the current signal status"""
        return SignalStatus((
            Side(self._obj.signals.current),
            Side(self._obj.signals.prev)
        ))

    def init_simulated_scale_out(
        self,
        stop_loss_col: str = 'stop_loss',
        target_r: float = 1.5
    ) -> t.Tuple[pd.Series, pd.Series]:
        """
        For all signals, calculate remaining position size as a percent
        if target price is hit.
        Note, simulation assumes order is filled exactly at order price
        """
        assert 1.0 < target_r < 2.0
        remaining_size_pct = 1 - (1 / target_r)
        size_percents_group = []
        target_group = []
        for signal_data in self._obj.signals.slices():
            size_percents = pd.Series(data=1, index=signal_data.index)
            initial_stop_price = signal_data[stop_loss_col][0]
            entry_price = signal_data.close[0]
            signal = signal_data.signal[0]

            target_gain = abs(initial_stop_price - entry_price) * target_r
            if signal == 1:
                target_price = entry_price + target_gain
                extremes = signal_data.high
                hits_target = (extremes - target_price).cummax() >= 0
            else:
                target_price = entry_price - target_gain
                extremes = signal_data.low
                hits_target = (extremes - target_price).cummax() <= 0

            size_percents.loc[hits_target] = remaining_size_pct
            size_percents_group.append(size_percents)
            target_group.append(pd.Series(data=target_price, index=signal_data.index))

        return pd.concat(size_percents_group), pd.concat(target_group)

    def equity_risk_weight(self, risk: float, stop_loss_col: str = 'stop_loss') -> pd.Series:
        position_weight = []
        for signal_data in self._obj.signals.slices():
            px_adj = signal_data.close.iloc[0]
            stop_loss = signal_data[stop_loss_col].iloc[0]

            # dsl = px_adj / stop_loss - 1  # distance to stop loss in currency adjusted relative
            dsl = (px_adj - stop_loss) / px_adj
            try:
                eqty_at_risk = risk / dsl  # weight in currency adjusted relative terms
            except ZeroDivisionError:
                eqty_at_risk = 0

            # weight = pd.Series(data=eqty_at_risk, index=signal_data.index)
            weight = pd.Series(data=eqty_at_risk, index=signal_data.index)
            position_weight.append(weight)

        return pd.concat(position_weight)

    def calc_lot(self, capital, weight_col: str, fx_rate=1, round_lot=1) -> pd.Series:
        """calculate the size in number of shares"""
        lots = []
        for signal_data in self._obj.signals.slices():
            weight = signal_data[weight_col].iloc[0]
            entry_price = signal_data.close.iloc[0]

            book_value = weight * capital
            shares = book_value * fx_rate / entry_price
            lot = round(shares // round_lot, 0) * round_lot

            lots.append(pd.Series(data=lot, index=signal_data.index))
        return pd.concat(lots).fillna(value=0)


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
        'signal',
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


@pd.api.extensions.register_dataframe_accessor('scan_data')
class ScanData(DfAccessorBase):
    mandatory_cols = [
        'symbol',
        'st',
        'mt'
    ]

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def by_symbol(self, symbol: str):
        return self._obj[self._obj.symbol == symbol]

    def get_symbols_score(self):
        pass


@pd.api.extensions.register_dataframe_accessor('stop_losses')
class StopLoss(DfAccessorBase):
    mandatory_cols = [
        'signal',
        'high',
        'low',
        'close',
        'sw_high',
        'sw_low',
        'stop_loss'
    ]

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def init_trail_stop(self) -> pd.Series:
        trail_stops = []
        for signal_data in self._obj.signals.slices():
            signal = signal_data.signal.iloc[-1]
            entry_price = signal_data.close.iloc[0]

            if signal == 1:
                # TODO practically, initial high should price at time of trail stop entry
                # initial high is the price we entered (approximately)
                extremes = signal_data.high.copy()
                extremes.iat[0] = entry_price
                cum_extreme = signal_data.high.cummax()
            else:
                extremes = signal_data.low.copy()
                extremes.iat[0] = entry_price
                cum_extreme = signal_data.low.cummin()

            # shift back to
            cum_delta_from_entry = (cum_extreme - entry_price)
            # stop loss increases/decreases by the current cumulative extreme from the beginning stop loss
            trail_stop = cum_delta_from_entry + signal_data.stop_loss

            trail_stops.append(trail_stop)

        return pd.concat(trail_stops)

    def trail_to_cost(self, trail_stop: pd.Series) -> t.Tuple[pd.Series, pd.Series, pd.Series]:
        """
        TODO does this need to be vectorized?
        TODO test stop loss generator
        trailing stop loss that is limited to entry price
        :param trail_stop:
        :return:
        """
        local_obj = self._obj.copy()
        local_obj['trail_stop'] = trail_stop

        trail_stops = []
        cropped_signals = []
        stop_status = []
        for signal_data in self._obj.signals.slices():
            signal = signal_data.signal.copy()
            signal_dir = signal.iloc[-1]
            entry_price = signal_data.close.iloc[0]
            trail_stop_at_cost = signal_data.trail_stop.copy()

            if signal_dir == 1:
                def crosses_cost():
                    return trail_stop_at_cost >= entry_price

                extreme = signal_data.low.copy()
                trip_stop_params = {
                    'higher': extreme,
                    'lower': trail_stop_at_cost
                }
            else:
                def crosses_cost():
                    return trail_stop_at_cost <= entry_price

                extreme = signal_data.high.copy()
                trip_stop_params = {
                    'higher': trail_stop_at_cost,
                    'lower': extreme
                }
            where_crosses_cost = crosses_cost()
            extreme.iat[0] = entry_price
            trail_stop_at_cost.loc[where_crosses_cost] = entry_price
            where_stop_trips = trips_price(**trip_stop_params)
            trail_stop_at_cost.loc[where_stop_trips] = np.nan
            signal.loc[where_stop_trips] = np.nan
            where_crosses_cost.loc[where_stop_trips] = np.nan

            trail_stops.append(trail_stop_at_cost)
            cropped_signals.append(signal)
            stop_status.append(where_crosses_cost)

        return (
            pd.concat(trail_stops),
            pd.concat(cropped_signals),
            pd.concat(stop_status)
        )

    def local_stop(self):
        s_low = self._obj.sw_low
        s_high = self._obj.sw_high
        signal = self._obj.signal
        close = self._obj.close

        stoploss = (
            s_low.add(s_high, fill_value=0)
        ).fillna(method='ffill')  # join all swings in 1 column
        stoploss[~((np.isnan(signal.shift(1))) & (~np.isnan(signal)))] = np.nan  # keep 1st sl by signal
        stoploss = stoploss.fillna(method='ffill')  # extend first value with fillna

        # Bull: lowest close, Bear: highest close
        close_max = close.groupby(stoploss).cummax()
        close_min = close.groupby(stoploss).cummin()
        cum_close = np.where(signal == 1, close_min, np.where(signal == -1, close_max, 0))

        # reset signal where stop loss is breached
        sl_delta = (cum_close - stoploss).fillna(0)
        sl_sign = signal * np.sign(sl_delta)
        signal[sl_sign == -1] = np.nan
        return stoploss


def trips_price(higher, lower):
    cum_delta = (higher - lower).cummin()
    return cum_delta <= 0
