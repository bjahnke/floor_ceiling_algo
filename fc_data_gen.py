import typing as t
from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime, timedelta

import pandas as pd
import tda.client
from pandas import Timedelta

import back_test_utils as btu
import tda_access
import tdargs
from matplotlib import pyplot as plt

import trade_stats

from strategy_utils import Side


class FcLosesToBuyHoldError(Exception):
    """fc data failed to initialize because it doesn't beat buy and hold"""


class NoSignalsError(Exception):
    """no signals were generated by the signal generator function"""


_bench_data = {}
_forex_data = {}
_configs = {}


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


def merge_copy(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    A copy of data with prefix appended to column names.
    Used for merging with other PriceDf data
    :return:
    """
    new_names = {column: f'{prefix}_{column}' for column in data.columns}
    return data.rename(columns=new_names)


def revert_copy(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    revert_names = {col: col.replace(f'{prefix}_', '') for col in data.columns}
    return data.rename(columns=revert_names)


def merge_reshape(price_data_prefix: str, price_data: pd.DataFrame, other_data: pd.DataFrame) -> pd.DataFrame:
    """
    :param price_data:
    :param price_data_prefix:
    :param other_data:
    :return:
    """
    data_cpy = merge_copy(price_data, price_data_prefix)
    data_copy_columns = list(data_cpy.columns)
    reshaped = other_data.join(data_cpy, how='left')
    reshaped = reshaped[data_copy_columns]
    reshaped = revert_copy(reshaped, prefix=price_data_prefix)
    return reshaped


class AccessorBase(metaclass=ABCMeta):
    mandatory_cols: t.List[str]

    def __init__(self, df: pd.DataFrame):
        self._validate(df)
        self._obj = df

    @classmethod
    def _validate(cls, df: pd.DataFrame):
        _validate(df, cls.mandatory_cols)


@pd.api.extensions.register_dataframe_accessor('swings')
class Swings(AccessorBase):
    mandatory_cols = [
        'high',
        'low',
    ]

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)


@pd.api.extensions.register_dataframe_accessor('fc')
class FcData(AccessorBase):
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
class Signals(AccessorBase):
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
        :return: if concat = True, a list of signals. if concat = False, a single dataframe
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
        """"""
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
class Stats(AccessorBase):
    mandatory_cols = ['close', 'b_close']

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    @property
    def cumulative_percent_returns(self):
        """"""
        return trade_stats.cum_return_percent(self.daily_log_returns)

    @property
    def daily_log_returns(self):
        return trade_stats.simple_returns(self._obj.b_close) * self._obj.signal


@pd.api.extensions.register_dataframe_accessor('lots')
class Lots(AccessorBase):
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


def init_fc_data(
        base_symbol: str,
        bench_symbol: str,
        freq_range: tdargs.FreqRangeArgs,
        equity: float,
        forex_symbol: t.Optional[str] = None,
        transaction_cost: t.Optional[float] = 0.025,
        percentile: float = 0.05,
        min_periods: int = 50,
        window: int = 200,
        limit: int = 5,
        arg_rel_window: int = 20,
        threshold: int = 1.5,  #
        st_dev_window: int = 63,  #
        st_list: range = range(10, 101, 10),  #
        mt_list: range = range(160, 201, 20),  #
        round_lot: int = 1,
        constant_risk: float = 0.25 / 100,
        constant_weight: float = 3 / 100,
        market_type: tda.client.Client.Markets = tda.client.Client.Markets.EQUITY
):
    """

    :param base_symbol:
    :param bench_symbol:
    :param freq_range:
    :param equity:
    :param forex_symbol:
    :param transaction_cost:
    :param percentile: tale ratio param
    :param min_periods: Minimum number of observations in window required to have a value (otherwise result is NA).
                        Used in various metric calculations
    :param window: rolling window for rolling_profits and tail ratio
    :param limit: tail ratio limit
    :param arg_rel_window: window for capturing relative extrema
    :param threshold: standard deviation threshold
    :param st_dev_window: window to calculate starndard deviation
    :param st_list: range of short term moving avg to test
    :param mt_list: range of mid term moving avg to test
    :param round_lot: round-by param for position sizer
    :param constant_risk:
    :param constant_weight:
    :param market_type:
    :return:
    """
    _configs[base_symbol] = locals()

    # get price history for symbol
    data = tda_access.LocalClient.price_history(
        symbol=base_symbol,
        freq_range=freq_range,
    )

    # benchmark history
    bench_data = _bench_data.get(bench_symbol, None)
    # TODO give market as parameter, will not always
    if bench_data is None or bench_data.update_check.is_ready(market_type):
        bench_data = tda_access.LocalClient.price_history(
            symbol=bench_symbol,
            freq_range=freq_range
        )
        _bench_data[bench_symbol] = bench_data

    forex_data = None
    if forex_symbol is not None:
        forex_data = _forex_data.get(forex_symbol, None)
        if forex_data is None:
            forex_data = tda_access.LocalClient.price_history(
                symbol=forex_symbol,
                freq_range=freq_range
            )
            _forex_data[forex_symbol] = forex_data

    # sometimes bench has less rows than base symbol data. Relative calc fails if columns lengths don't match
    data = merge_reshape(base_symbol, data, bench_data)

    rel_data = btu.relative_series(
        base_df=data,
        bench_df=bench_data,
        forex_df=forex_data,
        decimal=2
    )

    data_cpy = merge_copy(data, 'b')
    data = rel_data.join(data_cpy)
    data = data[
        ['open', 'high', 'low', 'close', 'b_close', 'b_high', 'b_low']
    ]

    try:
        data = btu.swings(
            df=data,
            high='high',
            low='low',
            arg_rel_window=arg_rel_window,
            prefix='sw'
        )
        data = btu.swings(
            df=data,
            high='b_high',
            low='b_low',
            arg_rel_window=arg_rel_window,
            prefix='sw'
        )
    except btu.NoSwingsError as err:
        # pass along the symbol that swing failed to calculate
        err.args += (base_symbol,)
        raise err

    data = btu.regime_fc(
        df=data,
        close='close',
        swing_low='sw_low',
        swing_high='sw_high',
        threshold=threshold,
        st_dev_window=st_dev_window
    )
    # data['regime_change'] = 0
    data = btu.init_fc_signal_stoploss(
        fc_data=data,
        symbol=base_symbol,
        base_close='b_close',
        relative_close='close',
        st_list=st_list,
        mt_list=mt_list,
        transaction_cost=transaction_cost,
        percentile=percentile,
        min_periods=min_periods,
        window=window,
        limit=limit,
        best_risk_adjusted_returns=0
    )[0]
    if data is None:
        raise FcLosesToBuyHoldError(f'{base_symbol} Floor/Ceiling does not beat buy and hold')

    data['ceiling'] = data.loc[data.regime_floorceiling == -1, 'regime_change']
    data['floor'] = data.loc[data.regime_floorceiling == 1, 'regime_change']

    # TODO this code is extremely brittle. Will break if columns are added or changed in df
    data = data.rename(columns={
        data.columns[10]: 'signal',
        data.columns[11]: 'stop_loss'
    })

    data['eqty_risk_lot'] = 0
    data['equal_weight_lot'] = 0
    signals = data[data.signal.shift(-1).isnull() & data.signal.notnull()]

    # don't waste time on pandas operations if there are no signals to
    # produce positions sizes
    if len(signals.index) > 0:
        data = btu.get_position_size(
            data=data,
            # TODO resolve difference
            capital=equity,
            constant_risk=constant_risk,
            constant_weight=constant_weight,
            stop_loss_col='stop_loss',
            round_lot=round_lot
        )
    else:
        raise NoSignalsError(f'{base_symbol} No signals generated')

    # data[['close', 'b_close', 'signal', 'stop_loss']].plot()

    return data


def new_init_fc_data(
        base_symbol: str,
        price_data: pd.DataFrame,
        equity: float,
        transaction_cost: t.Optional[float] = 0.025,
        percentile: float = 0.05,
        min_periods: int = 50,
        window: int = 200,
        limit: int = 5,
        arg_rel_window: int = 20,
        threshold: int = 1.5,  #
        st_dev_window: int = 63,  #
        st_list: range = range(10, 101, 10),  #
        mt_list: range = range(160, 201, 20),  #
        round_lot: int = 1,
        constant_risk: float = 0.25 / 100,
        constant_weight: float = 3 / 100,
):
    """

    :param base_symbol:
    :param price_data:
    :param equity:
    :param transaction_cost:
    :param percentile: tale ratio param
    :param min_periods: Minimum number of observations in window required to have a value (otherwise result is NA).
                        Used in various metric calculations
    :param window: rolling window for rolling_profits and tail ratio
    :param limit: tail ratio limit
    :param arg_rel_window: window for capturing relative extrema
    :param threshold: standard deviation threshold
    :param st_dev_window: window to calculate starndard deviation
    :param st_list: range of short term moving avg to test
    :param mt_list: range of mid term moving avg to test
    :param round_lot: round-by param for position sizer
    :param constant_risk:
    :param constant_weight:
    :return:
    """

    try:
        price_data = btu.swings(
            df=price_data,
            high='high',
            low='low',
            arg_rel_window=arg_rel_window,
            prefix='sw'
        )
        price_data = btu.swings(
            df=price_data,
            high='b_high',
            low='b_low',
            arg_rel_window=arg_rel_window,
            prefix='sw'
        )
    except btu.NoSwingsError as err:
        # pass along the symbol that swing failed to calculate
        err.args += (base_symbol,)
        raise err

    price_data = btu.regime_fc(
        df=price_data,
        close='close',
        swing_low='sw_low',
        swing_high='sw_high',
        threshold=threshold,
        st_dev_window=st_dev_window
    )
    # price_data['regime_change'] = 0
    price_data = btu.init_fc_signal_stoploss(
        fc_data=price_data,
        symbol=base_symbol,
        base_close='b_close',
        relative_close='close',
        st_list=st_list,
        mt_list=mt_list,
        transaction_cost=transaction_cost,
        percentile=percentile,
        min_periods=min_periods,
        window=window,
        limit=limit,
        best_risk_adjusted_returns=0
    )[0]
    if price_data is None:
        raise FcLosesToBuyHoldError(f'{base_symbol} Floor/Ceiling does not beat buy and hold')

    price_data['ceiling'] = price_data.loc[price_data.regime_floorceiling == -1, 'regime_change']
    price_data['floor'] = price_data.loc[price_data.regime_floorceiling == 1, 'regime_change']

    # TODO this code is extremely brittle. Will break if columns are added or changed in df
    price_data = price_data.rename(columns={
        price_data.columns[10]: 'signal',
        price_data.columns[11]: 'stop_loss'
    })

    price_data['eqty_risk_lot'] = 0
    price_data['equal_weight_lot'] = 0
    signals = price_data[price_data.signal.shift(-1).isnull() & price_data.signal.notnull()]

    # don't waste time on pandas operations if there are no signals to
    # produce positions sizes
    if len(signals.index) > 0:
        price_data = btu.get_position_size(
            data=price_data,
            # TODO resolve difference
            capital=equity,
            constant_risk=constant_risk,
            constant_weight=constant_weight,
            stop_loss_col='stop_loss',
            round_lot=round_lot
        )
    else:
        raise NoSignalsError(f'{base_symbol} No signals generated')

    # price_data[['close', 'b_close', 'signal', 'stop_loss']].plot()

    return price_data


def create_relative_data(
    base_symbol: str,
    price_data: pd.DataFrame,
    bench_data: pd.DataFrame,
    freq_range: tdargs.FreqRangeArgs,
    forex_symbol: t.Optional[str] = None,
):
    forex_data = None
    if forex_symbol is not None:
        forex_data = _forex_data.get(forex_symbol, None)
        if forex_data is None:
            forex_data = tda_access.LocalClient.price_history(
                symbol=forex_symbol,
                freq_range=freq_range
            )
            _forex_data[forex_symbol] = forex_data

    # sometimes bench has less rows than base symbol data. Relative calc fails if columns lengths don't match
    data = merge_reshape(base_symbol, price_data, bench_data)

    rel_data = btu.relative_series(
        base_df=data,
        bench_df=bench_data,
        forex_df=forex_data,
        decimal=2
    )

    data_cpy = merge_copy(data, 'b')
    data = rel_data.join(data_cpy)
    data = data[
        ['open', 'high', 'low', 'close', 'b_close', 'b_high', 'b_low']
    ]
    return data


def test():
    res = init_fc_data(
        base_symbol='AAPL',
        bench_symbol='SPX',
        equity=100000,
        freq_range=tdargs.freqs.day.range(tdargs.periods.y2)
    )
    print('Done.')


if __name__ == '__main__':
    test()
