import typing as t
import pandas as pd
import back_test_utils as btu
import tda_access
import tdargs
from matplotlib import pyplot as plt

class FcLosesToBuyHoldError(Exception):
    """fc data failed to initialize because it doesn't beat buy and hold"""

class NoSignalsError(Exception):
    """no signals were generated by the signal generator function"""


_bench_data = {}
_forex_data = {}
_configs = {}

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


@pd.api.extensions.register_dataframe_accessor('swings')
class Swings:
    mandatory_cols = [
        'high',
        'low',
    ]

    def __init__(self, df: pd.DataFrame):
        self._validate(df)
        self._obj = df

    @staticmethod
    def _validate(df: pd.DataFrame):
        _validate(df, Swings.mandatory_cols)


@pd.api.extensions.register_dataframe_accessor('fc')
class FcData:
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
        self._validate(df)
        self._obj = df

    @staticmethod
    def _validate(df: pd.DataFrame):
        _validate(df, FcData.mandatory_cols)

    @property
    def signals(self):
        """get all rows where a signal is generated"""
        return self._obj[self._obj.signal.shift(-1).isnull() & self._obj.signal.notnull()]

    @property
    def position_sizes(self):
        """"""
        return None


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
    constant_risk: float = 0.25/100,
    constant_weight: float = 3/100,
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
    if bench_data is None:
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
        ['open', 'high', 'low', 'close', 'b_close']
    ]


    data = btu.swings(
        df=data,
        high='high',
        low='low',
        arg_rel_window=arg_rel_window,
        prefix='sw'
    )
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
    btu.graph_regime_fc(
        ticker=base_symbol,
        df=data,
        y='close',
        th=threshold,
        sl='sw_low',
        sh='sw_high',
        clg='ceiling',
        flr='floor',
        st=data['st_ma'],
        mt=data['mt_ma'],
        bs='regime_change',
        rg='regime_floorceiling',
        bo=200
    )
    plt.show()

    data = data.rename(columns={
        data.columns[8]: 'signal',
        data.columns[9]: 'stop_loss'
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


if __name__ == '__main__':
    res = init_fc_data(
        base_symbol='AAPL',
        bench_symbol='SPX',
        equity=100000,
        freq_range=tdargs.freqs.day.range(tdargs.periods.y2)
    )
    print('Done.')



