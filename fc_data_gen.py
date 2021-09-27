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


def init_fc_data(
        base_symbol: str,
        price_data: pd.DataFrame,
        equity: float,
        transaction_cost: t.Optional[float] = 0,
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
    )[0]
    # if price_data is None:
    #     raise FcLosesToBuyHoldError(f'{base_symbol} Floor/Ceiling does not beat buy and hold')

    price_data['ceiling'] = price_data.loc[price_data.regime_floorceiling == -1, 'regime_change']
    price_data['floor'] = price_data.loc[price_data.regime_floorceiling == 1, 'regime_change']

    # TODO this code is extremely brittle. Will break if columns are added or changed in df
    price_data = price_data.rename(columns={
        price_data.columns[10]: 'signal',
        price_data.columns[11]: 'stop_loss'
    })

    price_data['eqty_risk_lot'] = 0
    price_data['equal_weight_lot'] = 0

    # don't waste time on pandas operations if there are no signals to
    # produce positions sizes
    if len(price_data.signals.slices()) > 0:
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
        # todo should return dataframe with empty signal col?
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

