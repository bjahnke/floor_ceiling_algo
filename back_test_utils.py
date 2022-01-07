from typing import List, Tuple, Union
import operator as op
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.signal
import tdargs
from datetime import datetime
from dataclasses import dataclass

from strategy_utils import Side
from tda_access import LocalClient
import itertools  # construct a list of permutations
from trade_stats import (
    simple_returns, cumulative_returns, cum_return_percent, count_signals,
    cumulative_returns_pct, rolling_profits, hit_rate, average_win,
    average_loss, george, grit_index, calmar_ratio, tail_ratio, t_stat,
    common_sense_ratio, equity_at_risk, get_round_lot
)
from trade_stats import ts2
import pd_accessors


@dataclass
class AnalysisData:
    df: pd.DataFrame
    stats: pd.DataFrame
    perf: pd.DataFrame
    best_rar: float
    # get mt lt sma params from table

    def __post_init__(self):
        self.stats = self.stats.sort_values(by=['score'], ascending=False)


class NoSwingsError(Exception):
    """FC calc cannot continue because no swings were detected"""


class StrNanInPriceError(Exception):
    """Relative price calculation fails because a string nan is in price history"""


def ma_crossover(slow_ma, fast_ma):
    """
    TODO test crossover with bounce (zeros included) vs no bounce
    Cross is defined as follows:
    diff = fast moving average - slow moving average
    A & B | C & D
    A = current_diff > 0
    B = prior_non_zero_diff was < 0
    C = current_dif < 0
    D = prior_non_zero_diff was > 0

    :param slow_ma: column of slow ma
    :param fast_ma: column of fast ma
    :return: indexes where a cross occurred
    """
    diff_from_fast = fast_ma - slow_ma
    # drop zeros
    diff_from_fast = diff_from_fast[diff_from_fast != 0]
    prev_diff = diff_from_fast.shift(1)
    crosses = diff_from_fast[(prev_diff < 0 < diff_from_fast) | (diff_from_fast < 0 < prev_diff)]
    crosses = crosses.rename("cross_val")
    return crosses


def patch_nans(iterable):
    prior_num = 0
    patched_list = []
    for i, item in enumerate(iterable):
        if not np.isnan(item):
            prior_num = item
        patched_list.append(prior_num)
    return patched_list


def relative_series(
        base_df: pd.DataFrame,
        bench_df: pd.DataFrame,
        forex_df=None,
        decimal=2
) -> pd.DataFrame:
    """
    my relative series
    TODO what to do when sizes are different?
    :param base_df:
    :param bench_df:
    :param forex_df:
    :param decimal:
    :return:
    """
    forex_values = 1 if forex_df is None else forex_df.values
    # 1. apply adjustment factors
    adjustments = bench_df * forex_values
    # 2. divide out adjustment factors from unadjusted and set to relative_df
    relative_df = base_df / adjustments.values
    # 3. rebase
    relative_df *= adjustments.iloc[0].values

    relative_df = round(relative_df, decimal)

    return relative_df


# Define relative function
def relative(stock_dataframe, benchmark_dataframe, benchmark_name, forex_dataframe, forex_name, start, end, decimals=2):
    # Slice dataframe from start to end period: either offset or datetime
    stock_dataframe = stock_dataframe[start:end]

    # Join the data set: Concatenation of benchmark, stock & currency
    data = pd.concat([stock_dataframe, forex_dataframe,
                      benchmark_dataframe], axis=1, sort=True).dropna()

    # Adjustment factor: Calculate the product of benchmark and currency
    data['adjustment_factor'] = data[benchmark_name] * (data[forex_name])

    # Relative series: Calculate the relative series by dividing the OHLC stock data with the adjustment factor
    data['relative_open'] = data['Open'] / data['adjustment_factor']
    data['relative_high'] = data['High'] / data['adjustment_factor']
    data['relative_low'] = data['Low'] / data['adjustment_factor']
    data['relative_close'] = data['Close'] / data['adjustment_factor']

    # Rebased series: Multiply relative series with the first value of the adjustment factor to get the rebased series
    data['rebased_open'] = data['relative_open'] * data['adjustment_factor'].iloc[0]
    data['rebased_high'] = data['relative_high'] * data['adjustment_factor'].iloc[0]
    data['rebased_low'] = data['relative_low'] * data['adjustment_factor'].iloc[0]
    data['rebased_close'] = data['relative_close'] * data['adjustment_factor'].iloc[0]

    data = round(data, decimals)

    return data


# Calculate simple moving average
def sma(
    df: pd.DataFrame,
    price,
    ma_per,
    min_per,
    decimals: Union[int, None]
):
    """
    Returns the simple moving average.
    price: column within the df
    ma_per: moving average periods
    min_per: minimum periods (expressed as 0<pct<1) to calculate moving average
    decimals: rounding number of decimals
    """
    res = df[price].rolling(window=ma_per, min_periods=int(round(ma_per * min_per, 0))).mean()
    if decimals is not None:
        res = round(res, decimals)
    return res

# Calculate exponential moving average
def ema(
    df: pd.DataFrame,
    price: str,
    ma_per: int,
    min_per,
    decimals: int
):
    """
    Returns exponentially weighted moving average.

    price: column within the df
    ma_per: moving average periods
    min_per: minimum periods (expressed as 0<pct<1) to calculate moving average
    decimals: rounding number of a

    """
    return round(
        df[price]
        .ewm(span=ma_per, min_periods=int(round(ma_per * min_per, 0)))
        .mean(),
        decimals,
    )


# Calculate regime simple moving average
def regime_sma(
    df: pd.DataFrame,
    price: str,
    short_term: int,
    long_term: int
):
    """
    when price >= sma bull +1, when price < sma: bear -1, fillna
    """
    # define rolling high/low
    sma_st = df[price].rolling(window=short_term, min_periods=short_term).mean()
    sma_lt = df[price].rolling(window=long_term, min_periods=long_term).mean()

    # when price>= sma: bull, when price<sma: bear
    df['regime_sma' + '_' + str(short_term) + '_' + str(long_term)] = np.where(
        sma_st >= sma_lt,
        1,
        np.where(
            sma_st < sma_lt,
            -1,
            np.nan
        )
    )
    df['regimes_sma' + '_' + str(short_term) + '_' + str(long_term)] = df[
        'regime_sma' + '_' + str(short_term) + '_' + str(long_term)
        ].fillna(method='ffill')
    return df


# Calculate regime simple moving average
def r_sma(price_view: pd.DataFrame, short_term: int, long_term: int) -> pd.DataFrame:
    """
    Brian's implement
    when price >= sma bull +1, when price < sma: bear -1, fillna
    :param price_view: subset of price history, index: date, col=a single price column
    :param short_term:
    :param long_term:
    :return:
    """
    # TODO meta data for all of these columns would be useful to know how/from where
    # TODO they were constructed
    # define rolling high/low
    sma_st = price_view.rolling(window=short_term, min_periods=short_term).mean()
    sma_st.name = 'sma_short'
    sma_lt = price_view.rolling(window=long_term, min_periods=long_term).mean()
    sma_lt.name = 'sma_long'
    price = price_view.copy(deep=True)
    price.name = 'price'

    ma_cross_df = pd.concat([price, sma_st, sma_lt], axis=1)

    # when price>= sma: bull, when price<sma: bear
    ma_cross_df['regime'] = np.where(
        sma_st >= sma_lt,
        1,
        np.where(
            sma_st < sma_lt,
            -1,
            np.nan
        )
    )
    ma_cross_df['regime'] = ma_cross_df['regime_sma'].fillna(method='ffill')
    return ma_cross_df


# Calculate regime exponential moving average
def regime_ema(df: pd.DataFrame, price: str, short_term:int, long_term: int) -> pd.DataFrame:
    """
    when price >= ema bull +1, when price < ema: bear -1, fillna
    """
    # define rolling high/low
    ema_st = df[price].ewm(span=short_term, min_periods=short_term).mean()
    ema_lt = df[price].ewm(span=long_term, min_periods=long_term).mean()

    # when price>= sma: bull, when price<sma: bear
    df['regime_ema' + '_' + str(short_term) + '_' + str(long_term)] = np.where(ema_st >= ema_lt, 1,
                                                                               np.where(ema_st < ema_lt, -1, np.nan))
    df['regime_ema' + '_' + str(short_term) + '_' + str(long_term)] = df[
        'regime_ema' + str(short_term) + str(long_term)].fillna(method='ffill')
    return df


# Plot regime
def graph_regime_fc(
    ticker: str,
    df: pd.DataFrame,
    y: str,
    th: float,
    sl: str,
    sh,
    clg,
    flr,
    bs,
    rg,
    st,
    mt,
    bo
):
    # fig = plt.figure(figsize=(10, 7))
    ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=2, colspan=2)


    date = df.index
    close = df[y]
    swing_low = df[sl]
    swing_high = df[sh]
    ceiling = df[clg]
    floor = df[flr]

    base = df[bs]
    regime = df[rg]

    ax1.plot_date(df.index, close, '-', color='k', label=ticker.upper() + ' stdev:' + str(th))
    ax1.plot(df.index, swing_low, '.', color='r', label='swing low', alpha=0.5)
    ax1.plot(df.index, swing_high, '.', color='g', label='swing high', alpha=0.5)
    plt.scatter(df.index, floor, c='b', marker='^', label='floor')
    plt.scatter(df.index, ceiling, c='m', marker='v', label='ceiling')

    ax1.plot([], [], linewidth=5, label='bear', color='m', alpha=0.1)
    ax1.plot([], [], linewidth=5, label='bull', color='b', alpha=0.1)
    ax1.fill_between(date, close, base, where=((regime == 1) & (close > base)), facecolor='b', alpha=0.1)
    #     ax1.fill_between(date, close, base,where=((regime==1)&(close<base)), facecolor='b', alpha=0.8)
    ax1.fill_between(date, close, base, where=((regime == -1) & (close < base)), facecolor='m', alpha=0.1)
    #     ax1.fill_between(date, close, base,where=((regime==-1)&(close>base)), facecolor='m', alpha=0.8)

    if np.sum(st) > 0:
        ax1.plot(df.index, st, '-', color='lime', label=' st')
        ax1.plot(df.index, mt, '-', color='green', label=' mt')  # 2. plot line
        # Profitable conditions
        ax1.fill_between(date, close, mt, where=((regime == 1) & (st >= mt) & (close >= mt)),
                         facecolor='green', alpha=0.3)
        ax1.fill_between(date, close, mt, where=((regime == -1) & (st <= mt) & (close <= mt)),
                         facecolor='red', alpha=0.3)
        # Unprofitable conditions
        ax1.fill_between(date, close, mt, where=((regime == 1) & (st >= mt) & (close < mt)),
                         facecolor='darkgreen', alpha=1)
        ax1.fill_between(date, close, mt, where=((regime == -1) & (st <= mt) & (close > mt)),
                         facecolor='darkred', alpha=1)

    if bo > 0:
        #         ax1.plot([],[],linewidth=5, label=str(bo)+' days high', color='m',alpha=0.3)
        #         ax1.plot([],[],linewidth=5, label=str(bo) + ' days low', color='b',alpha=0.3)
        rolling_min = close.rolling(window=bo).min()
        rolling_max = close.rolling(window=bo).max()
        ax1.fill_between(date, close, rolling_min,
                         where=((regime == 1) & (close > rolling_min)), facecolor='b', alpha=0.2)
        ax1.fill_between(date, close, rolling_max,
                         where=((regime == -1) & (close < rolling_max)), facecolor='m', alpha=0.2)
        ax1.plot(df.index, rolling_min, '-.', color='b', label=str(bo) + ' low', alpha=0.5)
        ax1.plot(df.index, rolling_max, '-.', color='m', label=str(bo) + ' low', alpha=0.5)

    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax1.grid(True)
    ax1.xaxis.label.set_color('g')
    ax1.yaxis.label.set_color('g')

    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.title('Floor and Ceiling', fontsize=16)
    plt.legend()


# Calculate regime breakout-breakdown
def regime_breakout_breakdown(
    breakout_period: int,
    df: pd.DataFrame,
    high: str,
    low: str
) -> pd.DataFrame:
    """
    :param breakout_period:
    :param df:
    :param high:
    :param low:
    :return:
    """
    """
    when new hi: bull +1, when new low: bear -1, fillna
    """

    # define rolling high/low
    rolling_high = df[high].rolling(window=breakout_period, min_periods=breakout_period).max()  #
    rolling_low = df[low].rolling(window=breakout_period, min_periods=breakout_period).min()

    # when new high bull, when new low: bear
    df['regime_breakout_breakdown' + '_' + str(breakout_period)] = np.where(df[high] >= rolling_high, 1,
                                                                            np.where(df[low] <= rolling_low, -1,
                                                                                     np.nan))
    df['regime_breakout_breakdown' + '_' + str(breakout_period)] = df[
        'regime_breakout_breakdown' + '_' + str(breakout_period)].fillna(method='ffill')
    return df


# Calculate swings
def swings(
    df: pd.DataFrame,
    high: str,
    low: str,
    arg_rel_window: int,
    prefix: str = 'sw'
) -> pd.DataFrame:
    """
    Will raise ValueError if no swings found above 0 on high or low side
    :param df:
    :param high: col name for highs of day
    :param low: col name for lows of day
    :param arg_rel_window: window to detect local highs and lows
    :param prefix: prefix for swing high and swing low column name
    :return:
    """
    # Create swings:

    # Step 1: copy existing df. We will manipulate and reduce this df and want to preserve the original
    high_low = df[[high, low]].copy()

    # Step 2: build 2 lists of highs and lows using argrelextrema
    highs_list = scipy.signal.argrelextrema(
        high_low[high].values, np.greater, order=arg_rel_window)
    lows_list = scipy.signal.argrelextrema(
        high_low[low].values, np.less, order=arg_rel_window)

    # Step 3: Create swing high and low columns and assign values from the lists
    swing_high = f'{prefix}_' + str(high)[-12:]
    swing_low = f'{prefix}_' + str(low)[-12:]
    high_low[swing_low] = high_low.iloc[lows_list[0], 1]
    high_low[swing_high] = high_low.iloc[highs_list[0], 0]

    # Alternation: We want highs to follow lows and keep the most extreme values

    # Step 4. Create a unified column with peaks<0 and troughs>0
    swing_high_low = str(high)[:2]+str(low)[:2]
    high_low[swing_high_low] = high_low[swing_low].sub(
        high_low[swing_high], fill_value=0)

    # Step 5: Reduce dataframe and alternation loop
    # Instantiate start
    i = 0
    # Drops all rows with no swing
    high_low = high_low.dropna(subset=[swing_high_low]).copy()
    while ((high_low[swing_high_low].shift(1) * high_low[swing_high_low] > 0)).any():
        # eliminate lows higher than highs
        high_low.loc[(high_low[swing_high_low].shift(1) * high_low[swing_high_low] < 0) &
                     (high_low[swing_high_low].shift(1) < 0) & (np.abs(high_low[swing_high_low].shift(1)) < high_low[swing_high_low]), swing_high_low] = np.nan
        # eliminate earlier lower values
        high_low.loc[(high_low[swing_high_low].shift(1) * high_low[swing_high_low] > 0) & (
                high_low[swing_high_low].shift(1) < high_low[swing_high_low]), swing_high_low] = np.nan
        # eliminate subsequent lower values
        high_low.loc[(high_low[swing_high_low].shift(-1) * high_low[swing_high_low] > 0) & (
                high_low[swing_high_low].shift(-1) < high_low[swing_high_low]), swing_high_low] = np.nan
        # reduce dataframe
        high_low = high_low.dropna(subset=[swing_high_low]).copy()
        i += 1
        if i == 4:  # avoid infinite loop
            break

    # Step 6: Join with existing dataframe as pandas cannot join columns with the same headers
    # First, we check if the columns are in the dataframe
    if swing_low in df.columns:
        # If so, drop them
        df.drop([swing_low, swing_high], axis=1, inplace=True)
    # Then, join columns
    df = df.join(high_low[[swing_low, swing_high]])

    # Last swing adjustment:

    # Step 7: Preparation for the Last swing adjustment
    high_low[swing_high_low] = np.where(
        np.isnan(high_low[swing_high_low]), 0, high_low[swing_high_low])
    # If last_sign <0: swing high, if > 0 swing low
    if len(high_low[swing_high_low]) == 0:
        raise NoSwingsError('Quit swing calc. high[swing_high_low] is empty')

    last_sign = np.sign(high_low[swing_high_low][-1])

    # Step 8: Instantiate last swing high and low dates
    last_slo_dt = df[df[swing_low] > 0].index.max()
    last_shi_dt = df[df[swing_high] > 0].index.max()
    if pd.isnull(last_slo_dt) or pd.isnull(last_shi_dt):
        raise NoSwingsError('Quit swing calc. [Swing High > 0 AND Swing Low] condition fails.')

    # Step 9: Test for extreme values
    if (last_sign == -1) & (last_shi_dt != df[last_slo_dt:][swing_high].idxmax()):
        # Reset swing_high to nan
        df.loc[last_shi_dt, swing_high] = np.nan
    elif (last_sign == 1) & (last_slo_dt != df[last_shi_dt:][swing_low].idxmax()):
        # Reset swing_low to nan
        df.loc[last_slo_dt, swing_low] = np.nan

    return df


# Calculate regime floor ceiling
def regime_fc(
    df: pd.DataFrame,
    close: str,
    swing_low: str,
    swing_high: str,
    threshold: float,
    st_dev_window: int,
    decimals: int = 2,
):

    # 1. copy swing lows and highs to a smaller df called regime
    regime = df[
        (df[swing_low] > 0) |
        (df[swing_high] > 0)
    ][
        [close, swing_low, swing_high]
    ].copy()

    # 2. calculate volatility from main df and populate regime df
    stdev = 'stdev'
    regime[stdev] = rolling_stdev(
        price_col=df[close],
        window=st_dev_window,
        min_periods=1,
        decimals=decimals
    )

    ceiling_found = False
    floor_found = False

    # 4.  instantiate columns
    rg_cols = [
        'floor',
        'ceiling',
        'regime_change',
        'regime_floorceiling',
        'floorceiling',
        'regime_breakout'
    ]

    # b) instantiate columns by concatenation
    regime = pd.concat(
        [
            regime,  # existing df
            # temporary df with same index, regime columns initialised at nan
            pd.DataFrame(
                np.nan, index=regime.index, columns=rg_cols
            )
        ],  # temp df
        axis=1
    )  # along the vertical axis

    # c) column variables names instantiation via a list comprehension
    floor, ceiling, regime_change, regime_floorceiling, floorceiling, regime_breakout = [
        list(rg_cols)[n] for n in range(len(rg_cols))
    ]

    # 5. Range initialisation to 1st swing
    floor_ix = regime.index[0]
    ceiling_ix = regime.index[0]

    # 6. Loop through swings
    for i in range(1, len(regime)):

        if regime[swing_high][i] > 0:  # ignores swing lows
            top = regime[floor_ix:regime.index[i]][swing_high].max()  # highest swing high from range[floor_ix:swing[i]]
            ceiling_test = round((regime[swing_high][i] - top) / regime[stdev][i], 1)  # test vs highest

            if ceiling_test <= -threshold:  # if swing <= top - x * stdev
                ceiling = regime[floor_ix:regime.index[i]][swing_high].max()  # ceiling = top
                ceiling_ix = regime[floor_ix:regime.index[i]][swing_high].idxmax()  # ceiling index
                regime.loc[ceiling_ix, ceiling] = ceiling  # assign ceiling

                if not ceiling_found:  # test met == ceiling found
                    rg_chg_ix = regime[swing_high].index[i]
                    _rg_chg = regime[swing_high][i]
                    regime.loc[rg_chg_ix, regime_change] = _rg_chg  # prints where/n ceiling found
                    regime.loc[rg_chg_ix, regime_floorceiling] = -1  # regime change
                    regime.loc[rg_chg_ix, floorceiling] = ceiling  # used in floor/ceiling breakout test

                    ceiling_found = True  # forces alternation btwn Floor & ceiling
                    floor_found = False

        if regime[swing_low][i] > 0:  # ignores swing highs
            bottom = regime[ceiling_ix:regime.index[i]][swing_low].min()  # lowest swing low from ceiling
            floor_test = round((regime[swing_low][i] - bottom) / regime[stdev][i], 1)  # test vs lowest

            if floor_test >= threshold:  # if swing > bottom + n * stdev
                floor = regime[ceiling_ix:regime.index[i]][swing_low].min()  # floor = bottom
                floor_ix = regime[ceiling_ix:regime.index[i]][swing_low].idxmin()
                regime.loc[floor_ix, floor] = floor  # assign floor

                if not floor_found:  # test met == floor found
                    rg_chg_ix = regime[swing_low].index[i]
                    _rg_chg = regime[swing_low][i]
                    regime.loc[rg_chg_ix, regime_change] = _rg_chg  # prints where/n floor found
                    regime.loc[rg_chg_ix, regime_floorceiling] = 1  # regime change
                    regime.loc[rg_chg_ix, floorceiling] = floor  # used in floor/ceiling breakdown test

                    ceiling_found = False  # forces alternation btwn floor/ceiling
                    floor_found = True

    # 8. drop rg_cols if already in df to avoid overlap error
    if regime_floorceiling in df.columns:  # drop columns if already in df before join, otherwise overlap error
        df = df.drop(rg_cols, axis=1)

    # 7. join regime to df
    df = df.join(regime[rg_cols], how='outer')

    # 9. forward fill regime 'rg_fc','rg_chg','fc', then fillna(0) from start to 1st value
    df[regime_floorceiling] = df[regime_floorceiling].fillna(method='ffill').fillna(0)  # regime
    df[regime_change] = df[regime_change].fillna(method='ffill').fillna(0)  # rg_chg
    df[floorceiling] = df[floorceiling].fillna(method='ffill').fillna(0)  # floor ceiling continuous line

    # 10. test breakout/down: if price crosses floor/ceiling, regime change
    close_max = df.groupby([floorceiling])[close].cummax()  # look for highest close for every floor/ceiling
    close_min = df.groupby([floorceiling])[close].cummin()  # look for lowest close for every floor/ceiling

    # 11. if rgme bull: assign lowest close, elif bear: highest close
    rgme_close = np.where(
        df[floorceiling] < df[regime_change],
        close_min,
        np.where(
            df[floorceiling] > df[regime_change], close_max, 0
        )
    )

    df[regime_breakout] = (rgme_close - df[floorceiling]).fillna(0)  # subtract from floor/ceiling & replace nan with 0
    df[regime_breakout] = np.sign(df[regime_breakout])  # if sign == -1 : bull breakout or bear breakdown
    df[regime_change] = np.where(
        np.sign(df[regime_floorceiling] * df[regime_breakout]) == -1,
        df[floorceiling],
        df[regime_change]  # re-assign fc
    )
    df[regime_floorceiling] = np.where(
        np.sign(df[regime_floorceiling] * df[regime_breakout]) == -1,
        df[regime_breakout], df[regime_floorceiling]  # rgme chg
    )

    return df


# Calculate rolling standard deviation
def rolling_stdev(
    price_col: pd.Series,
    window: int,
    min_periods: int,
    decimals: int
):
    """
    Rolling volatility rounded at 'decimals', starting at percentage 'min_per' of window 'window'
    """
    return round(
        price_col.rolling(
            window=window,
            min_periods=int(round(window * min_periods, 0))
        ).std(ddof=0),
        decimals
    )


# Calculates the signals
def signal_fcstmt(regime: pd.Series, st: pd.Series, mt: pd.Series, side: Side = None) -> pd.Series:
    """
    This function overimposes st/mt moving average cross condition on regime
    it will have an active position only if regime and moving averages are aligned
    Long : st-mt > 0 & regime == 1
    Short: st-mt < 0 & regime == -1
    TODO
        - check out the affect of this function in debugger
        - is double np.sign necessary?
        - is (regime * active) necessary after (regime * stmt_sign)?
    """
    # Calculate the sign of the stmt delta
    stmt_sign = np.sign((st - mt).fillna(0))
    # Calculate entries/exits based on regime and stmt delta

    if side is None:
        search = np.sign(regime * stmt_sign) == 1
    else:
        search = (np.sign(regime * stmt_sign) == 1 & (regime * side.value == 1))
    active = np.where(search, 1, 0)
    return regime * active


# Calculates the stop-loss
def stop_loss(
    signal: pd.Series,
    close: pd.Series,
    s_low: pd.Series,
    s_high: pd.Series
):
    """
    this function uses signals from previous function and swings to calculate stop loss
    1. join swing lows/highs, only keep value of first day of signal (to avoid reset)
    2. find cumulative lowest/highest close
    3. if close breaches stop loss, then crop signal column: np.nan thereafter
    4. return stop loss column
    """
    # stop loss calculation
    long_stops = s_low.loc[s_low.notna()]
    short_stops = s_high.loc[s_high.notna()]
    stoploss = pd.Series(index=signal.index)

    for signal_time, signal_value in pd.DataFrame(signal).signals.starts.signal.items():
        if signal_value == 1:
            prior_stops = long_stops.loc[:signal_time].to_list()
            opr = op.lt
        else:
            prior_stops = short_stops.loc[:signal_time].to_list()
            opr = op.gt

        entry = close.loc[signal_time]
        i = len(prior_stops) - 1
        while i >= 0:
            if opr(prior_stops[i], entry):
                stoploss.loc[signal_time] = prior_stops[i]
                break
            i -= 1

    stoploss = stoploss.fillna(method='ffill')  # extend first value with fillna

    # Bull: lowest close, Bear: highest close
    close_max = close.groupby(stoploss).cummax()
    close_min = close.groupby(stoploss).cummin()
    cum_close = np.where(signal == 1, close_min, np.where(signal == -1, close_max, 0))

    # reset signal where stop loss is breached
    sl_delta = (cum_close - stoploss).fillna(0)
    sl_sign = signal * np.sign(sl_delta)
    signal[sl_sign == -1] = 0
    return stoploss


# Calculates the transaction costs
def transaction_costs(data: pd.DataFrame, position_column: str, daily_return, transaction_cost: float):
    """
    identifies entries and exits by subtracting position column one row up .diff().
    fillna(0) is done before subtraction to avoid any residual na and after
    transaction_cost is subtracted from daily returns before compounding
    returns daily returns col adjusted for transactions
    """
    inout = data[position_column].fillna(0).diff().fillna(0) != 0  # boolean filter
    data[daily_return] = np.where(inout, data[daily_return] - float(transaction_cost), data[daily_return])
    return data[daily_return]


def init_fc_signal_stoploss(
    fc_data: pd.DataFrame,
    symbol: str,
    base_close: str,
    relative_close: str,
    ma_pairs: List[Tuple[float, float]],
    transaction_cost: float,
    percentile: float,
    min_periods: int,
    window: int,
    limit: int,
    side=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    initializes strategy for a given set of short term/ long term ma pairs.
    Selects the highest robustness score, also returns overview of all stats
    TODO return info on selected signal (smas, etc)
    :param side:
    :param ma_pairs:
    :param fc_data:
    :param symbol:
    :param base_close:
    :param relative_close:
    :param transaction_cost:
    :param percentile:
    :param min_periods:
    :param window:
    :param limit:
    :return:
    """

    # cumulative performance if multiple symbols are being analyzed
    best_rar = None
    # ==================================
    # Beginning of file loop in example
    # ==================================
    # Calculate returns for the relative closed price
    # fc_data['r_return_1d'] = returns(fc_data['rebased_close'])
    r_return_1d = 'r_return_1d'
    fc_data[r_return_1d] = simple_returns(fc_data[relative_close])

    # Calculate returns for the absolute closed price
    return_1d = 'return_1d'
    fc_data[return_1d] = simple_returns(fc_data[base_close])

    r_regime_floorceiling = 'regime_floorceiling'
    sw_rebased_low = 'sw_low'
    sw_rebased_high = 'sw_high'
    high_score = None
    stats = []
    best_stats = None

    for st, mt in ma_pairs:
        # Create dataframe
        data = fc_data[
            [
                relative_close,
                base_close,
                'high',
                'low',
                'sw_b_low',
                'sw_b_high',
                sw_rebased_low,
                sw_rebased_high,
                r_return_1d,
                return_1d,
                r_regime_floorceiling,
                'regime_change',
            ]
        ].copy()

        data = single_init_fc_signal_stoploss(
            fc_data=data,
            symbol=symbol,
            base_close=base_close,
            relative_close=relative_close,
            st_ma_per=int(st),
            mt_ma_per=int(mt),
            side=side
        )
        if data is None:
            continue

        stmt = str(st) + str(mt)
        signal_col = 'signal'
        daily_returns_col = 'd' + stmt

        # Date of initial position to calculate excess returns for passive
        # Passive stats are recalculated each time because start date changes with stmt sma
        first_position_dt = data.signals.slices()[0].index[0]
        data_sliced = data[first_position_dt:].copy()

        # Calculate daily & cumulative returns and include transaction costs
        # BJ: scale returns by a factor of the remaining position
        data_sliced[daily_returns_col] = (
            data_sliced['r_return_1d'] *
            data_sliced[signal_col].shift(1) *
            data_sliced.size_remaining
        ).fillna(0)

        data_sliced[daily_returns_col] = transaction_costs(
            data=data_sliced,
            position_column=signal_col,
            daily_return=daily_returns_col,
            transaction_cost=transaction_cost
        )

        # Cumulative performance must be higher than passive or regime (w/o transaction costs)

        full_stats = calc_stats(
            strategy_data=data_sliced,
            strategy_returns_col=daily_returns_col,
            passive_returns_col='r_return_1d',
            min_periods=min_periods,
            window=window,
            percentile=percentile,
            limit=limit
        )

        row = full_stats.iloc[-1].to_dict()

        row['ticker'] = symbol
        row['st'] = st
        row['mt'] = mt
        stats.append(row)

        # Save high_score for later use in the position sizing module
        if best_rar is None or row['risk_adjusted_returns'] > best_rar:
            best_rar = row['risk_adjusted_returns']
            high_score = data.copy()
            best_stats = full_stats

    stats = pd.DataFrame(stats)
    stats_out = stats.sort_values(by='risk_adjusted_returns').dropna(subset=['risk_adjusted_returns'])
    # TODO include interval in name
    try:

        stats_out.reset_index().to_feather(fr'C:\algo_data_store\strategy_data\strategy_scoreboard\{symbol}.ftr')
    except Exception as e:
        print(f'TODO catch exception {e}')
    try:
        best_stats.reset_index().to_feather(fr'C:\algo_data_store\strategy_data\full_stats\{symbol}.ftr')
    except Exception as e:
        print(f'TODO catch exception {e}')

    return (
        high_score,
        stats
    )


def calc_stats(
    strategy_data: pd.DataFrame,
    strategy_returns_col: str,
    passive_returns_col: str,
    min_periods: int,
    window: int,
    percentile: float,
    limit,
) -> pd.DataFrame:
    """
    get full stats of strategy, rolling and expanding
    :param strategy_data:
    :param strategy_returns_col:
    :param passive_returns_col:
    :param min_periods:
    :param window:
    :param percentile:
    :param limit:
    :return:
    """

    strategy_returns_1d = strategy_data[strategy_returns_col]
    passive_returns_1d = strategy_data[passive_returns_col]

    # Performance
    cumul_passive = cumulative_returns_pct(passive_returns_1d, min_periods)
    cumul_returns = cumulative_returns_pct(strategy_returns_1d, min_periods)
    cumul_excess = cumul_returns - cumul_passive - 1
    cumul_returns_pct = cumulative_returns_pct(strategy_returns_1d, min_periods)

    # Robustness metrics

    grit_expanding = ts2.expanding_grit(cumul_returns)
    grit_roll = ts2.rolling_grit(cumul_returns, window)

    tr_expanding = ts2.expanding_tail_ratio(cumul_returns, percentile, limit)
    tr_roll = ts2.rolling_tail_ratio(cumul_returns, window, percentile, limit)

    profits_expanding = ts2.expanding_profits(strategy_returns_1d)
    losses_expanding = ts2.expanding_losses(strategy_returns_1d)
    pr_expanding = ts2.profit_ratio(
        profits=profits_expanding,
        losses=losses_expanding
    )

    profits_roll = ts2.rolling_profits(strategy_returns_1d, window)
    losses_roll = ts2.rolling_losses(strategy_returns_1d, window)
    pr_roll = ts2.profit_ratio(
        profits=profits_roll,
        losses=losses_roll
    )

    # Cumulative t-stat
    win_count = strategy_returns_1d[strategy_returns_1d > 0].expanding().count().fillna(method='ffill')
    total_count = strategy_returns_1d[strategy_returns_1d != 0].expanding().count().fillna(method='ffill')

    csr_expanding = ts2.common_sense_ratio(pr_expanding, tr_expanding)
    csr_roll = ts2.common_sense_ratio(pr_roll, tr_roll)

    # Trade Count
    signals = strategy_data.signal.copy()
    trade_count = pd.Series(data=np.NAN, index=signals.index)
    # TODO check if first trade counts if first row is active signal
    trade_count.loc[signals.diff() != 0] = signals.loc[signals.diff() != 0].abs().cumsum()
    trade_count = trade_count.fillna(method='ffill')
    signal_roll = trade_count.diff(window)

    win_rate = (win_count / total_count).fillna(method='ffill')
    avg_win = profits_expanding / total_count
    avg_loss = losses_expanding / total_count
    edge_expanding = ts2.expectancy(win_rate, avg_win, avg_loss).fillna(method='ffill')
    sqn_expanding = ts2.t_stat(trade_count, edge_expanding)

    win_roll = strategy_returns_1d.copy()
    win_roll[win_roll < 0] = np.nan
    win_rate_roll = win_roll.rolling(window, min_periods=0).count() / window
    avg_win_roll = profits_roll / window
    avg_loss_roll = losses_roll / window

    edge_roll = ts2.expectancy(win_rate=win_rate_roll, avg_win=avg_win_roll, avg_loss=avg_loss_roll)
    sqn_roll = t_stat(signal_count=signal_roll, expectancy=edge_roll)

    score_expanding = ts2.robustness_score(grit_expanding, csr_expanding, sqn_expanding)
    score_roll = ts2.robustness_score(grit_roll, csr_roll, sqn_roll)

    stats = pd.DataFrame.from_dict({
        # Note: commented out items should be included afterwords
        # 'ticker': symbol,
        # 'tstmt': ticker_stmt,
        # 'st': st,
        # 'mt': mt,
        'perf': cumul_returns_pct,
        'excess': cumul_excess,

        # 'score': round(score_expanding[-1], 1),  # TODO remove (risk_adj_returns used for score)
        # 'score_roll': round(score_roll[-1], 1),  # TODO remove (risk_adj_returns used for score)

        'trades': trade_count,
        'win': win_rate,
        'win_roll': win_rate_roll,
        'avg_win': avg_win,
        'avg_win_roll': avg_win_roll,
        'avg_loss': avg_loss,
        'avg_loss_roll': avg_loss_roll,
        # 'geo_GE': round(geo_ge, 4),
        'expectancy': edge_expanding,
        'edge_roll': edge_roll,

        'grit': grit_expanding,
        'grit_roll': grit_roll,

        'csr': csr_expanding,
        'csr_roll': csr_roll,

        'pr': pr_expanding,
        'pr_roll': pr_roll,

        'tail': tr_expanding,
        'tail_roll': tr_roll,

        'sqn': sqn_expanding,
        'sqn_roll': sqn_roll,

        'risk_adjusted_returns': score_expanding,
        'risk_adj_returns_roll': score_roll,
    })
    return stats


def single_init_fc_signal_stoploss(
        fc_data: pd.DataFrame,
        symbol: str,
        base_close: str,
        relative_close: str,
        st_ma_per: int,
        mt_ma_per: int,
        side=None
):
    """

    :param side:
    :param mt_ma_per:
    :param st_ma_per:
    :param fc_data:
    :param symbol:
    :param base_close:
    :param relative_close:
    :return:
    """
    r_regime_floorceiling = 'regime_floorceiling'
    sw_rebased_low = 'sw_low'
    sw_rebased_high = 'sw_high'
    signal_col = 'signal'
    stop_loss_col = 'stop_loss'
    data = fc_data.copy()

    r_st_ma = sma(
        df=data,
        price=relative_close,
        ma_per=st_ma_per,
        min_per=1,
        decimals=None
    )
    data['st_ma'] = r_st_ma

    r_mt_ma = sma(
        df=data,
        price=relative_close,
        ma_per=mt_ma_per,
        min_per=1,
        decimals=None
    )
    data['mt_ma'] = r_mt_ma

    data[signal_col] = signal_fcstmt(
        regime=data[r_regime_floorceiling], st=r_st_ma, mt=r_mt_ma, side=side
    )

    if data.signals.count == 0:
        # no signals found, skip
        print('no initial signal found')
        return None

    # crop out first signal if data set does not capture where the first cross occurred
    stmt_sign = pd.Series(np.sign((r_st_ma - r_mt_ma))).backfill()
    ma_crosses = data.loc[stmt_sign != stmt_sign.shift(1).backfill()]
    try:
        first_cross_date = ma_crosses.index[0]
    except IndexError:
        if data.signals.count == 1:
            print(f'{symbol}: ma never crosses in data set')
            return None
    else:
        first_signal = data.signals.slices()[0]
        if first_signal.index[0] < first_cross_date:
            data.signal.loc[first_signal.index[0]: first_signal.index[-1]] = 0

        if data.signals.count == 0:
            # initial signal not valid. none left. skip
            print('no signal after first removed')
            return None

    # stop loss (relative)
    data[stop_loss_col] = stop_loss(
        signal=data[signal_col],
        close=data[relative_close],
        s_low=data[sw_rebased_low],
        s_high=data[sw_rebased_high]
    )

    # stop loss (absolute)
    data['stop_loss_base'] = stop_loss(
        signal=data[signal_col],
        close=data[base_close],
        s_low=data['sw_b_low'],
        s_high=data['sw_b_high']
    )

    if data.signals.count == 0:
        print(f'{symbol}: no signals after stop loss generation')
        # initial signal not valid. none left. skip
        return None

    data['trail_stop'] = data.stop_losses.init_trail_stop()
    (
        data['trail_stop_to_cost'],
        data['signal'],
        data['stop_status']
    ) = data.stop_losses.trail_to_cost(
        trail_stop=data.trail_stop
    )
    data.signal = data.signal.fillna(0)
    if data.signals.count == 0:
        print('no signals after trail stop generation')
        # initial signal not valid. none left. skip
        return None

    data['size_remaining'], data['target_price'] = data.signals.init_simulated_scale_out(
        stop_loss_col=stop_loss_col,
        # TODO target_r should be an input
        target_r=1.5
    )
    return data


def get_position_size(
    data: pd.DataFrame,
    constant_risk: float,
    round_lot: int,
    capital: Union[float, None] = None  # K
) -> pd.DataFrame:
    """
    :param round_lot:
    :param capital: total value of account
    :param data:
    :param constant_risk:
    :return:
    """
    data_cpy = data.copy()

    # Define posSizer weight
    data_cpy['eqty_risk'] = data_cpy.signals.equity_risk_weight(risk=constant_risk)

    # Instantiation of equity curves
    data_cpy['equity_at_risk'] = capital
    # data_cpy['equal_weight'] = capital
    data_cpy.eqty_risk_lot = data_cpy.signals.calc_lot(
        capital=data_cpy.equity_at_risk.iloc[0],
        weight_col='eqty_risk',
        round_lot=round_lot,
        fx_rate=1
    )

    return data_cpy


if __name__ == '__main__':
    LocalClient.price_history(
        'FNILX',
        freq_range=tdargs.freqs.day.range(tdargs.periods.y2, datetime.today())
    )

