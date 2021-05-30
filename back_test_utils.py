import matplotlib.pyplot as plt
import tda
import pandas as pd
import numpy as np
import selenium.webdriver
import scipy.signal
import tdargs
import trade_df
from datetime import datetime
from typing import List, Union
from tda_access import LocalClient
import itertools  # construct a list of permutations
from trade_stats import (
    simple_returns, cumulative_returns, cum_return_percent, count_signals,
    cumulative_returns_pct, rolling_profits, hit_rate, average_win,
    average_loss, george, grit_index, calmar_ratio, tail_ratio, t_stat,
    common_sense_ratio, equity_at_risk, round_lot

)


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
    crosses = diff_from_fast[(prev_diff < 0) & (0 < diff_from_fast) | (diff_from_fast < 0) & (0 < prev_diff)]
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
    if forex_df is None:
        forex_values = 1
    else:
        forex_values = forex_df.values

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
    decimals: int
):
    """
    Returns the simple moving average.
    price: column within the df
    ma_per: moving average periods
    min_per: minimum periods (expressed as 0<pct<1) to calculate moving average
    decimals: rounding number of decimals
    """
    sma_col = round(df[price].rolling(window=ma_per, min_periods=int(round(ma_per * min_per, 0))).mean(), decimals)
    return sma_col


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
    ema_col = round(df[price].ewm(span=ma_per, min_periods=int(round(ma_per * min_per, 0))).mean(), decimals)
    return ema_col


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
def graph_regime_fc(ticker, df, y, th, sl, sh, clg, flr, bs, rg, st, mt, bo):
    # fig = plt.figure(figsize=(10, 7))
    ax1 = plt.subplot2grid((1, 1), (0, 0))

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
    argrelwindow: int,
    prefix: str = 'sw'
) -> pd.DataFrame:
    """
    Will raise ValueError if no swings found above 0 on high or low side
    :param df:
    :param high: col name for highs of day
    :param low: col name for lows of day
    :param argrelwindow: window to detect local highs and lows
    :param prefix: prefix for swing high and swing low column name
    :return:
    """
    # Create swings:

    # Step 1: copy existing df. We will manipulate and reduce this df and want to preserve the original
    high_low = df[[high, low]].copy()

    # Step 2: build 2 lists of highs and lows using argrelextrema
    highs_list = scipy.signal.argrelextrema(
        high_low[high].values, np.greater, order=argrelwindow)
    lows_list = scipy.signal.argrelextrema(
        high_low[low].values, np.less, order=argrelwindow)

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
    last_sign = np.sign(high_low[swing_high_low][-1])

    # Step 8: Instantiate last swing high and low dates
    last_slo_dt = df[df[swing_low] > 0].index.max()
    last_shi_dt = df[df[swing_high] > 0].index.max()
    if pd.isnull(last_slo_dt) or pd.isnull(last_shi_dt):
        # TODO improve error if necessary
        raise ValueError('Quit swing calc. [Swing High > 0 AND Swing Low] condition fails.')

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
    stdev_window: int,
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
        window=stdev_window,
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

                if ceiling_found is False:  # test met == ceiling found
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

                if floor_found is False:  # test met == floor found
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
    stdev = round(
        price_col.rolling(
            window=window,
            min_periods=int(round(window * min_periods, 0))
        ).std(ddof=0),
        decimals
    )
    return stdev


# Calculates the signals
def signal_fcstmt(regime, st, mt):
    """
    This function overimposes st/mt moving average cross condition on regime
    it will have an active position only if regime and moving averages are aligned
    Long : st-mt > 0 & regime == 1
    Short: st-mt < 0 & regime == -1

    """
    # Calculate the sign of the stmt delta
    stmt_sign = np.sign((st - mt).fillna(0))

    # Calculate entries/exits based on regime and stmt delta
    active = np.where(np.sign(regime * stmt_sign) == 1, 1, np.nan)
    signal = regime * active

    return signal


# Calculates the stop-loss
def stop_loss(signal, close, s_low, s_high):
    """
    this function uses signals from previous function and swings to calculate stop loss
    1. join swing lows/highs, only keep value of first day of signal (to avoid reset)
    2. find cumulative lowest/highest close
    3. if close breaches stop loss, then crop signal column: np.nan thereafter
    4. return stop loss column
    """
    # stop loss calculation
    stoploss = (s_low.add(s_high, fill_value=0)).fillna(method='ffill')  # join all swings in 1 column
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


# Calculates the transaction costs
def transaction_costs(df, position_column, daily_return, tcs):
    """
    identifies entries and exits by subtracting position column one row up .diff().
    fillna(0) is done before subtraction to avoid any residual na and after
    tcs is subtracted from daily returns before compounding
    returns daily returns col adjusted for transactions
    """
    inout = df[position_column].fillna(0).diff().fillna(0) != 0  # boolean filter
    df[daily_return] = np.where(inout, df[daily_return] - float(tcs), df[daily_return])
    return df[daily_return]


def score_all(
    fc_data_list: List[pd.DataFrame],  # fc_data_list: List[trade_df.PriceMdf],
    base_close: str,
    relative_close: str,
    st_list: List[int],
    mt_list: List[int],
    tcs: float,
    percentile: float,
    minperiods: int,
    window: int,
    limit: int,
):

    """
    We loop through the four different stock data sets.
    Steps within the for loop: ​

        Create a relative series
        Calculate the swings
        Calculate the stock regime
        Calculate returns for the relative and absolute closed price
        Create another for loop ​

    Steps within the second for loop:

        Start with a condition: If short_term < mid_term,
        Create dataframe
        Calculate moving averages
        Calculate positions based on the regime and moving average cross
        Calculate excess returns for passive
        Calculate daily & cumulative returns and include transaction costs
        Calculate the gain expectancies, Performance and robustness.
        Append the performance to a list
    :return:
    """

    new = fc_data_list.copy()

    # Instantiate variables
    minperiods = 50
    window = 200
    percentile = 0.05
    limit = 5
    tcs = 0.0025

    # Instantiate the permutations of moving averages
    st_list = range(10, 101, 10)
    mt_list = range(50, 201, 20)

    # Instantiate list dictionary and perf dataframe
    list_dict = []
    col_order = ['stmt', 'score', 'csr', 'grit', 'sqn', 'perf']

    # Fields to optimise (here, we optimise for robustness)
    optimised_fields = ['score']
    display_rows = 10
    best_risk_adjusted_returns = 0

    scoreboard = pd.DataFrame(columns=col_order)
    high_score = pd.DataFrame()
    perf = pd.DataFrame()
    # TODO run score_card for all dfs
    for fc in fc_data_list:
        # res_perf, new_row, best_rar = init_fc_signal_stoploss(
        #     fc_data=fc.data,
        #     symbol=fc.symbol,
        #     base_close=base_close,
        #     relative_close=relative_close,
        #
        # )
        pass


def init_fc_signal_stoploss(
    fc_data: pd.DataFrame,
    symbol: str,
    base_close: str,
    relative_close: str,
    st_list: range,
    mt_list: range,
    tcs: float,
    percentile: float,
    minperiods: int,
    window: int,
    limit: int,
    best_risk_adjusted_returns: int,
):
    """
    TODO return info on selected signal
    :param fc_data:
    :param symbol:
    :param base_close:
    :param relative_close:
    :param st_list:
    :param mt_list:
    :param tcs:
    :param percentile:
    :param minperiods:
    :param window:
    :param limit:
    :param best_risk_adjusted_returns:
    :return:
    """
    perf = pd.DataFrame()
    best_rar = best_risk_adjusted_returns
    # ==================================
    # Beginning of file loop in example
    # ==================================
    # Calculate returns for the relative closed price
    # fc_data['r_return_1d'] = returns(fc_data['rebased_close'])
    # rets = pd.Series(fc_data['rebased_close'])
    rets = fc_data[relative_close]
    r_return_1d = 'r_return_1d'
    fc_data[r_return_1d] = np.log(rets / rets.shift(1))
    row = {}

    # Calculate returns for the absolute closed price
    # fc_data['return_1d'] = returns(fc_data['Close'])
    rets = fc_data[base_close]
    return_1d = 'return_1d'
    fc_data[return_1d] = np.log(rets / rets.shift(1))

    r_regime_floorceiling = 'regime_floorceiling'
    sw_rebased_low = 'sw_low'
    sw_rebased_high = 'sw_high'
    high_score = None
    for st, mt in itertools.product(st_list, mt_list):
        if not st < mt:
            continue
        # Create dataframe
        data = pd.DataFrame()
        data[
            [relative_close, base_close, r_return_1d, return_1d,
             r_regime_floorceiling, sw_rebased_low, sw_rebased_high]
        ] = fc_data[
            [relative_close, base_close, r_return_1d, return_1d,
             r_regime_floorceiling, sw_rebased_low, sw_rebased_high]
        ].copy()

        # Calculate moving averages
        stmt = str(st) + str(mt)
        r_st_ma = sma(
            df=data,
            price=relative_close,
            ma_per=st,
            min_per=1,
            decimals=2
        )
        r_mt_ma = sma(
            df=data,
            price=relative_close,
            ma_per=mt,
            min_per=1,
            decimals=2
        )

        # Calculate positions based on regime and ma cross
        data['s' + stmt] = signal_fcstmt(
            regime=data[r_regime_floorceiling], st=r_st_ma, mt=r_mt_ma
        )

        signals = data[pd.notnull(data['s' + stmt])]
        if len(signals) == 0:
            # no signals found, skip
            continue

        first_position_dt = signals.index[0]

        data['sl' + stmt] = stop_loss(
            signal=data['s' + stmt],
            close=data[relative_close],
            s_low=data[sw_rebased_low],
            s_high=data[sw_rebased_high]
        )

        # Date of initial position to calculate excess returns for passive
        # Passive stats are recalculated each time because start date changes with stmt sma
        # TODO move to first_position_dt
        data_sliced = data[first_position_dt:].copy()

        # Calculate daily & cumulative returns and include transaction costs
        data_sliced['d' + stmt] = data_sliced['r_return_1d'] * data_sliced['s' + stmt].shift(1)
        data_sliced['d' + stmt] = transaction_costs(
            df=data_sliced,
            position_column='s' + stmt,
            daily_return='d' + stmt,
            tcs=tcs
        )

        # Cumulative performance must be higher than passive or regime (w/o transaction costs)
        passive_1d = data_sliced['r_return_1d']
        returns = data_sliced['d' + stmt]

        # Performance
        trade_count = count_signals(signals=data_sliced['s' + stmt])
        cumul_passive = cumulative_returns(passive_1d, minperiods)
        cumul_returns = cumulative_returns(returns, minperiods)
        cumul_excess = cumul_returns - cumul_passive - 1
        cumul_returns_pct = cumulative_returns_pct(returns, minperiods)
        roll_profits = rolling_profits(
            returns, window).fillna(method='ffill')

        # Gain Expectancies
        _hit_rate = hit_rate(returns, minperiods)
        _avg_win = average_win(returns, minperiods)
        _avg_loss = average_loss(returns, minperiods)
        geo_ge = george(win_rate=_hit_rate, avg_win=_avg_win,
                        avg_loss=_avg_loss).apply(np.exp) - 1

        # Robustness metrics
        grit = grit_index(returns, minperiods)
        calmar = calmar_ratio(returns, minperiods)
        pr = roll_profits
        tr = tail_ratio(returns, window, percentile, limit)
        csr = common_sense_ratio(pr, tr)
        sqn = t_stat(signal_count=trade_count, expectancy=geo_ge)

        ticker_stmt = f'{symbol}_{str(stmt)}'

        # Add cumulative performance to the perf dataframe
        perf[ticker_stmt] = cumul_returns

        # Append list
        row = {
            'ticker': symbol,
            'tstmt': ticker_stmt,
            'stmt': stmt,
            'perf': round(cumul_returns_pct[-1], 3),
            'excess': round(cumul_excess[-1], 3),
            'score': round(grit[-1] * csr[-1] * sqn[-1], 1),
            'trades': trade_count[-1],
            'win': round(_hit_rate[-1], 3),
            'avg_win': round(_avg_win[-1], 3),
            'avg_loss': round(_avg_loss[-1], 3),
            'geo_GE': round(geo_ge[-1], 4),
            'grit': round(grit[-1], 1),
            'csr': round(csr[-1], 1),
            'p2l': round(pr[-1], 1),
            'tail': round(tr[-1], 1),
            'sqn': round(sqn[-1], 1),
            'risk_adjusted_returns': csr[-1] * sqn[-1] * grit[-1]
        }

        # Save high_score for later use in the position sizing module
        if row['risk_adjusted_returns'] > best_rar:
            best_rar = row['risk_adjusted_returns']
            high_score = data.copy()
            high_score['score'] = row['risk_adjusted_returns']
            high_score['trades'] = trade_count
            high_score['r_perf'] = cumul_returns
            high_score['csr'] = csr
            high_score['geo_GE'] = geo_ge
            high_score['sqn'] = sqn
    """
    perf, row, best_rar: variables at a higher state
    high_score: the best results in terms of robustness score()
    
    """
    return high_score, perf, row, best_rar


if __name__ == '__main__':
    LocalClient.price_history(
        'FNILX',
        freq_range=tdargs.freqs.day.range(tdargs.periods.y2, datetime.today())
    )

