# Data manipulation libraries
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import *
from scipy.signal import argrelextrema

# Imports charting libraries
import matplotlib.pyplot as plt

# The function 'read_data' will read csv files into a dataframe and plot the graphs
def read_data(file_name, column_name, y_label, title_name):
    relative_path = "../data_modules/"
    dataframe = pd.read_csv(
        relative_path + file_name + ".csv", index_col=0, parse_dates=True
    )
    file_name = dataframe[column_name].plot(figsize=(10, 7))
    plt.title(title_name, fontsize=16)
    plt.ylabel(y_label, fontsize=14)
    plt.xlabel("Date", fontsize=14)
    # Add legend to the plot
    plt.legend()
    # Add grid to the plot
    plt.grid()
    # Display the graph
    plt.show()

    return dataframe


# Define relative function


def relative(
    stock_dataframe,
    benchmark_dataframe,
    benchmark_name,
    forex_dataframe,
    forex_name,
    decimals,
    start,
    end,
):

    # Slice dataframe from start to end period: either offset or datetime
    stock_dataframe = stock_dataframe[start:end]

    # Join the dataset: Concatenation of benchmark, stock & currency
    data = pd.concat(
        [stock_dataframe, forex_dataframe, benchmark_dataframe], axis=1, sort=True
    ).dropna()

    # Adjustment factor: Calculate the product of benchmark and currency
    data["adjustment_factor"] = data[benchmark_name] * (data[forex_name])

    # Relative series: Calculate the relative series by dividing the OHLC stock data with the adjustment factor
    data["relative_open"] = data["Open"] / data["adjustment_factor"]
    data["relative_high"] = data["High"] / data["adjustment_factor"]
    data["relative_low"] = data["Low"] / data["adjustment_factor"]
    data["relative_close"] = data["Close"] / data["adjustment_factor"]

    # Rebased series: Multiply relative series with the first value of the adjustment factor to get the rebased series
    data["rebased_open"] = data["relative_open"] * data["adjustment_factor"].iloc[0]
    data["rebased_high"] = data["relative_high"] * data["adjustment_factor"].iloc[0]
    data["rebased_low"] = data["relative_low"] * data["adjustment_factor"].iloc[0]
    data["rebased_close"] = data["relative_close"] * data["adjustment_factor"].iloc[0]

    # Rounds values upto 2 decimal places
    decimals = 2
    data = round(data, decimals)

    return data


# Calculates the returns
def returns(prices):
    """
    calculates log returns based on price series
    """
    rets = pd.Series(prices)
    log_returns = np.log(rets / rets.shift(1))
    return log_returns


def cumulative_returns(returns):
    """
    Calculates cumulative (expanding from initial value) sum, applies exponential
    """
    rets = pd.Series(returns)
    cum_log_return = rets.cumsum().apply(np.exp)
    return cum_log_return


def cum_return_percent(returns):
    """
    Calculates cumulative returns and returns as percentage
    """
    rets = pd.Series(returns)
    cum_log_returns = round((rets.cumsum().apply(np.exp) - 1) * 100, 1)
    return cum_log_returns


# Calculate simple moving average
def sma(df, price, ma_per, min_per, decimals):
    """
    Returns the simple moving average.
    price: column within the df
    ma_per: moving average periods
    min_per: minimum periods (expressed as 0<pct<1) to calculate moving average
    decimals: rounding number of decimals
    
    
    """
    sma = round(
        df[price]
        .rolling(window=ma_per, min_periods=int(round(ma_per * min_per, 0)))
        .mean(),
        decimals,
    )
    return sma


# Calculate exponential moving average
def ema(df, price, ma_per, min_per, decimals):
    """
    Returns exponentially weighted moving average. 
    
    price: column within the df
    ma_per: moving average periods
    min_per: minimum periods (expressed as 0<pct<1) to calculate moving average
    decimals: rounding number of a
    
    """
    ema = round(
        df[price].ewm(span=ma_per, min_periods=int(round(ma_per * min_per, 0))).mean(),
        decimals,
    )
    return ema


# Calculate regime simple moving average
def regime_sma(df, price, short_term, long_term):
    """
    when price >= sma bull +1, when price < sma: bear -1, fillna 
    """
    # define rolling high/low
    sma_st = df[price].rolling(window=short_term, min_periods=short_term).mean()
    sma_lt = df[price].rolling(window=long_term, min_periods=long_term).mean()

    # when price>= sma: bull, when price<sma: bear
    df["regime_sma" + "_" + str(short_term) + "_" + str(long_term)] = np.where(
        sma_st >= sma_lt, 1, np.where(sma_st < sma_lt, -1, np.nan)
    )
    df["regimes_sma" + "_" + str(short_term) + "_" + str(long_term)] = df[
        "regime_sma" + "_" + str(short_term) + "_" + str(long_term)
    ].fillna(method="ffill")
    return df


# Calculate regime exponential moving average
def regime_ema(df, price, short_term, long_term):
    """
    when price >= ema bull +1, when price < ema: bear -1, fillna 
    """
    # define rolling high/low
    ema_st = df[price].ewm(span=short_term, min_periods=short_term).mean()
    ema_lt = df[price].ewm(span=long_term, min_periods=long_term).mean()

    # when price>= sma: bull, when price<sma: bear
    df["regime_ema" + "_" + str(short_term) + "_" + str(long_term)] = np.where(
        ema_st >= ema_lt, 1, np.where(ema_st < ema_lt, -1, np.nan)
    )
    df["regime_ema" + "_" + str(short_term) + "_" + str(long_term)] = df[
        "regime_ema" + str(short_term) + str(long_term)
    ].fillna(method="ffill")
    return df


# Plot regime
def graph_regime_fc(ticker, df, y, th, sl, sh, clg, flr, bs, rg, st, mt, bo):

    fig = plt.figure(figsize=(10, 7))
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    date = df.index
    close = df[y]
    swing_low = df[sl]
    swing_high = df[sh]
    ceiling = df[clg]
    floor = df[flr]

    base = df[bs]
    regime = df[rg]

    ax1.plot_date(
        df.index, close, "-", color="k", label=ticker.upper() + " stdev:" + str(th)
    )
    ax1.plot(df.index, swing_low, ".", color="r", label="swing low", alpha=0.5)
    ax1.plot(df.index, swing_high, ".", color="g", label="swing high", alpha=0.5)
    plt.scatter(df.index, floor, c="b", marker="^", label="floor")
    plt.scatter(df.index, ceiling, c="m", marker="v", label="ceiling")

    ax1.plot([], [], linewidth=5, label="bear", color="m", alpha=0.1)
    ax1.plot([], [], linewidth=5, label="bull", color="b", alpha=0.1)
    ax1.fill_between(
        date,
        close,
        base,
        where=((regime == 1) & (close > base)),
        facecolor="b",
        alpha=0.1,
    )
    #     ax1.fill_between(date, close, base,where=((regime==1)&(close<base)), facecolor='b', alpha=0.8)
    ax1.fill_between(
        date,
        close,
        base,
        where=((regime == -1) & (close < base)),
        facecolor="m",
        alpha=0.1,
    )
    #     ax1.fill_between(date, close, base,where=((regime==-1)&(close>base)), facecolor='m', alpha=0.8)

    if np.sum(st) > 0:
        ax1.plot(df.index, st, "-", color="lime", label=" st")
        ax1.plot(df.index, mt, "-", color="green", label=" mt")  # 2. plot line
        # Profitable conditions
        ax1.fill_between(
            date,
            close,
            mt,
            where=((regime == 1) & (st >= mt) & (close >= mt)),
            facecolor="green",
            alpha=0.3,
        )
        ax1.fill_between(
            date,
            close,
            mt,
            where=((regime == -1) & (st <= mt) & (close <= mt)),
            facecolor="red",
            alpha=0.3,
        )
        # Unprofitable conditions
        ax1.fill_between(
            date,
            close,
            mt,
            where=((regime == 1) & (st >= mt) & (close < mt)),
            facecolor="darkgreen",
            alpha=1,
        )
        ax1.fill_between(
            date,
            close,
            mt,
            where=((regime == -1) & (st <= mt) & (close > mt)),
            facecolor="darkred",
            alpha=1,
        )

    if bo > 0:
        #         ax1.plot([],[],linewidth=5, label=str(bo)+' days high', color='m',alpha=0.3)
        #         ax1.plot([],[],linewidth=5, label=str(bo) + ' days low', color='b',alpha=0.3)
        rolling_min = close.rolling(window=bo)._min()
        rolling_max = close.rolling(window=bo)._max()
        ax1.fill_between(
            date,
            close,
            rolling_min,
            where=((regime == 1) & (close > rolling_min)),
            facecolor="b",
            alpha=0.2,
        )
        ax1.fill_between(
            date,
            close,
            rolling_max,
            where=((regime == -1) & (close < rolling_max)),
            facecolor="m",
            alpha=0.2,
        )
        ax1.plot(
            df.index, rolling_min, "-.", color="b", label=str(bo) + " low", alpha=0.5
        )
        ax1.plot(
            df.index, rolling_max, "-.", color="m", label=str(bo) + " low", alpha=0.5
        )

    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax1.grid(True)
    ax1.xaxis.label.set_color("g")
    ax1.yaxis.label.set_color("g")

    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.title("Floor and Ceiling", fontsize=16)
    plt.legend()


# Calculate regime breakout-breakdown
def regime_breakout_breakdown(breakout_period, df, high, low):
    """
    when new hi: bull +1, when new low: bear -1, fillna 
    """

    # define rolling high/low
    rolling_high = (
        df[high].rolling(window=breakout_period, min_periods=breakout_period)._max()
    )  #
    rolling_low = (
        df[low].rolling(window=breakout_period, min_periods=breakout_period)._min()
    )

    # when new high bull, when new low: bear
    df["regime_breakout_breakdown" + "_" + str(breakout_period)] = np.where(
        df[high] >= rolling_high, 1, np.where(df[low] <= rolling_low, -1, np.nan)
    )
    df["regime_breakout_breakdown" + "_" + str(breakout_period)] = df[
        "regime_breakout_breakdown" + "_" + str(breakout_period)
    ].fillna(method="ffill")
    return df


# Calculate swings
def swings(df, high, low, argrel_window):

    # Create swings:

    # Step 1: copy existing df. We will manipulate and reduce this df and want to preserve the original
    high_low = df[[high, low]].copy()

    # Step 2: build 2 lists of highs and lows using argrelextrema
    highs_list = argrelextrema(high_low[high].values, np.greater, order=argrel_window)
    lows_list = argrelextrema(high_low[low].values, np.less, order=argrel_window)

    # Step 3: Create swing high and low columns and assign values from the lists
    swing_high = "s" + str(high)[-12:]
    swing_low = "s" + str(low)[-12:]
    high_low[swing_low] = high_low.iloc[lows_list[0], 1]
    high_low[swing_high] = high_low.iloc[highs_list[0], 0]

    # Alternation: We want highs to follow lows and keep the most extreme values

    # Step 4. Create a unified column with peaks<0 and troughs>0
    swing_high_low = str(high)[:2] + str(low)[:2]
    high_low[swing_high_low] = high_low[swing_low].sub(
        high_low[swing_high], fill_value=0
    )

    # Step 5: Reduce dataframe and alternation loop
    # Instantiate start
    i = 0
    # Drops all rows with no swing
    high_low = high_low.dropna(subset=[swing_high_low]).copy()
    while ((high_low[swing_high_low].shift(1) * high_low[swing_high_low] > 0)).any():
        # eliminate lows higher than highs
        high_low.loc[
            (high_low[swing_high_low].shift(1) * high_low[swing_high_low] < 0)
            & (high_low[swing_high_low].shift(1) < 0)
            & (np.abs(high_low[swing_high_low].shift(1)) < high_low[swing_high_low]),
            swing_high_low,
        ] = np.nan
        # eliminate earlier lower values
        high_low.loc[
            (high_low[swing_high_low].shift(1) * high_low[swing_high_low] > 0)
            & (high_low[swing_high_low].shift(1) < high_low[swing_high_low]),
            swing_high_low,
        ] = np.nan
        # eliminate subsequent lower values
        high_low.loc[
            (high_low[swing_high_low].shift(-1) * high_low[swing_high_low] > 0)
            & (high_low[swing_high_low].shift(-1) < high_low[swing_high_low]),
            swing_high_low,
        ] = np.nan
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
        np.isnan(high_low[swing_high_low]), 0, high_low[swing_high_low]
    )
    # If last_sign <0: swing high, if > 0 swing low
    last_sign = np.sign(high_low[swing_high_low][-1])

    # Step 8: Instantiate last swing high and low dates
    last_slo_dt = df[df[swing_low] > 0].index._max()
    last_shi_dt = df[df[swing_high] > 0].index._max()

    # Step 9: Test for extreme values
    if (last_sign == -1) & (last_shi_dt != df[last_slo_dt:][swing_high].idxmax()):
        # Reset swing_high to nan
        df.loc[last_shi_dt, swing_high] = np.nan
    elif (last_sign == 1) & (last_slo_dt != df[last_shi_dt:][swing_low].idxmax()):
        # Reset swing_low to nan
        df.loc[last_slo_dt, swing_low] = np.nan

    return df


# Calculate regime floor ceiling
def regime_fc(df, close, swing_low, swing_high, threshold, t_dev, decimals):
    # 1. copy swing lows and highs to a smaller df called regime
    regime = df[(df[swing_low] > 0) | (df[swing_high] > 0)][
        [close, swing_low, swing_high]
    ].copy()

    # 2. calculate volatility from main df and populate regime df
    stdev = "stdev"
    regime[stdev] = rolling_stdev(
        df=df, price=close, t_dev=t_dev, min_per=1, decimals=decimals
    )

    # 3. Variables declaration
    floor_ix, ceiling_ix, floor_test, ceiling_test = [], [], [], []

    ceiling_found = False
    floor_found = False

    # 4.  instantiate columns based on absolute or relative series
    if str(close)[0] == "r":  # a) test the first letter of the cl input variable
        # if 1st letter =='r', relative series, add 'r_' prefix
        rg_cols = [
            "r_floor",
            "r_ceiling",
            "r_regime_change",
            "r_regime_floorceiling",
            "r_floorceiling",
            "r_regime_breakout",
        ]
    else:  # absolute series
        rg_cols = [
            "floor",
            "ceiling",
            "regime_change",
            "regime_floorceiling",
            "floorceiling",
            "regime_breakout",
        ]

    # b) instantiate columns by concatenation
    regime = pd.concat(
        [
            regime,  # existing df
            # temporary df with same index, regime columns initialised at nan
            pd.DataFrame(np.nan, index=regime.index, columns=rg_cols),
        ],  # temp df
        axis=1,
    )  # along the vertical axis

    # c) column variables names instantiation via a list comprehension
    (
        floor,
        ceiling,
        regime_change,
        regime_floorceiling,
        floorceiling,
        regime_breakout,
    ) = [list(rg_cols)[n] for n in range(len(rg_cols))]

    # 5. Range initialisation to 1st swing
    floor_ix = regime.index[0]
    ceiling_ix = regime.index[0]

    # 6. Loop through swings
    for i in range(1, len(regime)):

        if regime[swing_high][i] > 0:  # ignores swing lows
            top = regime[floor_ix : regime.index[i]][
                swing_high
            ]._max()  # highest swing high from range[floor_ix:swing[i]]
            ceiling_test = round(
                (regime[swing_high][i] - top) / regime[stdev][i], 1
            )  # test vs highest

            if ceiling_test <= -threshold:  # if swing <= top - x * stdev
                ceiling = regime[floor_ix : regime.index[i]][
                    swing_high
                ]._max()  # ceiling = top
                ceiling_ix = regime[floor_ix : regime.index[i]][
                    swing_high
                ].idxmax()  # ceiling index
                regime.loc[ceiling_ix, ceiling] = ceiling  # assign ceiling

                if ceiling_found == False:  # test met == ceiling found
                    rg_chg_ix = regime[swing_high].index[i]
                    _rg_chg = regime[swing_high][i]
                    regime.loc[
                        rg_chg_ix, regime_change
                    ] = _rg_chg  # prints where/n ceiling found
                    regime.loc[rg_chg_ix, regime_floorceiling] = -1  # regime change
                    regime.loc[
                        rg_chg_ix, floorceiling
                    ] = ceiling  # used in floor/ceiling breakout test

                    ceiling_found = True  # forces alternation btwn Floor & ceiling
                    floor_found = False

        if regime[swing_low][i] > 0:  # ignores swing highs
            bottom = regime[ceiling_ix : regime.index[i]][
                swing_low
            ]._min()  # lowest swing low from ceiling
            floor_test = round(
                (regime[swing_low][i] - bottom) / regime[stdev][i], 1
            )  # test vs lowest

            if floor_test >= threshold:  # if swing > bottom + n * stdev
                floor = regime[ceiling_ix : regime.index[i]][
                    swing_low
                ]._min()  # floor = bottom
                floor_ix = regime[ceiling_ix : regime.index[i]][swing_low].idxmin()
                regime.loc[floor_ix, floor] = floor  # assign floor

                if floor_found == False:  # test met == floor found
                    rg_chg_ix = regime[swing_low].index[i]
                    _rg_chg = regime[swing_low][i]
                    regime.loc[
                        rg_chg_ix, regime_change
                    ] = _rg_chg  # prints where/n floor found
                    regime.loc[rg_chg_ix, regime_floorceiling] = 1  # regime change
                    regime.loc[
                        rg_chg_ix, floorceiling
                    ] = floor  # used in floor/ceiling breakdown test

                    ceiling_found = False  # forces alternation btwn floor/ceiling
                    floor_found = True

    # 7. join regime to df

    # 8. drop rg_cols if already in df to avoid overlap error
    if (
        regime_floorceiling in df.columns
    ):  # drop columns if already in df before join, otherwise overlap error
        df = df.drop(rg_cols, axis=1)
    df = df.join(regime[rg_cols], how="outer")

    # 9. forward fill regime 'rg_fc','rg_chg','fc', then fillna(0) from start to 1st value
    df[regime_floorceiling] = (
        df[regime_floorceiling].fillna(method="ffill").fillna(0)
    )  # regime
    df[regime_change] = df[regime_change].fillna(method="ffill").fillna(0)  # rg_chg
    df[floorceiling] = (
        df[floorceiling].fillna(method="ffill").fillna(0)
    )  # floor ceiling continuous line

    # 10. test reakout/down: if price crosses floor/ceiling, regime change
    close_max = df.groupby([floorceiling])[
        close
    ].cummax()  # look for highest close for every floor/ceiling
    close_min = df.groupby([floorceiling])[
        close
    ].cummin()  # look for lowest close for every floor/ceiling

    # 11. if rgme bull: assign lowest close, elif bear: highest close
    rgme_close = np.where(
        df[floorceiling] < df[regime_change],
        close_min,
        np.where(df[floorceiling] > df[regime_change], close_max, 0),
    )

    df[regime_breakout] = (rgme_close - df[floorceiling]).fillna(
        0
    )  # subtract from floor/ceiling & replace nan with 0
    df[regime_breakout] = np.sign(
        df[regime_breakout]
    )  # if sign == -1 : bull breakout or bear breakdown
    df[regime_change] = np.where(
        np.sign(df[regime_floorceiling] * df[regime_breakout]) == -1,
        df[floorceiling],
        df[regime_change],
    )  # re-assign fc
    df[regime_floorceiling] = np.where(
        np.sign(df[regime_floorceiling] * df[regime_breakout]) == -1,
        df[regime_breakout],
        df[regime_floorceiling],
    )  # rgme chg

    return df


# Calculate rolling standard deviation
def rolling_stdev(df, price, t_dev, min_per, decimals):
    """
    Rolling volatility rounded at 'decimals', starting at percentage 'min_per' of window 't_dev'
    """

    stdev = round(
        df[price]
        .rolling(window=t_dev, min_periods=int(round(t_dev * min_per, 0)))
        .std(ddof=0),
        decimals,
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
    stoploss = (s_low.add(s_high, fill_value=0)).fillna(
        method="ffill"
    )  # join all swings in 1 column
    stoploss[
        ~((np.isnan(signal.shift(1))) & (~np.isnan(signal)))
    ] = np.nan  # keep 1st sl by signal
    stoploss = stoploss.fillna(method="ffill")  # extend first value with fillna

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
def transaction_costs(df, position_column, daily_return, transaction_cost):
    """
    identifies entries and exits by subtracting position column one row up .diff().
    fillna(0) is done before subtraction to avoid any residual na and after 
    transaction_cost is subtracted from daily returns before compounding
    returns daily returns col adjusted for transactions
    """
    inout = df[position_column].fillna(0).diff().fillna(0) != 0  # boolean filter
    df[daily_return] = np.where(
        inout, df[daily_return] - float(transaction_cost), df[daily_return]
    )
    return df[daily_return]
