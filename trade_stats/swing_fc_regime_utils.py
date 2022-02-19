from dataclasses import dataclass
from datetime import datetime, timedelta
import typing as t
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import yfinance as yf
import pd_accessors as pda
import operator as op


def regime_breakout(df, _h, _l, window):
    hl = np.where(
        df[_h] == df[_h].rolling(window).max(),
        1,
        np.where(df[_l] == df[_l].rolling(window).min(), -1, np.nan),
    )
    roll_hl = pd.Series(index=df.index, data=hl).fillna(method="ffill")
    return roll_hl


def lower_upper_ohlc(df, is_relative=False):
    if is_relative == True:
        rel = "r"
    else:
        rel = ""
    if "Open" in df.columns:
        ohlc = [rel + "Open", rel + "High", rel + "Low", rel + "Close"]
    elif "open" in df.columns:
        ohlc = [rel + "open", rel + "high", rel + "low", rel + "close"]

    try:
        _o, _h, _l, _c = [ohlc[h] for h in range(len(ohlc))]
    except:
        _o = _h = _l = _c = np.nan
    return _o, _h, _l, _c


def regime_args(df, lvl, is_relative=False):
    if ("Low" in df.columns) & (is_relative == False):
        reg_val = [
            "Lo1",
            "Hi1",
            "Lo" + str(lvl),
            "Hi" + str(lvl),
            "rg",
            "clg",
            "flr",
            "rg_ch",
        ]
    elif ("low" in df.columns) & (is_relative == False):
        reg_val = [
            "lo1",
            "hi1",
            "lo" + str(lvl),
            "hi" + str(lvl),
            "rg",
            "clg",
            "flr",
            "rg_ch",
        ]
    elif ("Low" in df.columns) & (is_relative == True):
        reg_val = [
            "rL1",
            "rH1",
            "rL" + str(lvl),
            "rH" + str(lvl),
            "rrg",
            "rclg",
            "rflr",
            "rrg_ch",
        ]
    elif ("low" in df.columns) & (is_relative == True):
        reg_val = [
            "rl1",
            "rh1",
            "rl" + str(lvl),
            "rh" + str(lvl),
            "rrg",
            "rclg",
            "rflr",
            "rrg_ch",
        ]

    try:
        rt_lo, rt_hi, slo, shi, rg, clg, flr, rg_ch = [
            reg_val[s] for s in range(len(reg_val))
        ]
    except:
        rt_lo = rt_hi = slo = shi = rg = clg = flr = rg_ch = np.nan
    return rt_lo, rt_hi, slo, shi, rg, clg, flr, rg_ch


def hilo_alternation(hilo, dist=None, hurdle=None):
    i = 0
    while (
        np.sign(hilo.shift(1)) == np.sign(hilo)
    ).any():  # runs until duplicates are eliminated

        # removes swing lows > swing highs
        hilo.loc[
            (np.sign(hilo.shift(1)) != np.sign(hilo))
            & (hilo.shift(1) < 0)  # hilo alternation test
            & (np.abs(hilo.shift(1)) < np.abs(hilo))  # previous datapoint:  high
        ] = np.nan  # high[-1] < low, eliminate low

        hilo.loc[
            (np.sign(hilo.shift(1)) != np.sign(hilo))
            & (hilo.shift(1) > 0)  # hilo alternation
            & (np.abs(hilo) < hilo.shift(1))  # previous swing: low
        ] = np.nan  # swing high < swing low[-1]

        # alternation test: removes duplicate swings & keep extremes
        hilo.loc[
            (np.sign(hilo.shift(1)) == np.sign(hilo))
            & (hilo.shift(1) < hilo)  # same sign
        ] = np.nan  # keep lower one

        hilo.loc[
            (np.sign(hilo.shift(-1)) == np.sign(hilo))
            & (hilo.shift(-1) < hilo)  # same sign, forward looking
        ] = np.nan  # keep forward one

        # removes noisy swings: distance test
        if pd.notnull(dist):
            hilo.loc[
                (np.sign(hilo.shift(1)) != np.sign(hilo))
                & (np.abs(hilo + hilo.shift(1)).div(dist, fill_value=1) < hurdle)
            ] = np.nan

        # reduce hilo after each pass
        hilo = hilo.dropna().copy()
        i += 1
        if i == 4:  # breaks infinite loop
            break
        return hilo


def historical_swings(df, _o, _h, _l, _c, dist=None, hurdle=None, round_place=2):
    reduction = df[[_o, _h, _l, _c]].copy()

    reduction["avg_px"] = round(reduction[[_h, _l, _c]].mean(axis=1), round_place)
    highs = reduction["avg_px"].values
    lows = -reduction["avg_px"].values
    reduction_target = len(reduction) // 100
    #     print(reduction_target )

    n = 0
    while len(reduction) >= reduction_target:
        highs_list = find_peaks(highs, distance=1, width=0)
        lows_list = find_peaks(lows, distance=1, width=0)
        hilo = reduction.iloc[lows_list[0]][_l].sub(
            reduction.iloc[highs_list[0]][_h], fill_value=0
        )

        # Reduction dataframe and alternation loop
        hilo_alternation(hilo, dist=None, hurdle=None)
        reduction["hilo"] = hilo

        # Populate reduction df
        n += 1
        hi_lvl_col = str(_h)[:2] + str(n)
        lo_lvl_col = str(_l)[:2] + str(n)

        reduce_hi = reduction.loc[reduction["hilo"] < 0, _h]
        reduce_lo = reduction.loc[reduction["hilo"] > 0, _l]
        reduction[hi_lvl_col] = reduce_hi
        reduction[lo_lvl_col] = reduce_lo

        # Populate main dataframe
        df[hi_lvl_col] = reduce_hi
        df[lo_lvl_col] = reduce_lo

        # Reduce reduction
        reduction = reduction.dropna(subset=["hilo"])
        reduction.fillna(method="ffill", inplace=True)
        highs = reduction[hi_lvl_col].values
        lows = -reduction[lo_lvl_col].values

        if n >= 9:
            break

    return df


def cleanup_latest_swing(df, shi, slo, rt_hi, rt_lo):
    """
    removes false positives
    """
    # latest swing
    shi_dt = df.loc[pd.notnull(df[shi]), shi].index[-1]
    s_hi = df.loc[pd.notnull(df[shi]), shi][-1]
    slo_dt = df.loc[pd.notnull(df[slo]), slo].index[-1]
    s_lo = df.loc[pd.notnull(df[slo]), slo][-1]
    len_shi_dt = len(df[:shi_dt])
    len_slo_dt = len(df[:slo_dt])

    # Reset false positives to np.nan
    for i in range(2):

        if (len_shi_dt > len_slo_dt) & (
            (df.loc[shi_dt:, rt_hi].max() > s_hi) | (s_hi < s_lo)
        ):
            df.loc[shi_dt, shi] = np.nan
            len_shi_dt = 0
        elif (len_slo_dt > len_shi_dt) & (
            (df.loc[slo_dt:, rt_lo].min() < s_lo) | (s_hi < s_lo)
        ):
            df.loc[slo_dt, slo] = np.nan
            len_slo_dt = 0
        else:
            pass

    return df


def latest_swing_variables(df, shi, slo, rt_hi, rt_lo, _h, _l, _c):
    """
    Latest swings dates & values
    """
    shi_dt = df.loc[pd.notnull(df[shi]), shi].index[-1]
    slo_dt = df.loc[pd.notnull(df[slo]), slo].index[-1]
    s_hi = df.loc[pd.notnull(df[shi]), shi][-1]
    s_lo = df.loc[pd.notnull(df[slo]), slo][-1]

    if slo_dt > shi_dt:
        swg_var = [
            1,
            s_lo,
            slo_dt,
            rt_lo,
            shi,
            df.loc[slo_dt:, _h].max(),
            df.loc[slo_dt:, _h].idxmax(),
        ]
    elif shi_dt > slo_dt:
        swg_var = [
            -1,
            s_hi,
            shi_dt,
            rt_hi,
            slo,
            df.loc[shi_dt:, _l].min(),
            df.loc[shi_dt:, _l].idxmin(),
        ]
    else:
        ud = 0
    ud, bs, bs_dt, _rt, _swg, hh_ll, hh_ll_dt = [
        swg_var[h] for h in range(len(swg_var))
    ]

    return ud, bs, bs_dt, _rt, _swg, hh_ll, hh_ll_dt


def test_distance(ud, bs, hh_ll, dist_vol, dist_pct):
    # priority: 1. Vol 2. pct 3. dflt
    if dist_vol > 0:
        distance_test = np.sign(abs(hh_ll - bs) - dist_vol)
    elif dist_pct > 0:
        distance_test = np.sign(abs(hh_ll / bs - 1) - dist_pct)
    else:
        distance_test = np.sign(dist_pct)

    return int(max(distance_test, 0) * ud)


def average_true_range(df, _h, _l, _c, n):
    """
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr
    """
    atr = (
        (df[_h].combine(df[_c].shift(), max) - df[_l].combine(df[_c].shift(), min))
        .rolling(window=n)
        .mean()
    )
    return atr


def retest_swing(df, _sign, _rt, hh_ll_dt, hh_ll, _c, _swg):
    rt_sgmt = df.loc[hh_ll_dt:, _rt]

    if (rt_sgmt.count() > 0) & (_sign != 0):  # Retests exist and distance test met
        if _sign == 1:  #
            rt_list = [
                rt_sgmt.idxmax(),
                rt_sgmt.max(),
                df.loc[rt_sgmt.idxmax() :, _c].cummin(),
            ]

        elif _sign == -1:
            rt_list = [
                rt_sgmt.idxmin(),
                rt_sgmt.min(),
                df.loc[rt_sgmt.idxmin() :, _c].cummax(),
            ]
        rt_dt, rt_hurdle, rt_px = [rt_list[h] for h in range(len(rt_list))]

        if str(_c)[0] == "r":
            df.loc[rt_dt, "rrt"] = rt_hurdle
        elif str(_c)[0] != "r":
            df.loc[rt_dt, "rt"] = rt_hurdle

        if (np.sign(rt_px - rt_hurdle) == -np.sign(_sign)).any():
            df.at[hh_ll_dt, _swg] = hh_ll
    return df


def all_retest_swing(df, rt: str, dist_pct, retrace_pct, n_num, is_relative=False):
    """
    for back testing entries from swings
    get all retest values by working backward, storing the current retest value then
    slicing out data from current retest onward
    :return:
    """
    all_retests = pd.Series(data=np.NAN, index=df.index)
    working_df = df.copy()
    retest_count = 0
    index = 2220
    ax = None
    lvl4 = None
    while True:
        index += 1
        # working_df = working_df[['open', 'close', 'high', 'low']].copy()

        try:
            working_df = df[["open", "close", "high", "low"]].iloc[:index].copy()
            working_df = init_swings(
                working_df, dist_pct, retrace_pct, n_num, is_relative=is_relative
            )
            retest_val_lookup = ~pd.isna(working_df[rt])
            retest_value_row = working_df[rt].loc[retest_val_lookup]
            retest_value_index = retest_value_row.index[0]
            all_retests.at[retest_value_index] = retest_value_row
        except KeyError:
            # working_df = working_df.iloc[:-1]
            pass

        else:
            if ax is None:
                try:
                    ax = working_df[["close", "hi4", "lo4", "hi2", "lo2"]].plot(
                        style=["grey", "rv", "g^", "r.", "g.", "ko"],
                        figsize=(15, 5),
                        grid=True,
                        ax=ax,
                    )
                    ax = all_retests.plot(style=["k."], ax=ax)
                    plt.ion()
                    plt.show()
                    plt.pause(0.001)
                except:
                    print("lvl4 not in index")
                    pass
        if ax is not None:

            try:
                ax.clear()
                lvl4 = True
                working_df[["close", "hi4", "lo4", "hi2", "lo2"]].plot(
                    style=["grey", "rv", "g^", "r.", "g."],
                    figsize=(15, 5),
                    grid=True,
                    ax=ax,
                )
                all_retests.plot(style=["k."], ax=ax)
                plt.pause(0.001)
            except:
                if lvl4 is True:
                    print("switch")
                    lvl4 = False
                ax.clear()
                working_df[["close", "hi3", "lo3", "hi2", "lo2"]].plot(
                    style=["grey", "rv", "g^", "r.", "g."],
                    figsize=(15, 5),
                    grid=True,
                    ax=ax,
                )
                all_retests.plot(style=["k."], ax=ax)
                plt.pause(0.001)

            # plt.show()
            # plt.clear()
            # working_df = working_df.loc[:retest_value_index]

        # print(len(working_df.index.to_list()))
        count = all_retests.count()
        if count > retest_count:
            retest_count = count
            print(f"retest count: {retest_count}")

    # return all_retests


def retracement_swing(
    df, _sign, _swg, _c, hh_ll_dt, hh_ll, vlty, retrace_vol, retrace_pct
):
    if _sign == 1:  #
        retracement = df.loc[hh_ll_dt:, _c].min() - hh_ll

        if (
            (vlty > 0)
            & (retrace_vol > 0)
            & ((abs(retracement / vlty) - retrace_vol) > 0)
        ):
            df.at[hh_ll_dt, _swg] = hh_ll
        elif (retrace_pct > 0) & ((abs(retracement / hh_ll) - retrace_pct) > 0):
            df.at[hh_ll_dt, _swg] = hh_ll

    elif _sign == -1:
        retracement = df.loc[hh_ll_dt:, _c].max() - hh_ll
        if (
            (vlty > 0)
            & (retrace_vol > 0)
            & ((round(retracement / vlty, 1) - retrace_vol) > 0)
        ):
            df.at[hh_ll_dt, _swg] = hh_ll
        elif (retrace_pct > 0) & ((round(retracement / hh_ll, 4) - retrace_pct) > 0):
            df.at[hh_ll_dt, _swg] = hh_ll
    else:
        retracement = 0
    return df


def relative(
    df, _o, _h, _l, _c, bm_df, bm_col, ccy_df, ccy_col, dgt, start, end, rebase=True
):
    """
    df: df
    bm_df, bm_col: df benchmark dataframe & column name
    ccy_df,ccy_col: currency dataframe & column name
    dgt: rounding decimal
    start/end: string or offset
    rebase: boolean rebase to beginning or continuous series
    """
    # Slice df dataframe from start to end period: either offset or datetime
    df = df[start:end]

    # inner join of benchmark & currency: only common values are preserved
    df = df.join(bm_df[[bm_col]], how="inner")
    df = df.join(ccy_df[[ccy_col]], how="inner")

    # rename benchmark name as bm and currency as ccy
    df.rename(columns={bm_col: "bm", ccy_col: "ccy"}, inplace=True)

    # Adjustment factor: calculate the scalar product of benchmark and currency
    df["bmfx"] = round(df["bm"].mul(df["ccy"]), dgt).fillna(method="ffill")
    if rebase == True:
        df["bmfx"] = df["bmfx"].div(df["bmfx"][0])

    # Divide absolute price by fxcy adjustment factor and rebase to first value
    df["r" + str(_o)] = round(df[_o].div(df["bmfx"]), dgt)
    df["r" + str(_h)] = round(df[_h].div(df["bmfx"]), dgt)
    df["r" + str(_l)] = round(df[_l].div(df["bmfx"]), dgt)
    df["r" + str(_c)] = round(df[_c].div(df["bmfx"]), dgt)
    df = df.drop(["bm", "ccy", "bmfx"], axis=1)

    return df


def init_swings(
    df: pd.DataFrame,
    dist_pct: float,
    retrace_pct: float,
    n_num: int,
    is_relative=False,
    lvl=3,
):
    _o, _h, _l, _c = lower_upper_ohlc(df, is_relative=is_relative)
    # swings = ['hi3', 'lo3', 'hi1', 'lo1']
    swings = [f"hi{lvl}", f"lo{lvl}", "hi1", "lo1"]
    if is_relative:
        swings = [f"r_{name}" for name in swings]
    shi, slo, rt_hi, rt_lo = swings

    df = historical_swings(df, _o=_o, _h=_h, _l=_l, _c=_c, dist=None, hurdle=None)
    df = cleanup_latest_swing(df, shi=shi, slo=slo, rt_hi=rt_hi, rt_lo=rt_lo)
    ud, bs, bs_dt, _rt, _swg, hh_ll, hh_ll_dt = latest_swing_variables(
        df, shi, slo, rt_hi, rt_lo, _h, _l, _c
    )
    vlty = round(average_true_range(df=df, _h=_h, _l=_l, _c=_c, n=n_num)[hh_ll_dt], 2)
    dist_vol = 5 * vlty

    # _sign = test_distance(ud, bs, hh_ll, dist_vol, dist_pct)
    # df = retest_swing(df, _sign, _rt, hh_ll_dt, hh_ll, _c, _swg)
    # retrace_vol = 2.5 * vlty

    # df = retracement_swing(df, _sign, _swg, _c, hh_ll_dt, hh_ll, vlty, retrace_vol, retrace_pct)
    return df


def regime_floor_ceiling(
    df: pd.DataFrame,
    slo: str,
    shi: str,
    flr,
    clg,
    rg,
    rg_ch,
    stdev,
    threshold,
    _h: str = "high",
    _l: str = "low",
    _c: str = "close",
):
    # Lists instantiation
    threshold_test, rg_ch_ix_list, rg_ch_list = [], [], []
    floor_ix_list, floor_list, ceiling_ix_list, ceiling_list = [], [], [], []

    # Range initialisation to 1st swing
    floor_ix_list.append(df.index[0])
    ceiling_ix_list.append(df.index[0])

    # Boolean variables
    ceiling_found = floor_found = breakdown = breakout = False

    # Swings lists
    swing_highs = list(df[pd.notnull(df[shi])][shi])
    swing_highs_ix = list(df[pd.notnull(df[shi])].index)
    swing_lows = list(df[pd.notnull(df[slo])][slo])
    swing_lows_ix = list(df[pd.notnull(df[slo])].index)
    loop_size = np.maximum(len(swing_highs), len(swing_lows))

    # Loop through swings
    for i in range(loop_size):

        # asymetric swing list: default to last swing if shorter list
        try:
            s_lo_ix = swing_lows_ix[i]
            s_lo = swing_lows[i]
        except:
            s_lo_ix = swing_lows_ix[-1]
            s_lo = swing_lows[-1]

        try:
            s_hi_ix = swing_highs_ix[i]
            s_hi = swing_highs[i]
        except:
            s_hi_ix = swing_highs_ix[-1]
            s_hi = swing_highs[-1]

        swing_max_ix = np.maximum(s_lo_ix, s_hi_ix)  # latest swing index

        # CLASSIC CEILING DISCOVERY
        if ceiling_found == False:
            top = df[floor_ix_list[-1] : s_hi_ix][_h].max()
            ceiling_test = round((s_hi - top) / stdev[s_hi_ix], 1)

            # Classic ceiling test
            if ceiling_test <= -threshold:
                # Boolean flags reset
                ceiling_found = True
                floor_found = breakdown = breakout = False
                threshold_test.append(ceiling_test)

                # Append lists
                ceiling_list.append(top)
                ceiling_ix_list.append(df[floor_ix_list[-1] : s_hi_ix][_h].idxmax())
                rg_ch_ix_list.append(s_hi_ix)
                rg_ch_list.append(s_hi)

                # EXCEPTION HANDLING: price penetrates discovery swing
        # 1. if ceiling found, calculate regime since rg_ch_ix using close.cummin
        elif ceiling_found == True:
            close_high = df[rg_ch_ix_list[-1] : swing_max_ix][_c].cummax()
            df.loc[rg_ch_ix_list[-1] : swing_max_ix, rg] = np.sign(
                close_high - rg_ch_list[-1]
            )

            # 2. if price.cummax penetrates swing high: regime turns bullish, breakout
            if (df.loc[rg_ch_ix_list[-1] : swing_max_ix, rg] > 0).any():
                # Boolean flags reset
                floor_found = ceiling_found = breakdown = False
                breakout = True

        # 3. if breakout, test for bearish pullback from highest high since rg_ch_ix
        if breakout == True:
            brkout_high_ix = df.loc[rg_ch_ix_list[-1] : swing_max_ix, _c].idxmax()
            brkout_low = df[brkout_high_ix:swing_max_ix][_c].cummin()
            df.loc[brkout_high_ix:swing_max_ix, rg] = np.sign(
                brkout_low - rg_ch_list[-1]
            )

        # CLASSIC FLOOR DISCOVERY
        if floor_found == False:
            bottom = df[ceiling_ix_list[-1] : s_lo_ix][_l].min()
            floor_test = round((s_lo - bottom) / stdev[s_lo_ix], 1)

            # Classic floor test
            if floor_test >= threshold:
                # Boolean flags reset
                floor_found = True
                ceiling_found = breakdown = breakout = False
                threshold_test.append(floor_test)

                # Append lists
                floor_list.append(bottom)
                floor_ix_list.append(df[ceiling_ix_list[-1] : s_lo_ix][_l].idxmin())
                rg_ch_ix_list.append(s_lo_ix)
                rg_ch_list.append(s_lo)

        # EXCEPTION HANDLING: price penetrates discovery swing
        # 1. if floor found, calculate regime since rg_ch_ix using close.cummin
        elif floor_found == True:
            close_low = df[rg_ch_ix_list[-1] : swing_max_ix][_c].cummin()
            df.loc[rg_ch_ix_list[-1] : swing_max_ix, rg] = np.sign(
                close_low - rg_ch_list[-1]
            )

            # 2. if price.cummin penetrates swing low: regime turns bearish, breakdown
            if (df.loc[rg_ch_ix_list[-1] : swing_max_ix, rg] < 0).any():
                floor_found = floor_found = breakout = False
                breakdown = True

                # 3. if breakdown,test for bullish rebound from lowest low since rg_ch_ix
        if breakdown == True:
            brkdwn_low_ix = df.loc[
                rg_ch_ix_list[-1] : swing_max_ix, _c
            ].idxmin()  # lowest low
            breakdown_rebound = df[brkdwn_low_ix:swing_max_ix][_c].cummax()  # rebound
            df.loc[brkdwn_low_ix:swing_max_ix, rg] = np.sign(
                breakdown_rebound - rg_ch_list[-1]
            )
    #             breakdown = False
    #             breakout = True

    # POPULATE FLOOR,CEILING, RG CHANGE COLUMNS
    df.loc[floor_ix_list[1:], flr] = floor_list
    df.loc[ceiling_ix_list[1:], clg] = ceiling_list
    df.loc[rg_ch_ix_list, rg_ch] = rg_ch_list
    df[rg_ch] = df[rg_ch].fillna(method="ffill")

    # regime from last swing
    df.loc[swing_max_ix:, rg] = np.where(
        ceiling_found,  # if ceiling found, highest high since rg_ch_ix
        np.sign(df[swing_max_ix:][_c].cummax() - rg_ch_list[-1]),
        np.where(
            floor_found,  # if floor found, lowest low since rg_ch_ix
            np.sign(df[swing_max_ix:][_c].cummin() - rg_ch_list[-1]),
            np.sign(df[swing_max_ix:][_c].rolling(5).mean() - rg_ch_list[-1]),
        ),
    )
    df[rg] = df[rg].fillna(method="ffill")
    #     df[rg+'_no_fill'] = df[rg]
    return df


def update_sw_lag(swing_lags: pd.Series, swings: pd.Series, discovered_sw_dates):
    """
    Note this function updates args passed for swing_lags and discovered_sw_dates
    :param swing_lags:
    :param swings:
    :param discovered_sw_dates:
    :return:
    """
    swing_lags = swing_lags.reindex(swings.index)
    latest_sw = swings.loc[~pd.isna(swings)].iloc[-1:]
    if latest_sw.index not in discovered_sw_dates:
        swing_lags.loc[latest_sw.index[0] :] = latest_sw[0]
        discovered_sw_dates.append(latest_sw.index)

    return swing_lags


def historical_peak_discovery(df):
    pass


def full_peak_lag(df, asc_peaks) -> pd.DataFrame:
    """
    calculates distance from highest level peak to the time it was discovered
    value is not nan if peak, does not matter if swing low or swing high
    :param df:
    :param asc_peaks: peak level columns in ascending order
    :return:
    """
    # desire lag for all peak levels greater than 1,
    # so if [hi1, hi2, hi3] given,
    # group by [[hi1, hi2], [hi1, hi2, hi3]] to get lag for level 2 and level 3
    lag_groups = []
    for end_idx in range(1, len(asc_peaks)):
        lag_group = [asc_peaks[i] for i in range(end_idx + 1)]
        lag_groups.append(lag_group)
    full_pivot_table = pd.DataFrame(columns=["start", "end", "type"])

    for lag_group in lag_groups:
        highest_peak_col = lag_group[-1]
        highest_peak = df[highest_peak_col]
        prior_peaks = df[lag_group[-2]]

        # will hold the lowest level peaks
        follow_peaks, lag_pivot_table = get_follow_peaks(highest_peak, prior_peaks)
        i = len(lag_group) - 3
        while i >= 0:
            lag_pivot_table = lag_pivot_table.drop(columns=[prior_peaks.name])
            prior_peaks_col = lag_group[i]
            prior_peaks = df[prior_peaks_col]
            follow_peaks, short_lag_pivot_table = get_follow_peaks(
                follow_peaks, prior_peaks
            )
            lag_pivot_table[prior_peaks_col] = short_lag_pivot_table[prior_peaks_col]

            i -= 1
        lag_pivot_table = lag_pivot_table.melt(
            id_vars=[prior_peaks.name],
            value_vars=[highest_peak_col],
            var_name="type",
            value_name="start",
        )
        lag_pivot_table = lag_pivot_table.rename(columns={prior_peaks.name: "end"})
        full_pivot_table = pd.concat([full_pivot_table, lag_pivot_table])

    full_pivot_table = full_pivot_table[["start", "end", "type"]].reset_index(drop=True)
    full_pivot_table["lvl"] = pd.to_numeric(full_pivot_table.type.str.slice(start=-1))
    full_pivot_table["type"] = np.where(
        full_pivot_table.type.str.slice(stop=-1) == "hi", -1, 1
    )
    return full_pivot_table


def get_follow_peaks(
    current_peak: pd.Series, prior_peaks: pd.Series
) -> t.Tuple[pd.Series, pd.DataFrame]:
    """
    calculates lage between current peak and next level peak.
    helper function, must be used sequentially from current level down to lvl 1 peak
    to get full lag
    :param df:
    :param current_peak:
    :param prior_peaks:
    :return:
    """
    pivot_table = pd.DataFrame(columns=[current_peak.name, prior_peaks.name])
    follow_peaks = pd.Series(index=current_peak.index)

    for r in current_peak.dropna().iteritems():
        # slice df starting with r swing, then exclude r swing, drop nans, then only keep the first row
        # gets the first swing of the prior level after the current swing of the current level.
        current_peak_date = r[0]
        follow_peak = prior_peaks.loc[current_peak_date:].iloc[1:].dropna().iloc[:1]
        if len(follow_peak) > 0:
            follow_peaks.loc[follow_peak.index[0]] = follow_peak.iloc[0]
            pivot_table = pivot_table.append(
                {
                    current_peak.name: current_peak_date,
                    prior_peaks.name: follow_peak.index[0],
                },
                ignore_index=True,
            )
    return follow_peaks, pivot_table


def swing_signal_count(df, raw_signal_col, regime_col, regime_val=None) -> pd.Series:
    """

    :param df:
    :param raw_signal_col:
    :param regime_col:
    :param regime_val:
    :return:
    """
    regime_slices = pda.regime_slices(df, regime_col, regime_val)
    res = []
    for regime_slice in regime_slices:
        res.append(regime_slice[raw_signal_col].cummax())

    return pd.concat(res)


def pyramid(position, root=2):
    return 1 / (1 + position) ** (1 / root)


def assign_pyramid_weight(df, regime_col, entry_count_col, regime_val=None):
    res = []
    for regime_slice in pda.regime_slices(df, regime_col, regime_val):
        res.append(pyramid(regime_slice[entry_count_col]))


def signal_generator(df):
    pass


def unpivot(
    pivot_table: pd.DataFrame,
    start_date_col: str,
    end_date_col: str,
    new_date_col="date",
):
    """unpivot the given table given start and end dates"""
    unpivot_table = pivot_table.copy()
    unpivot_table[new_date_col] = unpivot_table.apply(
        lambda x: pd.date_range(x[start_date_col], x[end_date_col]), axis=1
    )
    unpivot_table = unpivot_table.explode(new_date_col, ignore_index=True).drop(
        columns=[start_date_col, end_date_col]
    )
    return unpivot_table


def regime_ranges(df, rg_col: str):
    start_col = "start"
    end_col = "end"
    loop_params = [(start_col, df[rg_col].shift(1)), (end_col, df[rg_col].shift(-1))]
    boundaries = {}
    for name, shift in loop_params:
        rg_boundary = df[rg_col].loc[
            ((df[rg_col] == -1) & (pd.isna(shift) | (shift != -1)))
            | ((df[rg_col] == 1) & ((pd.isna(shift)) | (shift != 1)))
        ]
        rg_df = pd.DataFrame(data={rg_col: rg_boundary})
        rg_df.index.name = name
        rg_df = rg_df.reset_index()
        boundaries[name] = rg_df

    boundaries[start_col][end_col] = boundaries[end_col][end_col]
    return boundaries[start_col][[start_col, end_col, rg_col]]


def get_all_entry_candidates(
    price: pd.DataFrame,
    regimes: pd.DataFrame,
    peaks: pd.DataFrame,
    entry_lvls: t.List[int],
    highest_peak_lvl: int,
):
    """
    set fixed stop for first signal in each regime to the recent lvl 3 peak
    build raw signal table, contains entry signal date and direction of trade
    regimes: start(date), end(date), rg(date)
    TODO
        - add fixed_stop_date to output
        - add trail_stop_date to output
    peaks: start(date: peak location), end(date: peak discovery), type
    :param partial_exit_r:
    :param highest_peak_lvl:
    :param entry_lvls:
    :param peaks:
    :param regimes:
    :return: raw_signals_df: entry, fixed stop, trail stop, dir
    """

    raw_signals_list = []

    # rename the table prior to collecting entries
    entry_table = peaks.rename(columns={"start": "trail_stop", "end": "entry"})

    for rg_idx, rg_info in regimes.iterrows():
        rg_entries = get_regime_signal_candidates(
            regime=rg_info,
            entry_table=entry_table,
            entry_lvls=entry_lvls,
            highest_peak_lvl=highest_peak_lvl,
        )
        rg_entries = validate_entries(price, rg_entries, rg_info.rg)
        raw_signals_list.append(rg_entries)

    signal_candidates = pd.concat(raw_signals_list).reset_index(drop=True)
    signal_candidates = signal_candidates.rename(
        columns={"start": "trail_stop", "end": "entry"}
    )
    signal_candidates = signal_candidates.drop(columns=["lvl", "type"])
    return signal_candidates


def get_regime_signal_candidates(
    regime: pd.Series, entry_table: pd.DataFrame, entry_lvls, highest_peak_lvl
):
    """get all regime candidates for a single regime"""
    # set 'start'
    rg_entries = entry_table.loc[
        regime.pivot_row.slice(entry_table.entry)
        & regime.pivot_row.slice(entry_table.trail_stop)
        & (entry_table.type == regime.rg)
        & (entry_table.lvl.isin(entry_lvls))
    ].copy()
    rg_entries["dir"] = regime.rg
    rg_entries["fixed_stop"] = rg_entries.trail_stop
    rg_entries = rg_entries.sort_values(by="trail_stop")
    first_sig = rg_entries.iloc[0]
    peaks_since_first_sig = entry_table.loc[
        entry_table.trail_stop < first_sig.trail_stop
    ]
    prior_major_peaks = peaks_since_first_sig.loc[
        (peaks_since_first_sig.lvl == highest_peak_lvl)
        & (peaks_since_first_sig.type == first_sig.type)
    ]
    rg_entries.fixed_stop.iat[0] = prior_major_peaks.trail_stop.iat[-1]
    return rg_entries


@dataclass
class TrailStop:
    """
    pos_price_col: price column to base trail stop movement off of
    neg_price_col: price column to check if stop was crossed
    cum_extreme: cummin/cummax, name of function to use to calculate trailing stop direction
    """

    neg_price_col: str
    pos_price_col: str
    cum_extreme: str
    dir: int

    def init_trail_stop(
        self, price: pd.DataFrame, initial_trail_price, entry_date, rg_end_date
    ) -> pd.Series:
        """
        :param price:
        :param initial_trail_price:
        :param entry_price:
        :param offset_pct: distance from the discovery peak to set the stop loss
        :return:
        """
        entry_price = price.close.loc[entry_date]
        trail_pct_from_entry = (entry_price - initial_trail_price) / entry_price
        extremes = price.loc[entry_date:rg_end_date, self.pos_price_col]

        # when short, pct should be negative, pushing modifier above one
        trail_modifier = 1 - trail_pct_from_entry
        # trail stop reaction must be delayed one bar since same bar reaction cannot be determined
        trail_stop: pd.Series = (
            getattr(extremes, self.cum_extreme)() * trail_modifier
        ).shift(1)
        trail_stop.iat[0] = trail_stop.iat[1]

        return trail_stop

    def exit_signal(self, price: pd.DataFrame, trail_stop: pd.Series) -> pd.Series:
        """detect where price has crossed price"""
        return ((trail_stop - price[self.neg_price_col]) * self.dir) >= 0

    def target_exit_signal(self, price: pd.DataFrame, target_price) -> pd.Series:
        """detect where price has crossed price"""
        return ((target_price - price[self.pos_price_col]) * self.dir) <= 0

    def get_stop_price(
        self, price: pd.DataFrame, stop_date, offset_pct: float
    ) -> float:
        """calculate stop price given a date and percent to offset the stop point from the peak"""
        pct_from_peak = 1 - (offset_pct * self.dir)
        return price[self.neg_price_col].at[stop_date] * pct_from_peak

    def cap_trail_stop(self, trail_data: pd.Series, cap_price) -> pd.Series:
        """apply cap to trail stop"""
        trail_data.loc[((trail_data - cap_price) * self.dir) > 0] = cap_price
        return trail_data

    def validate_entries(self, price, entry_candidates):
        """entry price must be within trail stop/fixed stop"""
        return validate_entries(price, entry_candidates, self.dir)


def validate_entries(price, entry_candidates, direction):
    """entry price must be within trail stop/fixed stop"""
    stop_price_col = "high" if direction == -1 else "low"
    entry_prices = price.loc[entry_candidates.entry, "close"]
    trail_prices = price.loc[entry_candidates.trail_stop, stop_price_col]
    valid_entry_query = ((entry_prices.values - trail_prices.values) * direction) > 0
    return entry_candidates.loc[valid_entry_query]


def get_target_price(stop_price, entry_price, r_multiplier):
    """
    get target price derived from distance from entry to stop loss times r
    :param entry_price:
    :param stop_price:
    :param r_multiplier: multiplier to apply to distance from entry and stop loss
    :return:
    """
    return entry_price + ((entry_price - stop_price) * r_multiplier)


def draw_stop_line(
    stop_calc: TrailStop,
    price: pd.DataFrame,
    trail_stop_date,
    fixed_stop_date,
    entry_date,
    offset_pct,
    r_multiplier,
    rg_end_date,
) -> t.Tuple[pd.Series, pd.Timestamp, pd.Timestamp, pd.Series]:
    """
    trail stop to entry price, then reset to fixed stop price after target price is reached
    :param r_multiplier:
    :param price:
    :param trail_stop_date:
    :param fixed_stop_date:
    :param entry_price:
    :param offset_pct:
    :return:
    """
    entry_price = price.close.loc[entry_date]
    trail_price = stop_calc.get_stop_price(price, trail_stop_date, offset_pct)
    stop_line = stop_calc.init_trail_stop(price, trail_price, entry_date, rg_end_date)
    stop_line = stop_calc.cap_trail_stop(stop_line, entry_price)

    fixed_stop_price = stop_calc.get_stop_price(price, fixed_stop_date, offset_pct)

    target_price = get_target_price(fixed_stop_price, entry_price, r_multiplier)

    target_exit_signal = stop_calc.target_exit_signal(price, target_price)
    partial_exit_date = stop_line.loc[target_exit_signal].first_valid_index()

    if partial_exit_date is None:
        partial_exit_date = price.index[-1]
    else:
        stop_line.loc[partial_exit_date:] = fixed_stop_price

    stop_loss_exit_signal = stop_calc.exit_signal(price, stop_line)
    exit_signal_date = stop_line.loc[stop_loss_exit_signal].first_valid_index()
    if exit_signal_date is None:
        exit_signal_date = price.index[-1]
    stop_line = stop_line.loc[:exit_signal_date]

    return stop_line, exit_signal_date, partial_exit_date, stop_loss_exit_signal


def process_signal_data(
    price_data: pd.DataFrame,
    regimes: pd.DataFrame,
    entry_candidates: pd.DataFrame,
    offset_pct=0.01,
    r_multiplier=1.5,
):
    # sourcery skip: merge-duplicate-blocks, remove-redundant-if
    """process signal data"""
    trail_map = {
        1: TrailStop(
            pos_price_col="high", neg_price_col="low", cum_extreme="cummax", dir=1
        ),
        -1: TrailStop(
            pos_price_col="low", neg_price_col="high", cum_extreme="cummin", dir=-1
        ),
    }
    valid_entries = pd.DataFrame()
    stop_lines = []

    for rg_idx, rg_info in regimes.iterrows():
        stop_calc = trail_map[rg_info.rg]
        start = rg_info.start
        end = rg_info.end

        # next candidate must be higher/lower than prev entry price depending on regime
        while True:
            rg_entry_candidates = entry_candidates.loc[
                pda.date_slice(start, end, entry_candidates.entry)
            ]
            # rg_price_data = price_data.loc[start:end]
            try:
                entry_prices = price_data.loc[rg_entry_candidates.entry, "close"]
                entry_query = ((entry_prices.values - entry_price) * rg_info.rg) > 0
                rg_entry_candidates = rg_entry_candidates.loc[entry_query]
            except NameError:
                pass
            if len(rg_entry_candidates) == 0:
                break

            entry_signal = rg_entry_candidates.iloc[0]

            entry_price = price_data.close.loc[entry_signal.entry]
            # TODO store this stuff signal/stop_loss dataframes?
            (
                stop_line,
                exit_signal_date,
                partial_exit_date,
                stop_loss_exit_signal,
            ) = draw_stop_line(
                stop_calc=stop_calc,
                price=price_data,
                trail_stop_date=entry_signal.trail_stop,
                fixed_stop_date=entry_signal.fixed_stop,
                entry_date=entry_signal.entry,
                offset_pct=offset_pct,
                r_multiplier=r_multiplier,
                rg_end_date=end,
            )
            # crop last signal at entry
            if len(stop_lines) > 0:
                try:
                    stop_lines[-1] = stop_lines[-1].iloc[
                        : stop_lines[-1].index.get_loc(entry_signal.entry)
                    ]
                except KeyError:
                    # unless we moved to the next regime
                    pass
            stop_lines.append(stop_line)
            entry_signal_data = entry_signal.copy()
            entry_signal_data["exit_signal_date"] = exit_signal_date
            entry_signal_data["partial_exit_date"] = partial_exit_date
            if valid_entries.empty:
                valid_entries = pd.DataFrame([entry_signal_data])
            else:
                valid_entries = valid_entries.append(
                    entry_signal_data, ignore_index=True
                )

            if exit_signal_date <= partial_exit_date:
                # start = rg_price_data.iloc[rg_price_data.index.get_loc(exit_signal_date) + 1].index[0]
                start = exit_signal_date
            else:
                # if exit greater than partial exit, then potentially another signal can be added
                start = partial_exit_date

    stop_prices = pd.concat(stop_lines)
    return valid_entries, stop_prices


if __name__ == "__main__":
    ticker = "AAPL"
    try:
        data = yf.ticker.Ticker(ticker).history(
            start=(datetime.now() - timedelta(days=58)),
            end=datetime.now(),
            interval="15m",
        )
    # if no internet, use cached data
    except:
        data = pd.read_excel("data.xlsx")
    else:
        data = data.tz_localize(None)
        data.to_excel("data.xlsx")

    data = data.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
    )
    data = data[["open", "high", "low", "close"]]
    _open = "open"
    _high = "high"
    _low = "low"
    _close = "close"

    # swing inputs
    distance_percent = 0.05
    retrace_percent = 0.05
    swing_window = 63

    sw_lvl = 3
    shi_col = f"hi{sw_lvl}"
    slo_col = f"lo{sw_lvl}"

    # regime inputs
    standard_dev = data[_close].rolling(swing_window).std(ddof=0)
    regime_threshold = 0.5

    use_index = False
    initial_size = 600
    plot_window = 250

    ndf = init_swings(
        df=data,
        dist_pct=distance_percent,
        retrace_pct=retrace_percent,
        n_num=swing_window,
        lvl=sw_lvl,
    )

    hi_peak_table = full_peak_lag(ndf, ["hi1", "hi2", "hi3"])
    lo_peak_table = full_peak_lag(ndf, ["lo1", "lo2", "lo3"])
    peak_table = pd.concat([hi_peak_table, lo_peak_table]).reset_index(drop=True)

    ndf = regime_floor_ceiling(
        df=ndf,
        slo=slo_col,
        shi=shi_col,
        flr="flr",
        clg="clg",
        rg="rg",
        rg_ch="rg_ch",
        stdev=standard_dev,
        threshold=regime_threshold,
    )

    rg_table = regime_ranges(ndf, "rg")
    raw_signals = get_all_entry_candidates(data, rg_table, peak_table, [2], 3)
    v_entries, stop_loss_series = process_signal_data(data, rg_table, raw_signals)
    ndf["stop_loss"] = stop_loss_series
    a = ndf[
        [_close, shi_col, slo_col, "clg", "flr", "rg_ch", "hi2", "lo2", "stop_loss"]
    ].plot(
        style=["grey", "ro", "go", "kv", "k^", "c:", "r.", "g."],
        figsize=(15, 5),
        grid=True,
        title=str.upper(ticker),
        use_index=use_index,
    )

    ndf["rg"].plot(
        style=["b-."],
        # figsize=(15, 5),
        # marker='o',
        secondary_y=["rg"],
        ax=a,
        use_index=use_index,
    )
    plt.show()
    # all_retest_swing(data, 'rt', distance_percent, retrace_percent, swing_window)
    # data[['close', 'hi3', 'lo3', 'rt']].plot(
    #     style=['grey', 'rv', 'g^', 'ko'],
    #     figsize=(10, 5), grid=True, title=str.upper(ticker))

    # data[['close', 'hi3', 'lo3']].plot(
    #     style=['grey', 'rv', 'g^'],
    #     figsize=(20, 5), grid=True, title=str.upper(ticker))

    # plt.show()
    """
    ohlc = ['Open','High','Low','Close']
_o,_h,_l,_c = [ohlc[h] for h in range(len(ohlc))]
rg_val = ['Hi3','Lo3','flr','clg','rg','rg_ch',1.5]
slo, shi,flr,clg,rg,rg_ch,threshold = [rg_val[s] for s in range(len(rg_val))]
stdev = df[_c].rolling(63).std(ddof=0)
df = regime_floor_ceiling(df,_h,_l,_c,slo, shi,flr,clg,rg,rg_ch,stdev,threshold)
 
df[[_c,'Hi3', 'Lo3','clg','flr','rg_ch','rg']].plot(    
style=['grey', 'ro', 'go', 'kv', 'k^','c:','y-.'],     
secondary_y= ['rg'],figsize=(20,5),    
grid=True, 
title = str.upper(ticker))

    """

    axis = None
    index = initial_size
    fp_rg = None
    hi2_lag = None
    lo2_lag = None
    hi2_discovery_dts = []
    lo2_discovery_dts = []
    d = data[[_open, _high, _low, _close]].copy().iloc[:index]
    for idx, row in data.iterrows():
        if (num := data.index.get_loc(idx)) <= index:
            print(f"iter index {num}")
            continue
        d.at[idx] = row
        try:
            d = init_swings(
                df=d,
                dist_pct=distance_percent,
                retrace_pct=retrace_percent,
                n_num=swing_window,
            )
            d = regime_floor_ceiling(
                df=d,
                shi=shi_col,
                slo=slo_col,
                flr="flr",
                clg="clg",
                rg="rg",
                rg_ch="rg_ch",
                stdev=standard_dev,
                threshold=regime_threshold,
            )
            if fp_rg is None:
                fp_rg = d.rg.copy()
                fp_rg = fp_rg.fillna(0)
                hi2_lag = d.hi2.copy()
                lo2_lag = d.lo2.copy()
            else:
                fp_rg = fp_rg.reindex(d.rg.index)
                new_val = d.rg.loc[pd.isna(fp_rg)][0]
                fp_rg.loc[idx] = new_val

                hi2_lag = update_sw_lag(hi2_lag, d.hi2, hi2_discovery_dts)
                lo2_lag = update_sw_lag(lo2_lag, d.lo2, lo2_discovery_dts)

        except KeyError:
            pass
        else:
            pass
            # live print procedure
            try:
                window = len(d.index) - plot_window
                if axis is None:
                    axis = (
                        d[[_close, shi_col, slo_col, "clg", "flr", "rg_ch", "rg"]]
                        .iloc[index - plot_window :]
                        .plot(
                            style=["grey", "ro", "go", "kv", "k^", "c:", "b-."],
                            figsize=(15, 5),
                            secondary_y=["rg"],
                            grid=True,
                            title=str.upper(ticker),
                            use_index=use_index,
                        )
                    )
                    fp_rg.iloc[window:].plot(
                        style="y-.", secondary_y=True, use_index=use_index, ax=axis
                    )
                    hi2_lag.iloc[window:].plot(style="r.", use_index=use_index, ax=axis)
                    lo2_lag.iloc[window:].plot(style="g.", use_index=use_index, ax=axis)
                    plt.ion()
                    plt.show()
                    plt.pause(0.001)
                else:
                    plt.gca().cla()
                    axis.clear()
                    d[[_close, shi_col, slo_col, "clg", "flr", "rg_ch", "rg"]].iloc[
                        window:
                    ].plot(
                        style=["grey", "ro", "go", "kv", "k^", "c:", "b-."],
                        figsize=(15, 5),
                        secondary_y=["rg"],
                        grid=True,
                        title=str.upper(ticker),
                        ax=axis,
                        use_index=use_index,
                    )
                    fp_rg.iloc[window:].plot(
                        style="y-.", secondary_y=True, use_index=use_index, ax=axis
                    )
                    hi2_lag.iloc[window:].plot(style="r.", use_index=use_index, ax=axis)
                    lo2_lag.iloc[window:].plot(style="g.", use_index=use_index, ax=axis)
                    # d.rt.iloc[window:].plot(style='k.', use_index=use_index, ax=axis)
                    plt.pause(0.001)
            except Exception as e:
                print(e)
        print(idx)

    # plt.close()
    a = ndf[[_close, shi_col, slo_col, "clg", "flr", "rg_ch"]].plot(
        style=["grey", "ro", "go", "kv", "k^", "c:"],
        figsize=(15, 5),
        grid=True,
        title=str.upper(ticker),
        use_index=use_index,
    )
    ndf["rg"].plot(
        style=["b-."],
        # figsize=(15, 5),
        # marker='o',
        secondary_y=["rg"],
        ax=a,
        use_index=use_index,
    )
    fp_rg.plot(style="y-.", secondary_y=True, use_index=use_index, ax=a)
    hi2_lag.plot(style="r.", use_index=use_index, ax=axis)
    lo2_lag.plot(style="g.", use_index=use_index, ax=axis)
    plt.show()
