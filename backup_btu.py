import matplotlib.pyplot as plt
import tda
import pandas as pd
import numpy as np
import selenium.webdriver
import scipy.signal
import tdargs
from enum import Enum
from back_test_utils import *

# Calculate regime floor ceiling
def gen_regime_fc(
    df,
    close,
    swing_low,
    swing_high,
    threshold,
    t_dev,
    decimals
):
    # new inputs
    price_df = pd.DataFrame()
    swing_df = pd.DataFrame()
    threshold = 1.5
    t_def = 63
    decimals = 3

    # 1. copy swing lows and highs to a smaller df called regime
    regime = price_df.merge(swing_df, on='time')
    regime = regime[
        (regime.swing_low > 0) |
        (regime.swing_high > 0)
        ][
        [close, swing_low, swing_high]
    ].copy()

    # 2. calculate volatility from main df and populate regime df
    stdev = 'stdev'
    regime[stdev] = rolling_stdev(
        price_col=close,
        window=t_dev,
        min_periods=1,
        decimals=decimals
    )

    ceiling_found = False
    floor_found = False

    # 4.  instantiate columns based on absolute or relative series
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

    class FcState(Enum):
        FLOOR = 0
        CEILING = 1
        NONE = 2

    last_found = FcState.NONE

    # 6. Loop through swings
    for i in range(1, len(regime)):
        """
        global state:
        floor_ix,
        ceiling_ix
        regime (duh)

        """

        if regime[swing_high][i] > 0:  # ignores swing lows
            top = regime[floor_ix:regime.index[i]][swing_high].max()  # highest swing high from range[floor_ix:swing[i]]
            ceiling_test = round((regime.swing_high[i] - top) / regime[stdev][i], 1)  # test vs highest

            # update top and top index
            if ceiling_test <= -threshold:  # if swing <= top - x * stdev
                ceiling = regime[floor_ix:regime.index[i]][swing_high].max()  # ceiling = top
                ceiling_ix = regime[floor_ix:regime.index[i]][swing_high].idxmax()  # ceiling index
                regime.loc[ceiling_ix, ceiling] = ceiling  # assign ceiling

                if last_found != FcState.CEILING:  # test met == ceiling found
                    rg_chg_ix = regime[swing_high].index[i]
                    _rg_chg = regime[swing_high][i]
                    regime.loc[rg_chg_ix, regime_change] = _rg_chg  # prints where/n ceiling found
                    regime.loc[rg_chg_ix, regime_floorceiling] = -1  # regime change
                    regime.loc[rg_chg_ix, floorceiling] = ceiling  # used in floor/ceiling breakout test

                    last_found = FcState.CEILING  # forces alternation btwn floor/ceiling

        if regime[swing_low][i] > 0:  # ignores swing highs
            bottom = regime[ceiling_ix:regime.index[i]][swing_low].min()  # lowest swing low from ceiling
            floor_test = round((regime[swing_low][i] - bottom) / regime[stdev][i], 1)  # test vs lowest

            if floor_test >= threshold:  # if swing > bottom + n * stdev
                floor = regime[ceiling_ix:regime.index[i]][swing_low].min()  # floor = bottom
                floor_ix = regime[ceiling_ix:regime.index[i]][swing_low].idxmin()
                regime.loc[floor_ix, floor] = floor  # assign floor

                if last_found != FcState.FLOOR:  # test met == floor found
                    rg_chg_ix = regime[swing_low].index[i]
                    _rg_chg = regime[swing_low][i]
                    regime.loc[rg_chg_ix, regime_change] = _rg_chg  # prints where/n floor found
                    regime.loc[rg_chg_ix, regime_floorceiling] = 1  # regime change
                    regime.loc[rg_chg_ix, floorceiling] = floor  # used in floor/ceiling breakdown test

                    last_found = FcState.FLOOR  # forces alternation btwn floor/ceiling

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