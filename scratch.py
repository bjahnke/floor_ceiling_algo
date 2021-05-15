

def hof_get_local_swings(np_compare):
    def get_local_swings(bar_extreme: pd.DataFrame, argrelwindow=20):
        """

        :param bar_extreme: index=date, cols=[high/low], df with daily extreme
        :param argrelwindow:
        :return:
        """
        local_swings_df = bar_extreme.copy(deep=True)
        # Step 2: build 2 lists of highs and lows using argrelextrema
        local_swings = scipy.signal.argrelextrema(
            bar_extreme.extreme.values, np_compare, order=argrelwindow
        )[0]
        # Step 3: Create swing high and low columns and assign values from the lists
        # in the new swing columns, print the low/high value for all indexes in the swing lists
        local_swings_df['swing'] = local_swings_df.iloc[local_swings, 0]

    return get_local_swings


def reduce_swings(swings_data: pd.DataFrame) -> pd.DataFrame:
    """
    :param swings_data:
    :return:
    """
    # Step 5: Reduce dataframe and alternation loop
    # Instantiate start
    i = 0
    # Drops all rows with no swing
    local_swings = swings_data.dropna().copy()
    while (local_swings.shift(1) * local_swings > 0).any():
        # eliminate lows higher than highs
        local_swings.loc[
            (local_swings.shift(1) * local_swings < 0) &
            (local_swings.shift(1) < 0) &
            (np.abs(local_swings.shift(1)) < local_swings)
            ] = np.nan
        # eliminate earlier lower values
        local_swings.loc[
            (local_swings[swings].shift(1) * local_swings[swings] > 0) &
            (local_swings[swings].shift(1) < local_swings[swings])
            ] = np.nan

        # eliminate subsequent lower values
        local_swings.loc[
            (local_swings[swings].shift(-1) * local_swings[swings] > 0) &
            (local_swings[swings].shift(-1) < local_swings[swings])
            ] = np.nan
        # reduce dataframe
        local_swings = local_swings.dropna().copy()
        i += 1
        if i == 4:  # avoid infinite loop
            break
    return local_swings


def init_swing_df(
    local_highs_df: pd.DataFrame,
    local_lows_df: pd.DataFrame,
    high_low_df: pd.DataFrame
) -> pd.DataFrame:
    local_highs_df = local_highs_df.rename({'extreme': 'swing_high'})
    local_lows_df = local_lows_df.rename({'extreme': 'swing_low'})
    high_low = pd.concat(
        [local_highs_df, local_lows_df, high_low_df],
        join='outer',
        axis=1,
    )
    return high_low


def init_extreme_df(col: pd.Series, col_name: str) -> pd.DataFrame:
    extreme_df = pd.DataFrame(col)
    extreme_df = extreme_df.rename(columns={col_name: 'extreme'})
    return extreme_df


def gen_swing_df(df: pd.DataFrame, argrelwindow=20) -> pd.DataFrame:
    """
    :param df: price df with high/low column
    :param argrelwindow:
    :return:
    """
    # todo constructor that renames to 'extreme'
    high_df = init_extreme_df(df.high, 'high')
    low_df = init_extreme_df(df.low, 'low')
    high_low = pd.DataFrame(index=df.index)

    local_highs = hof_get_local_swings(np.greater)(high_df, argrelwindow=argrelwindow)
    local_lows = hof_get_local_swings(np.less)(low_df, argrelwindow=argrelwindow)

    high_low['swings'] = local_lows.extreme.sub(
        local_highs.extreme, fill_value=0
    )
    high_low = reduce_swings(high_low)

    swing_df = init_swing_df(local_highs, local_lows, high_low)

    # Step 7: Preparation for the Last swing adjustment
    swing_df.swings = np.where(
        np.isnan(swing_df.swings), 0, swing_df.swings
    )
    # If last_sign <0: swing high, if > 0 swing low
    last_sign = np.sign(swing_df.swings[-1])

    # Step 8: Instantiate last swing high and low dates
    last_slo_dt = swing_df[swing_df.swing_low > 0].index.max()
    last_shi_dt = swing_df[swing_df.swing_high > 0].index.max()

    # Step 9: Test for extreme values
    if (last_sign == -1) & (last_shi_dt != swing_df[last_slo_dt:]['swing_high'].idxmax()):
        # Reset swing_high to nan
        swing_df.loc[last_shi_dt, 'swing_high'] = np.nan
    elif (last_sign == 1) & (last_slo_dt != swing_df[last_shi_dt:]['swing_low'].idxmax()):
        # Reset swing_low to nan
        swing_df.loc[last_slo_dt, 'swing_low'] = np.nan

    return swing_df

def cumulative_returns(raw_returns):
    """
    Calculates cumulative (expanding from initial value) sum, applies exponential
    """
    rets = pd.Series(raw_returns)
    cum_log_return = rets.cumsum().apply(np.exp)
    return cum_log_return

def gen_fc_data(relative_pd: pd.DataFrame, base_pd: pd.DataFrame):
    """
    :param relative_pd: relative price data
    :param base_pd: base price data
    :return:
    """
    # add swings
    fc_data = swings(
        df=relative_pd,
        high='high',
        low='low',
        argrelwindow=20
    )