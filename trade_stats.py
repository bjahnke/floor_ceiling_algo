"""
Pure functions for calculating performance of strategies
"""
import pandas as pd
import numpy as np
import typing as t

# Calculates the returns
def simple_returns(prices):
    """
    calculates log returns based on price series
    """
    rets = pd.Series(prices)
    log_returns = np.log(rets / rets.shift(1))
    return log_returns


def cum_return_percent(raw_returns):
    """
    Calculates cumulative returns and returns as percentage
    """
    rets = pd.Series(raw_returns)
    cum_log_returns = round((rets.cumsum().apply(np.exp) - 1) * 100, 1)
    return cum_log_returns

# =================
# Basic statistics
# =================


# Define a function 'hit_rate', calculates the hits
def hit_rate(returns, min_periods):
    hits = (returns[returns > 0].expanding(min_periods=min_periods).count() /
            returns.expanding(min_periods=min_periods).count()).fillna(method='ffill')
    return hits


# Define a function 'miss_rate', calculates the miss
def miss_rate(returns, min_periods):
    misses = (returns[returns < 0].expanding(min_periods=min_periods).count() /
              returns.expanding(min_periods=min_periods).count()).fillna(method='ffill')
    return misses


# Define a function 'avg_win', calculates the average win
def average_win(returns, min_periods):
    avg_win = (
        returns[returns > 0].expanding(min_periods=min_periods).sum() / returns.expanding(min_periods=min_periods).count()
    ).fillna(method='ffill')
    return avg_win


# Define a function 'avg_loss', calculates the average loss
def average_loss(returns, min_periods):
    avg_loss = (
        returns[returns < 0].expanding(min_periods=min_periods).sum() / returns.expanding(min_periods=min_periods).count()
    ).fillna(method='ffill')
    return avg_loss


# Define a function 'rolling_loss_rate', calculates the rolling loss rate
def rolling_loss_rate(returns, window):
    losing_days = returns.copy()
    losing_days[losing_days > 0] = np.nan
    losing_days_rolling = (losing_days.rolling(window).count()/window)
    return losing_days_rolling


# Define a function 'rolling_avg_loss', calculates the rolling average loss
def rolling_avg_loss(returns, window):
    avg_losing_day = returns.copy()
    avg_losing_day[avg_losing_day > 0] = 0
    _avg_losing_day = (avg_losing_day.rolling(window).sum()/window)
    return _avg_losing_day


# Define a function 'rolling_win_rate', calculates the rolling win rate
def rolling_win_rate(returns, window):
    good_days = returns.copy()
    good_days[good_days < 0] = np.nan
    good_days_rolling = (good_days.rolling(window).count()/window)
    return good_days_rolling


# Define a function 'rolling_avg_win', calculates the rolling average win
def rolling_avg_win(returns, window):
    avg_good_day = returns.copy()
    avg_good_day[avg_good_day < 0] = 0
    _avg_good_day = (avg_good_day.rolling(window).sum()/window)
    return _avg_good_day

# Gain expectancies and Kelly criterion

# Define a function 'arige'
def arige(win_rate, avg_win, avg_loss):  # win% * avg_win% - loss% * abs(avg_loss%)
    return win_rate * avg_win + (1-win_rate) * avg_loss


# Define a function 'george'
def george(win_rate, avg_win, avg_loss):  # (1+ avg_win%)** win% * (1- abs(avg_loss%)) ** loss%  -1
    return (1+avg_win) ** win_rate * (1 + avg_loss) ** (1 - win_rate) - 1


# Define a function 'kelly'
def kelly(win_rate, avg_win, avg_loss):  # Kelly = win% / abs(avg_loss%) - loss% / avg_win%
    return win_rate / np.abs(avg_loss) - (1-win_rate) / avg_win


# Define a function 'count_signals'
def count_signals(signals):
    signal = signals.copy()
    signal[~((pd.isnull(signal.shift(1))) & (pd.notnull(signal)))] = np.nan
    return signal.expanding().count()

# ======================
# Performance statistics
# ======================

# Define a function 'cumulative_returns'
def cumulative_returns(returns, min_periods):
    return returns.expanding(min_periods=min_periods).sum().apply(np.exp)


# Define a function 'cumulative_returns_pct'
def cumulative_returns_pct(returns, min_periods):
    return returns.expanding(min_periods=min_periods).sum().apply(np.exp) - 1


# Define a function 'average_returns'
def average_returns(returns, min_periods):
    avg_returns = (returns.expanding(min_periods=min_periods).sum() /
                   returns.expanding(min_periods=min_periods).count())
    return avg_returns


# Define a function 'stdev_returns'
def stdev_returns(returns, min_periods):
    std_returns = returns.expanding(min_periods=min_periods).std(ddof=0)
    return std_returns


# Define a function 'rolling_returns'
def rolling_returns(returns, window):
    return returns.rolling(window).sum().fillna(method='ffill')


# Define a function 'rolling_profits'
def rolling_profits(returns, window):
    profit_roll = returns.copy()
    profit_roll[profit_roll < 0] = 0
    profit_roll_sum = profit_roll.rolling(
        window, min_periods=1).sum().fillna(method='ffill')
    return profit_roll_sum


# Define a function 'rolling_losses'
def rolling_losses(returns, window):
    loss_roll = returns.copy()
    loss_roll[loss_roll > 0] = 0
    loss_roll_sum = loss_roll.rolling(
        window, min_periods=1).sum().fillna(method='ffill')
    return loss_roll_sum


# Define a function 'rolling_avg_returns'
def rolling_avg_returns(returns, window):
    roll_avg_returns = returns.rolling(
        window=window, min_periods=1).sum()/window
    roll_avg_returns = roll_avg_returns.fillna(method='ffill')
    return roll_avg_returns


# Define a function 'rolling_stdev_returns'
def rolling_stdev_returns(returns, window):
    return returns.rolling(window).std(ddof=0).fillna(method='ffill')


# Define a function 'rolling_sharpe'
def rolling_sharpe(returns, window):
    roll_sharpe = rolling_avg_returns(
        returns, window) / rolling_stdev_returns(returns, window)
    return roll_sharpe


# Define a function 'drawdown'
def drawdown(returns, min_periods):
    cum_rets = cumulative_returns(returns, min_periods)
    dd = cum_rets / cum_rets.cummax() - 1
    return dd


# Define a function 'max_drawdown'
def max_drawdown(returns, min_periods):
    max_dd = drawdown(returns, min_periods).cummin().fillna(method='ffill')
    return max_dd

# Robustness metrics


# Define a function 'ulcer_index'
def ulcer_index(returns, min_periods):
    cum_rets = cumulative_returns(returns, min_periods).fillna(method='ffill')
    peak_rets = cum_rets.cummax()
    dd = np.log((cum_rets/peak_rets).fillna(1)) ** 2
    ulcer = np.sqrt(dd.expanding(min_periods=min_periods).sum())
    return ulcer


# Define a function 'grit_index'
def grit_index(returns, min_periods):
    cum_rets = cumulative_returns(returns, min_periods).fillna(method='ffill')
    peak_rets = cum_rets.cummax()
    dd = np.log((cum_rets/peak_rets)) ** 2
    ulcer = np.sqrt(dd.expanding(min_periods=min_periods).sum())
    grit = cum_rets * ulcer ** -1
    return grit


# Define a function 't_stat'
def t_stat(signal_count, expectancy):
    sqn = (signal_count ** 0.5) * expectancy / expectancy.expanding().std(ddof=0)
    return sqn


# Define a function 'calmar_ratio'
def calmar_ratio(returns, min_periods):
    cum_rets = cumulative_returns(returns, min_periods).fillna(method='ffill')
    max_dd = np.abs(max_drawdown(returns, min_periods))
    calmar = cum_rets / max_dd
    return calmar


# Define a function 'rolling_profit_ratio'
def rolling_profit_ratio(returns, window):
    pr = (rolling_profits(returns, window).fillna(method='ffill') /
          abs(rolling_losses(returns, window).fillna(method='ffill')))
    return pr


# Define a function 'tail_ratio'
def tail_ratio(returns, window, percentile, limit):
    cumul_returns = returns.cumsum().fillna(method='ffill')
    left_tail = np.abs(cumul_returns.rolling(window).quantile(percentile))
    right_tail = cumul_returns.rolling(window).quantile(1-percentile)
    np.seterr(all='ignore')
    tail = np.maximum(np.minimum(right_tail / left_tail, limit), -limit)
    return tail


# Define a function 'common_sense_ratio'
def common_sense_ratio(pr, tr):
    return pr * tr


# Define a function 'round_lot'
def get_round_lot(
    weight: t.Union[pd.Series, float],
    capital: float,
    fx_rate: int,
    price_local: t.Union[pd.Series, float],
    roundlot: t.Union[pd.Series, int]
) -> t.Union[pd.Series, int]:
    """
    TODO add unit test:
        - output cannot be fraction?
    :param weight: percent of portfolio to risk
    :param capital: total capital in account
    :param fx_rate: TODO foreign exchange rate?
    :param price_local: base price at a given time
    :param roundlot: amount to round calculated position size to
    :return:
    """
    book_value = weight * capital
    shares = book_value * fx_rate / price_local
    lot = round(shares // roundlot, 0) * roundlot
    return lot


# Define a function 'equity_at_risk'
def equity_at_risk(
    px_adj: t.Union[pd.Series, float],
    stop_loss: t.Union[pd.Series, float],
    risk: float
) -> pd.Series:
    """
    :param px_adj: usually rebased close price is used
    :param stop_loss: stop loss data
    :param risk: risk percent to apply
    :return:
    """
    dsl = px_adj / stop_loss - 1  # distance to stop loss in currency adjusted relative
    try:
        eqty_at_risk = risk / dsl  # weight in currency adjusted relative terms
    except ZeroDivisionError:
        eqty_at_risk = 0
    else:
        try:
            eqty_at_risk = eqty_at_risk.replace([np.inf, -np.inf], 0)
        except AttributeError:
            # inputs were floats rather than series
            pass
    return eqty_at_risk
