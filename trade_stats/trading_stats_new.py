import numpy as np
import pandas as pd


def rolling_grit(cumul_returns, window):
    rolling_peak = cumul_returns.rolling(window).max()
    draw_down_squared = (cumul_returns - rolling_peak) ** 2
    ulcer = draw_down_squared.rolling(window).sum() ** 0.5
    grit = cumul_returns / ulcer
    return grit.replace([-np.inf, np.inf], np.NAN)


def expanding_grit(cumul_returns):
    tt_peak = cumul_returns.expanding().max()
    draw_down_squared = (cumul_returns - tt_peak) ** 2
    ulcer = draw_down_squared.expanding().sum() ** 0.5
    grit = cumul_returns / ulcer
    return grit.replace([-np.inf, np.inf], np.NAN)


def rolling_profits(returns, window):
    profit_roll = returns.copy()
    profit_roll[profit_roll < 0] = 0
    profit_roll_sum = profit_roll.rolling(window).sum().fillna(method="ffill")
    return profit_roll_sum


def rolling_losses(returns, window):
    loss_roll = returns.copy()
    loss_roll[loss_roll > 0] = 0
    loss_roll_sum = loss_roll.rolling(window).sum().fillna(method="ffill")
    return loss_roll_sum


def expanding_profits(returns):
    profit_roll = returns.copy()
    profit_roll[profit_roll < 0] = 0
    profit_roll_sum = profit_roll.expanding().sum().fillna(method="ffill")
    return profit_roll_sum


def expanding_losses(returns):
    loss_roll = returns.copy()
    loss_roll[loss_roll > 0] = 0
    loss_roll_sum = loss_roll.expanding().sum().fillna(method="ffill")
    return loss_roll_sum


def profit_ratio(profits, losses):
    pr = profits.fillna(method="ffill") / abs(losses.fillna(method="ffill"))
    return pr


def rolling_profit_ratio(returns, window):
    return profit_ratio(
        profits=rolling_profits(returns, window), losses=rolling_losses(returns, window)
    )


def expanding_profit_ratio(returns):
    return profit_ratio(
        profits=expanding_profits(returns), losses=expanding_losses(returns)
    )


def rolling_tail_ratio(cumul_returns, window, percentile, limit):
    left_tail = np.abs(cumul_returns.rolling(window).quantile(percentile))
    right_tail = cumul_returns.rolling(window).quantile(1 - percentile)
    np.seterr(all="ignore")
    tail = np.maximum(np.minimum(right_tail / left_tail, limit), -limit)
    return tail


def expanding_tail_ratio(cumul_returns, percentile, limit):
    left_tail = np.abs(cumul_returns.expanding().quantile(percentile))
    right_tail = cumul_returns.expanding().quantile(1 - percentile)
    np.seterr(all="ignore")
    tail = np.maximum(np.minimum(right_tail / left_tail, limit), -limit)
    return tail


def common_sense_ratio(pr, tr):
    return pr * tr


def expectancy(win_rate, avg_win, avg_loss):
    # win% * avg_win% - loss% * abs(avg_loss%)
    return win_rate * avg_win + (1 - win_rate) * avg_loss


def t_stat(signal_count, trading_edge):
    sqn = (signal_count ** 0.5) * trading_edge / trading_edge.std(ddof=0)
    return sqn


def robustness_score(grit, csr, sqn):
    # TODO should it start at 1?
    try:
        start_date = max(
            grit[pd.notnull(grit)].index[0],
            csr[pd.notnull(csr)].index[0],
            sqn[pd.notnull(sqn)].index[0],
        )
    except IndexError:
        score = pd.Series(data=np.NaN, index=grit.index)
    else:
        score = (
            grit * csr * sqn / (grit[start_date] * csr[start_date] * sqn[start_date])
        )
    return score
