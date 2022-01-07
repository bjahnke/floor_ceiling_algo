import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def graph_regime_combo(ticker ,df ,_c ,rg ,lo ,hi ,slo ,shi ,clg ,flr ,rg_ch,
                       ma_st ,ma_mt ,ma_lt ,lt_lo ,lt_hi ,st_lo ,st_hi):

    '''
    https://www.color-hex.com/color-names.html
    ticker,df,_c: _c is closing price
    rg: regime -1/0/1 using floor/ceiling method
    lo,hi: small, noisy highs/lows
    slo,shi: swing lows/highs
    clg,flr: ceiling/floor

    rg_ch: regime change base
    ma_st,ma_mt,ma_lt: moving averages ST/MT/LT
    lt_lo,lt_hi: range breakout High/Low LT
    st_lo,st_hi: range breakout High/Low ST
    '''
    fig = plt.figure(figsize=(20 ,8))
    ax1 = plt.subplot2grid((1 ,1), (0 ,0))
    date = df.index
    close = df[_c]
    ax1.plot_date(df.index, close ,'-', color='k',  label=ticker.upper())
    try:
        if pd.notnull(rg):
            base = df[rg_ch]
            regime = df[rg]

            if df[lo].count( ) >0:
                ax1.plot(df.index, df[lo] ,'.' ,color='r', label='swing low', alpha=0.6)
            if df[hi].count() > 0:
                ax1.plot(df.index, df[hi], '.', color='g', label='swing high', alpha=0.6)
            if df[slo].count() > 0:
                ax1.plot(df.index, df[slo], 'o', color='r', label='swing low', alpha=0.8)
            if df[shi].count() > 0:
                ax1.plot(df.index, df[shi], 'o', color='g', label='swing high', alpha=0.8)
            if df[flr].count() > 0:
                plt.scatter(df.index, df[flr], c='k', marker='^', label='floor')
            if df[clg].count() > 0:
                plt.scatter(df.index, df[clg], c='k', marker='v', label='ceiling')

            ax1.plot([], [], linewidth=5, label='bear', color='m', alpha=0.1)
            ax1.plot([], [], linewidth=5, label='bull', color='b', alpha=0.1)
            ax1.fill_between(date, close, base, where=((regime == 1) & (close > base)), facecolor='b', alpha=0.1)
            ax1.fill_between(date, close, base, where=((regime == 1) & (close < base)), facecolor='b', alpha=0.4)
            ax1.fill_between(date, close, base, where=((regime == -1) & (close < base)), facecolor='m', alpha=0.1)
            ax1.fill_between(date, close, base, where=((regime == -1) & (close > base)), facecolor='m', alpha=0.4)

        if np.sum(ma_st) > 0:
            ax1.plot(df.index, ma_st, '-', color='lime', label='ST MA')
            ax1.plot(df.index, ma_mt, '-', color='green', label='MT MA')
            ax1.plot(df.index, ma_lt, '-', color='red', label='LT MA')

            if pd.notnull(rg):  # floor/ceiling regime present
                # Profitable conditions
                ax1.fill_between(date, close, ma_mt, where=((regime == 1) & (ma_mt >= ma_lt) & (ma_st >= ma_mt)),
                                 facecolor='green', alpha=0.5)
                ax1.fill_between(date, close, ma_mt, where=((regime == -1) & (ma_mt <= ma_lt) & (ma_st <= ma_mt)),
                                 facecolor='red', alpha=0.5)
                # Unprofitable conditions
                ax1.fill_between(date, close, ma_mt,
                                 where=((regime == 1) & (ma_mt >= ma_lt) & (ma_st >= ma_mt) & (close < ma_mt)),
                                 facecolor='darkgreen', alpha=1)
                ax1.fill_between(date, close, ma_mt,
                                 where=((regime == -1) & (ma_mt <= ma_lt) & (ma_st <= ma_mt) & (close >= ma_mt)),
                                 facecolor='darkred', alpha=1)

            elif pd.isnull(rg):  # floor/ceiling regime absent
                # Profitable conditions
                ax1.fill_between(date, close, ma_mt, where=((ma_mt >= ma_lt) & (ma_st >= ma_mt)),
                                 facecolor='green', alpha=0.4)
                ax1.fill_between(date, close, ma_mt, where=((ma_mt <= ma_lt) & (ma_st <= ma_mt)),
                                 facecolor='red', alpha=0.4)
                # Unprofitable conditions
                ax1.fill_between(date, close, ma_mt, where=((ma_mt >= ma_lt) & (ma_st >= ma_mt) & (close < ma_mt)),
                                 facecolor='darkgreen', alpha=1)
                ax1.fill_between(date, close, ma_mt, where=((ma_mt <= ma_lt) & (ma_st <= ma_mt) & (close >= ma_mt)),
                                 facecolor='darkred', alpha=1)

        if (np.sum(lt_hi) > 0):  # LT range breakout
            ax1.plot([], [], linewidth=5, label=' LT High', color='m', alpha=0.2)
            ax1.plot([], [], linewidth=5, label=' LT Low', color='b', alpha=0.2)

            if pd.notnull(rg):  # floor/ceiling regime present
                ax1.fill_between(date, close, lt_lo,
                                 where=((regime == 1) & (close > lt_lo)),
                                 facecolor='b', alpha=0.2)
                ax1.fill_between(date, close, lt_hi,
                                 where=((regime == -1) & (close < lt_hi)),
                                 facecolor='m', alpha=0.2)
                if (np.sum(st_hi) > 0):  # ST range breakout
                    ax1.fill_between(date, close, st_lo,
                                     where=((regime == 1) & (close > st_lo)),
                                     facecolor='b', alpha=0.3)
                    ax1.fill_between(date, close, st_hi,
                                     where=((regime == -1) & (close < st_hi)),
                                     facecolor='m', alpha=0.3)

            elif pd.isnull(rg):  # floor/ceiling regime absent
                ax1.fill_between(date, close, lt_lo,
                                 where=((close > lt_lo)), facecolor='b', alpha=0.2)
                ax1.fill_between(date, close, lt_hi,
                                 where=((close < lt_hi)), facecolor='m', alpha=0.2)
                if (np.sum(st_hi) > 0):  # ST range breakout
                    ax1.fill_between(date, close, st_lo,
                                     where=((close > st_lo) & (st_lo >= lt_lo)), facecolor='b', alpha=0.3)
                    ax1.fill_between(date, close, st_hi,
                                     where=((close < st_hi) & (st_hi <= lt_hi)), facecolor='m', alpha=0.3)

            if (np.sum(st_hi) > 0):  # ST range breakout
                ax1.plot([], [], linewidth=5, label=' ST High', color='m', alpha=0.3)
                ax1.plot([], [], linewidth=5, label=' ST Low', color='b', alpha=0.3)

            ax1.plot(df.index, lt_lo, '-.', color='b', label='LT low', alpha=0.2)
            ax1.plot(df.index, lt_hi, '-.', color='m', label='LT high', alpha=0.2)
    except:
        pass

    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax1.grid(True)
    ax1.xaxis.label.set_color('k')
    ax1.yaxis.label.set_color('k')
    plt.xlabel('Date')
    plt.ylabel(str.upper(ticker) + ' Price')
    plt.title(str.upper(ticker))
    plt.legend()
