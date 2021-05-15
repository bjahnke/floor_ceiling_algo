import trade_df
from trade_df import tdargs
from typing import List

def init_mdfs(symbols: List[str], freq_range: tdargs.FreqRangeArgs) -> List[trade_df.PriceMdf]:
    """
    :param symbols:
    :param freq_range:
    :return:
    """
    mdfs = []
    for symbol in symbols:
        mdfs.append(
            trade_df.PriceMdf.init_base(
                symbol=symbol,
                freq_range=freq_range
            )
        )
    return mdfs

def test_match_shapes():
    base, bench = init_mdfs(
        symbols=['AAPL', 'SPX'],
        freq_range=tdargs.freqs.day.range(tdargs.periods.y2)
    )

    merged_dfs = base.data.merge(bench.data)


