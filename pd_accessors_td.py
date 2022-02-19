from datetime import datetime

import pandas as pd
from pandas import Timedelta

import tda_access


def get_minimum_freq(date_times: pd.Index) -> Timedelta:
    """
    get the minimum frequency across a series of timestamps.
    Used to determine frequency of a series while taking into
    account larger than normal differences in bar times due to
    weekends and holiday
    :param date_times:
    """
    minimum = datetime.today() - date_times[-10]
    for i, date in enumerate(date_times):
        if date == date_times[-1]:
            break
        current_diff = date_times[i + 1] - date
        minimum = min(minimum, current_diff)

    return minimum


@pd.api.extensions.register_dataframe_accessor("update_check")
class UpdateCheck:
    def __init__(self, df: pd.DataFrame):
        self._obj = df

    def is_ready(
        self,
        market_type: tda_access.tda.client.Client.Markets,
        data_freq: t.Optional[timedelta] = None,
    ):
        """check if new bar is ready to be retrieved, prevents redundant API calls"""
        if data_freq is None:
            data_freq = get_minimum_freq(self._opj.index)
        ready = False
        if (
            tda_access.LocalClient.market_is_open(market_type)
            or
            # Allow for extra time to get data because, theoretically, market
            # will be closed when the last bar closes
            tda_access.LocalClient.market_was_open(market_type, time_ago=data_freq)
        ):
            ready = (datetime.now() - self._data.index[-1]) > data_freq
        return ready
