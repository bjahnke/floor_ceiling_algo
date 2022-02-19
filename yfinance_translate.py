import json
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf
import typing as t


def format_price_data(data: pd.DataFrame) -> pd.DataFrame:
    """format price data to what is expected by scanner/strategy code"""

    data.columns = map(lambda x: x.lower(), data.columns.to_list())

    data["b_high"] = data.high
    data["b_low"] = data.low
    data["b_close"] = data.close

    # convert date time to timezone unaware
    try:
        data.index = data.index.tz_convert(None)
    except (AttributeError, TypeError):
        pass

    return data[
        ["open", "high", "low", "close", "b_high", "b_low", "b_close", "volume"]
    ]


def yf_price_history(symbol: str, freq_range, period="3mo", interval="1h"):
    try:
        price_data: pd.DataFrame = yf.Ticker(symbol).history(
            start=freq_range[0],
            end=freq_range[1],
            interval=freq_range[2]
            # period=period, interval=interval
        )
    except json.decoder.JSONDecodeError:
        return pd.DataFrame()
    except requests.exceptions.ConnectionError:
        return pd.DataFrame()
    except AttributeError:
        return pd.DataFrame()

    if symbol == "CCI30":
        price_data = pd.read_csv("cci30_OHLCV.csv")
        price_data.Date = pd.to_datetime(
            price_data.Date, infer_datetime_format=True, utc=True
        )
        price_data.index = price_data.Date
        price_data = price_data.sort_index()

    try:
        price_data.index = price_data.index.tz_convert(None)
    except AttributeError:
        pass
    price_data = price_data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    return format_price_data(price_data)


def yf_price_history_stream(symbol: str, interval: int, bars: int, interval_type="m"):
    """interval: bar interval in minutes"""
    interval_secs = interval * 60
    price_data = yf_price_history(
        symbol,
        freq_range=(
            datetime.now() - timedelta(seconds=interval_secs * bars),
            datetime.now(),
            f"{interval}{interval_type}",
        ),
    )
    price_data["symbol"] = symbol
    delay = datetime.utcnow() - price_data.index[-1] + timedelta(minutes=1)
    return price_data[["symbol", "open", "high", "low", "close"]].iloc[:-1], delay


def yf_get_delays(symbols: t.List[str], interval, days, interval_type="m"):
    delays = []
    for symbol in symbols:
        _, delay = yf_price_history_stream(symbol, interval, days, interval_type)
        delays.append((symbol, delay))
    return delays


# def yf_get_delays(symbols: t.List[str], interval, days, interval_type='m'):
#     price_data = yf.download(
#         symbols,
#         start=datetime.now() - timedelta(days=days),
#         end=datetime.now(),
#         interval=f'{interval}{interval_type}'
#     )
#     try:
#         price_data.index = price_data.index.tz_convert(None)
#     except (AttributeError, TypeError):
#         pass
#     data_split = []
#     for symbol in symbols:
#         close_data = price_data['Close'][symbol]
#         delay = datetime.utcnow() - close_data.index[-1] + timedelta(minutes=1)
#         data_split.append((symbol, delay))
#     return data_split
