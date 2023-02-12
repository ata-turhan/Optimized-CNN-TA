import random

import numpy as np
import pandas as pd
from ta.volatility import BollingerBands


def create_random_predictions(
    df: pd.DataFrame, hold_ratio: int = 10, buy_ratio: int = 1, sell_ratio: int = 1
) -> np.array:
    test_values = df
    predictions = np.array([0] * len(test_values))
    label_list = [0, 1, 2]
    weight_list = [hold_ratio, buy_ratio, sell_ratio]
    for i in range(test_values.shape[0]):
        predictions[i] = random.choices(label_list, weights=weight_list)[0]
    return predictions


def create_rsi_predictions(
    df: pd.DataFrame, period: int = 14, buy_value: int = 30, sell_value: int = 70
) -> np.array:
    rsi_values = df[[f"RSI-{period}"]]
    predictions = np.array([0] * len(rsi_values))
    last_label = 0
    for i in range(rsi_values.shape[0]):
        if (
            rsi_values.loc[rsi_values.index[i], f"RSI-{period}"] < buy_value
            and last_label != 1
        ):
            predictions[i] = 1
            last_label = 1
        elif (
            rsi_values.loc[rsi_values.index[i], f"RSI-{period}"] > sell_value
            and last_label != 2
        ):
            predictions[i] = 2
            last_label = 2
    return predictions


def create_ema_crossover_predictions(
    df: pd.DataFrame, short_period: int = 10, long_period: int = 20
) -> np.array:
    ema_values = df[[f"EMA-{short_period}", f"EMA-{long_period}"]]
    predictions = np.array([0] * len(ema_values))
    last_label = 0
    for i in range(ema_values.shape[0]):
        if (
            ema_values.loc[ema_values.index[i], f"EMA-{short_period}"]
            > ema_values.loc[ema_values.index[i], f"EMA-{long_period}"]
            and last_label != 1
        ):
            predictions[i] = 1
            last_label = 1
        elif (
            ema_values.loc[ema_values.index[i], f"EMA-{short_period}"]
            < ema_values.loc[ema_values.index[i], f"EMA-{long_period}"]
            and last_label != 2
        ):
            predictions[i] = 2
            last_label = 2
    return predictions


def create_bollinger_bands_predictions(
    df: pd.DataFrame, window: int = 20, window_dev: int = 2
) -> np.array:
    close_values = df[["Close"]]
    predictions = np.array([0] * len(close_values))
    indicator_bb = BollingerBands(
        close=close_values["Close"], window=window, window_dev=window_dev
    )
    close_values["bb_bbh"] = indicator_bb.bollinger_hband()
    close_values["bb_bbl"] = indicator_bb.bollinger_lband()
    last_label = 0
    for i in range(close_values.shape[0]):
        if (
            close_values.loc[close_values.index[i], "Close"]
            < close_values.loc[close_values.index[i], "bb_bbl"]
            and last_label != 1
        ):
            predictions[i] = 1
            last_label = 1
        elif (
            close_values.loc[close_values.index[i], "Close"]
            > close_values.loc[close_values.index[i], "bb_bbh"]
            and last_label != 2
        ):
            predictions[i] = 2
            last_label = 2
    return predictions
