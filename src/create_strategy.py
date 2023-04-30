import random

import numpy as np
import pandas as pd
from ta.volatility import BollingerBands
from .configurations import set_random_seed, shift_predictions
from Pattern import (
    DarkCloudCover,
    Engulf,
    Hammer_Hanging_Man,
    Harami,
    Inv_Hammer,
    Marubozu,
    PiercingPattern,
    Spinning_Top,
    doji,
    dragonfly_doji,
    gravestone_doji,
    longleg_doji,
)
from sklearn.cluster import AgglomerativeClustering
from .configurations import set_random_seed


def create_buy_and_hold_predictions(price_length: int):
    predictions = np.array([0] * price_length)
    predictions[0], predictions[-1] = 1, 2
    return predictions


def create_random_predictions(
    df: pd.DataFrame,
    hold_ratio: int = 10,
    buy_ratio: int = 1,
    sell_ratio: int = 1,
    seed: int = 42,
) -> np.array:
    set_random_seed(seed=seed)
    test_values = df
    predictions = np.array([0] * len(test_values))
    label_list = [0, 1, 2]
    weight_list = [hold_ratio, buy_ratio, sell_ratio]
    for i in range(test_values.shape[0]):
        predictions[i] = random.choices(label_list, weights=weight_list)[0]
    predictions = shift_predictions(predictions)
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
    predictions = shift_predictions(predictions)
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
    predictions = shift_predictions(predictions)
    return predictions


def create_bollinger_bands_predictions(
    df: pd.DataFrame, window: int = 20, window_dev: int = 2
) -> np.array:
    close_values = df[["Close"]]
    predictions = np.array([0] * len(close_values))
    indicator_bb = BollingerBands(
        close=close_values["Close"], window=window, window_dev=window_dev
    )
    close_values = close_values.assign(bb_bbh=indicator_bb.bollinger_hband())
    close_values = close_values.assign(bb_bbl=indicator_bb.bollinger_lband())
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
    predictions = shift_predictions(predictions)
    return predictions


def candlestick_pattern_trading(
    ohlcv: pd.DataFrame, buy_pattern: str, sell_pattern: str
):
    candlestick_func_dict = {
        "Doji": doji,
        "Gravestone Doji": gravestone_doji,
        "Dragonfly Doji": dragonfly_doji,
        "Longleg Doji": longleg_doji,
        "Hammer Hanging Man": Hammer_Hanging_Man,
        "Inverse Hammer": Inv_Hammer,
        "Spinning Top": Spinning_Top,
        "Dark Cloud Cover": DarkCloudCover,
        "Piercing Pattern": PiercingPattern,
        "Bullish Marubozu": Marubozu,
        "Bearish Marubozu": Marubozu,
        "Bullish Engulfing": Engulf,
        "Bearish Engulfing": Engulf,
        "Bullish Harami": Harami,
        "Bearish Harami": Harami,
    }
    candlestick_column_dict = {
        "Doji": "Doji",
        "Gravestone Doji": "Gravestone",
        "Dragonfly Doji": "Dragonfly",
        "Longleg Doji": "LongLeg",
        "Hammer Hanging Man": "Hammer",
        "Inverse Hammer": "Inv_Hammer",
        "Spinning Top": "Spinning",
        "Dark Cloud Cover": "DarkCloud",
        "Piercing Pattern": "Piercing",
        "Bullish Marubozu": "Bull_Marubozu",
        "Bearish Marubozu": "Bear_Marubouzu",
        "Bullish Engulfing": "BullEngulf",
        "Bearish Engulfing": "BearEngulf",
        "Bullish Harami": "BullHarami",
        "Bearish Harami": "BearHarami",
    }
    candlestick_func_dict[buy_pattern](ohlcv)
    candlestick_func_dict[sell_pattern](ohlcv)
    signals = pd.DataFrame(index=ohlcv.index, data={"Signals": np.zeros((len(ohlcv),))})
    for i in range(len(ohlcv) - 1):
        if ohlcv.at[ohlcv.index[i], candlestick_column_dict[buy_pattern]] == True:
            signals.at[ohlcv.index[i + 1], "Signals"] = 1
        elif ohlcv.at[ohlcv.index[i], candlestick_column_dict[sell_pattern]] == True:
            signals.at[ohlcv.index[i + 1], "Signals"] = 2
    predictions = np.array(signals["Signals"])
    predictions = shift_predictions(predictions)
    return predictions


def calculate_support_resistance(df, rolling_wave_length, num_clusters, area):
    set_random_seed(42)
    date = df.index
    # Reset index for merging
    df.reset_index(inplace=True)
    # Create min and max waves
    max_waves_temp = df.High.rolling(rolling_wave_length).max().rename("waves")
    min_waves_temp = df.Low.rolling(rolling_wave_length).min().rename("waves")
    max_waves = pd.concat(
        [max_waves_temp, pd.Series(np.zeros(len(max_waves_temp)) + 1)], axis=1
    )
    min_waves = pd.concat(
        [min_waves_temp, pd.Series(np.zeros(len(min_waves_temp)) + -1)], axis=1
    )
    #  Remove dups
    max_waves.drop_duplicates("waves", inplace=True)
    min_waves.drop_duplicates("waves", inplace=True)
    #  Merge max and min waves
    waves = pd.concat([max_waves, min_waves]).sort_index()
    waves = waves[waves[0] != waves[0].shift()].dropna()
    # Find Support/Resistance with clustering using the rolling stats
    # Create [x,y] array where y is always 1
    x = np.concatenate(
        (
            waves.waves.values.reshape(-1, 1),
            (np.zeros(len(waves)) + 1).reshape(-1, 1),
        ),
        axis=1,
    )
    # Initialize Agglomerative Clustering
    cluster = AgglomerativeClustering(
        n_clusters=num_clusters, affinity="euclidean", linkage="ward"
    )
    cluster.fit_predict(x)
    waves["clusters"] = cluster.labels_
    # Get index of the max wave for each cluster
    if area == "resistance":
        waves2 = waves.loc[waves.groupby("clusters")["waves"].idxmax()]
    if area == "support":
        waves2 = waves.loc[waves.groupby("clusters")["waves"].idxmin()]
    df.index = date
    df.drop("Date", axis=1, inplace=True)
    waves2.waves.drop_duplicates(keep="first", inplace=True)
    return waves2.reset_index().waves


def support_resistance_trading(
    ohlcv: pd.DataFrame, rolling_wave_length: int = 20, num_clusters: int = 4
):
    signals = pd.DataFrame(index=ohlcv.index, data={"Signals": np.zeros((len(ohlcv),))})
    supports = calculate_support_resistance(
        ohlcv, rolling_wave_length, num_clusters, "support"
    )
    resistances = calculate_support_resistance(
        ohlcv, rolling_wave_length, num_clusters, "resistance"
    )
    for i in range(rolling_wave_length * 2 + num_clusters, len(ohlcv) - 1):
        for support in supports:
            if (
                ohlcv.at[ohlcv.index[i - 1], "Close"] >= support
                and ohlcv.at[ohlcv.index[i], "Close"] < support
            ):
                signals.at[ohlcv.index[i + 1], "Signals"] = 2
        if signals.at[ohlcv.index[i + 1], "Signals"] == 0:
            for resistance in resistances:
                if (
                    ohlcv.at[ohlcv.index[i - 1], "Close"] <= resistance
                    and ohlcv.at[ohlcv.index[i], "Close"] > resistance
                ):
                    signals.at[ohlcv.index[i + 1], "Signals"] = 1
    predictions = np.array(signals["Signals"])
    predictions = shift_predictions(predictions)
    return predictions
