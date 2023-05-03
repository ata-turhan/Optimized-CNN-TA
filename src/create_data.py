import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import sklearn.linear_model
import talib
import tensorflow as tf
from sklearn.linear_model import LinearRegression
import datetime as dt
import json
import random
import pandas_datareader as pdr
import requests
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    f_classif,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from ta import add_all_ta_features
from .configurations import set_random_seed

fred_codes = {
    "FED 2Y Interest Rate": "DGS2",
    "FED 10Y Interest Rate": "DGS10",
    "30-Year Fixed Rate Mortgage Average in the United States": "MORTGAGE30US",
    "Unemployment Rate": "UNRATE",
    "Real Gross Domestic Product": "GDPC1",
    "Gross Domestic Product": "GDP",
    "10-Year Breakeven Inflation Rate": "T10YIE",
    "Median Sales Price of Houses Sold for the United States": "MSPUS",
    "Personal Saving Rate": "PSAVERT",
    "Deposits, All Commercial Banks": "DPSACBW027SBOG",
    "S&P 500": "SP500",
    "Federal Debt: Total Public Debt as Percent of Gross Domestic Product": "GFDEGDQ188S",
    "Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma": "DCOILWTICO",
    "Consumer Loans: Credit Cards and Other Revolving Plans, All Commercial Banks": "CCLACBW027SBOG",
    "Consumer Price Index for All Urban Consumers: All Items Less Food and Energy in U.S. City Average": "CPILFESL",
}


def HMA(df: pd.DataFrame, timeperiod: int = 14) -> float:
    """
    Hull Moving Average.
    Formula:
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    wma1 = talib.WMA(df, timeperiod=timeperiod // 2)
    wma2 = talib.WMA(df, timeperiod=timeperiod)
    return talib.WMA(2 * wma1 - wma2, timeperiod=int(timeperiod**0.5))


def money_flow_volume_series(df: pd.DataFrame) -> pd.Series:
    """
    Calculates money flow series
    """
    return (
        df["Volume"]
        * (2 * df["Close"] - df["High"] - df["Low"])
        / (df["High"] - df["Low"])
    )


def money_flow_volume(df: pd.DataFrame, timeperiod: int = 20) -> pd.Series:
    """
    Calculates money flow volume, or q_t in our formula
    """
    return money_flow_volume_series(df).rolling(timeperiod).sum()


def CMF(df: pd.DataFrame, timeperiod: int = 20) -> pd.Series:
    """
    Calculates the Chaikin money flow
    """
    return money_flow_volume(df, timeperiod) / df["Volume"].rolling(timeperiod).sum()


def pltcolor(lst: list) -> list:
    cols = []
    for i in range(lst.shape[0]):
        if lst.iloc[i] == 1:
            cols.append("green")
        elif lst.iloc[i] == 2:
            cols.append("red")
    return cols


def trendNormalizePrices(prices: pd.DataFrame, seed: int = 42) -> None:
    set_random_seed(seed)
    df = prices.copy()
    df["rowNumber"] = list(range(len(df)))
    df["TN_Open"] = list(range(len(df)))
    df["TN_High"] = list(range(len(df)))
    df["TN_Low"] = list(range(len(df)))
    df["TN_Close"] = list(range(len(df)))
    for i in range(29, len(df)):
        model = LinearRegression()
        model.fit(
            np.array(df["rowNumber"].iloc[i - 29 : i + 1]).reshape(-1, 1),
            np.array(df["Close"].iloc[i - 29 : i + 1]),
        )
        prediction = model.predict(np.array([df["rowNumber"].iloc[i]]).reshape(-1, 1))
        df.iloc[i, df.columns.get_loc("TN_Open")] = df["Open"].iloc[i] - prediction
        df.iloc[i, df.columns.get_loc("TN_High")] = df["High"].iloc[i] - prediction
        df.iloc[i, df.columns.get_loc("TN_Low")] = df["Low"].iloc[i] - prediction
        df.iloc[i, df.columns.get_loc("TN_Close")] = df["Close"].iloc[i] - prediction
    df["Open"] = df["TN_Open"]
    df["High"] = df["TN_High"]
    df["Low"] = df["TN_Low"]
    df["Close"] = df["TN_Close"]
    df = df.drop(index=df.index[:30], axis=0)
    df = df.drop(
        columns=["TN_Open", "TN_High", "TN_Low", "TN_Close", "rowNumber"], axis=1
    )
    return df


def create_labels(ohlcv: pd.DataFrame) -> None:
    df = ohlcv.copy()
    df["Label"] = [0] * df.shape[0]
    for i in range(df.shape[0] - 10):
        s = set(df["Close"].iloc[i : i + 11])
        minPrice, maxPrice = sorted(s)[0], sorted(s)[-1]
        for j in range(i, i + 11):
            if df["Close"].iloc[j] == minPrice and (j - i) == 5:
                df.iloc[j, df.columns.get_loc("Label")] = 1
            elif df["Close"].iloc[j] == maxPrice and (j - i) == 5:
                df.iloc[j, df.columns.get_loc("Label")] = 2
    return df


def reverse_one_hot(predictions: np.array) -> np.array:
    return np.argmax(predictions, axis=1)


def one_hot(predictions: np.array) -> np.array:
    predictions_one_hot = []
    for i in predictions:
        prediction = [0, 0, 0]
        prediction[int(i)] = 1
        predictions_one_hot.append(prediction)
    return np.array(predictions_one_hot)


def number_null_and_nan(df: pd.DataFrame) -> int:
    na = pd.isna(df).sum().sum()
    null = df.isnull().sum().sum()
    return na + null


def adjustPrices(ohlcv: pd.DataFrame) -> None:
    adjustedRatio = ohlcv["Adj Close"] / ohlcv["Close"]
    ohlcv["High"] = ohlcv["High"] * adjustedRatio
    ohlcv["Low"] = ohlcv["Low"] * adjustedRatio
    ohlcv["Open"] = ohlcv["Open"] * adjustedRatio
    ohlcv["Close"] = ohlcv["Close"] * adjustedRatio


def show_label_distribution(df: pd.DataFrame):
    label_and_counts = pd.DataFrame(
        {
            "Label Names": ["Hold", "Buy", "Sell"],
            "Label Counts": df["Label"].value_counts().values,
        }
    )
    fig = px.bar(label_and_counts, x="Label Names", y="Label Counts")
    fig.show()


def show_multiple_prices(
    data: pd.DataFrame, ticker: str, show_which_price: str
) -> None:
    colors = ["red", "green", "blue", "orange", "black", "magenta", "cyan"]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.1,
        subplot_titles=(f"Chart", f"Volume"),
        row_width=[1, 5],
    )
    if "Candlestick" in show_which_price:
        show_which_price.remove("Candlestick")
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )
    for column in show_which_price:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[column],
                mode="lines",
                line=dict(color=random.choice(colors)),
                name=f"{column}",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume"), row=2, col=1)
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(
        autosize=True,
        width=950,
        height=950,
    )
    return fig


def show_prices(ticker: str, df: pd.DataFrame, desc: str = ""):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            line=dict(color="darkblue"),
            name="Close Price",
        )
    )
    msg = desc or f"Close Price of '{ticker}'"
    fig.update_layout(title=msg, title_x=0.5)
    fig.show()


def show_price_and_labels(ticker: str, df: pd.DataFrame, desc: str = ""):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            line=dict(color="darkblue"),
            name="Close Price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[df["Label"] == 1].index,
            y=df.loc[df["Label"] == 1]["Close"],
            mode="markers",
            marker=dict(size=4, color="#2cc05c"),
            name="Buy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[df["Label"] == 2].index,
            y=df.loc[df["Label"] == 2]["Close"],
            mode="markers",
            marker=dict(size=4, color="#f62728"),
            name="Sell",
        )
    )
    msg = desc or f"Close Price with Labels of '{ticker}'"
    fig.update_layout(title=msg, title_x=0.5)
    fig.show()


def signal_smoothing(
    df: pd.DataFrame, smoothing_method: str = "None", parameters: dict = None
):
    data = df.copy(deep=True)
    if smoothing_method == "None":
        return data
    elif smoothing_method == "Moving Average":
        data["Open"] = data["Open"].rolling(parameters["window"]).mean()
        data["High"] = data["High"].rolling(parameters["window"]).mean()
        data["Low"] = data["Low"].rolling(parameters["window"]).mean()
        data["Close"] = data["Close"].rolling(parameters["window"]).mean()
        # data.dropna(inplace=True)
    elif smoothing_method == "Heikin-Ashi":
        data = data.assign(HeikinAshi_Open=np.zeros((data.shape[0])))
        data = data.assign(HeikinAshi_High=np.zeros((data.shape[0])))
        data = data.assign(HeikinAshi_Low=np.zeros((data.shape[0])))
        data = data.assign(HeikinAshi_Close=np.zeros((data.shape[0])))
        data.iloc[0, data.columns.get_loc("HeikinAshi_Open")] = (
            data["Open"].iloc[0] + data["Close"].iloc[0]
        ) / 2
        for i in range(data.shape[0]):
            if i != 0:
                data.iloc[i, data.columns.get_loc("HeikinAshi_Open")] = (
                    data["HeikinAshi_Open"].iloc[i - 1]
                    + data["HeikinAshi_Close"].iloc[i - 1]
                ) / 2
            data.iloc[i, data.columns.get_loc("HeikinAshi_Close")] = (
                data["Open"].iloc[i]
                + data["High"].iloc[i]
                + data["Low"].iloc[i]
                + data["Close"].iloc[i]
            ) / 4
            data.iloc[i, data.columns.get_loc("HeikinAshi_High")] = max(
                [
                    data["High"].iloc[i],
                    data["HeikinAshi_Open"].iloc[i],
                    data["HeikinAshi_Close"].iloc[i],
                ]
            )
            data.iloc[i, data.columns.get_loc("HeikinAshi_Low")] = min(
                [
                    data["Low"].iloc[i],
                    data["HeikinAshi_Open"].iloc[i],
                    data["HeikinAshi_Close"].iloc[i],
                ]
            )
        data.drop(index=data.index[:30], axis=0, inplace=True)
        data["Open"] = data["HeikinAshi_Open"]
        data["High"] = data["HeikinAshi_High"]
        data["Low"] = data["HeikinAshi_Low"]
        data["Close"] = data["HeikinAshi_Close"]
        data.drop(
            [
                "HeikinAshi_Open",
                "HeikinAshi_High",
                "HeikinAshi_Low",
                "HeikinAshi_Close",
            ],
            axis=1,
            inplace=True,
        )
    elif smoothing_method == "Trend Normalization":
        data["rowNumber"] = list(range(len(data)))
        data["TN_Open"] = list(range(len(data)))
        data["TN_High"] = list(range(len(data)))
        data["TN_Low"] = list(range(len(data)))
        data["TN_Close"] = list(range(len(data)))
        for i in range(29, len(data)):
            model = LinearRegression()
            model.fit(
                np.array(data["rowNumber"].iloc[i - 29 : i + 1]).reshape(-1, 1),
                np.array(data["Close"].iloc[i - 29 : i + 1]),
            )
            prediction = model.predict(
                np.array([data["rowNumber"].iloc[i]]).reshape(-1, 1)
            )
            data.iloc[i, data.columns.get_loc("TN_Open")] = (
                data["Open"].iloc[i] - prediction
            )
            data.iloc[i, data.columns.get_loc("TN_High")] = (
                data["High"].iloc[i] - prediction
            )
            data.iloc[i, data.columns.get_loc("TN_Low")] = (
                data["Low"].iloc[i] - prediction
            )
            data.iloc[i, data.columns.get_loc("TN_Close")] = (
                data["Close"].iloc[i] - prediction
            )
        data["Open"] = data["TN_Open"]
        data["High"] = data["TN_High"]
        data["Low"] = data["TN_Low"]
        data["Close"] = data["TN_Close"]
        data.drop(index=data.index[:30], axis=0, inplace=True)
        data.loc[data.index[:30], ["Open", "High", "Low", "Close"]] = 0
        data.drop(
            [
                "rowNumber",
                "TN_Open",
                "TN_High",
                "TN_Low",
                "TN_Close",
            ],
            axis=1,
            inplace=True,
        )
    return data


def create_fundamental_data(ohlcv: pd.DataFrame) -> pd.DataFrame:
    start = ohlcv.index[0]
    end = ohlcv.index[-1]
    data = fetch_fundamental_data(ohlcv, start, end)
    return data


def create_ta_technical_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        with st.spinner("Fetching indicator data"):
            data = add_all_ta_features(
                ohlcv,
                open="Open",
                high="High",
                low="Low",
                close="Close",
                volume="Volume",
                colprefix="ta_",
            )
            return data


def create_all_indicators_in_talib(
    df: pd.DataFrame, periods: list, include_periodless_indicators: bool = False
):
    df_with_indicators = df.copy()
    for i in periods:
        df_with_indicators = df_with_indicators.assign(
            **{
                f"RSI-{i}": talib.RSI(df["Close"], timeperiod=i),
                f"STOCHRSI-{i}": talib.STOCHRSI(
                    df_with_indicators["Close"], timeperiod=i
                )[0],
                f"WILLR-{i}": talib.WILLR(
                    df["High"], df["Low"], df["Close"], timeperiod=i
                ),
                f"STOCH-{i}": talib.STOCH(
                    df["High"],
                    df["Low"],
                    df["Close"],
                    fastk_period=i + 7,
                    slowk_period=i - 4,
                )[0],
                f"STOCHF-{i}": talib.STOCHF(
                    df["High"],
                    df["Low"],
                    df["Close"],
                    fastk_period=i - 2,
                    fastd_period=i - 4,
                )[0],
                f"PPO-{i}": talib.PPO(df["Close"], fastperiod=i, slowperiod=i + 14),
                f"ROC-{i}": talib.ROC(df_with_indicators["Close"], timeperiod=i),
                f"SMA-{i}": talib.SMA(df["Close"], timeperiod=i),
                f"EMA-{i}": talib.EMA(df["Close"], timeperiod=i),
                f"WMA-{i}": talib.WMA(df["Close"], timeperiod=i),
                f"HMA-{i}": HMA(df["Close"], timeperiod=i),
                f"TEMA-{i}": talib.TEMA(df["Close"], timeperiod=i),
                f"CMO-{i}": talib.CMO(df_with_indicators["Close"], timeperiod=i),
                f"MACD-{i}": talib.MACD(
                    df_with_indicators["Close"],
                    fastperiod=i,
                    slowperiod=i + 14,
                )[0],
                f"SAR-{i}": talib.SAR(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    maximum=1 / i,
                ),
                f"CCI-{i}": talib.CCI(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    df_with_indicators["Close"],
                    timeperiod=i,
                ),
                f"ADXR-{i}": talib.ADXR(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    df_with_indicators["Close"],
                    timeperiod=i,
                ),
                f"MAMA-{i}": talib.MAMA(
                    df_with_indicators["Close"],
                    fastlimit=1 / i,
                    slowlimit=1 / (i + 14),
                )[0],
                f"DX-{i}": talib.DX(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    df_with_indicators["Close"],
                    timeperiod=i,
                ),
                f"PLUS_DI-{i}": talib.PLUS_DI(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    df_with_indicators["Close"],
                    timeperiod=i,
                ),
                f"MINUS_DI-{i}": talib.MINUS_DI(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    df_with_indicators["Close"],
                    timeperiod=i,
                ),
                f"PLUS_DM-{i}": talib.PLUS_DM(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    timeperiod=i,
                ),
                f"AROONOSC-{i}": talib.AROONOSC(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    timeperiod=i,
                ),
                f"MIDPRICE-{i}": talib.MIDPRICE(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    timeperiod=i,
                ),
                f"MFI-{i}": talib.MFI(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    df_with_indicators["Close"],
                    df_with_indicators["Volume"],
                    timeperiod=i,
                ),
                f"CMF-{i}": CMF(df_with_indicators, timeperiod=i),
                f"ADOSC-{i}": talib.ADOSC(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    df_with_indicators["Close"],
                    df_with_indicators["Volume"],
                    fastperiod=i - 4,
                    slowperiod=i + 3,
                ),
                f"BBANDS-{i}": talib.BBANDS(df_with_indicators["Close"], timeperiod=i)[
                    1
                ],
                f"ATR-{i}": talib.ATR(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    df_with_indicators["Close"],
                    timeperiod=i,
                ),
            }
        )
    if include_periodless_indicators:
        df_with_indicators = df_with_indicators.assign(
            **{
                "BOP": talib.BOP(
                    df_with_indicators["Open"],
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    df_with_indicators["Close"],
                ),
                "TRANGE": talib.TRANGE(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    df_with_indicators["Close"],
                ),
                "SAREXT": talib.SAREXT(
                    df_with_indicators["High"], df_with_indicators["Low"]
                ),
                "AD": talib.AD(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    df_with_indicators["Close"],
                    df_with_indicators["Volume"],
                ),
                "OBV": talib.OBV(
                    df_with_indicators["Close"], df_with_indicators["Volume"]
                ),
            }
        )
    df_with_indicators.dropna(inplace=True)
    return df_with_indicators


def create_2d_data(datas: list, length: int):
    matrix_datas = []
    for data in datas:
        df = data[0].drop(columns=["Label"])
        df_matrix_train = pd.DataFrame(
            index=df.iloc[length - 1 :].index,
        )
        df_matrix_train["Image"] = [np.zeros((length, length))] * df.iloc[
            length - 1 :
        ].shape[0]

        for j in range(df_matrix_train.shape[0]):
            matrix = df.iloc[j : j + length].values.reshape((length, length, -1))
            df_matrix_train.iloc[j, 0] = matrix
        df_matrix_train["Label"] = data[0].iloc[length - 1 :, -1]

        df = data[1].drop(columns=["Label"])
        df_matrix_test = pd.DataFrame(
            index=df.iloc[length - 1 :].index,
        )
        df_matrix_test["Image"] = [np.zeros((length, length))] * df.iloc[
            length - 1 :
        ].shape[0]

        for j in range(df_matrix_test.shape[0]):
            matrix = df.iloc[j : j + length].values.reshape((length, length, -1))
            df_matrix_test.iloc[j, 0] = matrix
        df_matrix_test["Label"] = data[1].iloc[length - 1 :, -1]

        matrix_datas.append((df_matrix_train, df_matrix_test))

    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    for matrix_data in matrix_datas:
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for j in range(matrix_data[0].shape[0]):
            X_train.append(matrix_data[0].iloc[j, 0])
            y_train.append(
                tf.keras.utils.to_categorical(matrix_data[0].iloc[j, 1], num_classes=3)
            )
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(
            X_train, newshape=(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        )
        X_train_list.append(X_train)
        y_train_list.append(y_train)

        for j in range(matrix_data[1].shape[0]):
            X_test.append(matrix_data[1].iloc[j, 0])
            y_test.append(matrix_data[1].iloc[j, 1])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(
            X_test, newshape=(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
        )
        X_test_list.append(X_test)
        y_test_list.append(y_test)

    return [
        [X_train_list[i], y_train_list[i], X_test_list[i], y_test_list[i]]
        for i in range(len(datas))
    ]


def plot_confusion_matrix(cm, labels, title):
    # cm : confusion matrix list(list)
    # labels : name of the data list(str)
    # title : title for the heatmap
    data = go.Heatmap(z=cm, y=labels, x=labels)
    annotations = []
    for i, row in enumerate(cm):
        annotations.extend(
            {
                "x": labels[i],
                "y": labels[j],
                "font": {"color": "white"},
                "text": str(value),
                "xref": "x1",
                "yref": "y1",
                "showarrow": False,
            }
            for j, value in enumerate(row)
        )
    layout = {
        "title": title,
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"},
        "annotations": annotations,
    }
    return go.Figure(data=data, layout=layout)


def fetch_fred_data(start_date: str, end_date: str) -> pd.DataFrame:
    fred_data = pdr.fred.FredReader(
        symbols=fred_codes.values(), start=start_date, end=end_date
    ).read()
    fred_data.fillna(method="ffill", inplace=True)
    fred_data["10-2 Year Yield Difference"] = fred_data["DGS10"] - fred_data["DGS2"]
    fred_data.rename(
        columns={v: k for k, v in fred_codes.items()},
        inplace=True,
    )
    return fred_data


def fetch_fundamental_data(data: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    fed_data_exists = False
    cpi_data_exists = True
    fundamentals = [
        "FED 2Y Interest Rate",
        "FED 10Y Interest Rate",
        "Yield Difference",
    ]
    for fundamental in fundamentals[:3]:
        if fundamental in data.columns:
            fed_data_exists = True

    data = data.tz_localize(None)
    data.index.names = ["Date"]

    if not fed_data_exists:
        fed_data = fetch_fred_data(start_date - pd.DateOffset(months=6), end_date)
        data["merg_col"] = data.index.strftime("%Y-%m-%d")
        fed_data["merg_col"] = fed_data.index.strftime("%Y-%m-%d")
        data = (
            data.reset_index()
            .merge(fed_data, on="merg_col", how="left")
            .set_index("Date")
            .drop(columns="merg_col")
        )

    if not cpi_data_exists:
        cpi_data = fetch_cpi_data(start_date - pd.DateOffset(months=1), end_date)
        data.index = data.index - pd.DateOffset(months=1)
        data["merg_col"] = data.index.strftime("%Y-%m")
        cpi_data["merg_col"] = cpi_data.index.strftime("%Y-%m")
        data = (
            data.reset_index()
            .merge(cpi_data, on="merg_col", how="left")
            .set_index("Date")
            .drop(columns="merg_col")
        )
        data.index = data.index + pd.DateOffset(months=1)
    return data
