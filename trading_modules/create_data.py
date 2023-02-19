import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import talib
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go


def HMA(df: pd.DataFrame, timeperiod:int = 14) -> float:
    """
    Hull Moving Average.
    Formula:
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    wma1 = talib.WMA(df, timeperiod=int(timeperiod/2))
    wma2 = talib.WMA(df, timeperiod=timeperiod)
    return talib.WMA(2*wma1 - wma2, timeperiod=int(timeperiod**0.5))


def money_flow_volume_series(df: pd.DataFrame) -> pd.Series:
    """
    Calculates money flow series
    """
    mfv = (
        df["Volume"]
        * (2 * df["Close"] - df["High"] - df["Low"])
        / (df["High"] - df["Low"])
    )
    return mfv


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


def trendNormalizePrices(prices: pd.DataFrame) -> None:
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


def create_labels(prices: pd.DataFrame) -> None:
    df = prices.copy()
    df["Label"] = [0] * df.shape[0]
    for i in range(df.shape[0] - 10):
        s = set(df["Close"].iloc[i : i + 11])
        minPrice = sorted(s)[0]
        maxPrice = sorted(s)[-1]
        for j in range(i, i + 11):
            if df["Close"].iloc[j] == minPrice and (j - i) == 5:
                df.iloc[j, df.columns.get_loc("Label")] = 1
            elif df["Close"].iloc[j] == maxPrice and (j - i) == 5:
                df.iloc[j, df.columns.get_loc("Label")] = 2
    return df.iloc[6:-6]


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


def show_label_distribution(df:pd.DataFrame):
    label_and_counts = pd.DataFrame(
    {"Label Names": ["Hold", "Buy", "Sell"], "Label Counts": df["Label"].value_counts().values})
    fig = px.bar(label_and_counts, x="Label Names", y="Label Counts")
    fig.show()


def show_prices(ticker: str, df: pd.DataFrame, desc:str=""):
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
    if desc == "":
        msg = f"Close Price of '{ticker}'"
    else:
        msg = desc
    fig.update_layout(title=msg, title_x=0.5)
    fig.show()


def show_price_and_labels(ticker: str, df: pd.DataFrame, desc:str=""):
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
    if desc == "":
        msg = f"Close Price with Labels of '{ticker}'"
    else:
        msg = desc
    fig.update_layout(title=msg, title_x=0.5)
    fig.show()


def create_all_indicators_in_talib(df: pd.DataFrame, periods: list):
    df_with_indicators = df.copy()
    for i in periods:
        df_with_indicators = df_with_indicators.assign(
            **{
                f"RSI-{i}": talib.RSI(df["Close"], timeperiod=i),
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
                f"SMA-{i}": talib.SMA(df["Close"], timeperiod=i),
                f"EMA-{i}": talib.EMA(df["Close"], timeperiod=i),
                f"WMA-{i}": talib.WMA(df["Close"], timeperiod=i),
                f"HMA-{i}": HMA(df["Close"], timeperiod=i),
                f"TEMA-{i}": talib.TEMA(df["Close"], timeperiod=i),
                f"PPO-{i}": talib.PPO(df["Close"], fastperiod=i, slowperiod=i + 14),
                f"ROC-{i}": talib.ROC(df_with_indicators["Close"], timeperiod=i),
                f"CMO-{i}": talib.CMO(df_with_indicators["Close"], timeperiod=i),
                f"MACD-{i}": talib.MACD(
                    df_with_indicators["Close"], fastperiod=i, slowperiod=i + 14
                )[0],
                f"MAMA-{i}": talib.MAMA(
                    df_with_indicators["Close"], fastlimit=1 / i, slowlimit=1 / (i + 14)
                )[0],
                f"STOCHRSI-{i}": talib.STOCHRSI(
                    df_with_indicators["Close"], timeperiod=i
                )[0],
                f"DX-{i}": talib.DX(
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
                f"CCI-{i}": talib.CCI(
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
                f"ATR-{i}": talib.ATR(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    df_with_indicators["Close"],
                    timeperiod=i,
                ),
                f"SAR-{i}": talib.SAR(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    maximum=1 / i,
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
                f"CMF-{i}": CMF(df_with_indicators, timeperiod=i),
                f"BOP": talib.BOP(
                    df_with_indicators["Open"],
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    df_with_indicators["Close"],
                ),
                f"TRANGE": talib.TRANGE(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    df_with_indicators["Close"],
                ),
                f"SAREXT": talib.SAREXT(
                    df_with_indicators["High"], df_with_indicators["Low"]
                ),
                f"AD": talib.AD(
                    df_with_indicators["High"],
                    df_with_indicators["Low"],
                    df_with_indicators["Close"],
                    df_with_indicators["Volume"],
                ),
                f"OBV": talib.OBV(
                    df_with_indicators["Close"], df_with_indicators["Volume"]
                ),
            }
        )
    df_with_indicators.dropna(inplace=True)
    return df_with_indicators


def create_2d_data(datas: list, length: int):
    matrix_datas = []
    for i in range(len(datas)):
        df = datas[i][0].drop(columns=["Label"])
        df_matrix_train = pd.DataFrame(
            index=df.iloc[length - 1 :].index,
        )
        df_matrix_train["Image"] = [np.zeros((length, length))] * df.iloc[
            length - 1 :
        ].shape[0]

        for j in range(df_matrix_train.shape[0]):
            matrix = df.iloc[j : j + length].values.reshape((length, length, -1))
            df_matrix_train.iloc[j, 0] = matrix
        df_matrix_train["Label"] = datas[i][0].iloc[length - 1 :, -1]

        df = datas[i][1].drop(columns=["Label"])
        df_matrix_test = pd.DataFrame(
            index=df.iloc[length - 1 :].index,
        )
        df_matrix_test["Image"] = [np.zeros((length, length))] * df.iloc[
            length - 1 :
        ].shape[0]

        for j in range(df_matrix_test.shape[0]):
            matrix = df.iloc[j : j + length].values.reshape((length, length, -1))
            df_matrix_test.iloc[j, 0] = matrix
        df_matrix_test["Label"] = datas[i][1].iloc[length - 1 :, -1]

        matrix_datas.append((df_matrix_train, df_matrix_test))

    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    for i in range(len(matrix_datas)):
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for j in range(matrix_datas[i][0].shape[0]):
            X_train.append(matrix_datas[i][0].iloc[j, 0])
            y_train.append(
                tf.keras.utils.to_categorical(
                    matrix_datas[i][0].iloc[j, 1], num_classes=3
                )
            )
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(
            X_train, newshape=(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        )
        X_train_list.append(X_train)
        y_train_list.append(y_train)

        for j in range(matrix_datas[i][1].shape[0]):
            X_test.append(matrix_datas[i][1].iloc[j, 0])
            y_test.append(matrix_datas[i][1].iloc[j, 1])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(
            X_test, newshape=(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
        )
        X_test_list.append(X_test)
        y_test_list.append(y_test)

    datas_2d = []

    for i in range(len(datas_1d)):
        datas_2d.append(
            [X_train_list[i], y_train_list[i], X_test_list[i], y_test_list[i]]
        )

    return datas_2d
