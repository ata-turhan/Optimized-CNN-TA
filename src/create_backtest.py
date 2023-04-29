import datetime as dt
import math
import random
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import yfinance as yf
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# import sys
# sys.path.insert(1, "trading_modules")
from .configurations import set_random_seed


def second_2_minute_converter(seconds: int) -> str:
    minutes = seconds // 60
    return f"{int(minutes)} minutes and {round(seconds%60, 2)} seconds"


def volatility(portfolio_returns: np.array) -> float:
    return np.std(portfolio_returns)


def beta(portfolio_returns: np.array, benchmark_returns: np.array) -> float:
    covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
    variance = np.var(benchmark_returns)
    return covariance / variance


# Higher and Lower Partial Moments


def hpm(portfolio_returns: np.array, threshold: float = 0, order: int = 1) -> float:
    threshold_array = np.empty(len(portfolio_returns))
    threshold_array.fill(threshold)
    diff = portfolio_returns - threshold_array
    diff = diff.clip(min=0)
    return np.sum(diff**order) / len(portfolio_returns)


def lpm(portfolio_returns: np.array, threshold: float = 0, order: int = 1) -> float:
    threshold_array = np.empty(len(portfolio_returns))
    threshold_array.fill(threshold)
    diff = threshold_array - portfolio_returns
    diff = diff.clip(min=0)
    return np.sum(diff**order) / len(portfolio_returns)


# Expected Shortfall Measures


def VaR(portfolio_returns: np.array, alpha: float = 0.05) -> float:
    sorted_returns = np.sort(portfolio_returns)
    index = int(alpha * len(sorted_returns))
    return abs(sorted_returns[index])


def cVaR(portfolio_returns: np.array, alpha: float = 0.05) -> float:
    sorted_returns = np.sort(portfolio_returns)
    index = int(alpha * len(sorted_returns))
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    return abs(sum_var / index)


# DrawDown Measures


def max_drawdown(portfolio_returns: pd.Series, risk_free_rate: float = 0.01) -> float:
    wealth_index = (1 + portfolio_returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    return drawdown.min()


def average_drawdown(
    portfolio_returns: pd.Series, risk_free_rate: float = 0.01
) -> float:
    wealth_index = (1 + portfolio_returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    return drawdown.mean()


def average_drawdown_squared(
    portfolio_returns: pd.Series, risk_free_rate: float = 0.01
) -> float:
    wealth_index = (1 + portfolio_returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown_squared = ((wealth_index - previous_peaks) / previous_peaks).pow(2)
    return drawdown_squared.mean()


# Measures Of Risk-Adjusted Return Based On Volatility


def treynor_ratio(
    portfolio_returns: np.array,
    benchmark_returns: np.array,
    risk_free_rate: float = 0.01,
) -> float:
    return (
        (portfolio_returns.mean() - risk_free_rate)
        / beta(portfolio_returns, benchmark_returns)
        * math.sqrt(252)
    )


def sharpe_ratio(portfolio_returns: np.array, risk_free_rate: float = 0.01) -> float:
    return (
        (portfolio_returns.mean() - risk_free_rate)
        / volatility(portfolio_returns)
        * math.sqrt(252)
    )


def information_ratio(
    portfolio_returns: np.array, benchmark_returns: np.array
) -> float:
    diff = portfolio_returns - benchmark_returns
    return np.mean(diff) / np.std(diff) * math.sqrt(252)


def modigliani_ratio(
    portfolio_returns: np.array,
    benchmark_returns: np.array,
    risk_free_rate: float = 0.01,
) -> float:
    np_rf = np.empty(len(portfolio_returns))
    np_rf.fill(risk_free_rate)
    rdiff = portfolio_returns - np_rf
    bdiff = benchmark_returns - np_rf
    return (portfolio_returns.mean() - risk_free_rate) * (
        np.std(rdiff) / np.std(bdiff)
    ) + risk_free_rate


# Measure Of Risk-Adjusted Return Based On Value At Risk


def excess_var(
    portfolio_returns: np.array, risk_free_rate: float = 0.01, alpha: float = 0.05
) -> float:
    VaR_val = VaR(portfolio_returns, alpha)
    return (portfolio_returns.mean() - risk_free_rate) / (
        VaR_val if VaR_val != 0 else 10e-5
    )


def conditional_sharpe_ratio(
    portfolio_returns: np.array, risk_free_rate: float = 0.01, alpha: float = 0.05
) -> float:
    cVaR_val = cVaR(portfolio_returns, alpha)
    return (portfolio_returns.mean() - risk_free_rate) / (
        cVaR_val if cVaR_val != 0 else 10e-5
    )


# Measures Of Risk-Adjusted Return Based On Partial Moments


def omega_ratio(
    portfolio_returns: np.array, risk_free_rate: float = 0.01, threshold: float = 0
) -> float:
    return (portfolio_returns.mean() - risk_free_rate) / lpm(
        portfolio_returns, threshold, 1
    )


def sortino_ratio(
    portfolio_returns: np.array, risk_free_rate: float = 0.01, threshold: float = 0
) -> float:
    return (portfolio_returns.mean() - risk_free_rate) / math.sqrt(
        lpm(portfolio_returns, threshold, 2)
    )


def kappa_three_ratio(
    portfolio_returns: np.array, risk_free_rate: float = 0.01, threshold: float = 0
) -> float:
    return (portfolio_returns.mean() - risk_free_rate) / math.pow(
        lpm(portfolio_returns, threshold, 3), 1 / 3
    )


def gain_loss_ratio(portfolio_returns: np.array, threshold: float = 0) -> float:
    return hpm(portfolio_returns, threshold, 1) / lpm(portfolio_returns, threshold, 1)


def upside_potential_ratio(portfolio_returns: np.array, threshold: float = 0) -> float:
    return hpm(portfolio_returns, threshold, 1) / math.sqrt(
        lpm(portfolio_returns, threshold, 2)
    )


# Measures Of Risk-Adjusted Return Based On Drawdown Risk


def calmar_ratio(portfolio_returns: pd.Series, risk_free_rate: float = 0.01) -> float:
    return (
        (np.array(portfolio_returns).mean() - risk_free_rate)
        / abs(max_drawdown(portfolio_returns))
        * math.sqrt(252)
    )


def sterling_ratio(portfolio_returns: pd.Series, risk_free_rate: float = 0.01) -> float:
    return (portfolio_returns.mean() - risk_free_rate) / average_drawdown(
        pd.Series(portfolio_returns), risk_free_rate=0.01
    )


def burke_ratio(portfolio_returns: pd.Series, risk_free_rate: float = 0.01) -> float:
    return (portfolio_returns.mean() - risk_free_rate) / math.sqrt(
        average_drawdown_squared(pd.Series(portfolio_returns), risk_free_rate=0.01)
    )


def plot_init(
    ticker: str,
    benchmark_ticker: str,
    risk_free_rate: float,
    start_date: list,
    end_date: list,
    initial_capital: float,
    commission: float,
    alpha: float,
    threshold: float,
    order: int,
    short: bool,
    short_fee: float,
    standard_take_profit: bool,
    trailing_take_profit: bool,
    take_profit_ratio: float,
    standard_stop_loss: bool,
    trailing_stop_loss: bool,
    stop_loss_ratio: float,
    leverage: int,
) -> None:
    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        specs=[[{"type": "table"}]],
    )

    fig.add_trace(
        go.Table(
            columnwidth=[3, 1],
            header=dict(
                values=["Configurations", "Values"],
                line_color="darkslategray",
                fill_color="lightgreen",
                align="center",
            ),
            cells=dict(
                values=[
                    [
                        "Ticker",
                        "Benchmark Ticker",
                        "Risk-free Rate",
                        "Start Date",
                        "End Date",
                        "Inital Capital",
                        "Commission",
                        "Alpha",
                        "Threshold",
                        "Order",
                        "Short",
                        "Short Fee",
                        "Standard Take Profit",
                        "Trailing Take Profit",
                        "Take Profit Ratio",
                        "Standard Stop Loss",
                        "Trailing Stop Loss",
                        "Stop Loss Ratio",
                        "Leverage Ratio",
                    ],
                    [
                        f"{ticker}",
                        f"{benchmark_ticker}",
                        round(risk_free_rate, 2),
                        start_date,
                        end_date,
                        f"{initial_capital}$",
                        f"{commission}$",
                        alpha,
                        threshold,
                        order,
                        "is used" if short else "is not used",
                        short_fee,
                        "is used" if standard_take_profit else "is not used",
                        "is used" if trailing_take_profit else "is not used",
                        f"%{take_profit_ratio}",
                        "is used" if standard_stop_loss else "is not used",
                        "is used" if trailing_stop_loss else "is not used",
                        f"%{stop_loss_ratio}",
                        leverage,
                    ],
                ],
                line_color="darkslategray",
                fill_color="lightgoldenrodyellow",
                align="center",
            ),
        ),
        row=1,
        col=1,
    )
    fig.update_layout(
        width=700, height=700, title_text="INITIAL CONFIGURATION", title_x=0.5
    )
    fig.show()


def plot_tables(
    portfolio_value: np.array,
    benchmark_index: np.array,
    close_prices: np.array,
    total_trade_made: int,
    total_day_position_open: int,
    risk_free_rate: float = 0.01,
    alpha: float = 0.05,
    threshold: float = 0,
    order: int = 1,
    precision_point: int = 3,
    show_tables: bool = False,
) -> dict:
    portfolio_returns = np.diff(portfolio_value) / portfolio_value[:-1]
    benchmark_returns = np.diff(benchmark_index) / benchmark_index[:-1]

    benchmark_return_value = round(
        (benchmark_index[-1] - benchmark_index[0]) / benchmark_index[0] * 100,
        precision_point,
    )
    buy_hold_return = round(
        (close_prices[-1] - close_prices[0]) / close_prices[0] * 100, precision_point
    )
    total_return_value = round(
        (portfolio_value[-1] - portfolio_value[0]) / portfolio_value[0] * 100,
        precision_point,
    )
    year_count = len(close_prices) / 252
    try:
        annualized_return_value = round(
            (math.pow(total_return_value / 100 + 1, 1 / year_count) - 1) * 100,
            precision_point,
        )
        daily_return_value = round(
            (math.pow(total_return_value / 100 + 1, 1 / (year_count * 365)) - 1) * 100,
            precision_point,
        )
    except Exception:
        annualized_return_value = 0
        daily_return_value = 0
    profit_trade = (portfolio_returns > 0).sum()
    loss_trade = (portfolio_returns < 0).sum() - total_trade_made
    average_return_value = round(total_return_value / (profit_trade + loss_trade), 2)

    treynor_ratio_value = round(
        treynor_ratio(portfolio_returns, benchmark_returns, risk_free_rate),
        precision_point,
    )
    sharpe_ratio_value = round(
        sharpe_ratio(portfolio_returns, risk_free_rate), precision_point
    )
    information_ratio_value = round(
        information_ratio(portfolio_returns, benchmark_returns), precision_point
    )
    modigliani_ratio_value = round(
        modigliani_ratio(portfolio_returns, benchmark_returns, risk_free_rate),
        precision_point,
    )
    excess_var_value = round(
        excess_var(portfolio_returns, risk_free_rate, alpha), precision_point
    )
    conditional_sharpe_ratio_value = round(
        conditional_sharpe_ratio(portfolio_returns, risk_free_rate, alpha),
        precision_point,
    )
    omega_ratio_value = round(
        omega_ratio(portfolio_returns, risk_free_rate, threshold), precision_point
    )
    sortino_ratio_value = round(
        sortino_ratio(portfolio_returns, risk_free_rate, threshold), precision_point
    )
    kappa_3_ratio_value = round(
        kappa_three_ratio(portfolio_returns, risk_free_rate, threshold), precision_point
    )
    gain_loss_ratio_value = round(
        gain_loss_ratio(portfolio_returns, threshold), precision_point
    )
    upside_potential_ratio_value = round(
        upside_potential_ratio(portfolio_returns, threshold), precision_point
    )
    calmar_ratio_value = round(
        calmar_ratio(pd.Series(portfolio_returns), risk_free_rate), precision_point
    )
    sterling_ratio_value = round(
        sterling_ratio(pd.Series(portfolio_returns), risk_free_rate), precision_point
    )
    burke_ratio_value = round(
        burke_ratio(pd.Series(portfolio_returns), risk_free_rate), precision_point
    )

    volatility_value = round(volatility(portfolio_returns), precision_point)
    beta_value = round(beta(portfolio_returns, benchmark_returns), precision_point)
    hpm_value = round(hpm(portfolio_returns, threshold, order=1), precision_point)
    lpm_value = round(lpm(portfolio_returns, threshold, order=1), precision_point)
    var_value = round(VaR(portfolio_returns, alpha), precision_point)
    cvar_value = round(cVaR(portfolio_returns, alpha), precision_point)
    average_dd_value = round(
        average_drawdown(pd.Series(portfolio_returns), risk_free_rate), precision_point
    )
    average_dd_2_value = round(
        average_drawdown_squared(pd.Series(portfolio_returns), risk_free_rate),
        precision_point,
    )
    max_dd_value = round(
        max_drawdown(pd.Series(portfolio_returns), risk_free_rate), precision_point
    )

    total_trade_count = round(profit_trade + loss_trade, precision_point)
    annualized_total_trade_count = round(
        total_trade_count / year_count, precision_point
    )
    average_trade_duration_day = round(
        total_day_position_open / total_trade_count, precision_point
    )

    max_profit_trade = round(portfolio_returns.max() * 100, precision_point)
    max_loss_trade = round(portfolio_returns.min() * 100, precision_point)
    peak_capital = round(portfolio_value.max(), precision_point)
    through_capital = round(portfolio_value.min(), precision_point)

    if show_tables:
        fig = make_subplots(
            rows=3,
            cols=2,
            shared_xaxes=True,
            vertical_spacing=0.05,
            specs=[
                [{"type": "table"}, {"type": "table"}],
                [{"type": "table"}, {"type": "table"}],
                [{"type": "table"}, {"type": "table"}],
            ],
        )

        fig.add_trace(
            go.Table(
                columnwidth=[3, 1],
                header=dict(
                    values=["Returns", "Values"],
                    line_color="darkslategray",
                    fill_color="lightskyblue",
                    align="center",
                ),
                cells=dict(
                    values=[
                        [
                            "Benchmark Return",
                            "Buy&Hold Return",
                            "Total Return",
                            "Annualized Return",
                            "Daily Return",
                            "Average Return per Trade",
                        ],
                        [
                            f"%{benchmark_return_value}",
                            f"%{buy_hold_return}",
                            f"%{total_return_value}",
                            f"%{annualized_return_value}",
                            f"%{daily_return_value}",
                            f"%{average_return_value}",
                        ],
                    ],
                    line_color="darkslategray",
                    fill_color="lightcyan",
                    align="center",
                ),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Table(
                columnwidth=[3, 1],
                header=dict(
                    values=["Risk-Adjusted Return Metrics", "Values"],
                    line_color="darkslategray",
                    fill_color="lightskyblue",
                    align="center",
                ),
                cells=dict(
                    values=[
                        [
                            "Treynor Ratio",
                            "Sharpe Ratio",
                            "Information Ratio",
                            "Modigliani Ratio",
                            f"Excess VaR (alpha={alpha})",
                            f"Conditional Sharpe Ratio (alpha={alpha})",
                            f"Omega Ratio (threshold={threshold})",
                            f"Sortino Ratio (threshold={threshold})",
                            f"Kappa 3 Ratio (threshold={threshold})",
                            f"Gain Loss Ratio (threshold={threshold})",
                            f"Upside Potential Ratio (threshold={threshold})",
                            "Calmar Ratio",
                            "Sterling Ratio",
                            "Burke Ratio",
                        ],
                        [
                            treynor_ratio_value,
                            sharpe_ratio_value,
                            information_ratio_value,
                            modigliani_ratio_value,
                            excess_var_value,
                            conditional_sharpe_ratio_value,
                            omega_ratio_value,
                            sortino_ratio_value,
                            kappa_3_ratio_value,
                            gain_loss_ratio_value,
                            upside_potential_ratio_value,
                            calmar_ratio_value,
                            sterling_ratio_value,
                            burke_ratio_value,
                        ],
                    ],
                    line_color="darkslategray",
                    fill_color="lightcyan",
                    align="center",
                ),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Table(
                columnwidth=[3, 1],
                header=dict(
                    values=["Risk Metrics", "Values"],
                    line_color="darkslategray",
                    fill_color="lightskyblue",
                    align="center",
                ),
                cells=dict(
                    values=[
                        [
                            "Volatility",
                            "Beta",
                            f"Higher Partial Moment (threshold={threshold}, order={order})",
                            f"Lower Partial Moment (threshold={threshold}, order={order})",
                            f"VaR (alpha={alpha})",
                            f"CVaR (alpha={alpha})",
                            "Average Drawdown",
                            "Average Drawdown Squared",
                            "Maximum Drawdown",
                        ],
                        [
                            volatility_value,
                            beta_value,
                            hpm_value,
                            lpm_value,
                            var_value,
                            cvar_value,
                            average_dd_value,
                            average_dd_2_value,
                            max_dd_value,
                        ],
                    ],
                    line_color="darkslategray",
                    fill_color="lightcyan",
                    align="center",
                ),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Table(
                columnwidth=[3, 1],
                header=dict(
                    values=["Counts", "Values"],
                    line_color="darkslategray",
                    fill_color="lightskyblue",
                    align="center",
                ),
                cells=dict(
                    values=[
                        [
                            "Total trade count",
                            "Profitable trade count",
                            "Loss trade count",
                            "Win Rate",
                            "Annualized total trade count",
                            "Average trade duration day",
                            "Exposure Time",
                        ],
                        [
                            total_trade_count,
                            profit_trade,
                            loss_trade,
                            f"%{round(profit_trade/total_trade_count*100, precision_point)}",
                            annualized_total_trade_count,
                            average_trade_duration_day,
                            f"%{round(total_day_position_open/len(close_prices), precision_point)}",
                        ],
                    ],
                    line_color="darkslategray",
                    fill_color="lightcyan",
                    align="center",
                ),
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Table(
                columnwidth=[3, 1],
                header=dict(
                    values=["Extremums", "Values"],
                    line_color="darkslategray",
                    fill_color="lightskyblue",
                    align="center",
                ),
                cells=dict(
                    values=[
                        [
                            "Max profit for a trade",
                            "Max loss for a trade",
                            "Max capital value",
                            "Min capital value",
                        ],
                        [
                            f"%{max_profit_trade}",
                            f"%{max_loss_trade}",
                            peak_capital,
                            through_capital,
                        ],
                    ],
                    line_color="darkslategray",
                    fill_color="lightcyan",
                    align="center",
                ),
            ),
            row=3,
            col=1,
        )
        fig.update_layout(
            width=1000, height=1000, title_text="BACKTEST RESULTS", title_x=0.5
        )
        fig.show()

    return {
        "Total Return": total_return_value,
        "Treynor Ratio": treynor_ratio_value,
        "Sharpe Ratio": sharpe_ratio_value,
        "Information Ratio": information_ratio_value,
        "Modigliani Ratio": modigliani_ratio_value,
        "Excess VaR": excess_var_value,
        "Conditional Sharpe Ratio": conditional_sharpe_ratio_value,
        "Omega Ratio": omega_ratio_value,
        "Sortino Ratio": sortino_ratio_value,
        "Kappa 3 Ratio": kappa_3_ratio_value,
        "Gain Loss Ratio": gain_loss_ratio_value,
        "Upside Potential Ratio": upside_potential_ratio_value,
        "Calmar Ratio": calmar_ratio_value,
        "Sterling Ratio": sterling_ratio_value,
        "Burke Ratio": burke_ratio_value,
        "Volatility": volatility_value,
        "Beta": beta_value,
        "Higher Partial Moment": hpm_value,
        "Lower Partial Moment": lpm_value,
        "VaR": var_value,
        "CVaR": cvar_value,
        "Average Drawdown": average_dd_value,
        "Average Drawdown Squared": average_dd_2_value,
        "Maximum Drawdown": max_dd_value,
    }


def plot_charts(
    ticker: str,
    ohlcv: pd.DataFrame,
    predictions: np.array,
    portfolio_value: np.array,
    liquidated: bool,
) -> None:
    if liquidated:
        print(
            "\n----------------------------\nThis strategy is liquidated\n-------------------------------"
        )
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.1,
        subplot_titles=(f"OHLC of '{ticker}'", f"Volume of '{ticker}'"),
        row_width=[1, 5],
    )
    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv["Open"],
            high=ohlcv["High"],
            low=ohlcv["Low"],
            close=ohlcv["Close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Bar(x=ohlcv.index, y=ohlcv["Volume"], name="Volume"), row=2, col=1)
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(
        autosize=True,
        width=950,
        height=950,
    )
    fig.show()

    fig = go.Figure()
    buy_labels = predictions == 1
    sell_labels = predictions == 2
    fig.add_trace(
        go.Scatter(
            x=ohlcv.index,
            y=ohlcv["Close"],
            mode="lines",
            line=dict(color="#222266"),
            name="Close Price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ohlcv[buy_labels].index,
            y=ohlcv[buy_labels]["Close"],
            mode="markers",
            marker=dict(size=6, color="#2cc05c"),
            name="Buy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ohlcv[sell_labels].index,
            y=ohlcv[sell_labels]["Close"],
            mode="markers",
            marker=dict(size=6, color="#f62728"),
            name="Sell",
        )
    )
    fig.update_layout(title=f"Close Price with Predictions of '{ticker}'", title_x=0.5)
    fig.show()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ohlcv.index,
            y=portfolio_value,
            mode="lines",
            line=dict(color="#222266"),
            name="Portfolio Value",
        )
    )
    fig.update_layout(title="Portfolio Value", title_x=0.5)
    fig.show()


def financial_evaluation(
    ticker: str = "SPY",
    benchmark_ticker: str = "SPY",
    ohlcv: pd.DataFrame = None,
    predictions: np.array = None,
    risk_free_rate: float = 0.01 / 252,
    initial_capital: float = 1000,
    commission: float = 1,
    alpha: float = 0.05,
    threshold: float = 0,
    order: int = 1,
    order_type: str = "market",
    short: bool = False,
    short_fee: float = 1,
    standard_take_profit: bool = False,
    trailing_take_profit: bool = False,
    take_profit_ratio: float = 10,
    standard_stop_loss: bool = False,
    trailing_stop_loss: bool = False,
    stop_loss_ratio: float = 10,
    leverage: int = 1,
    miss_rate: int = 10,
    show_initial_configuration: bool = True,
    show_tables: bool = True,
    show_charts: bool = True,
    show_time: bool = True,
    precision_point: int = 3,
    seed: int = 42,
) -> dict:
    set_random_seed(42)
    HOLD_LABEL = 0
    BUY_LABEL = 1
    SELL_LABEL = 2
    start = time.time()
    if ohlcv.empty:
        print("OHLCV data is empty")
        return
    if predictions.all() == None:
        print("Predictions data is empty")
        return
    start_date = ohlcv.index[0] - dt.timedelta(days=5)
    end_date = ohlcv.index[-1] + dt.timedelta(days=5)
    benchmark_index = yf.download(
        benchmark_ticker, start_date, end_date, progress=False, interval="1d"
    )["Adj Close"]
    benchmark_index = benchmark_index.tz_localize(None)
    benchmark_index.index.names = ["Date"]
    benchmark_index = benchmark_index.loc[ohlcv.index]
    benchmark_index = np.array(benchmark_index)
    open_prices = ohlcv["Open"].values
    high_prices = ohlcv["High"].values
    low_prices = ohlcv["Low"].values
    close_prices = ohlcv["Close"].values
    capital = initial_capital
    long_open = False
    short_open = False
    long_price = 0.0
    short_price = 0.0
    stop_loss_price = 0.0
    take_profit_price = 0.0
    total_day_position_open = 0
    total_trade_made = 0
    portfolio_value = np.zeros(len(predictions) + 1)
    portfolio_value[0] = initial_capital
    liquidated = False
    trade_day = 0
    for i in range(len(predictions)):
        trade_day = i
        if short == False:
            change = 0
            if (
                standard_take_profit == True
                and standard_stop_loss == True
                and long_open == True
                and (
                    low_prices[i] <= stop_loss_price <= high_prices[i]
                    or low_prices[i] <= take_profit_price <= high_prices[i]
                )
            ):
                predictions[i] = SELL_LABEL
            if predictions[i] == BUY_LABEL and long_open == False:
                if order_type == "market":
                    long_open = True
                    long_price = round(random.uniform(low_prices[i], high_prices[i]), 6)
                elif order_type == "limit" and random.randint(1, miss_rate) != 1:
                    long_open = True
                    long_price = open_prices[i]
                if long_open == True:
                    stop_loss_price = long_price * (1 - stop_loss_ratio / 100)
                    take_profit_price = long_price * (1 + take_profit_ratio / 100)
                    capital -= commission
            elif predictions[i] == SELL_LABEL and long_open == True:
                if order_type == "market":
                    long_open = False
                    s = round(random.uniform(low_prices[i], high_prices[i]), 6)
                    change = (s - long_price) / long_price * leverage
                elif order_type == "limit" and random.randint(1, miss_rate) != 1:
                    long_open = False
                    change = (open_prices[i] - long_price) / long_price * leverage
                if long_open == False:
                    capital *= 1 + change
                    capital -= commission
                    if capital <= 0:
                        liquidated = True
                        break
                    total_trade_made += 1
            if long_open == True:
                total_day_position_open += 1
            portfolio_value[i + 1] = capital
            if long_open == True and trailing_stop_loss == True:
                stop_loss_price = close_prices[i] * (1 - stop_loss_ratio / 100)
            if long_open == True and trailing_take_profit == True:
                take_profit_price = close_prices[i] * (1 + take_profit_ratio / 100)
        elif short == True:
            change = 0
            if (
                predictions[i] != HOLD_LABEL
                and long_open == False
                and short_open == False
            ):
                if predictions[i] == BUY_LABEL:
                    if order_type == "market":
                        long_open = True
                        long_price = round(
                            random.uniform(low_prices[i], high_prices[i]), 6
                        )
                    elif order_type == "limit" and random.randint(1, miss_rate) != 1:
                        long_open = True
                        long_price = open_prices[i]
                    if long_open == True:
                        capital -= commission
                        total_trade_made += 1
                        stop_loss_price = long_price * (1 - stop_loss_ratio / 100)
                        take_profit_price = long_price * (1 + take_profit_ratio / 100)
                elif predictions[i] == SELL_LABEL:
                    if order_type == "market":
                        short_open = True
                        short_price = round(
                            random.uniform(low_prices[i], high_prices[i]), 6
                        )
                    elif order_type == "limit" and random.randint(1, miss_rate) != 1:
                        short_open = True
                        short_price = open_prices[i]
                    if short_open == True:
                        capital -= commission + short_fee
                        total_trade_made += 1
                        stop_loss_price = short_price * (1 + stop_loss_ratio / 100)
                        take_profit_price = short_price * (1 - take_profit_ratio / 100)
            if (
                standard_take_profit == True
                and standard_stop_loss == True
                and long_open == True
                and (
                    low_prices[i] <= stop_loss_price <= high_prices[i]
                    or low_prices[i] <= take_profit_price <= high_prices[i]
                )
            ):
                predictions[i] = SELL_LABEL
            if (
                standard_take_profit == True
                and standard_stop_loss == True
                and short_open == True
                and (
                    low_prices[i] <= stop_loss_price <= high_prices[i]
                    or low_prices[i] <= take_profit_price <= high_prices[i]
                )
            ):
                predictions[i] = BUY_LABEL
            if (
                predictions[i] == SELL_LABEL
                and long_open == True
                and short_open == False
            ):
                if order_type == "market":
                    long_open = False
                    short_open = True
                    short_price = round(
                        random.uniform(low_prices[i], high_prices[i]), 6
                    )
                    change = (short_price - long_price) / long_price * leverage
                elif order_type == "limit" and random.randint(1, miss_rate) != 1:
                    long_open = False
                    short_open = True
                    short_price = open_prices[i]
                    change = (short_price - long_price) / long_price * leverage
                if long_open == False and short_open == True:
                    capital *= 1 + change
                    capital -= commission + short_fee
                    if capital <= 0:
                        liquidated = True
                        break
                    total_trade_made += 1
                    stop_loss_price = short_price * (1 + stop_loss_ratio / 100)
                    take_profit_price = short_price * (1 - take_profit_ratio / 100)
            elif (
                predictions[i] == BUY_LABEL
                and long_open == False
                and short_open == True
            ):
                if order_type == "market":
                    long_open = True
                    short_open = False
                    long_price = round(random.uniform(low_prices[i], high_prices[i]), 6)
                    change = (short_price - long_price) / short_price * leverage
                elif order_type == "limit" and random.randint(1, miss_rate) != 1:
                    long_open = True
                    short_open = False
                    long_price = open_prices[i]
                    change = (short_price - long_price) / short_price * leverage
                if long_open == True and short_open == False:
                    capital *= 1 + change
                    capital -= commission
                    if capital <= 0:
                        liquidated = True
                        break
                    total_trade_made += 1
                    stop_loss_price = long_price * (1 - stop_loss_ratio / 100)
                    take_profit_price = long_price * (1 + take_profit_ratio / 100)
            if long_open == True or short_open == True:
                total_day_position_open += 1
            portfolio_value[i + 1] = capital
            if trailing_stop_loss == True and short_open == False and long_open == True:
                stop_loss_price = close_prices[i] * (1 - stop_loss_ratio / 100)
            if trailing_stop_loss == True and short_open == True and long_open == False:
                stop_loss_price = close_prices[i] * (1 + stop_loss_ratio / 100)
            if (
                trailing_take_profit == True
                and short_open == False
                and long_open == True
            ):
                take_profit_price = close_prices[i] * (1 + take_profit_ratio / 100)
            if (
                trailing_take_profit == True
                and short_open == True
                and long_open == False
            ):
                take_profit_price = close_prices[i] * (1 - take_profit_ratio / 100)
    trade_day += 2
    if total_trade_made == 0:
        print("No trade executed")
        return {}
    end = time.time()
    if show_time == True:
        print(f"\nBacktest was completed in {second_2_minute_converter(end-start)}.\n")
    if show_initial_configuration:
        plot_init(
            ticker,
            benchmark_ticker,
            risk_free_rate,
            start_date,
            end_date,
            initial_capital,
            commission,
            alpha,
            threshold,
            order,
            short,
            short_fee,
            standard_take_profit,
            trailing_take_profit,
            take_profit_ratio,
            standard_stop_loss,
            trailing_stop_loss,
            stop_loss_ratio,
            leverage,
        )
    metrics = plot_tables(
        portfolio_value[1:trade_day],
        benchmark_index[:trade_day],
        close_prices[:trade_day],
        total_trade_made,
        total_day_position_open,
        risk_free_rate,
        alpha,
        threshold,
        order,
        precision_point,
        show_tables,
    )
    if show_charts:
        plot_charts(
            "SPY",
            ohlcv[:trade_day],
            predictions[:trade_day],
            portfolio_value[1:trade_day],
            liquidated,
        )
    return metrics


def plain_classification_report(test_labels, predictions):
    print(
        classification_report(
            test_labels["Label"], predictions, target_names=["Hold", "Buy", "Sell"]
        )
    )


def classification_report_display(test_labels, predictions):
    sns.set(rc={"figure.figsize": (5, 5)})
    sampleClassificationReport = classification_report(
        test_labels["Label"],
        predictions,
        target_names=["Hold", "Buy", "Sell"],
        output_dict=True,
    )
    sns.heatmap(pd.DataFrame(sampleClassificationReport).iloc[:-1, :].T, annot=True)


def confusion_matrix_display(test_labels, predictions):
    disp = ConfusionMatrixDisplay.from_predictions(
        test_labels["Label"],
        predictions,
        display_labels=["Hold", "Buy", "Sell"],
        cmap=plt.cm.Blues,
        normalize="true",
    )
    fig = disp.ax_.get_figure()
    disp.ax_.grid(False)
    fig.set_figwidth(5)
    fig.set_figheight(5)
