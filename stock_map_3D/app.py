"""
日付・出来高・VWAP の 3D グラフ（1時間足・日足対応）Dash アプリ。
データは CSV 優先、なければ yfinance で取得して保存。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go

from fetch_stock_data import fetch_and_save, compute_vwap

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 銘柄リスト（ドロップダウン用）
TICKER_OPTIONS = [
    {"label": "AAPL (Apple)", "value": "AAPL"},
    {"label": "MSFT (Microsoft)", "value": "MSFT"},
    {"label": "7203.T (トヨタ)", "value": "7203.T"},
    {"label": "GOOGL (Google)", "value": "GOOGL"},
]

INTERVAL_OPTIONS = [
    {"label": "日足", "value": "1d"},
    {"label": "1時間足（最大約60日）", "value": "1h"},
]

PERIOD_OPTIONS = [
    {"label": "1年", "value": "1y"},
    {"label": "2年", "value": "2y"},
    {"label": "5年", "value": "5y"},
    {"label": "60日（1h用）", "value": "60d"},
]


def load_from_csv(ticker: str, interval: str) -> dict[str, Any] | None:
    """
    data/{ticker}_{interval}.csv を読み込むのみ（取得は行わない）。
    VWAP が無ければ (High+Low+Close)/3 で計算。
    返却: {"date_labels": [...], "volume": ndarray, "vwap": ndarray} または None（ファイルなし・失敗時）
    """
    csv_path = DATA_DIR / f"{ticker}_{interval}.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty:
        return None
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "VWAP" not in df.columns and all(c in df.columns for c in ("High", "Low", "Close")):
        df["VWAP"] = compute_vwap(df)
    if "VWAP" not in df.columns or "Volume" not in df.columns:
        return None
    df = df.dropna(subset=["VWAP", "Volume"])
    if len(df) == 0:
        return None
    date_labels = df["Date"].dt.strftime("%Y-%m-%d %H:%M").str.rstrip(" 00:00").tolist()
    return {
        "date_labels": date_labels,
        "volume": df["Volume"].to_numpy(dtype=float),
        "vwap": df["VWAP"].to_numpy(dtype=float),
        "n": len(df),
    }


def _no_data_figure(title: str = "データがありません") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=f"{title}<br>銘柄・時間足・期間を選び、左の「データを取得」ボタンで取得してください。",
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(size=14), align="center",
    )
    fig.update_layout(template="plotly_dark", margin=dict(t=40, b=40, l=40, r=40))
    return fig


def create_3d_figure(
    data: dict[str, Any],
    y_start: int | None = None,
    y_end: int | None = None,
    ticker_label: str = "",
) -> go.Figure:
    """3D: X=日付インデックス, Y=Volume, Z=VWAP, 線でつなぐ。"""
    if data is None or data["n"] == 0:
        return _no_data_figure()
    n = data["n"]
    indices = np.arange(n)
    if y_start is not None:
        y_start = max(0, min(y_start, n - 1))
    if y_end is not None:
        y_end = max(0, min(y_end, n - 1))
    if y_start is None:
        y_start = 0
    if y_end is None:
        y_end = n - 1
    if y_start > y_end:
        y_start, y_end = y_end, y_start
    idx_slice = slice(y_start, y_end + 1)
    x = data["vwap"][idx_slice]
    y = indices[idx_slice]
    z = data["volume"][idx_slice]
    labels = data["date_labels"]

    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines+markers",
        line=dict(width=4, color="cyan"),
        marker=dict(size=3),
        text=[labels[i] for i in range(y_start, y_end + 1)],
        hovertemplate="日付: %{text}<br>VWAP: %{x:.2f}<br>出来高: %{z:,.0f}<extra></extra>",
    )
    fig = go.Figure(data=[trace])
    step = max(1, (y_end - y_start + 1) // 10)
    tick_indices = list(range(y_start, y_end + 1, step))
    if tick_indices and tick_indices[-1] != y_end:
        tick_indices.append(y_end)
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="VWAP",
            yaxis_title="日付（インデックス）",
            zaxis_title="出来高",
            yaxis=dict(tickmode="array", tickvals=tick_indices, ticktext=[labels[i] for i in tick_indices]),
        ),
        template="plotly_dark",
        title=f"{ticker_label} VWAP × 日付 × 出来高（3D）",
    )
    return fig


def create_price_figure(data: dict[str, Any] | None, point_index: int, ticker_label: str = "") -> go.Figure:
    """2D: 日付 × VWAP。point_index に縦線とマーカー。"""
    if data is None or data["n"] == 0:
        return _no_data_figure()
    n = data["n"]
    idx = max(0, min(point_index, n - 1))
    dates = data["date_labels"]
    vwap = data["vwap"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(n)),
            y=vwap,
            mode="lines",
            hovertext=dates,
            hovertemplate="%{hovertext}<br>VWAP: %{y:.2f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=idx, line=dict(color="red", width=1, dash="dash"))
    fig.add_trace(
        go.Scatter(
            x=[idx],
            y=[vwap[idx]],
            mode="markers",
            marker=dict(color="red", size=10),
            showlegend=False,
            hovertext=[dates[idx]],
            hovertemplate="%{hovertext}<br>VWAP: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis=dict(
            title="時間",
            tickmode="array",
            tickvals=list(range(0, n, max(1, n // 10))),
            ticktext=[dates[i] for i in range(0, n, max(1, n // 10))],
        ),
        yaxis_title="VWAP",
        template="plotly_dark",
        showlegend=False,
        title=f"{ticker_label} VWAP 時系列（{dates[idx]} を強調）",
    )
    return fig


def create_volume_figure(data: dict[str, Any] | None, point_index: int, ticker_label: str = "") -> go.Figure:
    """2D: 日付 × 出来高。point_index に縦線とマーカー。"""
    if data is None or data["n"] == 0:
        return _no_data_figure()
    n = data["n"]
    idx = max(0, min(point_index, n - 1))
    dates = data["date_labels"]
    vol = data["volume"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(n)),
            y=vol,
            mode="lines",
            hovertext=dates,
            hovertemplate="%{hovertext}<br>出来高: %{y:,.0f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=idx, line=dict(color="red", width=1, dash="dash"))
    fig.add_trace(
        go.Scatter(
            x=[idx],
            y=[vol[idx]],
            mode="markers",
            marker=dict(color="red", size=10),
            showlegend=False,
            hovertext=[dates[idx]],
            hovertemplate="%{hovertext}<br>出来高: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis=dict(
            title="時間",
            tickmode="array",
            tickvals=list(range(0, n, max(1, n // 10))),
            ticktext=[dates[i] for i in range(0, n, max(1, n // 10))],
        ),
        yaxis_title="出来高",
        template="plotly_dark",
        showlegend=False,
        title=f"{ticker_label} 出来高時系列（{dates[idx]} を強調）",
    )
    return fig


_NO_SELECT = {
    "userSelect": "none",
    "WebkitUserSelect": "none",
    "MozUserSelect": "none",
    "msUserSelect": "none",
}

app = Dash(__name__)

app.layout = html.Div(
    style={
        "display": "flex",
        "height": "100vh",
        "backgroundColor": "#111",
        **_NO_SELECT,
    },
    children=[
        html.Div(
            style={
                "flex": "0 0 40%",
                "display": "flex",
                "flexDirection": "column",
                "padding": "10px",
                "backgroundColor": "#111",
                **_NO_SELECT,
            },
            children=[
                html.Div(
                    style={"marginBottom": "10px", **_NO_SELECT},
                    children=[
                        html.Label("銘柄"),
                        dcc.Dropdown(
                            id="ticker-dropdown",
                            options=TICKER_OPTIONS,
                            value="AAPL",
                            clearable=False,
                            style={"minWidth": "160px", "color": "#111", "backgroundColor": "#fff"},
                            className="ticker-dropdown-dark-text",
                        ),
                        html.Label("時間足", style={"marginTop": "10px"}),
                        dcc.Dropdown(
                            id="interval-dropdown",
                            options=INTERVAL_OPTIONS,
                            value="1d",
                            clearable=False,
                            style={"minWidth": "160px", "color": "#111", "backgroundColor": "#fff"},
                            className="ticker-dropdown-dark-text",
                        ),
                        html.Label("期間", style={"marginTop": "10px"}),
                        dcc.Dropdown(
                            id="period-dropdown",
                            options=PERIOD_OPTIONS,
                            value="1y",
                            clearable=False,
                            style={"minWidth": "120px", "color": "#111", "backgroundColor": "#fff"},
                            className="ticker-dropdown-dark-text",
                        ),
                        html.Div(
                            style={"marginTop": "12px"},
                            children=[
                                html.Button(
                                    id="fetch-button",
                                    children="データを取得",
                                    style={
                                        "padding": "8px 16px",
                                        "backgroundColor": "#2196F3",
                                        "color": "#fff",
                                        "border": "none",
                                        "borderRadius": "4px",
                                        "cursor": "pointer",
                                        "fontSize": "14px",
                                    },
                                ),
                                html.Span(
                                    id="fetch-status",
                                    style={"marginLeft": "10px", "fontSize": "12px", "color": "#aaa"},
                                ),
                            ],
                        ),
                        html.Div(
                            style={"marginTop": "10px"},
                            children=[
                                html.Label("日付範囲（インデックス）"),
                                dcc.RangeSlider(
                                    id="date-range-slider",
                                    min=0,
                                    max=0,
                                    value=[0, 0],
                                    allowCross=False,
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Graph(id="price-graph", style={"flex": "1 1 50%", "marginTop": "16px", **_NO_SELECT}),
                dcc.Graph(id="volume-graph", style={"flex": "1 1 50%", "marginTop": "20px", **_NO_SELECT}),
            ],
        ),
        html.Div(
            style={"flex": "1 1 60%", "padding": "10px", **_NO_SELECT},
            children=[
                dcc.Graph(id="surface-graph", style={"height": "100%", **_NO_SELECT}),
            ],
        ),
        dcc.Store(id="fetch-done-store", data=0),
    ],
)


def _period_for_fetch(interval: str, period: str) -> str:
    """1h の場合は最大約60日でトリム。"""
    if interval != "1h":
        return period or "1y"
    if period in ("60d", "1mo"):
        return period
    try:
        if "y" in (period or ""):
            days = int((period or "1").replace("y", "")) * 365
        else:
            days = int((period or "60").replace("d", "") or "60")
        if days > 60:
            return "60d"
    except ValueError:
        pass
    return period or "60d"


@app.callback(
    Output("fetch-done-store", "data"),
    Output("fetch-status", "children"),
    Input("fetch-button", "n_clicks"),
    State("ticker-dropdown", "value"),
    State("interval-dropdown", "value"),
    State("period-dropdown", "value"),
    State("fetch-done-store", "data"),
    prevent_initial_call=True,
)
def on_fetch_click(n_clicks: int | None, ticker: str, interval: str, period: str, store: int | None):
    if not n_clicks:
        return (store or 0), ""
    ticker = ticker or "AAPL"
    interval = interval or "1d"
    period = period or "1y"
    period_use = _period_for_fetch(interval, period)
    try:
        fetch_and_save(ticker=ticker, interval=interval, period=period_use)
        return (store or 0) + 1, "取得しました"
    except Exception as e:
        return store or 0, f"取得失敗: {e!s}"


@app.callback(
    Output("date-range-slider", "min"),
    Output("date-range-slider", "max"),
    Output("date-range-slider", "value"),
    Output("date-range-slider", "marks"),
    Input("ticker-dropdown", "value"),
    Input("interval-dropdown", "value"),
    Input("period-dropdown", "value"),
    Input("fetch-done-store", "data"),
)
def init_date_slider(ticker: str, interval: str, period: str, _fetch_done: int | None):
    data = load_from_csv(ticker or "AAPL", interval or "1d")
    if data is None or data["n"] == 0:
        return 0, 0, [0, 0], {}
    n = data["n"]
    min_idx, max_idx = 0, n - 1
    step = max(1, n // 10)
    def _mark_style():
        return {
            "transform": "translateX(-100%) rotate(-45deg)",
            "transformOrigin": "100% 50%",
            "textAlign": "right",
            "whiteSpace": "nowrap",
            "fontSize": "10px",
            "display": "inline-block",
        }
    marks = {i: {"label": data["date_labels"][i], "style": _mark_style()} for i in range(0, n, step)}
    marks[max_idx] = {"label": data["date_labels"][max_idx], "style": _mark_style()}
    return min_idx, max_idx, [min_idx, max_idx], marks


@app.callback(
    Output("surface-graph", "figure"),
    Input("ticker-dropdown", "value"),
    Input("interval-dropdown", "value"),
    Input("period-dropdown", "value"),
    Input("date-range-slider", "value"),
    Input("fetch-done-store", "data"),
)
def update_3d(ticker: str, interval: str, period: str, slider_value: list | None, _fetch_done: int | None):
    data = load_from_csv(ticker or "AAPL", interval or "1d")
    ticker_label = ticker or "AAPL"
    if data is None:
        return _no_data_figure()
    y_start, y_end = None, None
    if slider_value and len(slider_value) >= 2:
        y_start, y_end = int(slider_value[0]), int(slider_value[1])
    return create_3d_figure(data, y_start, y_end, ticker_label=ticker_label)


@app.callback(
    Output("price-graph", "figure"),
    Output("volume-graph", "figure"),
    Input("ticker-dropdown", "value"),
    Input("interval-dropdown", "value"),
    Input("period-dropdown", "value"),
    Input("surface-graph", "hoverData"),
    Input("date-range-slider", "value"),
    Input("fetch-done-store", "data"),
)
def update_2d(ticker: str, interval: str, period: str, hover_data: dict | None, slider_value: list | None, _fetch_done: int | None):
    data = load_from_csv(ticker or "AAPL", interval or "1d")
    ticker_label = ticker or "AAPL"
    if data is None:
        return _no_data_figure(), _no_data_figure()
    n = data["n"]
    y_start = 0
    y_end = n - 1
    if slider_value and len(slider_value) >= 2:
        y_start = max(0, min(int(slider_value[0]), n - 1))
        y_end = max(y_start, min(int(slider_value[1]), n - 1))
    point_index = y_end
    if hover_data and hover_data.get("points"):
        pt = hover_data["points"][0]
        # Scatter3d の pointNumber が点のインデックス
        if "pointNumber" in pt:
            point_index = int(pt["pointNumber"])
        elif "x" in pt and isinstance(pt["x"], (int, float)):
            point_index = int(round(pt["x"]))
    point_index = max(y_start, min(point_index, y_end))
    return (
        create_price_figure(data, point_index, ticker_label=ticker_label),
        create_volume_figure(data, point_index, ticker_label=ticker_label),
    )


if __name__ == "__main__":
    app.run(debug=True)
