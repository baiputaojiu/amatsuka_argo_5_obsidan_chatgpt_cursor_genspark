# stock_map_3D: 日付×株価×出来高の3D山脈と2Dグラフ
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAVORITES_PATH = DATA_DIR / "favorites.json"

# ---------------------------------------------------------------------------
# お気に入り
# ---------------------------------------------------------------------------
def load_favorites() -> list[str]:
    if not FAVORITES_PATH.exists():
        FAVORITES_PATH.parent.mkdir(parents=True, exist_ok=True)
        fav = ["7203.T"]
        with open(FAVORITES_PATH, "w", encoding="utf-8") as f:
            import json
            json.dump(fav, f, ensure_ascii=False, indent=2)
        return fav
    with open(FAVORITES_PATH, "r", encoding="utf-8") as f:
        import json
        return json.load(f)

def save_favorites(fav: list[str]) -> None:
    FAVORITES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAVORITES_PATH, "w", encoding="utf-8") as f:
        import json
        json.dump(fav, f, ensure_ascii=False, indent=2)

def add_favorite(ticker: str) -> list[str]:
    fav = load_favorites()
    ticker = (ticker or "").strip().upper()
    if ticker and ticker not in fav:
        fav.append(ticker)
        save_favorites(fav)
    return load_favorites()

# ---------------------------------------------------------------------------
# 5分足取得・CSV
# ---------------------------------------------------------------------------
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

def csv_path(ticker: str) -> Path:
    # ファイル名に使えない文字を置換（. はそのまま）
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in ticker)
    return DATA_DIR / f"{safe}_5m.csv"

def fetch_5m_full(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period="60d", interval="5m", group_by="column", progress=False, threads=False)
        if df is None or df.empty:
            return None
        df = _flatten_columns(df)
        df = df.dropna(how="all")
        if df.empty:
            return None
        return df
    except Exception:
        return None

def fetch_5m_incremental(ticker: str, last_ts: pd.Timestamp) -> pd.DataFrame | None:
    try:
        start = last_ts + pd.Timedelta(minutes=5)
        end = pd.Timestamp.now(tz=start.tzinfo) if start.tz else pd.Timestamp.now()
        if start >= end:
            return pd.DataFrame()
        df = yf.download(ticker, start=start, end=end, interval="5m", group_by="column", progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = _flatten_columns(df)
        df = df.dropna(how="all")
        return df
    except Exception:
        return None

def load_csv(ticker: str) -> pd.DataFrame | None:
    path = csv_path(ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        if df.empty:
            return None
        return df
    except Exception:
        return None

def save_csv(ticker: str, df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = csv_path(ticker)
    df.to_csv(path)

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            return pd.DataFrame()
    return df

# サーバー側キャッシュ（dcc.Store のサイズ制限で全期間が切られないように）
_DISPLAY_CACHE: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# 描画用データ計算
# ---------------------------------------------------------------------------
def compute_display_data(df: pd.DataFrame) -> dict[str, Any] | None:
    df = ensure_columns(df)
    if df.empty:
        return None
    df = df.sort_index()
    typical = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
    last_close = float(df["Close"].iloc[-1])
    step = max(last_close * 0.05, 1e-6)
    t_min = typical.min()
    t_max = typical.max()
    bins_start = np.floor(t_min / step) * step
    bins_end = np.ceil(t_max / step) * step
    price_bins = np.arange(bins_start, bins_end + step * 0.5, step)
    if len(price_bins) == 0:
        price_bins = np.array([bins_start])

    # 各行の日付（タイムゾーンはそのまま、日付のみ取得）
    dates_in_data = [pd.Timestamp(ts).date() for ts in df.index]
    min_date = min(dates_in_data)
    max_date = max(dates_in_data)
    all_dates = pd.date_range(min_date, max_date, freq="D").date.tolist()
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    Z = np.zeros((len(all_dates), len(price_bins)))
    daily_volume = {d: 0.0 for d in all_dates}
    daily_close = {}  # その日の最後の Close

    for ts, row in df.iterrows():
        d = pd.Timestamp(ts).date()
        if d not in date_to_idx:
            continue
        i = date_to_idx[d]
        vol = row["Volume"]
        if pd.isna(vol):
            vol = 0
        tp = typical.loc[ts]
        if pd.isna(tp):
            continue
        j = np.searchsorted(price_bins, tp) - 1
        if j < 0:
            j = 0
        if j >= len(price_bins):
            j = len(price_bins) - 1
        Z[i, j] += vol
        daily_volume[d] = daily_volume.get(d, 0) + vol
        daily_close[d] = row["Close"]

    date_labels = [str(d) for d in all_dates]
    daily_vol_list = [float(daily_volume[d]) for d in all_dates]
    daily_price_list = [float(daily_close.get(d, np.nan)) for d in all_dates]
    for i, p in enumerate(daily_price_list):
        if np.isnan(p) and i > 0:
            daily_price_list[i] = daily_price_list[i - 1]

    return {
        "date_labels": date_labels,
        "all_dates": date_labels,
        "price_bins": price_bins.tolist(),
        "Z": Z.tolist(),
        "daily_volume": daily_vol_list,
        "daily_price": daily_price_list,
        "last_close": last_close,
        "step": step,
        "n_dates": len(all_dates),
        "n_bins": len(price_bins),
    }

# ---------------------------------------------------------------------------
# 空グラフ
# ---------------------------------------------------------------------------
def _no_data_figure(title: str = "データがありません") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=title,
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(size=14), align="center",
    )
    fig.update_layout(template="plotly_dark", margin=dict(t=40, b=40, l=40, r=40))
    return fig

# ---------------------------------------------------------------------------
# 3D サーフェス
# ---------------------------------------------------------------------------
def create_surface_figure(
    data: dict[str, Any] | None,
    ticker: str,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
) -> go.Figure:
    if data is None or data["n_dates"] == 0 or data["n_bins"] == 0:
        return _no_data_figure("データがありません")
    date_labels = data["date_labels"]
    price_bins = np.array(data["price_bins"])
    Z = np.array(data["Z"])
    y_indices = np.arange(len(date_labels))

    surface = go.Surface(
        x=price_bins,
        y=y_indices,
        z=Z,
        colorscale="Viridis",
        colorbar=dict(title="出来高"),
        showscale=True,
        hovertemplate="株価: %{x:.2f}<br>日付: %{y}<br>出来高: %{z:.0f}<extra></extra>",
    )
    fig = go.Figure(data=[surface])

    # 日付軸: 表示範囲（y_range）に合わせて tickvals / ticktext を設定
    y_min = int(y_range[0]) if y_range is not None else 0
    y_max = int(y_range[1]) if y_range is not None else max(0, len(date_labels) - 1)
    y_min = max(0, min(y_min, len(date_labels) - 1))
    y_max = max(0, min(y_max, len(date_labels) - 1))
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    visible_len = y_max - y_min + 1
    y_step = max(1, visible_len // 10)
    y_tick_indices = list(range(y_min, y_max + 1, y_step))
    if y_tick_indices and y_tick_indices[-1] != y_max:
        y_tick_indices.append(y_max)
    y_tickvals = [i for i in y_tick_indices if i < len(date_labels)]
    y_ticktext = [date_labels[i] for i in y_tickvals]

    layout_scene = dict(
        xaxis_title="X: 株価",
        yaxis_title="Y: 日付",
        zaxis_title="Z: 出来高",
        yaxis=dict(
            tickmode="array",
            tickvals=y_tickvals,
            ticktext=y_ticktext,
            range=[y_min, y_max],
        ),
    )
    if x_range is not None:
        layout_scene["xaxis"] = layout_scene.get("xaxis", {}) or {}
        layout_scene["xaxis"]["range"] = list(x_range)
    if y_range is not None:
        layout_scene["yaxis"] = layout_scene.get("yaxis", {}) or {}
        layout_scene["yaxis"]["range"] = list(y_range)
    if z_range is not None:
        layout_scene["zaxis"] = dict(layout_scene.get("zaxis", {}), range=list(z_range), title="Z: 出来高")

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        scene=layout_scene,
        template="plotly_dark",
        title=f"{ticker} 日付×株価×出来高",
    )
    return fig

# ---------------------------------------------------------------------------
# 2D グラフ（日付-出来高、日付-株価、出来高-株価）
# ---------------------------------------------------------------------------
def create_date_volume_figure(
    data: dict[str, Any] | None,
    hover_date_idx: int | None,
) -> go.Figure:
    if data is None or not data["date_labels"]:
        return _no_data_figure()
    x = list(range(len(data["date_labels"])))
    y = data["daily_volume"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines",
            hovertemplate="日付: %{text}<br>出来高: %{y:,.0f}<extra></extra>",
            text=data["date_labels"],
            showlegend=False,
        )
    )
    if hover_date_idx is not None and 0 <= hover_date_idx < len(x):
        fig.add_vline(x=hover_date_idx, line=dict(color="red", width=1, dash="dash"))
        fig.add_trace(
            go.Scatter(
                x=[hover_date_idx], y=[y[hover_date_idx]],
                mode="markers", marker=dict(color="red", size=10),
                hovertemplate="日付: %{text}<br>出来高: %{y:,.0f}<extra></extra>",
                text=[data["date_labels"][hover_date_idx]],
                showlegend=False,
            )
        )
    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis_title="日付", yaxis_title="出来高",
        template="plotly_dark", title="日付-出来高",
        xaxis=dict(tickmode="array", tickvals=x[:: max(1, len(x) // 8)], ticktext=[data["date_labels"][i] for i in range(0, len(x), max(1, len(x) // 8))]),
    )
    return fig

def create_date_price_figure(
    data: dict[str, Any] | None,
    hover_date_idx: int | None,
) -> go.Figure:
    if data is None or not data["date_labels"]:
        return _no_data_figure()
    x = list(range(len(data["date_labels"])))
    y = data["daily_price"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines",
            hovertemplate="日付: %{text}<br>株価: %{y:.2f}<extra></extra>",
            text=data["date_labels"],
            showlegend=False,
        )
    )
    # 最新株価の緑点線
    if data.get("last_close") is not None:
        fig.add_hline(
            y=data["last_close"],
            line=dict(color="green", width=1, dash="dot"),
            annotation_text="最新株価",
        )
    if hover_date_idx is not None and 0 <= hover_date_idx < len(x):
        fig.add_vline(x=hover_date_idx, line=dict(color="red", width=1, dash="dash"))
        fig.add_trace(
            go.Scatter(
                x=[hover_date_idx], y=[y[hover_date_idx]],
                mode="markers", marker=dict(color="red", size=10),
                hovertemplate="日付: %{text}<br>株価: %{y:.2f}<extra></extra>",
                text=[data["date_labels"][hover_date_idx]],
                showlegend=False,
            )
        )
    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis_title="日付", yaxis_title="株価",
        template="plotly_dark", title="日付-株価",
        xaxis=dict(tickmode="array", tickvals=x[:: max(1, len(x) // 8)], ticktext=[data["date_labels"][i] for i in range(0, len(x), max(1, len(x) // 8))]),
    )
    return fig

def create_volume_price_figure(
    data: dict[str, Any] | None,
    df_day: pd.DataFrame | None,
    hover_date_idx: int | None,
) -> go.Figure:
    if data is None:
        return _no_data_figure()
    if df_day is None or df_day.empty:
        fig = go.Figure()
        fig.add_annotation(text="ホバーした日のデータなし", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", title="出来高-株価")
        return fig
    typical = (df_day["Open"] + df_day["High"] + df_day["Low"] + df_day["Close"]) / 4.0
    vol = df_day["Volume"].fillna(0)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vol, y=typical, mode="markers",
            hovertemplate="出来高: %{x:,.0f}<br>株価: %{y:.2f}<extra></extra>",
            showlegend=False,
        )
    )
    if data.get("last_close") is not None:
        fig.add_hline(
            y=data["last_close"],
            line=dict(color="green", width=1, dash="dot"),
            annotation_text="最新株価",
        )
    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis_title="出来高", yaxis_title="株価",
        template="plotly_dark", title="出来高-株価（ホバーした日）",
    )
    return fig

# ホバーした日付インデックスから、その日の 5分足 DataFrame を返す（CSV 由来の df と all_dates から）
def get_day_df(df: pd.DataFrame | None, date_labels: list, date_idx: int) -> pd.DataFrame | None:
    if df is None or not date_labels or date_idx < 0 or date_idx >= len(date_labels):
        return None
    d_str = date_labels[date_idx]
    idx = pd.to_datetime(df.index)
    mask = idx.strftime("%Y-%m-%d") == d_str
    return df.loc[mask] if mask.any() else None

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
_NO_SELECT = {
    "userSelect": "none",
    "WebkitUserSelect": "none",
    "MozUserSelect": "none",
    "msUserSelect": "none",
}

app = Dash(__name__)

app.layout = html.Div(
    style={"display": "flex", "height": "100vh", "backgroundColor": "#111", **_NO_SELECT},
    children=[
        html.Div(
            id="updating-indicator",
            style={
                "position": "fixed", "bottom": "10px", "right": "10px",
                "backgroundColor": "rgba(0,0,0,0.8)", "color": "#fff", "padding": "6px 12px",
                "borderRadius": "4px", "fontSize": "12px", "zIndex": 9999,
                "display": "none",
            },
            children="更新中…",
        ),
        html.Div(
            style={"flex": "0 0 40%", "display": "flex", "flexDirection": "column", "padding": "10px", "backgroundColor": "#111", **_NO_SELECT},
            children=[
                html.Div(
                    style={"marginBottom": "8px", **_NO_SELECT},
                    children=[
                        html.Label("銘柄コード"),
                        html.Div(
                            style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
                            children=[
                                dcc.Input(id="ticker-input", type="text", placeholder="例: 7203.T", value="7203.T", style={"width": "100px", "color": "#111"}),
                                html.Button("取得", id="btn-fetch", n_clicks=0),
                                html.Button("更新", id="btn-update", n_clicks=0),
                            ],
                        ),
                        html.Div(style={"marginTop": "6px"}, children=[
                            html.Label("お気に入り"),
                            dcc.Dropdown(
                                id="favorites-dropdown",
                                options=[{"label": t, "value": t} for t in load_favorites()],
                                value=None,
                                clearable=True,
                                style={"minWidth": "140px", "color": "#111", "backgroundColor": "#fff"},
                                className="ticker-dropdown-dark-text",
                            ),
                        ]),
                        html.Button("お気に入りに追加", id="btn-add-favorite", n_clicks=0, style={"marginTop": "6px"}),
                        html.Div(id="step-display", style={"marginTop": "6px", "fontSize": "12px", "color": "#ccc"}),
                        html.Div(
                            style={"marginTop": "8px"},
                            children=[
                                html.Label("X軸(日付)範囲"),
                                dcc.RangeSlider(id="range-x", min=0, max=1, value=[0, 1], allowCross=False, tooltip={"placement": "bottom"}),
                            ],
                        ),
                        html.Div(
                            style={"marginTop": "4px"},
                            children=[
                                html.Label("Y軸(株価)範囲"),
                                dcc.RangeSlider(id="range-y", min=0, max=1, value=[0, 1], allowCross=False, tooltip={"placement": "bottom"}),
                            ],
                        ),
                        html.Div(
                            style={"marginTop": "4px"},
                            children=[
                                html.Label("Z軸(出来高)範囲"),
                                dcc.RangeSlider(id="range-z", min=0, max=1, value=[0, 1], allowCross=False, tooltip={"placement": "bottom"}),
                            ],
                        ),
                        html.Button("全体表示に戻す", id="btn-reset-range", n_clicks=0, style={"marginTop": "4px"}),
                    ],
                ),
                dcc.Graph(id="graph-date-volume", style={"flex": "1 1 33%", "minHeight": "120px", **_NO_SELECT}),
                dcc.Graph(id="graph-date-price", style={"flex": "1 1 33%", "minHeight": "120px", **_NO_SELECT}),
                dcc.Graph(id="graph-volume-price", style={"flex": "1 1 33%", "minHeight": "120px", **_NO_SELECT}),
            ],
        ),
        html.Div(
            style={"flex": "1 1 60%", "padding": "10px", **_NO_SELECT},
            children=[dcc.Graph(id="surface-graph", style={"height": "100%", **_NO_SELECT})],
        ),
        dcc.Store(id="store-ticker", data=None),
        dcc.Store(id="store-display-data", data=None),
    ],
)

@app.callback(
    Output("store-ticker", "data"),
    Output("store-display-data", "data"),
    Output("step-display", "children"),
    Output("favorites-dropdown", "options"),
    Input("btn-fetch", "n_clicks"),
    Input("favorites-dropdown", "value"),
    State("ticker-input", "value"),
    State("store-ticker", "data"),
    prevent_initial_call=False,
)
def on_fetch_or_select(n_fetch, fav_value, input_ticker, current_ticker):
    from dash import ctx
    ticker = None
    if ctx.triggered_id == "btn-fetch" and input_ticker:
        ticker = (input_ticker or "").strip().upper()
    elif ctx.triggered_id == "favorites-dropdown" and fav_value:
        ticker = fav_value
    elif ctx.triggered_id is None and not current_ticker:
        # 起動時: 7203.T のCSVがあれば読んで表示
        ticker = "7203.T"

    if not ticker:
        if current_ticker:
            return current_ticker, None, "", [{"label": t, "value": t} for t in load_favorites()]
        return None, None, "", [{"label": t, "value": t} for t in load_favorites()]

    # 取得ボタン: CSV が無ければ全量取得
    if ctx.triggered_id == "btn-fetch":
        existing = load_csv(ticker)
        if existing is None or existing.empty:
            df = fetch_5m_full(ticker)
            if df is None or df.empty:
                return current_ticker, None, "取得できませんでした", [{"label": t, "value": t} for t in load_favorites()]
            save_csv(ticker, df)
            existing = load_csv(ticker)
        data = compute_display_data(existing)
        if data is None:
            return ticker, None, "データなし", [{"label": t, "value": t} for t in load_favorites()]
        _DISPLAY_CACHE[ticker] = data
        return ticker, {"ticker": ticker}, f"価格刻み: {data['step']:.2f}（直近終値 {data['last_close']:.2f} の5%）", [{"label": t, "value": t} for t in load_favorites()]

    # お気に入り選択 or 起動時: 既存CSVのみ読む
    df = load_csv(ticker)
    if df is None or df.empty:
        return ticker, None, "データがありません。取得または更新してください。", [{"label": t, "value": t} for t in load_favorites()]
    data = compute_display_data(df)
    if data is None:
        return ticker, None, "", [{"label": t, "value": t} for t in load_favorites()]
    _DISPLAY_CACHE[ticker] = data
    return ticker, {"ticker": ticker}, f"価格刻み: {data['step']:.2f}（直近終値 {data['last_close']:.2f} の5%）", [{"label": t, "value": t} for t in load_favorites()]


def _get_cached_data(ticker: str | None) -> dict[str, Any] | None:
    if not ticker:
        return None
    return _DISPLAY_CACHE.get(ticker)


@app.callback(
    Output("store-ticker", "data", allow_duplicate=True),
    Output("store-display-data", "data", allow_duplicate=True),
    Output("step-display", "children", allow_duplicate=True),
    Input("btn-update", "n_clicks"),
    State("store-ticker", "data"),
    prevent_initial_call=True,
)
def on_update(n, current_ticker):
    if not current_ticker:
        return current_ticker, None, ""
    existing = load_csv(current_ticker)
    if existing is None or existing.empty:
        df = fetch_5m_full(current_ticker)
        if df is None or df.empty:
            return current_ticker, None, "更新できませんでした"
        save_csv(current_ticker, df)
    else:
        last_ts = existing.index[-1]
        new_df = fetch_5m_incremental(current_ticker, last_ts)
        if new_df is not None and not new_df.empty:
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            save_csv(current_ticker, combined)
            existing = combined
        df = existing
    data = compute_display_data(df)
    if data is None:
        return current_ticker, None, ""
    _DISPLAY_CACHE[current_ticker] = data
    return current_ticker, {"ticker": current_ticker}, f"価格刻み: {data['step']:.2f}（直近終値 {data['last_close']:.2f} の5%）"

@app.callback(
    Output("favorites-dropdown", "options", allow_duplicate=True),
    Input("btn-add-favorite", "n_clicks"),
    State("store-ticker", "data"),
    prevent_initial_call=True,
)
def on_add_favorite(n, ticker):
    if ticker:
        add_favorite(ticker)
    return [{"label": t, "value": t} for t in load_favorites()]

# 軸範囲スライダー初期化（データ変更時）
@app.callback(
    Output("range-x", "min"), Output("range-x", "max"), Output("range-x", "value"),
    Output("range-y", "min"), Output("range-y", "max"), Output("range-y", "value"),
    Output("range-z", "min"), Output("range-z", "max"), Output("range-z", "value"),
    Input("store-display-data", "data"),
    State("store-ticker", "data"),
)
def init_axis_sliders(_display_data, ticker):
    data = _get_cached_data(ticker)
    if data is None or data.get("n_dates", 0) == 0:
        return 0, 1, [0, 1], 0, 1, [0, 1], 0, 1, [0, 1]
    n_d = data["n_dates"]
    pb = data["price_bins"]
    z_arr = np.array(data["Z"])
    z_min = float(np.nanmin(z_arr)) if z_arr.size else 0.0
    z_max = float(np.nanmax(z_arr)) if z_arr.size else 1.0
    return (
        0, max(0, n_d - 1), [0, max(0, n_d - 1)],
        0, max(0, len(pb) - 1), [0, max(0, len(pb) - 1)],
        0, 1, [0, 1],
    )

# 全体表示に戻す
@app.callback(
    Output("range-x", "value", allow_duplicate=True),
    Output("range-y", "value", allow_duplicate=True),
    Output("range-z", "value", allow_duplicate=True),
    Input("btn-reset-range", "n_clicks"),
    State("range-x", "max"), State("range-y", "max"), State("range-z", "max"),
    prevent_initial_call=True,
)
def on_reset_range(n, x_max, y_max, z_max):
    return [0, max(0, x_max)], [0, max(0, y_max)], [0, 1]

# 3D サーフェス更新（store + 軸範囲）
@app.callback(
    Output("surface-graph", "figure"),
    Input("store-display-data", "data"),
    Input("store-ticker", "data"),
    Input("range-x", "value"),
    Input("range-y", "value"),
    Input("range-z", "value"),
)
def update_surface(_display_data, ticker, rx, ry, rz):
    data = _get_cached_data(ticker)
    if data is None or not ticker:
        return _no_data_figure()
    pb = data["price_bins"]
    nb = len(pb)
    j0 = max(0, min(int(ry[0]), nb - 1)) if ry else 0
    j1 = max(0, min(int(ry[1]), nb - 1)) if ry else max(0, nb - 1)
    x_range = (float(pb[j0]), float(pb[j1])) if nb else None
    y_range = (int(rx[0]), int(rx[1])) if rx else None
    Z_arr = np.array(data["Z"])
    z_max = float(np.nanmax(Z_arr)) if Z_arr.size else 1.0
    z_min = float(np.nanmin(Z_arr)) if Z_arr.size else 0.0
    if rz and len(rz) == 2:
        z_range = (z_min + (z_max - z_min) * rz[0], z_min + (z_max - z_min) * rz[1])
    else:
        z_range = (z_min, z_max)
    return create_surface_figure(data, ticker, x_range=x_range, y_range=y_range, z_range=z_range)

# 2D グラフ更新（store + 3D hoverData）
@app.callback(
    Output("graph-date-volume", "figure"),
    Output("graph-date-price", "figure"),
    Output("graph-volume-price", "figure"),
    Input("store-display-data", "data"),
    Input("store-ticker", "data"),
    Input("surface-graph", "hoverData"),
)
def update_2d_graphs(_display_data, ticker, hover_data):
    data = _get_cached_data(ticker)
    hover_idx = None
    if data and hover_data and hover_data.get("points"):
        pt = hover_data["points"][0]
        y_val = pt.get("y")
        if isinstance(y_val, (int, float)):
            hover_idx = int(round(y_val))
        pi = pt.get("pointIndex")
        if isinstance(pi, (list, tuple)) and len(pi) >= 1:
            hover_idx = int(pi[0])
        if hover_idx is not None and data.get("n_dates"):
            hover_idx = max(0, min(hover_idx, data["n_dates"] - 1))

    df = load_csv(ticker) if ticker else None
    day_df = get_day_df(df, data["date_labels"] if data else [], hover_idx) if data and hover_idx is not None else None
    f1 = create_date_volume_figure(data, hover_idx)
    f2 = create_date_price_figure(data, hover_idx)
    f3 = create_volume_price_figure(data, day_df, hover_idx)
    return f1, f2, f3

if __name__ == "__main__":
    app.run(debug=True)
