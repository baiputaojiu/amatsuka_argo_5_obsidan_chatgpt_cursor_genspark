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
    step = max(last_close * 0.01, 1e-6)
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
    y_range: tuple[int, int] | None = None,
    z_range: tuple[float, float] | None = None,
) -> go.Figure:
    if data is None or data["n_dates"] == 0 or data["n_bins"] == 0:
        return _no_data_figure("データがありません")
    date_labels = data["date_labels"]
    price_bins = np.array(data["price_bins"])
    Z = np.array(data["Z"])

    # 範囲でスライス（描画データを絞るので確実に反映される）
    i0 = int(y_range[0]) if y_range is not None else 0
    i1 = int(y_range[1]) if y_range is not None else len(date_labels) - 1
    i0 = max(0, min(i0, len(date_labels) - 1))
    i1 = max(0, min(i1, len(date_labels) - 1))
    if i0 > i1:
        i0, i1 = i1, i0
    date_slice = date_labels[i0 : i1 + 1]
    Z_date = Z[i0 : i1 + 1, :]

    if x_range is not None:
        p_min, p_max = float(x_range[0]), float(x_range[1])
        if p_min > p_max:
            p_min, p_max = p_max, p_min
        j_mask = (price_bins >= p_min) & (price_bins <= p_max)
        if not np.any(j_mask):
            j_mask = np.ones(len(price_bins), dtype=bool)
        price_slice = price_bins[j_mask]
        Z_slice = Z_date[:, j_mask]
    else:
        price_slice = price_bins
        Z_slice = Z_date

    if len(date_slice) == 0 or len(price_slice) == 0:
        return _no_data_figure("データがありません")
    y_indices = np.arange(len(date_slice))

    # Z軸範囲でクリップ（表示用）
    if z_range is not None:
        z_lo, z_hi = float(z_range[0]), float(z_range[1])
        Z_slice = np.clip(Z_slice, z_lo, z_hi)

    surface = go.Surface(
        x=price_slice,
        y=y_indices,
        z=Z_slice,
        colorscale="Viridis",
        colorbar=dict(title="出来高"),
        showscale=True,
        hovertemplate="株価: %{x:.2f}<br>日付: %{y}<br>出来高: %{z:.0f}<extra></extra>",
    )
    traces: list = [surface]

    # 現在価格の位置に半透明の壁（表示範囲内の場合のみ）
    last_close = data.get("last_close")
    p_lo, p_hi = float(np.min(price_slice)), float(np.max(price_slice))
    if last_close is not None and p_lo <= last_close <= p_hi:
        y_max = len(date_slice) - 1
        if z_range is not None:
            z_lo, z_hi = float(z_range[0]), float(z_range[1])
            z_max = z_hi * 3
        else:
            z_max = float(np.nanmax(Z_slice)) * 1.1 if Z_slice.size else 1.0
        wall_x = [[last_close, last_close], [last_close, last_close]]
        wall_y = [[0, 0], [y_max, y_max]]
        wall_z = [[0, z_max], [0, z_max]]
        wall = go.Surface(
            x=wall_x,
            y=wall_y,
            z=wall_z,
            surfacecolor=[[1, 1], [1, 1]],
            cmin=0,
            cmax=1,
            colorscale=[[0, "rgba(0,255,200,0.5)"], [1, "rgba(0,255,200,0.5)"]],
            showscale=False,
            opacity=0.4,
            hovertemplate="現在価格: %{x:.2f}<extra></extra>",
        )
        traces.append(wall)

    fig = go.Figure(data=traces)

    y_step = max(1, len(date_slice) // 10)
    y_tick_indices = list(range(0, len(date_slice), y_step))
    if y_tick_indices and y_tick_indices[-1] != len(date_slice) - 1:
        y_tick_indices.append(len(date_slice) - 1)
    y_tickvals = y_tick_indices
    y_ticktext = [date_slice[i] for i in y_tickvals]

    layout_scene = dict(
        xaxis_title="X: 株価",
        yaxis_title="Y: 日付",
        zaxis_title="Z: 出来高",
        xaxis=dict(autorange="reversed", tickmode="auto"),
        yaxis=dict(
            tickmode="array",
            tickvals=y_tickvals,
            ticktext=y_ticktext,
        ),
    )
    if z_range is not None:
        z_lo, z_hi = float(z_range[0]), float(z_range[1])
        # 表示上限を3倍に広げて山を1/3の高さで表示
        layout_scene["zaxis"] = dict(range=[z_lo, z_hi * 3], title="Z: 出来高")

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
    date_range: tuple[int, int] | None = None,
) -> go.Figure:
    if data is None or not data["date_labels"]:
        return _no_data_figure()
    labels = data["date_labels"]
    vol = data["daily_volume"]
    if date_range is not None:
        i0, i1 = max(0, date_range[0]), min(len(labels) - 1, date_range[1])
        if i0 > i1:
            i0, i1 = i1, i0
        labels = labels[i0 : i1 + 1]
        vol = vol[i0 : i1 + 1]
        if hover_date_idx is not None and i0 <= hover_date_idx <= i1:
            hover_date_idx = hover_date_idx - i0
        else:
            hover_date_idx = None
    x = list(range(len(labels)))
    y = vol
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
                text=[labels[hover_date_idx]],
                showlegend=False,
            )
        )
    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis_title="日付", yaxis_title="出来高",
        template="plotly_dark", title="日付-出来高",
        xaxis=dict(tickmode="array", tickvals=x[:: max(1, len(x) // 8)], ticktext=[labels[i] for i in range(0, len(x), max(1, len(x) // 8))]),
    )
    return fig

def create_date_price_figure(
    data: dict[str, Any] | None,
    hover_date_idx: int | None,
    date_range: tuple[int, int] | None = None,
) -> go.Figure:
    if data is None or not data["date_labels"]:
        return _no_data_figure()
    labels = data["date_labels"]
    price = data["daily_price"]
    if date_range is not None:
        i0, i1 = max(0, date_range[0]), min(len(labels) - 1, date_range[1])
        if i0 > i1:
            i0, i1 = i1, i0
        labels = labels[i0 : i1 + 1]
        price = price[i0 : i1 + 1]
        if hover_date_idx is not None and i0 <= hover_date_idx <= i1:
            hover_date_idx = hover_date_idx - i0
        else:
            hover_date_idx = None
    x = list(range(len(labels)))
    y = price
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
                text=[labels[hover_date_idx]],
                showlegend=False,
            )
        )
    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis_title="日付", yaxis_title="株価",
        template="plotly_dark", title="日付-株価",
        xaxis=dict(tickmode="array", tickvals=x[:: max(1, len(x) // 8)], ticktext=[labels[i] for i in range(0, len(x), max(1, len(x) // 8))]),
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
                                dcc.DatePickerRange(
                                    id="range-x-date",
                                    start_date=None,
                                    end_date=None,
                                    display_format="YYYY-MM-DD",
                                    style={"color": "#111"},
                                ),
                            ],
                        ),
                        html.Div(
                            style={"marginTop": "4px"},
                            children=[
                                html.Label("Y軸(株価)範囲"),
                                html.Div(
                                    style={"display": "flex", "gap": "8px", "alignItems": "center"},
                                    children=[
                                        dcc.Input(id="range-y-min", type="number", placeholder="最小", style={"width": "80px", "color": "#111"}),
                                        dcc.Input(id="range-y-max", type="number", placeholder="最大", style={"width": "80px", "color": "#111"}),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            style={"marginTop": "4px"},
                            children=[
                                html.Label("Z軸(出来高)範囲"),
                                html.Div(
                                    style={"display": "flex", "gap": "8px", "alignItems": "center"},
                                    children=[
                                        dcc.Input(id="range-z-min", type="number", placeholder="最小", style={"width": "80px", "color": "#111"}),
                                        dcc.Input(id="range-z-max", type="number", placeholder="最大", style={"width": "80px", "color": "#111"}),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            style={"display": "flex", "gap": "8px", "marginTop": "4px"},
                            children=[
                                html.Button("範囲を適用", id="btn-apply-range", n_clicks=0),
                                html.Button("全体表示に戻す", id="btn-reset-range", n_clicks=0),
                            ],
                        ),
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
        dcc.Store(id="store-current-range", data=None),
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
        return ticker, {"ticker": ticker}, f"価格刻み: {data['step']:.2f}（直近終値 {data['last_close']:.2f} の1%）", [{"label": t, "value": t} for t in load_favorites()]

    # お気に入り選択 or 起動時: 既存CSVのみ読む
    df = load_csv(ticker)
    if df is None or df.empty:
        return ticker, None, "データがありません。取得または更新してください。", [{"label": t, "value": t} for t in load_favorites()]
    data = compute_display_data(df)
    if data is None:
        return ticker, None, "", [{"label": t, "value": t} for t in load_favorites()]
    _DISPLAY_CACHE[ticker] = data
    return ticker, {"ticker": ticker}, f"価格刻み: {data['step']:.2f}（直近終値 {data['last_close']:.2f} の1%）", [{"label": t, "value": t} for t in load_favorites()]


def _get_cached_data(ticker: str | None) -> dict[str, Any] | None:
    if not ticker:
        return None
    return _DISPLAY_CACHE.get(ticker)


def _save_graph_data_csv(ticker: str, date_slice: list, price_slice: np.ndarray, Z_slice: np.ndarray) -> None:
    """描画用Z行列を data/{ticker}_graph_data.csv に保存する。"""
    if not date_slice or price_slice is None or Z_slice is None or len(Z_slice) == 0:
        return
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in ticker)
    path = DATA_DIR / f"{safe}_graph_data.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(Z_slice, index=date_slice, columns=[f"{p:.2f}" for p in price_slice])
    df.index.name = "日付"
    df.to_csv(path, encoding="utf-8-sig")


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
    return current_ticker, {"ticker": current_ticker}, f"価格刻み: {data['step']:.2f}（直近終値 {data['last_close']:.2f} の1%）"

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

# 軸範囲入力の初期化（データ変更時）。X初期範囲は山脈が0でない範囲
@app.callback(
    Output("range-x-date", "start_date"),
    Output("range-x-date", "end_date"),
    Output("range-x-date", "min_date_allowed"),
    Output("range-x-date", "max_date_allowed"),
    Output("range-y-min", "value"),
    Output("range-y-max", "value"),
    Output("range-z-min", "value"),
    Output("range-z-max", "value"),
    Output("store-current-range", "data"),
    Input("store-display-data", "data"),
    State("store-ticker", "data"),
)
def init_axis_inputs(_display_data, ticker):
    data = _get_cached_data(ticker)
    if data is None or data.get("n_dates", 0) == 0:
        return None, None, None, None, None, None, None, None, None
    date_labels = data["date_labels"]
    Z_arr = np.array(data["Z"])
    row_sums = np.sum(Z_arr, axis=1)
    nonzero = np.where(row_sums > 0)[0]
    if len(nonzero) == 0:
        first_date = date_labels[0]
        last_date = date_labels[-1]
        i0, i1 = 0, len(date_labels) - 1
    else:
        i0, i1 = int(nonzero[0]), int(nonzero[-1])
        first_date = date_labels[i0]
        last_date = date_labels[i1]
    pb = data["price_bins"]
    price_min = float(min(pb)) if pb else None
    price_max = float(max(pb)) if pb else None
    z_min = float(np.nanmin(Z_arr)) if Z_arr.size else None
    z_max = float(np.nanmax(Z_arr)) if Z_arr.size else None
    range_data = {"y_range": [i0, i1], "x_range": [price_min, price_max], "z_range": [z_min, z_max]}
    return (
        first_date, last_date, date_labels[0], date_labels[-1],
        price_min, price_max, z_min, z_max,
        range_data,
    )

# 全体表示に戻す
@app.callback(
    Output("range-x-date", "start_date", allow_duplicate=True),
    Output("range-x-date", "end_date", allow_duplicate=True),
    Output("range-y-min", "value", allow_duplicate=True),
    Output("range-y-max", "value", allow_duplicate=True),
    Output("range-z-min", "value", allow_duplicate=True),
    Output("range-z-max", "value", allow_duplicate=True),
    Input("btn-reset-range", "n_clicks"),
    State("store-ticker", "data"),
    prevent_initial_call=True,
)
def on_reset_range(n, ticker):
    data = _get_cached_data(ticker)
    if data is None or data.get("n_dates", 0) == 0:
        return None, None, None, None, None, None
    date_labels = data["date_labels"]
    pb = data["price_bins"]
    z_arr = np.array(data["Z"])
    z_min = float(np.nanmin(z_arr)) if z_arr.size else None
    z_max = float(np.nanmax(z_arr)) if z_arr.size else None
    return (
        date_labels[0], date_labels[-1],
        float(min(pb)) if pb else None, float(max(pb)) if pb else None,
        z_min, z_max,
    )

# 3D サーフェス更新＋現在範囲を store に保存（「範囲を適用」クリック or 銘柄変更時）
@app.callback(
    Output("surface-graph", "figure"),
    Output("store-current-range", "data", allow_duplicate=True),
    Input("btn-apply-range", "n_clicks"),
    Input("store-display-data", "data"),
    Input("store-ticker", "data"),
    State("range-x-date", "start_date"),
    State("range-x-date", "end_date"),
    State("range-y-min", "value"),
    State("range-y-max", "value"),
    State("range-z-min", "value"),
    State("range-z-max", "value"),
    prevent_initial_call="initial_duplicate",
)
def update_surface(_apply_clicks, _display_data, ticker, start_date, end_date, y_min_val, y_max_val, z_min_val, z_max_val):
    data = _get_cached_data(ticker)
    if data is None or not ticker:
        return _no_data_figure(), None
    date_labels = data["date_labels"]
    pb = data["price_bins"]
    Z_arr = np.array(data["Z"])
    data_z_min = float(np.nanmin(Z_arr)) if Z_arr.size else 0.0
    data_z_max = float(np.nanmax(Z_arr)) if Z_arr.size else 1.0

    # X軸(日付): カレンダーで選択した日付をインデックスに変換
    y_range = (0, len(date_labels) - 1)
    if start_date is not None and end_date is not None:
        try:
            start_s = (start_date[:10] if isinstance(start_date, str) else str(start_date)[:10]).strip()
            end_s = (end_date[:10] if isinstance(end_date, str) else str(end_date)[:10]).strip()
            if start_s in date_labels and end_s in date_labels:
                i0 = date_labels.index(start_s)
                i1 = date_labels.index(end_s)
                y_range = (min(i0, i1), max(i0, i1))
            elif start_s in date_labels:
                i0 = date_labels.index(start_s)
                y_range = (i0, len(date_labels) - 1)
            elif end_s in date_labels:
                i1 = date_labels.index(end_s)
                y_range = (0, i1)
        except (ValueError, TypeError, AttributeError):
            pass

    # Y軸(株価): 数値入力（片方だけの場合はデータ範囲で補う）
    x_range = None
    p_data_min = float(min(pb)) if pb else 0.0
    p_data_max = float(max(pb)) if pb else 0.0
    if pb is not None and len(pb) > 0:
        p_min = float(y_min_val) if y_min_val is not None and str(y_min_val).strip() != "" else p_data_min
        p_max = float(y_max_val) if y_max_val is not None and str(y_max_val).strip() != "" else p_data_max
        x_range = (min(p_min, p_max), max(p_min, p_max))

    # Z軸(出来高): 数値入力（片方だけの場合はデータ範囲で補う）
    z_min_in = float(z_min_val) if z_min_val is not None and str(z_min_val).strip() != "" else data_z_min
    z_max_in = float(z_max_val) if z_max_val is not None and str(z_max_val).strip() != "" else data_z_max
    z_range = (min(z_min_in, z_max_in), max(z_min_in, z_max_in))

    range_data = {"y_range": list(y_range), "x_range": list(x_range) if x_range else [p_data_min, p_data_max], "z_range": list(z_range)}
    date_slice, price_slice, Z_slice, _, _ = _sliced_display_data(data, range_data)
    if date_slice is not None and price_slice is not None and Z_slice is not None:
        _save_graph_data_csv(ticker, date_slice, price_slice, Z_slice)
    return create_surface_figure(data, ticker, x_range=x_range, y_range=y_range, z_range=z_range), range_data

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


def _sliced_display_data(data: dict[str, Any], range_data: dict | None):
    """store-current-range に従って日付・株価でスライスしたデータを返す。"""
    if not data or not data.get("date_labels"):
        return None, None, None, None, None
    date_labels = data["date_labels"]
    price_bins = np.array(data["price_bins"])
    Z = np.array(data["Z"])
    daily_volume = data.get("daily_volume", [])
    daily_price = data.get("daily_price", [])

    i0, i1 = 0, len(date_labels) - 1
    j0, j1 = 0, len(price_bins) - 1
    if range_data:
        yr = range_data.get("y_range")
        if yr and len(yr) >= 2:
            i0 = max(0, min(int(yr[0]), len(date_labels) - 1))
            i1 = max(0, min(int(yr[1]), len(date_labels) - 1))
            if i0 > i1:
                i0, i1 = i1, i0
        xr = range_data.get("x_range")
        if xr and len(xr) >= 2 and len(price_bins) > 0:
            p_min, p_max = float(xr[0]), float(xr[1])
            j_mask = (price_bins >= p_min) & (price_bins <= p_max)
            if np.any(j_mask):
                j_idx = np.where(j_mask)[0]
                j0, j1 = int(j_idx[0]), int(j_idx[-1])

    date_slice = date_labels[i0 : i1 + 1]
    price_slice = price_bins[j0 : j1 + 1]
    Z_slice = Z[i0 : i1 + 1, j0 : j1 + 1]
    vol_slice = daily_volume[i0 : i1 + 1] if daily_volume else []
    price_slice_daily = daily_price[i0 : i1 + 1] if daily_price else []
    return date_slice, price_slice, Z_slice, vol_slice, price_slice_daily


if __name__ == "__main__":
    app.run(debug=True)
