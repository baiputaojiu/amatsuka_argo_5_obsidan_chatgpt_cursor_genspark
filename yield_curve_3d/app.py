import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

# #region agent log
DEBUG_LOG = Path(__file__).resolve().parent.parent / ".cursor" / "debug.log"
def _dlog(location: str, message: str, data: dict, hypothesis_id: str):
    try:
        payload = {"sessionId": "debug-session", "runId": "run1", "hypothesisId": hypothesis_id, "location": location, "message": message, "data": data, "timestamp": __import__("time").time() * 1000}
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# #endregion
import pandas as pd
from dash import Dash, callback_context, dcc, html, Input, Output, State
import plotly.graph_objs as go


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


def _empty_dataset(country_key: str, display_name: str) -> Dict[str, Any]:
    """CSV が無い場合の空データセット。"""
    return {
        "country": country_key,
        "display_name": display_name,
        "dates": [],
        "date_values": pd.Series(dtype="datetime64[ns]"),
        "maturity_years": np.array([], dtype=float),
        "maturity_labels": [],
        "z": np.zeros((0, 0), dtype=float),
        "ts_col_index": 0,
    }


def _load_yield_curve(
    country_key: str,
    display_name: str,
    filename: str,
    z_label: str = "利回り (%)",
    chart_suffix: str = "イールドカーブ",
) -> Dict[str, Any]:
    """
    統一形式のCSVを読み込む。
    形式: 1行目ヘッダー（Date + 残存期間を数値年で表した列名）、値は YYYY-MM-DD と数値。
    z_label: Z軸・Y軸のラベル（利回り%/先物価格等）
    chart_suffix: グラフタイトル用（イールドカーブ/フォワードカーブ）
    """
    try:
        df = pd.read_csv(DATA_DIR / filename)
    except FileNotFoundError:
        return _empty_dataset(country_key, display_name)
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    if len(df) == 0:
        return _empty_dataset(country_key, display_name)
    maturity_cols = [c for c in df.columns if c != "Date"]
    valid_maturity = []
    for c in maturity_cols:
        try:
            float(c)
            valid_maturity.append(c)
        except (ValueError, TypeError):
            pass
    maturity_cols = valid_maturity
    if not maturity_cols:
        return _empty_dataset(country_key, display_name)
    for c in maturity_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    maturity_years = np.array([float(c) for c in maturity_cols], dtype=float)
    maturity_labels = [str(c) for c in maturity_cols]
    date_values = pd.to_datetime(df["Date"])
    date_labels = date_values.dt.strftime("%Y-%m-%d").tolist()
    z = df[maturity_cols].to_numpy(dtype=float)
    ts_col_index = int(np.argmin(np.abs(maturity_years - 10.0)))
    if len(maturity_years) > 0 and maturity_years.max() < 2:
        ts_col_index = int(np.argmin(np.abs(maturity_years - 0.25)))
    return {
        "country": country_key,
        "display_name": display_name,
        "dates": date_labels,
        "date_values": date_values,
        "maturity_years": maturity_years,
        "maturity_labels": maturity_labels,
        "z": z,
        "ts_col_index": ts_col_index,
        "z_label": z_label,
        "chart_suffix": chart_suffix,
    }


# 全国とも日本に合わせて 3D 日付軸は古い→新しい（y_min→y_max）。反転なし。
DATASETS: Dict[str, Dict[str, Any]] = {
    "japan": _load_yield_curve("japan", "日本", "japan_yield_curve.csv"),
    "usa": _load_yield_curve("usa", "米国", "usa_yield_curve.csv"),
    "uk": _load_yield_curve("uk", "英国", "uk_yield_curve.csv"),
    "euro": _load_yield_curve("euro", "ユーロ圏", "euro_yield_curve.csv"),
    "china": _load_yield_curve("china", "中国", "china_yield_curve.csv"),
    "india": _load_yield_curve("india", "インド", "india_yield_curve.csv"),
    "gold": _load_yield_curve(
        "gold", "ゴールド先物", "gold_forward_curve.csv",
        z_label="先物価格 (USD/oz)", chart_suffix="フォワードカーブ",
    ),
    "silver": _load_yield_curve(
        "silver", "シルバー先物", "silver_forward_curve.csv",
        z_label="先物価格 (USD/oz)", chart_suffix="フォワードカーブ",
    ),
}


def _no_data_figure(title: str = "データがありません") -> go.Figure:
    """データが無い場合に表示する空の Figure（注釈のみ）。"""
    fig = go.Figure()
    fig.add_annotation(
        text=f"{title}<br>READMEの手順でCSVを用意してください。",
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(size=14), align="center",
    )
    fig.update_layout(template="plotly_dark", margin=dict(t=40, b=40, l=40, r=40))
    return fig


def _build_wireframe_traces(
    maturity_years: np.ndarray,
    y_indices: np.ndarray,
    z: np.ndarray,
    z_label: str,
) -> list:
    """ワイヤーフレーム用の Scatter3d トレースを生成。有効なzの点同士のみ線で接続。"""
    valid = ~np.isnan(z) & np.isfinite(z)
    x_list, y_list, z_list = [], [], []
    # 水平線（満期方向）: 同一日付で隣接満期を結ぶ
    for j in range(z.shape[0]):
        for i in range(z.shape[1] - 1):
            if valid[j, i] and valid[j, i + 1]:
                x_list.extend([maturity_years[i], maturity_years[i + 1], None])
                y_list.extend([y_indices[j], y_indices[j], None])
                z_list.extend([z[j, i], z[j, i + 1], None])
    # 垂直線（日付方向）: 同一満期で隣接日付を結ぶ
    for i in range(z.shape[1]):
        for j in range(z.shape[0] - 1):
            if valid[j, i] and valid[j + 1, i]:
                x_list.extend([maturity_years[i], maturity_years[i], None])
                y_list.extend([y_indices[j], y_indices[j + 1], None])
                z_list.extend([z[j, i], z[j + 1, i], None])
    if not x_list:
        return []
    line_trace = go.Scatter3d(
        x=x_list,
        y=y_list,
        z=z_list,
        mode="lines",
        line=dict(color="rgba(100,150,255,0.8)", width=1),
        hoverinfo="skip",
    )
    # 有効な点をマーカーで表示
    yy, xx = np.where(valid)
    z_vals = z[yy, xx]
    point_trace = go.Scatter3d(
        x=maturity_years[xx],
        y=y_indices[yy],
        z=z_vals,
        mode="markers",
        marker=dict(size=2, color=z_vals, colorscale="RdBu", reversescale=True),
        text=[f"残存: {maturity_years[xi]:.2f}年<br>日付idx: {y_indices[yi]}<br>{z_label}: {z[yi, xi]:.3f}" for yi, xi in zip(yy, xx)],
        hovertemplate="%{text}<extra></extra>",
    )
    return [line_trace, point_trace]


def create_surface_figure(
    country_key: str, y_start=None, y_end=None, view_mode: str = "surface"
) -> go.Figure:
    data = DATASETS[country_key]
    dates = data["dates"]
    maturity_years = data["maturity_years"]
    z_full = data["z"]

    if len(dates) == 0:
        return _no_data_figure(f"{data['display_name']}のデータがありません")

    y_indices_full = np.arange(len(dates))
    # 日付範囲指定があればスライス（表示パフォーマンスのため）
    if y_start is not None and y_end is not None:
        y_min = max(0, int(y_start))
        y_max = min(len(dates) - 1, int(y_end))
    else:
        y_min = 0
        y_max = len(dates) - 1
    y_indices = y_indices_full[y_min : y_max + 1]
    z = z_full[y_min : y_max + 1, :]

    z_label = data.get("z_label", "利回り (%)")

    if view_mode == "wireframe":
        traces = _build_wireframe_traces(maturity_years, y_indices, z, z_label)
        if not traces:
            return _no_data_figure("表示できるデータがありません")
        fig = go.Figure(data=traces)
        # ワイヤーフレームは point に colorscale があるので colorbar を point に付与
        fig.update_traces(
            marker_showscale=True,
            selector=dict(mode="markers"),
        )
        fig.update_traces(
            marker_colorbar=dict(title=z_label),
            selector=dict(mode="markers"),
        )
    else:
        surface = go.Surface(
            x=maturity_years,
            y=y_indices,
            z=z,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title=z_label),
            showscale=True,
        )
        fig = go.Figure(data=[surface])

    visible_indices = np.arange(y_min, y_max + 1) if y_max >= y_min else np.array([y_min])
    y_step = max(1, len(visible_indices) // 10)
    tick_indices = visible_indices[::y_step]

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="残存期間 (年)",
            xaxis=dict(autorange="reversed"),
            yaxis_title="日付",
            zaxis_title=z_label,
            yaxis=dict(
                tickmode="array",
                tickvals=tick_indices,
                ticktext=[dates[int(i)] for i in tick_indices],
                range=[y_min, y_max],
            ),
        ),
        template="plotly_dark",
        title=f"{data['display_name']} {data.get('chart_suffix', 'イールドカーブ')}（3D）",
    )
    return fig


def create_curve_figure(country_key: str, row_index: int, col_index: int) -> go.Figure:
    data = DATASETS[country_key]
    dates = data["dates"]
    maturity_years = data["maturity_years"]
    maturity_labels = data["maturity_labels"]
    z = data["z"]

    if len(dates) == 0:
        return _no_data_figure(f"{data['display_name']}のデータがありません")
    row_index = max(0, min(row_index, len(dates) - 1))
    col_index = max(0, min(col_index, len(maturity_years) - 1))

    y_values = z[row_index, :]
    date_label = dates[row_index]
    maturity_label = maturity_labels[col_index]
    maturity_x = maturity_years[col_index]
    current_y = y_values[col_index]
    z_label = data.get("z_label", "利回り (%)")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=maturity_years,
            y=y_values,
            mode="lines+markers",
            text=maturity_labels,
            hovertemplate=f"残存期間: %{{text}}<br>{z_label}: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        )
    )

    # ホバーしている残存期間を強調表示（縦線 + 赤丸）
    fig.add_vline(
        x=maturity_x,
        line=dict(color="red", width=1, dash="dash"),
    )
    fig.add_trace(
        go.Scatter(
            x=[maturity_x],
            y=[current_y],
            mode="markers",
            marker=dict(color="red", size=10),
            showlegend=False,
            hovertext=[maturity_label],
            hovertemplate=f"残存期間: %{{hovertext}}<br>{z_label}: %{{y:.3f}}<extra></extra>",
        )
    )

    chart_suffix = data.get("chart_suffix", "イールドカーブ")
    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis_title="残存期間 (年)",
        yaxis_title=z_label,
        template="plotly_dark",
        showlegend=False,
        title=f"{date_label} の{chart_suffix}断面（{maturity_label} を強調）",
    )
    return fig


def create_timeseries_figure(
    country_key: str,
    col_index: int,
    row_index: int,
    y_start: int | None = None,
    y_end: int | None = None,
) -> go.Figure:
    data = DATASETS[country_key]
    dates = data["dates"]
    z = data["z"]
    maturity_years = data["maturity_years"]
    maturity_labels = data["maturity_labels"]

    if len(dates) == 0:
        return _no_data_figure(f"{data['display_name']}のデータがありません")
    # インデックスの安全な範囲チェック
    col_index = max(0, min(col_index, len(maturity_years) - 1))
    row_index = max(0, min(row_index, len(dates) - 1))

    y_values = z[:, col_index]
    maturity_label = maturity_labels[col_index]
    z_label = data.get("z_label", "利回り (%)")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(dates))),
            y=y_values,
            mode="lines",
            hovertext=dates,
            hovertemplate=f"日付: %{{hovertext}}<br>{z_label}: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        )
    )

    # 現在のホバー位置を強調表示
    current_y = y_values[row_index]

    # 縦線
    fig.add_vline(
        x=row_index,
        line=dict(color="red", width=1, dash="dash"),
    )

    # 該当ポイントに赤いマーカー
    fig.add_trace(
        go.Scatter(
            x=[row_index],
            y=[current_y],
            mode="markers",
            marker=dict(color="red", size=10),
            showlegend=False,
            hovertext=[dates[row_index]],
            hovertemplate=f"日付: %{{hovertext}}<br>{z_label}: %{{y:.3f}}<extra></extra>",
        )
    )

    # x 軸の表示範囲（インデックス）を指定
    if y_start is None:
        y_start = 0
    if y_end is None:
        y_end = len(dates) - 1

    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis=dict(
            title="時間",
            tickmode="array",
            tickvals=list(range(0, len(dates), max(1, len(dates) // 10))),
            ticktext=[dates[i] for i in range(0, len(dates), max(1, len(dates) // 10))],
            range=[y_start, y_end],
        ),
        yaxis_title=z_label,
        template="plotly_dark",
        showlegend=False,
        title=f"{maturity_label} の推移（{dates[row_index]} を強調）",
    )
    return fig


app = Dash(__name__)

# ドラッグ時にテキスト選択・コピーにならないよう選択を無効化
_NO_SELECT = {
    "userSelect": "none",
    "WebkitUserSelect": "none",
    "MozUserSelect": "none",
    "msUserSelect": "none",
}

app.layout = html.Div(
    style={
        "display": "flex",
        "height": "100vh",
        "backgroundColor": "#111",
        **_NO_SELECT,
    },
    children=[
        # 左側: 2D グラフ
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
                        html.Label("国を選択"),
                        dcc.Dropdown(
                            id="country-dropdown",
                            options=[
                                {"label": "日本", "value": "japan"},
                                {"label": "米国", "value": "usa"},
                                {"label": "英国", "value": "uk"},
                                {"label": "ユーロ圏", "value": "euro"},
                                {"label": "中国", "value": "china"},
                                {"label": "インド", "value": "india"},
                                {"label": "ゴールド先物", "value": "gold"},
                                {"label": "シルバー先物", "value": "silver"},
                            ],
                            value="japan",
                            clearable=False,
                            style={
                                "minWidth": "140px",
                                "color": "#111",
                                "backgroundColor": "#fff",
                            },
                            className="country-dropdown-dark-text",
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
                        html.Div(
                            style={"marginTop": "10px"},
                            children=[
                                html.Label("日付範囲（西暦）"),
                                dcc.DatePickerRange(
                                    id="date-range",
                                    display_format="YYYY-MM-DD",
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Graph(
                    id="curve-graph",
                    style={"flex": "1 1 50%", "marginTop": "16px", **_NO_SELECT},
                ),
                dcc.Graph(
                    id="ts-graph",
                    style={"flex": "1 1 50%", "marginTop": "20px", **_NO_SELECT},
                ),
            ],
        ),
        # 右側: 3D サーフェス（ドラッグで回転するため選択無効）
        html.Div(
            style={
                "flex": "1 1 60%",
                "padding": "10px",
                **_NO_SELECT,
            },
            children=[
                dcc.Store(id="surface-view-mode", data="surface"),
                html.Div(
                    id="surface-view-buttons",
                    style={"marginBottom": "8px", "display": "flex", "gap": "8px", "alignItems": "center"},
                    children=[
                        html.Span("3D表示:", style={"fontSize": "12px"}),
                        html.Button("面", id="btn-view-surface", n_clicks=0),
                        html.Button("ワイヤーフレーム", id="btn-view-wireframe", n_clicks=0),
                    ],
                ),
                dcc.Graph(
                    id="surface-graph",
                    style={"height": "100%", **_NO_SELECT},
                )
            ],
        ),
    ],
)


@app.callback(
    Output("date-range-slider", "min"),
    Output("date-range-slider", "max"),
    Output("date-range-slider", "value"),
    Output("date-range-slider", "marks"),
    Input("country-dropdown", "value"),
)
def init_date_slider(country_key: str):
    # #region agent log
    _dlog("app.py:init_date_slider", "entry", {"country_key": country_key}, "H2")
    # #endregion
    if country_key not in DATASETS:
        country_key = "japan"
    data = DATASETS[country_key]
    dates = data["dates"]
    n = len(dates)
    if n == 0:
        # #region agent log
        _dlog("app.py:init_date_slider", "exit n==0", {"min": 0, "max": 0, "value": [0, 0], "marks_len": 0}, "H2")
        # #endregion
        return 0, 0, [0, 0], {}

    min_idx = 0
    max_idx = n - 1
    # だいたい 10 個くらいの目盛りを出す
    step = max(1, n // 10)
    def _mark_style():
        # 文字のお尻（右端）とスライダーの点が同じになるよう、1.2ラベル分左にずらしてから回転
        return {
            "transform": "translateX(-100%) rotate(-45deg)",
            "transformOrigin": "100% 50%",
            "textAlign": "right",
            "whiteSpace": "nowrap",
            "fontSize": "10px",
            "display": "inline-block",
        }

    marks = {
        i: {
            "label": dates[i],
            "style": _mark_style(),
        }
        for i in range(0, n, step)
    }
    marks[max_idx] = {
        "label": dates[max_idx],
        "style": _mark_style(),
    }
    out_value = [min_idx, max_idx]
    # #region agent log
    _dlog("app.py:init_date_slider", "exit", {"n": n, "min": min_idx, "max": max_idx, "value": out_value, "marks_type": str(type(marks)), "marks_keys_sample": list(marks.keys())[:3]}, "H2")
    # #endregion
    return min_idx, max_idx, out_value, marks


_BTN_BASE = {"fontSize": "12px", "padding": "4px 12px", "cursor": "pointer"}
_BTN_ACTIVE = {**_BTN_BASE, "backgroundColor": "#4a7", "borderColor": "#6c9"}
_BTN_INACTIVE = {**_BTN_BASE, "backgroundColor": "#333", "borderColor": "#555"}


@app.callback(
    Output("surface-view-mode", "data"),
    Input("btn-view-surface", "n_clicks"),
    Input("btn-view-wireframe", "n_clicks"),
)
def update_view_mode(_n_surface, _n_wireframe):
    ctx = callback_context
    if not ctx.triggered:
        return "surface"
    tid = ctx.triggered[0]["prop_id"].split(".")[0]
    return "surface" if tid == "btn-view-surface" else "wireframe"


@app.callback(
    Output("btn-view-surface", "style"),
    Output("btn-view-wireframe", "style"),
    Input("surface-view-mode", "data"),
)
def update_button_styles(view_mode: str):
    if view_mode == "wireframe":
        return _BTN_INACTIVE, _BTN_ACTIVE
    return _BTN_ACTIVE, _BTN_INACTIVE


@app.callback(
    Output("surface-graph", "figure"),
    Input("country-dropdown", "value"),
    Input("date-range-slider", "value"),
    Input("surface-view-mode", "data"),
)
def update_surface(country_key: str, slider_value, view_mode: str):
    # #region agent log
    _dlog("app.py:update_surface", "entry", {"country_key": country_key, "slider_value": repr(slider_value), "slider_value_is_none": slider_value is None}, "H1")
    # #endregion
    if country_key not in DATASETS:
        country_key = "japan"
    data = DATASETS[country_key]
    n = len(data["dates"])
    mode = view_mode if view_mode in ("surface", "wireframe") else "surface"
    if not slider_value or n == 0:
        # #region agent log
        _dlog("app.py:update_surface", "branch no_slider_or_n0", {"returning_full_figure": True}, "H3")
        # #endregion
        return create_surface_figure(country_key, view_mode=mode)
    start_idx = max(0, min(int(slider_value[0]), n - 1))
    end_idx = max(start_idx, min(int(slider_value[1]), n - 1))
    # #region agent log
    _dlog("app.py:update_surface", "before_create", {"start_idx": start_idx, "end_idx": end_idx}, "H3")
    # #endregion
    return create_surface_figure(country_key, start_idx, end_idx, view_mode=mode)


@app.callback(
    Output("curve-graph", "figure"),
    Output("ts-graph", "figure"),
    Input("country-dropdown", "value"),
    Input("surface-graph", "hoverData"),
    Input("date-range-slider", "value"),
)
def update_2d_graphs(country_key: str, hover_data: Dict[str, Any] | None, slider_value):
    # #region agent log
    _dlog("app.py:update_2d_graphs", "entry", {"country_key": country_key, "slider_value": repr(slider_value), "has_hover": hover_data is not None and bool(hover_data.get("points"))}, "H4")
    # #endregion
    if country_key not in DATASETS:
        country_key = "japan"
    data = DATASETS[country_key]
    dates = data["dates"]
    date_values = data["date_values"]
    maturity_years = data["maturity_years"]
    maturity_labels = data["maturity_labels"]
    num_dates = len(dates)
    num_maturities = len(maturity_years)

    # スライダーの範囲（インデックス）
    if not slider_value or num_dates == 0:
        y_start = 0
        y_end = max(0, num_dates - 1)
    else:
        y_start = max(0, min(int(slider_value[0]), num_dates - 1))
        y_end = max(y_start, min(int(slider_value[1]), num_dates - 1))
    # #region agent log
    _dlog("app.py:update_2d_graphs", "after_range", {"y_start": y_start, "y_end": y_end, "slider_was_none": slider_value is None}, "H4")
    # #endregion

    # hoverData が無ければ「範囲の最後の日 × デフォルト（10 年付近）の残存期間」を使う
    row_index = y_end
    col_index = data.get("ts_col_index", 0)

    if hover_data and "points" in hover_data and hover_data["points"]:
        point = hover_data["points"][0]

        # 3D Surface の hoverData は Dash / Plotly のバージョンで形式が違うので、
        # いくつかのパターンを順番に試して行インデックス（row_index）を推定する。

        # 1) y 値がそのままインデックスとして返ってくる場合
        y_val = point.get("y", None)
        if isinstance(y_val, (int, float)):
            row_index = int(round(y_val))

        # 2) y が日付文字列で返ってくる場合
        elif isinstance(y_val, str):
            try:
                row_index = dates.index(y_val)
            except ValueError:
                pass

        # 3) Surface 特有の pointIndex = [row, col] がある場合
        point_index = point.get("pointIndex")
        if isinstance(point_index, (list, tuple)) and len(point_index) >= 1:
            row_index = int(point_index[0])

        # 4) pointNumber から行インデックスを復元する場合
        point_number = point.get("pointNumber")
        if isinstance(point_number, int) and num_maturities > 0:
            row_index = int(point_number // num_maturities)

        # === 列インデックス（col_index）の推定 ===
        # 1) x 値（残存期間の年数）から、一番近いカラムを選ぶ
        x_val = point.get("x", None)
        if isinstance(x_val, (int, float)) and num_maturities > 0:
            col_index = int(np.argmin(np.abs(maturity_years - float(x_val))))

        # 2) x がラベル文字列の場合（ほぼ無いが一応）
        elif isinstance(x_val, str):
            try:
                col_index = maturity_labels.index(x_val)
            except ValueError:
                pass

        # 3) pointIndex = [row, col] の col を使う
        if isinstance(point_index, (list, tuple)) and len(point_index) >= 2:
            col_index = int(point_index[1])

        # 4) pointNumber から列インデックスを復元する
        if isinstance(point_number, int) and num_maturities > 0:
            col_index = int(point_number % num_maturities)

    # インデックスが範囲外に出ないようにクランプ（かつスライダー範囲内に制限）
    row_index = max(y_start, min(row_index, y_end))
    col_index = max(0, min(col_index, num_maturities - 1))

    curve_fig = create_curve_figure(country_key, row_index, col_index)
    ts_fig = create_timeseries_figure(country_key, col_index, row_index, y_start, y_end)
    return curve_fig, ts_fig


if __name__ == "__main__":
    app.run(debug=True, port=8051)

