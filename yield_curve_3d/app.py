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
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


def _parse_japanese_era_date(s: str) -> pd.Timestamp:
    """
    "S49.9.24" や "R7.12.30" のような元号表記を西暦 Timestamp に変換する。
    対応: S(昭和), H(平成), R(令和)
    """
    if not isinstance(s, str) or not s:
        return pd.NaT
    s = s.strip()
    era = s[0]
    try:
        rest = s[1:]
        y_str, m_str, d_str = rest.split(".")
        era_year = int(y_str)
        month = int(m_str)
        day = int(d_str)
    except Exception:
        return pd.NaT

    if era == "S":  # 昭和 (1926-01-01〜)
        year = 1925 + era_year  # S1=1926
    elif era == "H":  # 平成 (1989-01-08〜)
        year = 1988 + era_year  # H1=1989
    elif era == "R":  # 令和 (2019-05-01〜)
        year = 2018 + era_year  # R1=2019
    else:
        return pd.NaT

    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except Exception:
        return pd.NaT


def _load_japan() -> Dict[str, Any]:
    """
    日本国債イールドカーブを読み込み、3D サーフェス用に整形する。

    CSV の形式（japan_yield_curve.csv）:
      1 行目: タイトル行（国債金利情報...）
      2 行目: ヘッダー行（基準日,1年,2年,...,40年）
      3 行目以降: データ
    """
    df = pd.read_csv(DATA_DIR / "japan_yield_curve.csv", header=1)
    # 元号表記の基準日を西暦 Timestamp に変換
    df = df.rename(columns={"基準日": "date_raw"})
    date_values = df["date_raw"].map(_parse_japanese_era_date)
    date_labels = date_values.dt.strftime("%Y-%m-%d").tolist()

    # 残存期間カラム（"1年", "2年", ...）を抽出
    maturity_cols = [c for c in df.columns if c != "date_raw"]

    # パーセント表記を float に変換（"-" 等は NaN にしておく）
    for c in maturity_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 横軸に使う「年数」
    # 日本国債は「1年, 2年, ..., 40年」と想定
    maturity_years = []
    maturity_labels = []
    for c in maturity_cols:
        # "1年" → 1, "10年" → 10 という雑なパースだが十分
        num = "".join(ch for ch in c if ch.isdigit())
        if num == "":
            continue
        maturity_years.append(float(num))
        maturity_labels.append(c)

    maturity_years = np.array(maturity_years, dtype=float)

    # 行: 日付, 列: 残存期間 の 2D 配列
    z = df[maturity_cols].to_numpy(dtype=float)

    # 10 年物カラム（なければ最も近い年限）
    target_label = None
    for label in maturity_labels:
        if "10" in label:
            target_label = label
            break
    if target_label is None and maturity_labels:
        target_label = min(
            maturity_labels,
            key=lambda lbl: abs(float("".join(ch for ch in lbl if ch.isdigit())) - 10.0),
        )
    ts_col_index = maturity_labels.index(target_label) if target_label else 0

    return {
        "country": "japan",
        "display_name": "日本",
        # 画面表示用は西暦 YYYY-MM-DD
        "dates": date_labels,
        # 範囲指定などに使う生の Timestamp
        "date_values": date_values,
        "maturity_years": maturity_years,
        "maturity_labels": maturity_labels,
        "z": z,
        "ts_col_index": ts_col_index,
    }


def _load_usa() -> Dict[str, Any]:
    """
    米国債イールドカーブを読み込み、3D サーフェス用に整形する。

    CSV の形式（usa_yield_curve.csv）:
      1 行目: ヘッダー行（Date,1 Mo,1.5 Month,2 Mo,...,30 Yr）
      2 行目以降: データ
    """
    df = pd.read_csv(DATA_DIR / "usa_yield_curve.csv")
    df = df.rename(columns={"Date": "date_raw"})

    maturity_cols = [c for c in df.columns if c != "date_raw"]

    for c in maturity_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 残存期間（年ベース）に変換
    def _col_to_years(col: str) -> float:
        col = col.strip()
        if "Mo" in col or "Month" in col:
            # "1 Mo", "1.5 Month" など
            num = "".join(ch if (ch.isdigit() or ch == ".") else " " for ch in col)
            try:
                months = float(num.split()[0])
                return months / 12.0
            except Exception:
                return np.nan
        if "Yr" in col or "Year" in col or "Years" in col:
            num = "".join(ch if (ch.isdigit() or ch == ".") else " " for ch in col)
            try:
                years = float(num.split()[0])
                return years
            except Exception:
                return np.nan
        return np.nan

    maturity_years = np.array([_col_to_years(c) for c in maturity_cols], dtype=float)
    maturity_labels = maturity_cols
    z = df[maturity_cols].to_numpy(dtype=float)
    date_values = pd.to_datetime(df["date_raw"])
    date_labels = date_values.dt.strftime("%Y-%m-%d").tolist()

    # 10 年物
    target_label = None
    for label in maturity_labels:
        if "10" in label and ("Yr" in label or "Year" in label):
            target_label = label
            break
    if target_label is None and maturity_labels:
        target_label = min(
            maturity_labels,
            key=lambda lbl: abs(_col_to_years(lbl) - 10.0),
        )
    ts_col_index = maturity_labels.index(target_label) if target_label else 0

    return {
        "country": "usa",
        "display_name": "米国",
        "dates": date_labels,
        "date_values": date_values,
        "maturity_years": maturity_years,
        "maturity_labels": maturity_labels,
        "z": z,
        "ts_col_index": ts_col_index,
    }


def _col_to_years_generic(col) -> float:
    """残存期間列名を年数に変換（UK, Euro, China, India 用）。数値列名(0.5, 1, 10 等)も可。"""
    if isinstance(col, (int, float)):
        return float(col)
    col = str(col).strip()
    if "M" in col.upper() and "Y" not in col.upper():
        num = "".join(ch for ch in col if ch.isdigit() or ch == ".")
        try:
            return float(num or 0) / 12.0
        except ValueError:
            return np.nan
    if "Y" in col.upper() or "Yr" in col or "Year" in col:
        num = "".join(ch for ch in col if ch.isdigit() or ch == ".")
        try:
            return float(num or 0)
        except ValueError:
            return np.nan
    try:
        return float(col)
    except (ValueError, TypeError):
        return np.nan


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


def _load_uk() -> Dict[str, Any]:
    """
    英国国債イールドカーブを読み込む。
    CSV: Date + 残存期間列（例: 5Y, 10Y, 30Y）。DMO/BoE 形式。
    """
    try:
        df = pd.read_csv(DATA_DIR / "uk_yield_curve.csv")
    except FileNotFoundError:
        return _empty_dataset("uk", "英国")
    df = df.rename(columns={df.columns[0]: "date_raw"})
    maturity_cols = [c for c in df.columns if c != "date_raw"]
    for c in maturity_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    maturity_years = np.array([_col_to_years_generic(c) for c in maturity_cols], dtype=float)
    maturity_labels = maturity_cols
    date_values = pd.to_datetime(df["date_raw"], errors="coerce")
    df = df.dropna(subset=["date_raw"])
    date_values = pd.to_datetime(df["date_raw"], errors="coerce")
    date_labels = date_values.dt.strftime("%Y-%m-%d").tolist()
    z = df[maturity_cols].to_numpy(dtype=float)
    if len(date_labels) == 0:
        return _empty_dataset("uk", "英国")
    ts_col_index = 0
    for i, lbl in enumerate(maturity_labels):
        if abs(_col_to_years_generic(lbl) - 10.0) < 0.5:
            ts_col_index = i
            break
    return {
        "country": "uk",
        "display_name": "英国",
        "dates": date_labels,
        "date_values": date_values,
        "maturity_years": maturity_years,
        "maturity_labels": maturity_labels,
        "z": z,
        "ts_col_index": ts_col_index,
    }


def _load_euro() -> Dict[str, Any]:
    """
    ユーロ圏（ECB AAA 国債）イールドカーブを読み込む。
    CSV: 日付列 + 残存期間列。ECB 形式。データは 2004年9月〜。
    """
    try:
        df = pd.read_csv(DATA_DIR / "euro_yield_curve.csv")
    except FileNotFoundError:
        return _empty_dataset("euro", "ユーロ圏")
    df = df.rename(columns={df.columns[0]: "date_raw"})
    maturity_cols = [c for c in df.columns if c != "date_raw"]
    for c in maturity_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    maturity_years = np.array([_col_to_years_generic(c) for c in maturity_cols], dtype=float)
    maturity_labels = maturity_cols
    date_values = pd.to_datetime(df["date_raw"], errors="coerce")
    df = df.dropna(subset=["date_raw"])
    date_values = pd.to_datetime(df["date_raw"], errors="coerce")
    date_labels = date_values.dt.strftime("%Y-%m-%d").tolist()
    z = df[maturity_cols].to_numpy(dtype=float)
    if len(date_labels) == 0:
        return _empty_dataset("euro", "ユーロ圏")
    ts_col_index = 0
    for i, lbl in enumerate(maturity_labels):
        if abs(_col_to_years_generic(lbl) - 10.0) < 0.5:
            ts_col_index = i
            break
    return {
        "country": "euro",
        "display_name": "ユーロ圏",
        "dates": date_labels,
        "date_values": date_values,
        "maturity_years": maturity_years,
        "maturity_labels": maturity_labels,
        "z": z,
        "ts_col_index": ts_col_index,
    }


def _load_china() -> Dict[str, Any]:
    """
    中国国債イールドカーブを読み込む。
    CSV: Date + 3M, 6M, 1Y, 2Y, ... など。ChinaBond 形式。
    """
    try:
        df = pd.read_csv(DATA_DIR / "china_yield_curve.csv")
    except FileNotFoundError:
        return _empty_dataset("china", "中国")
    df = df.rename(columns={df.columns[0]: "date_raw"})
    maturity_cols = [c for c in df.columns if c != "date_raw"]
    for c in maturity_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    maturity_years = np.array([_col_to_years_generic(c) for c in maturity_cols], dtype=float)
    maturity_labels = maturity_cols
    date_values = pd.to_datetime(df["date_raw"], errors="coerce")
    df = df.dropna(subset=["date_raw"])
    date_values = pd.to_datetime(df["date_raw"], errors="coerce")
    date_labels = date_values.dt.strftime("%Y-%m-%d").tolist()
    z = df[maturity_cols].to_numpy(dtype=float)
    if len(date_labels) == 0:
        return _empty_dataset("china", "中国")
    ts_col_index = 0
    for i, lbl in enumerate(maturity_labels):
        if abs(_col_to_years_generic(lbl) - 10.0) < 0.5:
            ts_col_index = i
            break
    return {
        "country": "china",
        "display_name": "中国",
        "dates": date_labels,
        "date_values": date_values,
        "maturity_years": maturity_years,
        "maturity_labels": maturity_labels,
        "z": z,
        "ts_col_index": ts_col_index,
    }


def _load_india() -> Dict[str, Any]:
    """
    インド国債イールドカーブを読み込む。
    CSV: Date + 1Y, 5Y, 10Y など。RBI 等の形式。フルカーブが無い場合は 10Y のみでも可。
    """
    try:
        df = pd.read_csv(DATA_DIR / "india_yield_curve.csv")
    except FileNotFoundError:
        return _empty_dataset("india", "インド")
    df = df.rename(columns={df.columns[0]: "date_raw"})
    maturity_cols = [c for c in df.columns if c != "date_raw"]
    for c in maturity_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    maturity_years = np.array([_col_to_years_generic(c) for c in maturity_cols], dtype=float)
    maturity_labels = maturity_cols
    date_values = pd.to_datetime(df["date_raw"], errors="coerce")
    df = df.dropna(subset=["date_raw"])
    date_values = pd.to_datetime(df["date_raw"], errors="coerce")
    date_labels = date_values.dt.strftime("%Y-%m-%d").tolist()
    z = df[maturity_cols].to_numpy(dtype=float)
    if len(date_labels) == 0:
        return _empty_dataset("india", "インド")
    ts_col_index = 0
    for i, lbl in enumerate(maturity_labels):
        if abs(_col_to_years_generic(lbl) - 10.0) < 0.5:
            ts_col_index = i
            break
    return {
        "country": "india",
        "display_name": "インド",
        "dates": date_labels,
        "date_values": date_values,
        "maturity_years": maturity_years,
        "maturity_labels": maturity_labels,
        "z": z,
        "ts_col_index": ts_col_index,
    }


DATASETS: Dict[str, Dict[str, Any]] = {
    "japan": _load_japan(),
    "usa": _load_usa(),
    "uk": _load_uk(),
    "euro": _load_euro(),
    "china": _load_china(),
    "india": _load_india(),
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


def create_surface_figure(country_key: str, y_start=None, y_end=None) -> go.Figure:
    data = DATASETS[country_key]
    dates = data["dates"]
    maturity_years = data["maturity_years"]
    z = data["z"]

    if len(dates) == 0:
        return _no_data_figure(f"{data['display_name']}のデータがありません")

    # y 軸はインデックス（0,1,2,...) を使い、ラベルに日付を表示
    y_indices = np.arange(len(dates))

    surface = go.Surface(
        x=maturity_years,
        y=y_indices,
        z=z,
        colorscale="RdBu",
        reversescale=True,
        colorbar=dict(title="利回り (%)"),
        showscale=True,
    )

    fig = go.Figure(data=[surface])

    # 日付範囲（インデックス）指定があれば、そこにズーム
    if y_indices.size == 0:
        y_min = 0
        y_max = 0
    else:
        y_min = int(y_indices[0])
        y_max = int(y_indices[-1])
    if y_start is not None:
        y_min = max(y_min, int(y_start))
    if y_end is not None:
        y_max = min(y_max, int(y_end))

    # 可視範囲の長さに応じて y 軸ラベルの間隔を調整
    visible_indices = np.arange(y_min, y_max + 1) if y_max >= y_min else np.array([y_min])
    y_step = max(1, len(visible_indices) // 10)
    tick_indices = visible_indices[::y_step]

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="残存期間 (年)",
            xaxis=dict(
                autorange="reversed",  # 残存期間軸を反転（短期→長期 を右方向に）
            ),
            yaxis_title="日付",
            zaxis_title="利回り (%)",
            yaxis=dict(
                tickmode="array",
                tickvals=tick_indices,
                ticktext=[dates[int(i)] for i in tick_indices],
                range=[y_max, y_min] if country_key in ("usa", "uk", "euro") else [y_min, y_max],
            ),
        ),
        template="plotly_dark",
        title=f"{data['display_name']} イールドカーブ（3D）",
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

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=maturity_years,
            y=y_values,
            mode="lines+markers",
            text=maturity_labels,
            hovertemplate="残存期間: %{text}<br>利回り: %{y:.3f}%<extra></extra>",
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
            hovertemplate="残存期間: %{hovertext}<br>利回り: %{y:.3f}%<extra></extra>",
        )
    )

    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis_title="残存期間 (年)",
        yaxis_title="利回り (%)",
        template="plotly_dark",
        showlegend=False,
        title=f"{date_label} のイールドカーブ断面（{maturity_label} を強調）",
    )
    return fig


def create_timeseries_figure(country_key: str, col_index: int, row_index: int) -> go.Figure:
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

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(dates))),
            y=y_values,
            mode="lines",
            hovertext=dates,
            hovertemplate="日付: %{hovertext}<br>利回り: %{y:.3f}%<extra></extra>",
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
            hovertemplate="日付: %{hovertext}<br>利回り: %{y:.3f}%<extra></extra>",
        )
    )

    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis=dict(
            title="時間",
            tickmode="array",
            tickvals=list(range(0, len(dates), max(1, len(dates) // 10))),
            ticktext=[dates[i] for i in range(0, len(dates), max(1, len(dates) // 10))],
        ),
        yaxis_title="利回り (%)",
        template="plotly_dark",
        showlegend=False,
        title=f"{maturity_label} の利回りの推移（{dates[row_index]} を強調）",
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


@app.callback(
    Output("surface-graph", "figure"),
    Input("country-dropdown", "value"),
    Input("date-range-slider", "value"),
)
def update_surface(country_key: str, slider_value):
    # #region agent log
    _dlog("app.py:update_surface", "entry", {"country_key": country_key, "slider_value": repr(slider_value), "slider_value_is_none": slider_value is None}, "H1")
    # #endregion
    if country_key not in DATASETS:
        country_key = "japan"
    data = DATASETS[country_key]
    n = len(data["dates"])
    if not slider_value or n == 0:
        # #region agent log
        _dlog("app.py:update_surface", "branch no_slider_or_n0", {"returning_full_figure": True}, "H3")
        # #endregion
        return create_surface_figure(country_key)
    start_idx = max(0, min(int(slider_value[0]), n - 1))
    end_idx = max(start_idx, min(int(slider_value[1]), n - 1))
    # #region agent log
    _dlog("app.py:update_surface", "before_create", {"start_idx": start_idx, "end_idx": end_idx}, "H3")
    # #endregion
    return create_surface_figure(country_key, start_idx, end_idx)


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
    num_dates = len(data["dates"])
    maturity_years = data["maturity_years"]
    maturity_labels = data["maturity_labels"]
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
                row_index = data["dates"].index(y_val)
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
    ts_fig = create_timeseries_figure(country_key, col_index, row_index)
    return curve_fig, ts_fig


if __name__ == "__main__":
    app.run(debug=True)

