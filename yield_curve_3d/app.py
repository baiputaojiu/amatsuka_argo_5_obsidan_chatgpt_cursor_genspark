from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


def _load_japan() -> Dict[str, Any]:
    """
    日本国債イールドカーブを読み込み、3D サーフェス用に整形する。

    CSV の形式（japan_yield_curve.csv）:
      1 行目: タイトル行（国債金利情報...）
      2 行目: ヘッダー行（基準日,1年,2年,...,40年）
      3 行目以降: データ
    """
    df = pd.read_csv(DATA_DIR / "japan_yield_curve.csv", header=1)
    df = df.rename(columns={"基準日": "date"})

    # 残存期間カラム（"1年", "2年", ...）を抽出
    maturity_cols = [c for c in df.columns if c != "date"]

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

    dates = df["date"].astype(str).tolist()

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
        "dates": dates,
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
    df = df.rename(columns={"Date": "date"})

    maturity_cols = [c for c in df.columns if c != "date"]

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
    dates = df["date"].astype(str).tolist()

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
        "dates": dates,
        "maturity_years": maturity_years,
        "maturity_labels": maturity_labels,
        "z": z,
        "ts_col_index": ts_col_index,
    }


DATASETS: Dict[str, Dict[str, Any]] = {
    "japan": _load_japan(),
    "usa": _load_usa(),
}


def create_surface_figure(country_key: str) -> go.Figure:
    data = DATASETS[country_key]
    dates = data["dates"]
    maturity_years = data["maturity_years"]
    z = data["z"]

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
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="残存期間 (年)",
            yaxis_title="日付",
            zaxis_title="利回り (%)",
            yaxis=dict(
                tickmode="array",
                tickvals=y_indices[:: max(1, len(y_indices) // 10)],
                ticktext=[dates[i] for i in y_indices[:: max(1, len(y_indices) // 10)]],
            ),
        ),
        template="plotly_dark",
        title=f"{data['display_name']} イールドカーブ（3D）",
    )
    return fig


def create_curve_figure(country_key: str, row_index: int) -> go.Figure:
    data = DATASETS[country_key]
    dates = data["dates"]
    maturity_years = data["maturity_years"]
    maturity_labels = data["maturity_labels"]
    z = data["z"]

    row_index = max(0, min(row_index, len(dates) - 1))
    y_values = z[row_index, :]
    date_label = dates[row_index]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=maturity_years,
            y=y_values,
            mode="lines+markers",
            text=maturity_labels,
            hovertemplate="残存期間: %{text}<br>利回り: %{y:.3f}%<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis_title="残存期間 (年)",
        yaxis_title="利回り (%)",
        template="plotly_white",
        title=f"{date_label} のイールドカーブ断面",
    )
    return fig


def create_timeseries_figure(country_key: str, row_index: int) -> go.Figure:
    data = DATASETS[country_key]
    dates = data["dates"]
    z = data["z"]
    ts_col_index = data["ts_col_index"]

    y_values = z[:, ts_col_index]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(dates))),
            y=y_values,
            mode="lines",
            hovertext=dates,
            hovertemplate="日付: %{hovertext}<br>利回り: %{y:.3f}%<extra></extra>",
        )
    )

    # 現在のホバー位置に縦線を描く
    row_index = max(0, min(row_index, len(dates) - 1))
    fig.add_vline(
        x=row_index,
        line=dict(color="red", width=1, dash="dash"),
    )

    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis=dict(
            title="時間",
            tickmode="array",
            tickvals=list(range(0, len(dates), max(1, len(dates) // 10))),
            ticktext=[dates[i] for i in range(0, len(dates), max(1, len(dates) // 10))],
        ),
        yaxis_title="10年金利 (%)",
        template="plotly_white",
        title="10年金利の推移",
    )
    return fig


app = Dash(__name__)

app.layout = html.Div(
    style={"display": "flex", "height": "100vh", "backgroundColor": "#111"},
    children=[
        # 左側: 2D グラフ
        html.Div(
            style={
                "flex": "0 0 40%",
                "display": "flex",
                "flexDirection": "column",
                "padding": "10px",
                "backgroundColor": "#f5f5f5",
            },
            children=[
                html.Div(
                    style={"marginBottom": "10px"},
                    children=[
                        html.Label("国を選択"),
                        dcc.RadioItems(
                            id="country-radio",
                            options=[
                                {"label": "日本", "value": "japan"},
                                {"label": "米国", "value": "usa"},
                            ],
                            value="japan",
                            labelStyle={"display": "inline-block", "marginRight": "10px"},
                        ),
                    ],
                ),
                dcc.Graph(
                    id="curve-graph",
                    style={"flex": "1 1 50%"},
                ),
                dcc.Graph(
                    id="ts-graph",
                    style={"flex": "1 1 50%"},
                ),
            ],
        ),
        # 右側: 3D サーフェス
        html.Div(
            style={"flex": "1 1 60%", "padding": "10px"},
            children=[
                dcc.Graph(
                    id="surface-graph",
                    style={"height": "100%"},
                )
            ],
        ),
    ],
)


@app.callback(
    Output("surface-graph", "figure"),
    Input("country-radio", "value"),
)
def update_surface(country_key: str):
    return create_surface_figure(country_key)


@app.callback(
    Output("curve-graph", "figure"),
    Output("ts-graph", "figure"),
    Input("country-radio", "value"),
    Input("surface-graph", "hoverData"),
)
def update_2d_graphs(country_key: str, hover_data: Dict[str, Any] | None):
    data = DATASETS[country_key]
    num_dates = len(data["dates"])
    num_maturities = len(data["maturity_years"])

    # hoverData が無ければ最新の日付を使う
    row_index = num_dates - 1

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

    # インデックスが範囲外に出ないようにクランプ
    row_index = max(0, min(row_index, num_dates - 1))

    curve_fig = create_curve_figure(country_key, row_index)
    ts_fig = create_timeseries_figure(country_key, row_index)
    return curve_fig, ts_fig


if __name__ == "__main__":
    app.run(debug=True)

