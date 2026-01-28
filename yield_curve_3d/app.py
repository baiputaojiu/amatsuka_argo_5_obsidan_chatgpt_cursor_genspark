from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


def _parse_japanese_era_date(s: str) -> pd.Timestamp:
    """
    \"S49.9.24\" や \"R7.12.30\" のような元号表記を西暦の Timestamp に変換する。
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
    # 日本国債の基準日（元号表記）を西暦 Timestamp に変換
    df = df.rename(columns={"基準日": "date_raw"})
    date_values = df["date_raw"].map(_parse_japanese_era_date)
    # 表示用は YYYY-MM-DD
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
        "dates": date_labels,
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


DATASETS: Dict[str, Dict[str, Any]] = {
    "japan": _load_japan(),
    "usa": _load_usa(),
}


def create_surface_figure(country_key: str, y_start: int | None = None, y_end: int | None = None) -> go.Figure:
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

    # 表示範囲（インデックス）の指定があれば yaxis.range に反映
    if y_start is None:
        y_start = int(y_indices[0]) if len(y_indices) > 0 else 0
    if y_end is None:
        y_end = int(y_indices[-1]) if len(y_indices) > 0 else 0

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
                tickvals=y_indices[:: max(1, len(y_indices) // 10)],
                ticktext=[dates[i] for i in y_indices[:: max(1, len(y_indices) // 10)]],
                range=[y_start, y_end],
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
        template="plotly_white",
        title=f"{date_label} のイールドカーブ断面（{maturity_label} を強調）",
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
        yaxis_title="利回り (%)",
        template="plotly_white",
        title=f"{maturity_label} の利回りの推移（{dates[row_index]} を強調）",
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
    Output("date-range", "min_date_allowed"),
    Output("date-range", "max_date_allowed"),
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
    Input("country-radio", "value"),
)
def sync_date_range(country_key: str):
    data = DATASETS[country_key]
    date_values = data["date_values"]
    if len(date_values) == 0:
        return None, None, None, None
    start = date_values.iloc[0]
    end = date_values.iloc[-1]
    # 初期値: 全期間
    return (
        start.date(),
        end.date(),
        start.date(),
        end.date(),
    )


@app.callback(
    Output("surface-graph", "figure"),
    Input("country-radio", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def update_surface(country_key: str, start_date: str | None, end_date: str | None):
    data = DATASETS[country_key]
    date_values = data["date_values"]
    num_dates = len(date_values)
    if num_dates == 0:
        return create_surface_figure(country_key)

    y_start = 0
    y_end = num_dates - 1

    # start_date, end_date は \"YYYY-MM-DD\" 形式
    if start_date:
        start_ts = pd.to_datetime(start_date)
        y_start = int(np.searchsorted(date_values.values, start_ts.to_datetime64(), side="left"))
    if end_date:
        end_ts = pd.to_datetime(end_date)
        y_end = int(np.searchsorted(date_values.values, end_ts.to_datetime64(), side="right") - 1)

    y_start = max(0, min(y_start, num_dates - 1))
    y_end = max(y_start, min(y_end, num_dates - 1))

    return create_surface_figure(country_key, y_start, y_end)


@app.callback(
    Output("curve-graph", "figure"),
    Output("ts-graph", "figure"),
    Input("country-radio", "value"),
    Input("surface-graph", "hoverData"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def update_2d_graphs(
    country_key: str,
    hover_data: Dict[str, Any] | None,
    start_date: str | None,
    end_date: str | None,
):
    data = DATASETS[country_key]
    dates = data["dates"]
    date_values = data["date_values"]
    maturity_years = data["maturity_years"]
    maturity_labels = data["maturity_labels"]
    num_dates = len(dates)
    num_maturities = len(maturity_years)

    if num_dates == 0 or num_maturities == 0:
        empty_fig = go.Figure()
        return empty_fig, empty_fig

    # 日付範囲に対応するインデックス（y_start, y_end）を算出
    y_start = 0
    y_end = num_dates - 1
    if start_date:
        start_ts = pd.to_datetime(start_date)
        y_start = int(np.searchsorted(date_values.values, start_ts.to_datetime64(), side="left"))
    if end_date:
        end_ts = pd.to_datetime(end_date)
        y_end = int(np.searchsorted(date_values.values, end_ts.to_datetime64(), side="right") - 1)
    y_start = max(0, min(y_start, num_dates - 1))
    y_end = max(y_start, min(y_end, num_dates - 1))

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

    # インデックスが範囲外に出ないようにクランプ（かつ選択範囲内に制限）
    row_index = max(y_start, min(row_index, y_end))
    col_index = max(0, min(col_index, num_maturities - 1))

    curve_fig = create_curve_figure(country_key, row_index, col_index)
    ts_fig = create_timeseries_figure(country_key, col_index, row_index, y_start, y_end)
    return curve_fig, ts_fig


if __name__ == "__main__":
    app.run(debug=True)

