"""
シルバー/ゴールド 1ヶ月フォワードレート時系列チャート（テスト用）

画像のXAG1M=TTKL風グラフを再現。金属切り替えでゴールドも表示。
- データ: silver_forward_curve.csv / gold_forward_curve.csv（0.0=現物、1=フロント月）
- レート = (フロント月 - 現物) / 現物 * 100 (%)
- ダークテーマ、Y=0（Par）基準線、期間切り替え（6ヶ月など）
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, no_update

import plotly.graph_objs as go

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "yield_curve_3d" / "data"
YIELD_CURVE_3D_DIR = BASE_DIR.parent / "yield_curve_3d"

# 金属ごとのCSVと表示名
METAL_OPTIONS = [
    {"label": "シルバー (XAG)", "value": "silver"},
    {"label": "ゴールド (XAU)", "value": "gold"},
]
METAL_FILES = {"silver": "silver_forward_curve.csv", "gold": "gold_forward_curve.csv"}
METAL_LABELS = {"silver": "シルバー", "gold": "ゴールド"}
METAL_SYMBOLS = {"silver": "XAG", "gold": "XAU"}

# 期間オプション（日数）
PERIOD_OPTIONS = [
    {"label": "1日", "value": 1},
    {"label": "5日", "value": 5},
    {"label": "10日", "value": 10},
    {"label": "1ヶ月", "value": 21},
    {"label": "3ヶ月", "value": 63},
    {"label": "6ヶ月", "value": 126},
    {"label": "2025年4月〜", "value": "2025-04"},
    {"label": "年初来", "value": "ytd"},
    {"label": "1年", "value": 252},
    {"label": "最大", "value": "max"},
]


def get_data_path(metal: str) -> Path:
    """金属に応じたCSVパスを返す"""
    f = METAL_FILES.get(metal, METAL_FILES["silver"])
    return DATA_DIR / f


def load_data(metal: str = "silver") -> pd.DataFrame:
    """フォワードカーブCSVを読み、(フロント月-現物)/現物*100 でフォワードレート (%) を計算"""
    path = get_data_path(metal)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    spot = pd.to_numeric(df["0.0"], errors="coerce")
    front = pd.to_numeric(df["1"], errors="coerce")
    # レート = (フロント月 - 現物) / 現物 * 100
    df["Rate_1M"] = (front - spot) / spot * 100
    df["Rate_1M"] = df["Rate_1M"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["0.0", "1", "Rate_1M"])
    return df.reset_index(drop=True)


def get_missing_spot_dates(metal: str) -> list[dict]:
    """現物(0.0)が欠損している日付のリストをドロップダウン用に返す。NaT・不正値は除外する。"""
    path = get_data_path(metal)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "0.0" not in df.columns or "Date" not in df.columns:
        return []
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["_spot"] = pd.to_numeric(df["0.0"], errors="coerce")
    # 日付が有効で、かつ 0.0 が NaN の行のみ
    missing = df.loc[df["Date"].notna() & df["_spot"].isna(), "Date"]
    missing = missing.drop_duplicates().sort_values()
    # 文字列に変換（NaT は上で除外済み）
    labels = missing.dt.strftime("%Y-%m-%d").tolist()
    return [{"label": s, "value": s} for s in labels if s]


def save_spot_price(metal: str, date_str: str, value: float) -> bool:
    """指定日の現物(0.0)をCSVに上書き保存する"""
    path = get_data_path(metal)
    if not path.exists():
        return False
    df = pd.read_csv(path)
    if "0.0" not in df.columns or "Date" not in df.columns:
        return False
    # 日付は文字列またはパースして比較
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    mask = df["Date"].notna() & (df["Date"].dt.strftime("%Y-%m-%d") == date_str)
    if not mask.any():
        return False
    df.loc[mask, "0.0"] = value
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df.to_csv(path, index=False)
    return True


def run_curve_update() -> str:
    """update_forward_curves.py を実行し、結果メッセージを返す"""
    script = YIELD_CURVE_3D_DIR / "update_forward_curves.py"
    if not script.exists():
        return "更新スクリプトが見つかりません"
    try:
        r = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(YIELD_CURVE_3D_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode != 0:
            return f"更新エラー: {r.stderr or r.stdout or 'unknown'}"
        return (r.stdout or "").strip() or "更新完了"
    except subprocess.TimeoutExpired:
        return "更新がタイムアウトしました"
    except Exception as e:
        return f"更新失敗: {e}"


def create_figure(df: pd.DataFrame, period: str | int, metal: str = "silver") -> go.Figure:
    """フォワードレート時系列グラフを作成"""
    label = METAL_LABELS.get(metal, "シルバー")
    symbol = METAL_SYMBOLS.get(metal, "XAG")
    csv_name = METAL_FILES.get(metal, "silver_forward_curve.csv")

    if df.empty or len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text=f"データがありません<br>{csv_name} を確認してください",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14), align="center",
        )
        fig.update_layout(template="plotly_dark", margin=dict(t=40, b=40, l=40, r=40))
        return fig

    # 期間でフィルタ
    df = df.copy()
    if period == "max":
        pass
    elif period == "ytd":
        y = pd.Timestamp.now().year
        df = df[df["Date"].dt.year == y]
    elif period == "2025-04":
        df = df[df["Date"] >= pd.Timestamp("2025-04-01")]
    else:
        n = int(period) if isinstance(period, (int, float)) else 126
        df = df.tail(n)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Rate_1M"],
            name="1M Forward Rate",
            marker_color=["rgb(70, 130, 255)" if y >= 0 else "rgb(220, 100, 100)" for y in df["Rate_1M"]],
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
        )
    )

    # Y=0 の基準線（Par）
    fig.add_hline(
        y=0,
        line=dict(color="rgba(150,150,150,0.6)", width=1, dash="dash"),
        annotation_text="Par (0%)",
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"{symbol} 1 Month Forward Rate（{label}1ヶ月フォワードレート）",
            font=dict(size=16),
            x=0.5,
            xanchor="center",
        ),
        margin=dict(l=60, r=20, t=50, b=60),
        xaxis=dict(
            title="日付",
            showgrid=True,
            gridcolor="rgba(100,100,100,0.3)",
            tickformat="%Y-%m",
        ),
        yaxis=dict(
            title="フォワードレート (%)",
            showgrid=True,
            gridcolor="rgba(100,100,100,0.3)",
            zeroline=True,
            zerolinecolor="rgba(100,150,255,0.5)",
            range=[-40, 15],
            dtick=5,
        ),
        hovermode="x unified",
        showlegend=False,
        plot_bgcolor="rgba(17,17,17,1)",
        paper_bgcolor="rgba(17,17,17,1)",
    )

    return fig


# --- Dash アプリ ---
app = Dash(__name__)

app.layout = html.Div(
    style={
        "backgroundColor": "#111",
        "minHeight": "100vh",
        "padding": "20px",
        "fontFamily": "Segoe UI, sans-serif",
    },
    children=[
        html.H2(
            "1ヶ月フォワードレート テスト（シルバー / ゴールド）",
            style={"color": "#eee", "marginBottom": "10px"},
        ),
        html.Div(
            style={"marginBottom": "16px", "display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "center"},
            children=[
                html.Label("金属:", style={"color": "#aaa", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="metal-dropdown",
                    options=[{"label": o["label"], "value": o["value"]} for o in METAL_OPTIONS],
                    value="silver",
                    clearable=False,
                    style={"minWidth": "160px", "color": "#111"},
                ),
                html.Label("表示期間:", style={"color": "#aaa", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="period-dropdown",
                    options=[{"label": o["label"], "value": o["value"]} for o in PERIOD_OPTIONS],
                    value=126,
                    clearable=False,
                    style={"minWidth": "120px", "color": "#111"},
                ),
                html.Button(
                    "限月データを更新",
                    id="update-btn",
                    n_clicks=0,
                    style={
                        "padding": "8px 16px",
                        "backgroundColor": "#2a7",
                        "color": "#fff",
                        "border": "none",
                        "borderRadius": "4px",
                        "cursor": "pointer",
                    },
                ),
                html.Span(id="update-status", style={"color": "#8c8", "marginLeft": "8px", "fontSize": "12px"}),
            ],
        ),
        dcc.Store(id="last-update-clicks", data=0),
        dcc.Graph(
            id="forward-rate-graph",
            style={"height": "500px"},
            config={"displayModeBar": True, "displaylogo": False},
        ),
        html.P(
            id="data-source-footer",
            children="データ: yield_curve_3d/data/silver_forward_curve.csv（0.0=現物、1=フロント月、レート=(1-0.0)/0.0*100）",
            style={"color": "#666", "fontSize": "12px", "marginTop": "12px"},
        ),
        html.Hr(style={"borderColor": "#333", "marginTop": "24px", "marginBottom": "16px"}),
        html.H4("現在価格が無い日（手入力）", style={"color": "#ccc", "marginBottom": "8px"}),
        html.Div(
            style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "alignItems": "center", "marginBottom": "8px"},
            children=[
                html.Label("日付:", style={"color": "#aaa"}),
                dcc.Dropdown(
                    id="missing-spot-date",
                    options=[],
                    placeholder="日付を選択",
                    style={"minWidth": "140px", "color": "#111"},
                ),
                html.Label("現物価格 (0.0):", style={"color": "#aaa"}),
                dcc.Input(
                    id="spot-value",
                    type="number",
                    placeholder="数値",
                    min=0,
                    step="any",
                    style={"width": "100px", "color": "#111", "padding": "4px"},
                ),
                html.Button(
                    "保存",
                    id="save-spot-btn",
                    n_clicks=0,
                    style={
                        "padding": "6px 12px",
                        "backgroundColor": "#46a",
                        "color": "#fff",
                        "border": "none",
                        "borderRadius": "4px",
                        "cursor": "pointer",
                    },
                ),
                html.Span(id="spot-save-status", style={"color": "#8c8", "fontSize": "12px"}),
            ],
        ),
    ],
)


@app.callback(
    [
        Output("forward-rate-graph", "figure"),
        Output("data-source-footer", "children"),
        Output("update-status", "children"),
        Output("last-update-clicks", "data"),
    ],
    [
        Input("metal-dropdown", "value"),
        Input("period-dropdown", "value"),
        Input("update-btn", "n_clicks"),
    ],
    [State("last-update-clicks", "data")],
)
def update_graph(metal, period, update_n_clicks, last_update_clicks):
    metal = metal or "silver"
    last = last_update_clicks or 0
    status_out = no_update
    store_out = no_update

    if update_n_clicks and update_n_clicks > last:
        store_out = update_n_clicks
        msg = run_curve_update()
        status_out = msg

    df = load_data(metal)
    csv_name = METAL_FILES.get(metal, "silver_forward_curve.csv")
    footer = f"データ: yield_curve_3d/data/{csv_name}（0.0=現物、1=フロント月、レート=(1-0.0)/0.0*100）"
    return create_figure(df, period, metal), footer, status_out, store_out


@app.callback(
    [
        Output("missing-spot-date", "options"),
        Output("spot-save-status", "children"),
    ],
    [
        Input("metal-dropdown", "value"),
        Input("save-spot-btn", "n_clicks"),
    ],
    [State("missing-spot-date", "value"), State("spot-value", "value")],
)
def update_missing_spot_and_save(metal, save_n_clicks, selected_date, spot_value):
    from dash import ctx

    metal = metal or "silver"
    try:
        options = get_missing_spot_dates(metal)
    except Exception:
        options = []
    if not isinstance(options, list):
        options = []
    save_status = ""

    if ctx.triggered_id == "save-spot-btn" and save_n_clicks and selected_date is not None and spot_value is not None:
        try:
            v = float(spot_value)
            if save_spot_price(metal, str(selected_date), v):
                save_status = "保存しました"
                try:
                    options = get_missing_spot_dates(metal)
                except Exception:
                    pass
        except (ValueError, TypeError):
            save_status = "無効な値です"

    return options, save_status


if __name__ == "__main__":
    app.run(debug=True, port=8052)
