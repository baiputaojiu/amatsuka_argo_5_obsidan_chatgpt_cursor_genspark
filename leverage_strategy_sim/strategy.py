"""
レバレッジ戦略の計算ロジック。

- データ取得: yfinance で ^N225 または ^TPX を取得
- 4戦略: 現物1倍、現物2倍、ブル2倍、ベア2倍ショート
- ドローダウン: 高値からの最大下落率
"""

from __future__ import annotations

from datetime import date

import pandas as pd


TICKERS = {"^N225": "日経平均", "^TPX": "TOPIX"}


def fetch_data(
    ticker: str,
    start_date: str | date,
    end_date: str | date | None = None,
) -> pd.DataFrame | None:
    """
    yfinance で指数データを取得する。

    Args:
        ticker: ^N225 または ^TPX
        start_date: 開始日
        end_date: 終了日（None の場合は本日まで）

    Returns:
        Date インデックス、Close 列を持つ DataFrame。失敗時は None。
    """
    import yfinance as yf

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else None

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    df = df.reset_index()
    date_col = df.columns[0]

    if "Close" not in df.columns:
        return None

    result = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col]),
        "Close": pd.to_numeric(df["Close"], errors="coerce"),
    })
    for c in ("Open", "High", "Low"):
        if c in df.columns:
            result[c] = pd.to_numeric(df[c], errors="coerce")
    result = result.dropna(subset=["Date", "Close"])
    result = result.sort_values("Date").reset_index(drop=True)
    return result


def compute_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """
    4つの戦略の時系列を計算する。

    Args:
        df: Date, Close 列を持つ DataFrame

    Returns:
        Date, spot_1x, spot_2x, bull_2x, bear_2x_short 列を持つ DataFrame
    """
    out = df[["Date", "Close"]].copy()

    close = df["Close"]
    first_close = close.iloc[0]

    # 現物1倍: 初日=1 で正規化
    out["spot_1x"] = close / first_close

    # 現物2倍（リバランスなし）: 初日に2倍数量を購入してホールド
    out["spot_2x"] = 2 * (close / first_close)

    # 日次騰落率
    daily_return = close.pct_change().fillna(0)

    # ブル2倍ロング（日次リバランス）
    bull_2x = (1 + daily_return * 2).cumprod()
    out["bull_2x"] = bull_2x

    # ベア2倍: (1 + daily_return * -2).cumprod()
    bear_2x = (1 + daily_return * -2).cumprod()
    # ベア2倍ショート: 初日に空売りした場合のポートフォリオ価値 = 2 - bear_2x
    out["bear_2x_short"] = 2 - bear_2x

    return out


def compute_max_drawdown(series: pd.Series) -> float:
    """
    高値からの最大下落率（%）を返す。

    Args:
        series: 価格時系列（初日=1 に正規化されている想定）

    Returns:
        最大ドローダウン（%）。例: -15.5 は 15.5% 下落。
    """
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    return float(drawdown.min() * 100)


def compute_total_return(series: pd.Series) -> float:
    """
    合計リターン（%）を返す。

    Args:
        series: 価格時系列（初日=1 に正規化されている想定）

    Returns:
        合計リターン（%）。例: 10.5 は 10.5% 上昇。
    """
    if len(series) < 2:
        return 0.0
    return float((series.iloc[-1] / series.iloc[0] - 1) * 100)


def compute_summary(result: pd.DataFrame) -> pd.DataFrame:
    """
    各戦略の合計リターンと最大ドローダウンを集計する。

    Args:
        result: compute_strategies の戻り値

    Returns:
        戦略名、合計リターン(%)、最大ドローダウン(%) の DataFrame
    """
    strategies = [
        ("現物1倍", "spot_1x"),
        ("現物2倍", "spot_2x"),
        ("ブル2倍ロング", "bull_2x"),
        ("ベア2倍ショート", "bear_2x_short"),
    ]
    rows = []
    for name, col in strategies:
        s = result[col]
        rows.append({
            "戦略": name,
            "合計リターン (%)": round(compute_total_return(s), 2),
            "最大ドローダウン (%)": round(compute_max_drawdown(s), 2),
        })
    return pd.DataFrame(rows)


def compute_divergence(result: pd.DataFrame) -> pd.DataFrame:
    """
    現物2倍に対するブル2倍・ベア2倍ショートの乖離（パーセント差）を計算。

    Args:
        result: compute_strategies の戻り値

    Returns:
        Date, bull_excess, bear_excess 列を持つ DataFrame
    """
    out = result[["Date"]].copy()
    spot_2x = result["spot_2x"]
    bull = result["bull_2x"]
    bear = result["bear_2x_short"]

    out["bull_excess"] = (bull - spot_2x) / spot_2x * 100
    out["bear_excess"] = (bear - spot_2x) / spot_2x * 100
    return out
