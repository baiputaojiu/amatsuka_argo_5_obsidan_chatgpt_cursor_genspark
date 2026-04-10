"""
日経平均 月末・月初変動率ツール

過去10年間の各月について、
「月末3日間の最後の取引日」から「翌月月初3日間の最初の取引日」までの
変動率（%）を計算し、リスト表示・CSV・画像で保存する。
"""

from __future__ import annotations

import argparse
import calendar
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
TICKER = "^N225"
DEBUG_LOG = BASE_DIR.parent / "debug-d77705.log"

def _debug_log(msg: str, data: dict, hypothesis_id: str) -> None:
    import json
    payload = {"sessionId": "d77705", "runId": "run1", "hypothesisId": hypothesis_id, "location": "app.py:fetch_nikkei", "message": msg, "data": data, "timestamp": __import__("time").time() * 1000}
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def fetch_nikkei(years: int = 10) -> pd.DataFrame | None:
    """
    yfinance で日経平均（^N225）を過去 years 年分取得する。

    Returns:
        Date, Close 列を持つ DataFrame。失敗時は None。
    """
    import yfinance as yf

    end_d = date.today()
    start_d = end_d - timedelta(days=years * 365)
    start_str = start_d.isoformat()
    end_str = end_d.isoformat()

    # #region agent log
    _debug_log("before fetch", {"start": start_str, "end": end_str}, "H1")
    # #endregion
    try:
        # #region agent log
        _debug_log("before fetch with session", {}, "H6")
        # #endregion
        from curl_cffi import requests as curl_requests
        session = curl_requests.Session(impersonate="chrome")
        # download() に session を渡す経路（Ticker().history() は ^N225 で NoneType が出るため）
        df = yf.download(
            TICKER,
            start=start_str,
            end=end_str,
            progress=False,
            auto_adjust=True,
            threads=False,
            session=session,
        )
        # #region agent log
        _debug_log("after download", {}, "H8")
        # #endregion
    except Exception as e:
        _debug_log("exception in try", {"type": type(e).__name__, "repr": str(e)[:200]}, "H5")
        print(f"取得エラー: {e}")
        return None

    # #region agent log
    _debug_log("after fetch", {"type": type(df).__name__, "is_None": df is None, "is_DataFrame": isinstance(df, pd.DataFrame) if df is not None else False, "empty": df.empty if df is not None and isinstance(df, pd.DataFrame) else "n/a", "columns_type": type(df.columns).__name__ if df is not None else "n/a", "columns_repr": str(df.columns)[:300] if df is not None else "n/a"}, "H2")
    # #endregion
    if df is None or (not isinstance(df, pd.DataFrame)) or df.empty:
        return None

    # #region agent log
    _debug_log("before copy", {}, "H4")
    # #endregion
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # #region agent log
        _debug_log("before get_level_values", {"columns_repr": str(df.columns)[:300]}, "H3")
        # #endregion
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    date_col = df.columns[0]
    if "Close" not in df.columns:
        return None

    result = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col]).dt.normalize(),
        "Close": pd.to_numeric(df["Close"], errors="coerce"),
    })
    result = result.dropna(subset=["Date", "Close"])
    result = result.sort_values("Date").reset_index(drop=True)
    return result


def last_trading_day_in_last_3_days(df: pd.DataFrame, year: int, month: int) -> tuple[pd.Timestamp | None, float | None]:
    """
    指定月の「月末3日間」に含まれる取引日のうち、最後の日の Date と Close を返す。
    月末3日間 = その月の (末日-2) 〜 末日（存在する日付のみ）。
    """
    _, last_day = calendar.monthrange(year, month)
    day_start = max(1, last_day - 2)
    period_start = pd.Timestamp(year=year, month=month, day=day_start)
    period_end = pd.Timestamp(year=year, month=month, day=last_day)

    in_period = (df["Date"] >= period_start) & (df["Date"] <= period_end)
    subset = df.loc[in_period]
    if subset.empty:
        return None, None
    row = subset.iloc[-1]
    return row["Date"], float(row["Close"])


def first_trading_day_in_first_3_days(df: pd.DataFrame, year: int, month: int) -> tuple[pd.Timestamp | None, float | None]:
    """
    指定月の「月初3日間」（1〜3日）に含まれる取引日のうち、最初の日の Date と Close を返す。
    """
    period_start = pd.Timestamp(year=year, month=month, day=1)
    period_end = pd.Timestamp(year=year, month=month, day=3)

    in_period = (df["Date"] >= period_start) & (df["Date"] <= period_end)
    subset = df.loc[in_period]
    if subset.empty:
        return None, None
    row = subset.iloc[0]
    return row["Date"], float(row["Close"])


def compute_month_turn_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    各月について「月末3日間の最後の取引日 → 翌月月初3日間の最初の取引日」の変動率を計算する。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    min_date = df["Date"].min()
    max_date = df["Date"].max()
    min_ym = (min_date.year, min_date.month)
    max_ym = (max_date.year, max_date.month)

    rows = []
    y, m = min_ym[0], min_ym[1]
    while (y, m) <= max_ym:
        end_d, close_end = last_trading_day_in_last_3_days(df, y, m)
        # 翌月
        if m == 12:
            next_y, next_m = y + 1, 1
        else:
            next_y, next_m = y, m + 1
        start_d, close_start = first_trading_day_in_first_3_days(df, next_y, next_m)

        if end_d is not None and start_d is not None and close_end is not None and close_start is not None and close_end != 0:
            change_pct = (close_start - close_end) / close_end * 100.0
            year_month = f"{y}-{m:02d}"
            rows.append({
                "year_month": year_month,
                "last_date_end_of_month": end_d.strftime("%Y-%m-%d"),
                "first_date_start_of_month": start_d.strftime("%Y-%m-%d"),
                "close_end": round(close_end, 2),
                "close_start": round(close_start, 2),
                "change_pct": round(change_pct, 4),
            })
        y, m = next_y, next_m
        if y > max_ym[0] or (y == max_ym[0] and m > max_ym[1]):
            break

    return pd.DataFrame(rows)


def print_list(result: pd.DataFrame) -> None:
    """変動率一覧を標準出力に表示する。"""
    if result.empty:
        print("データがありません。")
        return
    print("\n--- 日経平均 月末→月初 変動率（%）---\n")
    for _, row in result.iterrows():
        print(
            f"  {row['year_month']}  "
            f"月末: {row['last_date_end_of_month']} (終値 {row['close_end']})  →  "
            f"月初: {row['first_date_start_of_month']} (終値 {row['close_start']})  "
            f"変動: {row['change_pct']:+.2f}%"
        )
    print()


def save_csv(result: pd.DataFrame, out_dir: Path) -> Path:
    """CSV を output に保存する。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "nikkei_month_turn.csv"
    result.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_chart(result: pd.DataFrame, out_dir: Path) -> Path:
    """変動率の棒グラフを PNG で保存する。"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if result.empty:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "nikkei_month_turn.png"

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(result))
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in result["change_pct"]]
    ax.bar(x, result["change_pct"], color=colors, edgecolor="gray", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(result["year_month"], rotation=45, ha="right")
    ax.set_ylabel("変動率（%）")
    ax.set_xlabel("年月")
    ax.set_title("日経平均 月末3日間→月初3日間 変動率（過去10年）")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="日経平均 月末・月初変動率（yfinance）")
    parser.add_argument("--years", type=int, default=10, help="取得する年数（デフォルト: 10）")
    args = parser.parse_args()

    print(f"日経平均（{TICKER}）を過去 {args.years} 年分取得しています...")
    df = fetch_nikkei(years=args.years)
    if df is None or df.empty:
        print("データの取得に失敗しました。")
        raise SystemExit(1)

    print(f"  取得件数: {len(df)} 件, 期間: {df['Date'].min().date()} 〜 {df['Date'].max().date()}\n")
    result = compute_month_turn_changes(df)
    if result.empty:
        print("変動率を計算できる月がありませんでした。")
        raise SystemExit(1)

    print_list(result)
    csv_path = save_csv(result, OUTPUT_DIR)
    print(f"CSV 保存: {csv_path}")
    img_path = save_chart(result, OUTPUT_DIR)
    if img_path:
        print(f"画像保存: {img_path}")


if __name__ == "__main__":
    main()
