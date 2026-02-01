"""
ゴールド先物フォワードカーブを取得する。

- 0.0: 現物価格（空。Stooq で手動取得し merge_spot_into_forward_curve.py でマージ）
- 1: 最も近い限月の価格
- 2: 2番目に近い限月の価格
- 3, 4, ... : 順に次の限月の価格

各列は異なる限月なので、同じ値が横に並ぶことはない。
ゴールドは偶数月が主要限月のため偶数月のティッカーのみ使用。

使い方:
  python fetch_gold_data.py
  python fetch_gold_data.py --since 2024
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_CSV = DATA_DIR / "gold_forward_curve.csv"

CONTRACT_PREFIX = "GC"

MONTH_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
CODE_TO_LETTER = {v: k for k, v in MONTH_CODES.items()}
EVEN_MONTHS = [2, 4, 6, 8, 10, 12]

MAX_CONTRACTS = 24


def _round_sigfig(x: float, sigfig: int = 3) -> float:
    if x is None or (isinstance(x, float) and (np.isnan(x) or x != x)):
        return np.nan
    if x == 0:
        return 0.0
    return float(f"{x:.{sigfig}g}")


def _generate_contract_tickers(start_date: date, end_date: date) -> list[str]:
    """ゴールド先物ティッカー（偶数月のみ）"""
    tickers = []
    end_year = end_date.year + 3
    for y in range(start_date.year, end_year + 1):
        for m in EVEN_MONTHS:
            deliv = date(y, m, 1)
            if deliv >= start_date:
                letter = CODE_TO_LETTER[m]
                yy = str(y)[-2:]
                t = f"{CONTRACT_PREFIX}{letter}{yy}.CMX"
                if t not in tickers:
                    tickers.append(t)
    return tickers


def _get_delivery_date(ticker: str) -> date | None:
    if not ticker.startswith(CONTRACT_PREFIX) or ".CMX" not in ticker:
        return None
    try:
        body = ticker.replace(".CMX", "")[len(CONTRACT_PREFIX):]
        letter = body[0]
        yy = int("20" + body[1:3])
        m = MONTH_CODES.get(letter)
        if m is None:
            return None
        if m == 12:
            return date(yy, 12, 31)
        return date(yy, m + 1, 1) - timedelta(days=1)
    except Exception:
        return None


def _years_to_delivery(obs_date: date, delivery_date: date) -> float:
    delta = (delivery_date - obs_date).days
    return max(0.0, delta / 365.25)


def _fetch_futures_prices(since_year: int) -> pd.DataFrame:
    import yfinance as yf

    start_date = date(since_year, 1, 1)
    end_date = date.today()
    tickers = _generate_contract_tickers(start_date, end_date)

    print(f"先物価格を取得中: {len(tickers)} 限月")
    try:
        df = yf.download(
            tickers,
            start=f"{since_year}-01-01",
            end=None,
            progress=False,
            auto_adjust=True,
            group_by="column",
            threads=True,
        )
    except Exception as e:
        print(f"先物価格の取得エラー: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    records = []
    df = df.reset_index()
    date_col = df.columns[0]

    if isinstance(df.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                close_series = df["Close"][ticker]
            except (KeyError, TypeError):
                continue
            sub = pd.DataFrame({"Date": df[date_col], "Price": close_series})
            sub["Ticker"] = ticker
            sub["Date"] = pd.to_datetime(sub["Date"]).dt.date
            sub = sub.dropna(subset=["Date", "Price"])
            if len(sub) > 0:
                records.append(sub)
    else:
        price_col = "Close" if "Close" in df.columns else None
        if price_col:
            sub = df[[date_col, price_col]].copy()
            sub = sub.rename(columns={date_col: "Date", price_col: "Price"})
            sub["Ticker"] = tickers[0] if tickers else ""
            sub["Date"] = pd.to_datetime(sub["Date"]).dt.date
            sub = sub.dropna(subset=["Date", "Price"])
            if len(sub) > 0:
                records.append(sub)

    if not records:
        return pd.DataFrame()

    combined = pd.concat(records, ignore_index=True)

    def _calc_years(row):
        dd = _get_delivery_date(row["Ticker"])
        if dd is None:
            return np.nan
        return _years_to_delivery(row["Date"], dd)

    combined["YearsToDelivery"] = combined.apply(_calc_years, axis=1)
    combined = combined.dropna(subset=["YearsToDelivery"])
    combined = combined[combined["YearsToDelivery"] >= 0.01]

    result_rows = []
    for dt, grp in combined.groupby("Date"):
        sorted_grp = grp.sort_values("YearsToDelivery").reset_index(drop=True)
        prices = sorted_grp["Price"].tolist()

        row = {"Date": dt}
        row["0.0"] = np.nan
        for i, price in enumerate(prices[:MAX_CONTRACTS]):
            row[str(i + 1)] = _round_sigfig(float(price))
        result_rows.append(row)

    out = pd.DataFrame(result_rows)

    for i in range(1, MAX_CONTRACTS + 1):
        col = str(i)
        if col not in out.columns:
            out[col] = np.nan

    return out


def _fetch_all(since_year: int) -> pd.DataFrame | None:
    """先物のみ取得。0.0 列は空（Stooq 手動取得後に merge_spot_into_forward_curve.py でマージ）"""
    futures_df = _fetch_futures_prices(since_year)
    if futures_df.empty:
        return None

    out = futures_df.copy()
    out = out.sort_values("Date", ascending=True)
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    cols = ["Date", "0.0"] + [str(i) for i in range(1, MAX_CONTRACTS + 1)]
    out = out[[c for c in cols if c in out.columns]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="ゴールド先物フォワードカーブ取得")
    parser.add_argument("--since", type=int, default=2024, help="取得開始年")
    args = parser.parse_args()
    since_year = args.since

    print("ゴールド フォワードカーブを取得しています...")
    df = _fetch_all(since_year=since_year)

    if df is None or len(df) == 0:
        print(
            "自動取得に失敗しました。\n"
            "  pip install yfinance で yfinance をインストールしてください。"
        )
        raise SystemExit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    n = len(df)
    cols = [c for c in df.columns if c != "Date"]
    print(f"保存: {OUTPUT_CSV}")
    print(f"  行数: {n}, 列数: {len(cols)}")
    if n > 0:
        print(f"  期間: {df['Date'].iloc[0]} 〜 {df['Date'].iloc[-1]}")


if __name__ == "__main__":
    main()
