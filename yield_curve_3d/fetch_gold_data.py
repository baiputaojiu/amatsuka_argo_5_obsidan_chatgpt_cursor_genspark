"""
ゴールド先物（CME GC）複数限月を取得し、
本格的なフォワードカーブ用の gold_forward_curve.csv を生成する。

データソース: Yahoo Finance (GC{月}{年}.CMX)
複数限月（フロント〜約2年先）を取得し、各標準満期には最も近い限月の価格をそのまま割り当てて統一形式で出力（補間なし）。

使い方:
  python fetch_gold_data.py
      # 自動取得（直近約2年分 + 複数限月）
  python fetch_gold_data.py --since 2020
      # 開始年を指定
  python fetch_gold_data.py 手動DL.csv
      # 統一形式（Date + 数値年列）のCSVを正規化して保存
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_CSV = DATA_DIR / "gold_forward_curve.csv"

# CME 月コード
MONTH_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
MONTH_LETTERS = list(MONTH_CODES.keys())

# 出力する標準満期（年）
TARGET_MATURITIES = [0.083, 0.25, 0.5, 1.0, 2.0]


def _generate_contract_tickers(start_date: date, end_date: date) -> list[str]:
    """取得対象の限月ティッカーを生成。GC=F（フロント月）を先頭に、個別限月を追加。"""
    tickers = ["GC=F"]  # フロント月は常に取得（短満期用）
    end_year = end_date.year + 2
    for y in range(start_date.year, end_year + 1):
        for m_letter in MONTH_LETTERS:
            m = MONTH_CODES[m_letter]
            deliv = date(y, m, 1)
            if deliv >= start_date:
                yy = str(y)[-2:]
                t = f"GC{m_letter}{yy}.CMX"
                if t not in tickers:
                    tickers.append(t)
    return tickers


def _get_delivery_date(ticker: str) -> date | None:
    """ティッカーから限月の満期日（月の最終日）を返す。GC=F は None（別扱い）。"""
    if ticker == "GC=F":
        return None
    # GCJ25.CMX → April 2025
    if not ticker.startswith("GC") or ".CMX" not in ticker:
        return None
    try:
        body = ticker.replace(".CMX", "")[2:]
        m_letter = body[0]
        yy = int("20" + body[1:3])
        m = MONTH_CODES.get(m_letter)
        if m is None:
            return None
        # 月の最終日
        if m == 12:
            return date(yy, 12, 31)
        return date(yy, m + 1, 1) - __import__("datetime").timedelta(days=1)
    except Exception:
        return None


def _years_to_delivery(obs_date: date, delivery_date: date) -> float:
    """観測日から満期までの年数。"""
    delta = (delivery_date - obs_date).days
    return max(0.0, delta / 365.25)


def _map_to_maturities(years: np.ndarray, prices: np.ndarray, targets: list[float]) -> list[float]:
    """targets の各満期について、最も近い限月の価格をそのまま返す。補間は行わない。"""
    if len(years) == 0 or len(prices) == 0:
        return [np.nan] * len(targets)
    result = []
    for t in targets:
        idx = np.argmin(np.abs(years - t))
        result.append(float(prices[idx]))
    return result


def _fetch_multiple_contracts(since_year: int) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance をインストールしてください: pip install yfinance")
        return None

    start_date = date(since_year, 1, 1)
    end_date = date.today()
    tickers = _generate_contract_tickers(start_date, end_date)
    # 重複削除・順序維持
    seen = set()
    unique_tickers = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)

    print(f"取得中: {len(unique_tickers)} 限月 ({unique_tickers[0]} 〜 {unique_tickers[-1]})")
    try:
        df = yf.download(
            unique_tickers,
            start=f"{since_year}-01-01",
            end=None,
            progress=False,
            auto_adjust=True,
            group_by="column",
            threads=True,
        )
    except Exception as e:
        print(f"ダウンロードエラー: {e}")
        return None

    if df is None or df.empty:
        return None

    # yfinance: group_by='column' → (PriceType, Ticker)、単一ティッカーなら通常列
    records = []
    df = df.reset_index()
    date_col = df.columns[0]

    if isinstance(df.columns, pd.MultiIndex):
        for ticker in unique_tickers:
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
        price_col = "Close" if "Close" in df.columns else [c for c in df.columns if c != date_col][0]
        sub = df[[date_col, price_col]].copy()
        sub = sub.rename(columns={price_col: "Price"})
        sub["Ticker"] = unique_tickers[0] if unique_tickers else "GC=F"
        sub["Date"] = pd.to_datetime(sub["Date"]).dt.date
        sub = sub.dropna(subset=["Date", "Price"])
        if len(sub) > 0:
            records.append(sub)

    if not records:
        return None
    combined = pd.concat(records, ignore_index=True)

    # 各 (Date, Ticker) について満期年数と価格を計算
    def _calc_years(row):
        dd = _get_delivery_date(row["Ticker"])
        if dd is None:
            return 0.083  # GC=F フロント月 ≒ 1ヶ月
        return _years_to_delivery(row["Date"], dd)

    combined["YearsToDelivery"] = combined.apply(_calc_years, axis=1)
    combined = combined[combined["YearsToDelivery"] >= 0.01]  # 満期過ぎは除外

    # 日付ごとに (years, prices) を集約し、各標準満期に最も近い限月の価格を割り当て
    result_rows = []
    for dt, grp in combined.groupby("Date"):
        years = grp["YearsToDelivery"].to_numpy()
        prices = grp["Price"].to_numpy()
        mapped = _map_to_maturities(years, prices, TARGET_MATURITIES)
        row = {"Date": dt} | {str(t): v for t, v in zip(TARGET_MATURITIES, mapped)}
        result_rows.append(row)

    out = pd.DataFrame(result_rows)
    out = out.sort_values("Date", ascending=True)
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="ゴールド先物フォワードカーブ取得")
    parser.add_argument("--since", type=int, default=2023, help="取得開始年（デフォルト: 2023）")
    parser.add_argument("file", nargs="?", help="手動DLのCSV（指定時は正規化のみ）")
    args = parser.parse_args()
    since_year = args.since

    if args.file:
        path = Path(args.file).resolve()
        if not path.exists():
            print(f"ファイルが見つかりません: {path}")
            raise SystemExit(1)
        print(f"読み込み中: {path}")
        df = pd.read_csv(path)
        df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df[df["Date"].dt.year >= since_year]
        df = df.sort_values("Date", ascending=True)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    else:
        print("ゴールド先物フォワードカーブ（複数限月）を取得しています...")
        df = _fetch_multiple_contracts(since_year=since_year)
        if df is None or len(df) == 0:
            print(
                "自動取得に失敗しました。\n"
                "  pip install yfinance で yfinance をインストールしてください。\n"
                "手動DLの場合: python fetch_gold_data.py ファイル.csv"
            )
            raise SystemExit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    n = len(df)
    cols = [c for c in df.columns if c != "Date"]
    if n > 0:
        print(f"保存: {OUTPUT_CSV}（{n} 行、満期 {len(cols)} 点、{df['Date'].iloc[0]} 〜 {df['Date'].iloc[-1]}）")
    else:
        print("データが 0 行でした。")


if __name__ == "__main__":
    main()
