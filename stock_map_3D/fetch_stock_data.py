"""
yfinance で株価データを取得し、VWAP 列を付与して data/{ticker}_{interval}.csv に保存する。
VWAP = (High + Low + Close) / 3（各足の典型価格）。
1時間足は最大約60日分の制限あり（yfinance の intraday 制約）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 1h の最大日数（yfinance の intraday 制約）
INTRADAY_MAX_DAYS = 60


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """各足の典型価格 (H+L+C)/3 を VWAP として返す。"""
    return (df["High"] + df["Low"] + df["Close"]) / 3.0


def fetch_and_save(
    ticker: str,
    interval: str,
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    銘柄・interval（1h or 1d）・期間を指定して yfinance で取得し、
    VWAP 列を付与して data/{ticker}_{interval}.csv に保存する。
    interval が 1h の場合は start/end を推奨（最大約60日）。
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if interval == "1h":
        if period and period != "60d" and "d" in period:
            try:
                days = int(period.replace("d", ""))
                if days > INTRADAY_MAX_DAYS:
                    print(f"警告: 1時間足は最大約{INTRADAY_MAX_DAYS}日分です。{period} は {INTRADAY_MAX_DAYS}d に制限します。")
                    period = f"{INTRADAY_MAX_DAYS}d"
            except ValueError:
                period = f"{INTRADAY_MAX_DAYS}d"
        if not start and not end and not period:
            period = f"{INTRADAY_MAX_DAYS}d"

    kwargs = {"interval": interval, "progress": False}
    if start:
        kwargs["start"] = start
    if end:
        kwargs["end"] = end
    if period and not start and not end:
        kwargs["period"] = period

    df = yf.download(ticker, **kwargs, auto_adjust=False, group_by="ticker", threads=False)
    if df.empty:
        raise ValueError(f"データが取得できませんでした: {ticker} interval={interval}")

    # 単一ティッカー時は列が MultiIndex の場合があるのでフラット化
    if isinstance(df.columns, pd.MultiIndex):
        df = df[ticker].copy() if ticker in df.columns.get_level_values(0) else df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["VWAP"] = compute_vwap(df)
    out_path = DATA_DIR / f"{ticker}_{interval}.csv"
    df.to_csv(out_path, index=False)
    print(f"保存しました: {out_path} ({len(df)} 行)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="株価データを yfinance で取得し CSV に保存")
    parser.add_argument("ticker", help="銘柄コード（例: AAPL, 7203.T）")
    parser.add_argument(
        "interval",
        choices=["1h", "1d"],
        help="時間足: 1h または 1d（1h は最大約60日）",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--period", "-p", help="期間（例: 5d, 1mo, 1y, 5y）。1h の場合は 60d まで")
    group.add_argument("--start", "-s", help="開始日 YYYY-MM-DD")
    group.add_argument("--end", "-e", help="終了日 YYYY-MM-DD")
    args = parser.parse_args()

    if args.interval == "1h" and not args.period and not args.start and not args.end:
        args.period = "60d"
    fetch_and_save(
        ticker=args.ticker,
        interval=args.interval,
        period=args.period,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
