"""
日経平均の現物(^N225)と先物(NIY=F)を yfinance で取得し、
フォワードレート用CSV（Date, 0.0=現物, 1=フロント先物）を出力する。

使い方:
  python fetch_nikkei_data.py
  python fetch_nikkei_data.py --since 2023
"""

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_CSV = DATA_DIR / "nikkei_forward_curve.csv"

SPOT_TICKER = "^N225"
FUTURES_TICKER = "NIY=F"


def _round_sigfig(x: float, sigfig: int = 4) -> float:
    if x is None or (isinstance(x, float) and (pd.isna(x) or x != x)):
        return float("nan")
    if x == 0:
        return 0.0
    return float(f"{x:.{sigfig}g}")


def fetch_nikkei_forward_curve(since_year: int) -> pd.DataFrame | None:
    import yfinance as yf

    start = f"{since_year}-01-01"
    tickers = [SPOT_TICKER, FUTURES_TICKER]

    try:
        df = yf.download(
            tickers,
            start=start,
            end=None,
            progress=False,
            auto_adjust=True,
            group_by="column",
            threads=False,
        )
    except Exception as e:
        print(f"取得エラー: {e}")
        return None

    if df is None or df.empty:
        return None

    df = df.reset_index()
    date_col = df.columns[0]

    # MultiIndex: (Price, Ticker) または 単一ティッカー時の列名
    if isinstance(df.columns, pd.MultiIndex):
        close_spot = None
        close_fut = None
        for col in df.columns:
            if col[0] == "Close":
                if col[1] == SPOT_TICKER:
                    close_spot = df[col].copy()
                elif col[1] == FUTURES_TICKER:
                    close_fut = df[col].copy()
        if close_spot is None or close_fut is None:
            print("Close 列が見つかりません")
            return None
        out = pd.DataFrame({
            "Date": pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d"),
            "0.0": close_spot.values,
            "1": close_fut.values,
        })
    else:
        # 単一ティッカーの場合（想定外だが念のため）
        if "Close" not in df.columns:
            return None
        out = pd.DataFrame({
            "Date": pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d"),
            "0.0": df["Close"].values,
            "1": float("nan"),
        })

    out["0.0"] = pd.to_numeric(out["0.0"], errors="coerce")
    out["1"] = pd.to_numeric(out["1"], errors="coerce")
    out = out.dropna(subset=["0.0", "1"])
    out["0.0"] = out["0.0"].apply(_round_sigfig)
    out["1"] = out["1"].apply(_round_sigfig)
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="日経フォワードカーブ取得 (^N225 + NIY=F)")
    parser.add_argument("--since", type=int, default=2023, help="取得開始年")
    args = parser.parse_args()

    print("日経 フォワードカーブを取得しています...")
    print(f"  現物: {SPOT_TICKER}, 先物: {FUTURES_TICKER}")
    df = fetch_nikkei_forward_curve(since_year=args.since)

    if df is None or len(df) == 0:
        print("取得に失敗しました。yfinance を確認してください。")
        raise SystemExit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    n = len(df)
    print(f"保存: {OUTPUT_CSV}")
    print(f"  行数: {n}, 期間: {df['Date'].iloc[0]} 〜 {df['Date'].iloc[-1]}")


if __name__ == "__main__":
    main()
