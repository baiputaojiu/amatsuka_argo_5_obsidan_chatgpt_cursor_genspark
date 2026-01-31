"""
中国国債イールドカーブを ChinaBond 等から取得し、
2000年以降を china_yield_curve.csv に保存する。

利用元: ChinaBond (中国債券信息網)
https://yield.chinabond.com.cn/cbweb-mn/yc/downYearBzqxList (年別CSV等)

使い方:
  python fetch_china_data.py
      # 自動ダウンロードを試行（接続・形式により失敗する場合あり）
  python fetch_china_data.py ダウンロードした.csv
      # 手動DLしたCSVを正規化して data/china_yield_curve.csv に保存
"""

import sys
from io import StringIO
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_CSV = DATA_DIR / "china_yield_curve.csv"

# app.py が期待する形式: Date + 残存期間列（例: 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y）


def _normalize_china_csv(df: pd.DataFrame, since_year: int = 2000) -> pd.DataFrame:
    """日付列を Date に、残存期間列を数値化。since_year 以降に絞る。"""
    df = df.copy()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"].dt.year >= since_year]
    maturity_cols = [c for c in df.columns if c != "Date"]
    for c in maturity_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("Date", ascending=False)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


def _fetch_auto(since_year: int = 2000) -> pd.DataFrame | None:
    """ChinaBond 等から自動取得を試みる。URL・形式が変わりやすいため失敗しやすい。"""
    # ChinaBond の公的ダウンロードURLは認証やセッションが必要な場合があり、
    # ここでは手動DLを主とする。必要に応じて URL を追加。
    return None


def main() -> None:
    since_year = 2000
    if len(sys.argv) >= 2:
        path = Path(sys.argv[1]).resolve()
        if not path.exists():
            print(f"ファイルが見つかりません: {path}")
            raise SystemExit(1)
        print(f"読み込み中: {path}")
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except (UnicodeDecodeError, Exception):
                continue
        else:
            print("CSV の文字コードを判定できませんでした。")
            raise SystemExit(1)
        df = _normalize_china_csv(df, since_year=since_year)
    else:
        print("中国イールドカーブ（2000年〜）を取得しています...")
        df = _fetch_auto(since_year=since_year)
        if df is None or len(df) == 0:
            print(
                "自動ダウンロードは未実装または失敗しました。\n"
                "手動で ChinaBond 等から CSV をダウンロードし、\n"
                "  python fetch_china_data.py ダウンロードしたファイル.csv\n"
                "で正規化して保存してください。\n"
                "  ChinaBond: https://yield.chinabond.com.cn/cbweb-mn/yc/downYearBzqxList"
            )
            raise SystemExit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    n = len(df)
    if n > 0:
        print(f"保存しました: {OUTPUT_CSV}（{n} 行、{df['Date'].iloc[-1]} 〜 {df['Date'].iloc[0]}）")
    else:
        print("データが 0 行でした。")


if __name__ == "__main__":
    main()
