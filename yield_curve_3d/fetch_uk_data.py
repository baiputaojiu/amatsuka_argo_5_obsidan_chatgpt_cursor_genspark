"""
英国国債（Gilt）イールドカーブを BoE / DMO から取得し、
2000年以降を uk_yield_curve.csv に保存する。

利用元:
- Bank of England: https://www.bankofengland.co.uk/statistics/yield-curves
- DMO: https://www.dmo.gov.uk/data/ExportReport?reportCode=D4H

使い方:
  python fetch_uk_data.py
      # 自動ダウンロードを試行（失敗時は手動DL案内を表示）
  python fetch_uk_data.py ダウンロードした.csv
      # 手動DLしたCSVを正規化して data/uk_yield_curve.csv に保存
"""

import sys
from io import StringIO
from pathlib import Path
import zipfile

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_CSV = DATA_DIR / "uk_yield_curve.csv"

# BoE 最新イールドカーブ（ZIP 内に CSV）
BOE_ZIP_URL = "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/latest-yield-curve-data.zip"

# app.py が期待する形式: Date + 残存期間列（例: 5Y, 10Y, 30Y）


def _download_bytes(url: str, timeout: int = 120) -> bytes:
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "*/*",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.read()
    except Exception as e:
        raise RuntimeError(f"ダウンロードに失敗しました: {url}") from e


def _normalize_uk_csv(df: pd.DataFrame, since_year: int = 2000) -> pd.DataFrame:
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


def _fetch_from_boe_zip(since_year: int = 2000) -> pd.DataFrame | None:
    """BoE の ZIP を取得し、イールドカーブ CSV を探して正規化する。"""
    try:
        raw = _download_bytes(BOE_ZIP_URL)
    except Exception:
        return None
    try:
        z = zipfile.ZipFile(BytesIO(raw), "r")
    except Exception:
        return None
    for name in z.namelist():
        if not name.lower().endswith(".csv"):
            continue
        try:
            text = z.read(name).decode("utf-8", errors="replace")
            df = pd.read_csv(StringIO(text))
            if df.empty or len(df.columns) < 2:
                continue
            df = _normalize_uk_csv(df, since_year=since_year)
            if len(df) > 0:
                return df
        except Exception:
            continue
    return None


def main() -> None:
    since_year = 2000
    if len(sys.argv) >= 2:
        path = Path(sys.argv[1]).resolve()
        if not path.exists():
            print(f"ファイルが見つかりません: {path}")
            raise SystemExit(1)
        print(f"読み込み中: {path}")
        df = pd.read_csv(path)
        df = _normalize_uk_csv(df, since_year=since_year)
    else:
        print("英国イールドカーブ（2000年〜）を取得しています...")
        df = _fetch_from_boe_zip(since_year=since_year)
        if df is None or len(df) == 0:
            print(
                "自動ダウンロードに失敗しました。\n"
                "手動で以下からデータを取得し、\n"
                "  python fetch_uk_data.py ダウンロードしたファイル.csv\n"
                "で正規化して保存できます。\n"
                f"  BoE: {BOE_ZIP_URL}\n"
                "  DMO: https://www.dmo.gov.uk/data/ExportReport?reportCode=D4H"
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
