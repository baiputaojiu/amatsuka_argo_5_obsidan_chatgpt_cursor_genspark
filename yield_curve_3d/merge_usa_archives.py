"""
Treasury の年別 CSV（1990-1999, 2000-2009, 2010-2019）と
既存 usa_yield_curve.csv を正規化して1つにマージし、data/usa_yield_curve.csv に保存する。

使い方:
  python merge_usa_archives.py
  （ダウンロードした3ファイルのパスはコード内で指定）
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_CSV = DATA_DIR / "usa_yield_curve.csv"

# 想定する列順（app.py と一致）
TARGET_COLS = [
    "Date", "1 Mo", "1.5 Month", "2 Mo", "3 Mo", "4 Mo", "6 Mo",
    "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr",
]


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """日付を YYYY-MM-DD にし、列を TARGET_COLS に揃える（無い列は NaN）。"""
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    out = pd.DataFrame()
    for col in TARGET_COLS:
        if col == "Date":
            out["Date"] = df["Date"].values
        elif col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").values
        else:
            out[col] = float("nan")
    return out[["Date"] + [c for c in TARGET_COLS if c != "Date"]]


def main():
    downloads = [
        Path(r"d:\Downloads\par-yield-curve-rates-1990-1999.csv"),
        Path(r"d:\Downloads\par-yield-curve-rates-2000-2009.csv"),
        Path(r"d:\Downloads\par-yield-curve-rates-2010-2019.csv"),
    ]
    for p in downloads:
        if not p.exists():
            raise FileNotFoundError(f"ファイルがありません: {p}")

    frames = []
    for p in downloads:
        df = pd.read_csv(p)
        frames.append(_normalize_df(df))

    # 既存 usa_yield_curve.csv（先頭行の不具合を補正）
    if OUTPUT_CSV.exists():
        from io import StringIO
        raw = OUTPUT_CSV.read_text(encoding="utf-8", errors="replace")
        first = raw.split("\n")[0]
        if "python" in first.lower():
            first = first.replace("python fetch_usa_data.py", "").strip()
            if not first.startswith("Date"):
                first = "Date,1 Mo,1.5 Month,2 Mo,3 Mo,4 Mo,6 Mo,1 Yr,2 Yr,3 Yr,5 Yr,7 Yr,10 Yr,20 Yr,30 Yr"
            raw = first + "\n" + raw.split("\n", 1)[1]
        df_existing = pd.read_csv(StringIO(raw))
        if "Date" in df_existing.columns and len(df_existing) > 0:
            df_existing = _normalize_df(df_existing)
            frames.append(df_existing)

    merged = pd.concat(frames, ignore_index=True)
    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged = merged.dropna(subset=["Date"])
    merged = merged.drop_duplicates(subset=["Date"], keep="first")
    merged = merged.sort_values("Date", ascending=False)
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    n = len(merged)
    print(f"保存しました: {OUTPUT_CSV}（{n} 行、{merged['Date'].iloc[-1]} 〜 {merged['Date'].iloc[0]}）")


if __name__ == "__main__":
    main()
