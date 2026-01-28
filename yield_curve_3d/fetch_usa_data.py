"""
米国国債イールドカーブ（Par Yield Curve）を財務省アーカイブから取得し、
2000年〜現在まで usa_yield_curve.csv を生成する。

利用元: U.S. Treasury Daily Treasury Par Yield Curve Rate Archives
https://home.treasury.gov/interest-rates-data-csv-archive

使い方:
  python fetch_usa_data.py
      # 2000年〜現在のデータをダウンロードして保存（接続失敗時は手動DL案内を表示）
  python fetch_usa_data.py ダウンロードした.csv
      # 手動DLしたCSVを正規化し、2000年以降だけ保存。既存 usa_yield_curve.csv があれば
      # その中でより新しい日付は残してマージする。
"""

import sys
from io import StringIO
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_CSV = DATA_DIR / "usa_yield_curve.csv"

# 財務省アーカイブ（Par Yield Curve）のURL
# 1990-2022 は一括、2023以降は別ファイルの可能性あり
ARCHIVE_BASE = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rate-archives"

# 取得するアーカイブファイル（年範囲）
# 1990-2022 で 2000 以降を含む。2023+ は別URLで取得を試みる
ARCHIVE_URLS = [
    f"{ARCHIVE_BASE}/par-yield-curve-rates-1990-2022.csv",
    f"{ARCHIVE_BASE}/par-yield-curve-rates-2023-2024.csv",
]

# app.py が期待する列順（既存 usa_yield_curve.csv と同じ）
TARGET_COLS = [
    "Date",
    "1 Mo",
    "1.5 Month",
    "2 Mo",
    "3 Mo",
    "4 Mo",
    "6 Mo",
    "1 Yr",
    "2 Yr",
    "3 Yr",
    "5 Yr",
    "7 Yr",
    "10 Yr",
    "20 Yr",
    "30 Yr",
]


def _download_csv(url: str) -> bytes:
    """CSV をダウンロードする。"""
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/csv,*/*",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as res:
            return res.read()
    except Exception as e:
        raise RuntimeError(f"ダウンロードに失敗しました: {url}") from e


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    財務省CSVの列名を app が期待する形式に正規化する。
    列名は大文字・スペースの違いがあり得る（例: 1 MO → 1 Mo）。
    """
    # 日付列を探す
    date_col = None
    for c in df.columns:
        if c.strip().lower() == "date":
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # 残存期間列: 既存形式に近い名前へマッピング
    # Treasury は "1 Mo", "2 Mo", "3 Mo", "6 Mo", "1 Yr", "2 Yr", ... など
    col_map = {}
    for c in df.columns:
        if c == "Date":
            continue
        key = c.strip()
        # 既に目標名と一致
        if key in TARGET_COLS:
            col_map[c] = key
            continue
        # 数字 + " Mo" / " Month" / " Yr" などを正規化
        key_lower = key.lower()
        if "mo" in key_lower or "month" in key_lower:
            num = "".join(ch for ch in key if ch.isdigit() or ch == ".")
            if num:
                try:
                    f = float(num)
                    if f == 1.5:
                        col_map[c] = "1.5 Month"
                    elif f == 1:
                        col_map[c] = "1 Mo"
                    elif f == 2:
                        col_map[c] = "2 Mo"
                    elif f == 3:
                        col_map[c] = "3 Mo"
                    elif f == 4:
                        col_map[c] = "4 Mo"
                    elif f == 6:
                        col_map[c] = "6 Mo"
                    else:
                        col_map[c] = c
                except ValueError:
                    col_map[c] = c
            else:
                col_map[c] = c
        elif "yr" in key_lower or "year" in key_lower:
            num = "".join(ch for ch in key if ch.isdigit() or ch == ".")
            if num:
                try:
                    f = float(num)
                    col_map[c] = f"{int(f) if f == int(f) else f} Yr"
                    if col_map[c] not in TARGET_COLS:
                        col_map[c] = c
                except ValueError:
                    col_map[c] = c
            else:
                col_map[c] = c
        else:
            col_map[c] = c

    df = df.rename(columns=col_map)

    # 目標列だけに揃え（無い列は NaN）
    out = pd.DataFrame()
    for col in TARGET_COLS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[col] = float("nan")
    out["Date"] = df["Date"].values
    # Date を先頭に
    out = out[["Date"] + [c for c in TARGET_COLS if c != "Date"]]
    return out


def _fetch_and_merge(since_year: int = 2000) -> pd.DataFrame:
    """アーカイブから取得し、since_year 以降のデータをマージする。"""
    frames = []
    for url in ARCHIVE_URLS:
        try:
            raw = _download_csv(url)
            text = raw.decode("utf-8", errors="replace")
            df = pd.read_csv(StringIO(text))
            if df.empty:
                continue
            df = _normalize_columns(df)
            df = df.dropna(subset=["Date"])
            df["_date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df[df["_date"].dt.year >= since_year].drop(columns=["_date"])
            frames.append(df)
        except Exception as e:
            # 2023-2024 など存在しないファイルはスキップ
            print(f"スキップ: {url} - {e}")
            continue

    if not frames:
        print(
            "いずれのアーカイブも取得できませんでした（ネットワーク・ファイアウォール等を確認してください）。\n"
            "手動で以下から CSV をダウンロードし、\n"
            "  python fetch_usa_data.py ダウンロードしたファイル.csv\n"
            " で 2000年〜の形式に正規化して保存できます。\n"
            f"  {ARCHIVE_BASE}\n"
            "  例: par-yield-curve-rates-1990-2022.csv"
        )
        raise SystemExit(1)

    merged = pd.concat(frames, ignore_index=True)
    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged = merged.dropna(subset=["Date"])
    merged = merged.drop_duplicates(subset=["Date"], keep="first")
    merged = merged.sort_values("Date", ascending=False)
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
    return merged


def _load_and_normalize_file(path: Path) -> pd.DataFrame:
    """手動DLしたCSVを読み、正規化する。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    df = pd.read_csv(StringIO(text))
    if df.empty:
        raise ValueError(f"ファイルにデータがありません: {path}")
    return _normalize_columns(df)


def _merge_with_existing(df: pd.DataFrame, since_year: int = 2000) -> pd.DataFrame:
    """既存 usa_yield_curve.csv のうち、df より新しい日付をマージする。"""
    df = df.dropna(subset=["Date"])
    df["_date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["_date"].dt.year >= since_year]

    if not OUTPUT_CSV.exists():
        df = df.drop(columns=["_date"]).sort_values("Date", ascending=False)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return df

    try:
        existing = pd.read_csv(OUTPUT_CSV)
        existing = _normalize_columns(existing)
        existing["_date"] = pd.to_datetime(existing["Date"], errors="coerce")
        if df.empty:
            return existing.drop(columns=["_date"])
        max_new = df["_date"].max()
        old_only = existing[existing["_date"] > max_new].drop(columns=["_date"])
        df = df.drop(columns=["_date"])
        merged = pd.concat([old_only, df], ignore_index=True)
        merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
        merged = merged.drop_duplicates(subset=["Date"], keep="first")
        merged = merged.sort_values("Date", ascending=False)
        merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
        return merged
    except Exception:
        df = df.drop(columns=["_date"]).sort_values("Date", ascending=False)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return df


def main() -> None:
    since_year = 2000
    if len(sys.argv) >= 2:
        # 手動DLしたCSVを正規化して保存（既存があればマージ）
        path = Path(sys.argv[1]).resolve()
        if not path.exists():
            print(f"ファイルが見つかりません: {path}")
            raise SystemExit(1)
        print(f"読み込み中: {path}")
        df = _load_and_normalize_file(path)
        df = _merge_with_existing(df, since_year=since_year)
    else:
        print("米国イールドカーブ（2000年〜）を取得しています...")
        df = _fetch_and_merge(since_year=since_year)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    n = len(df)
    if n > 0:
        print(f"保存しました: {OUTPUT_CSV}（{n} 行、{df['Date'].iloc[-1]} 〜 {df['Date'].iloc[0]}）")
    else:
        print("データが 0 行でした。")


if __name__ == "__main__":
    main()
