"""
ユーロ圏（ECB AAA 国債）イールドカーブを ECB Data Portal から取得し、
2004年9月〜を euro_yield_curve.csv に保存する。

利用元: ECB Euro area yield curves - Par curve (AAA-rated government bonds)
https://data.ecb.europa.eu/data/data-categories/financial-markets-and-interest-rates/euro-area-yield-curves/aaa-rated-government-bonds-yield-curve/par-curve

データは 2004年9月〜 のみ公表。

使い方:
  python fetch_ecb_data.py
      # 自動ダウンロードを試行（失敗時は手動DL案内を表示）
  python fetch_ecb_data.py ダウンロードした.csv
      # 手動DLしたCSVを正規化して data/euro_yield_curve.csv に保存
"""

import sys
from io import StringIO
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_CSV = DATA_DIR / "euro_yield_curve.csv"

# ECB SDMX API: YC = Yield curves, B.U2.EUR.4F.G_N_A = AAA, SV_C_YM = Svensson, SR_* = spot rate
ECB_API_BASE = "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM"
# 残存期間コード → 列表示名（app.py の _col_to_years_generic で解釈可能）
ECB_MATURITY_CODES = [
    ("SR_3M", "3M"),
    ("SR_6M", "6M"),
    ("SR_1Y", "1Y"),
    ("SR_2Y", "2Y"),
    ("SR_3Y", "3Y"),
    ("SR_4Y", "4Y"),
    ("SR_5Y", "5Y"),
    ("SR_6Y", "6Y"),
    ("SR_7Y", "7Y"),
    ("SR_8Y", "8Y"),
    ("SR_9Y", "9Y"),
    ("SR_10Y", "10Y"),
    ("SR_15Y", "15Y"),
    ("SR_20Y", "20Y"),
    ("SR_25Y", "25Y"),
    ("SR_30Y", "30Y"),
]


def _download_csv(url: str, timeout: int = 120) -> bytes:
    try:
        import ssl
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/csv,*/*",
            },
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as res:
            return res.read()
    except Exception as e:
        raise RuntimeError(f"ダウンロードに失敗しました: {url}") from e


def _normalize_ecb_csv(df: pd.DataFrame, since_year: int = 2004) -> pd.DataFrame:
    """日付列を Date に、残存期間列を数値化。since_year 以降に絞る。"""
    df = df.copy()
    # ECB CSV は列が TIME_PERIOD, 残存期間名 等になり得る
    date_col = None
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower() or c == "Observed period" or c == "TIME_PERIOD":
            date_col = c
            break
    if date_col is None:
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


def _fetch_one_series(code: str, since_year: int) -> pd.DataFrame | None:
    """1残存期間のシリーズを取得し、Date と OBS_VALUE の DataFrame を返す。"""
    url = f"{ECB_API_BASE}.{code}?startPeriod={since_year}-01-01&format=csvdata"
    try:
        raw = _download_csv(url, timeout=90)
        text = raw.decode("utf-8", errors="replace")
        df = pd.read_csv(StringIO(text))
        if df.empty or "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
            return None
        df = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
        df = df.rename(columns={"TIME_PERIOD": "Date", "OBS_VALUE": code})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df[code] = pd.to_numeric(df[code], errors="coerce")
        return df
    except Exception:
        return None


def _fetch_from_ecb(since_year: int = 2004) -> pd.DataFrame | None:
    """ECB API から各残存期間を取得し、Date × 残存期間の表にまとめる。"""
    merged = None
    for api_code, col_name in ECB_MATURITY_CODES:
        df = _fetch_one_series(api_code, since_year)
        if df is None or len(df) == 0:
            continue
        df = df.rename(columns={api_code: col_name})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df[["Date", col_name]], on="Date", how="outer")
    if merged is None or len(merged) == 0:
        return None
    return _normalize_ecb_csv(merged, since_year=since_year)


def main() -> None:
    since_year = 2004
    if len(sys.argv) >= 2:
        path = Path(sys.argv[1]).resolve()
        if not path.exists():
            print(f"ファイルが見つかりません: {path}")
            raise SystemExit(1)
        print(f"読み込み中: {path}")
        df = pd.read_csv(path)
        df = _normalize_ecb_csv(df, since_year=since_year)
    else:
        print("ユーロ圏イールドカーブ（2004年9月〜）を取得しています...")
        df = _fetch_from_ecb(since_year=since_year)
        if df is None or len(df) == 0:
            print(
                "自動ダウンロードに失敗しました。\n"
                "手動で以下から CSV をダウンロードし、\n"
                "  python fetch_ecb_data.py ダウンロードしたファイル.csv\n"
                "で正規化して保存できます。\n"
                "  ECB Data Portal (Par curve, AAA):\n"
                "  https://data.ecb.europa.eu/data/data-categories/financial-markets-and-interest-rates/"
                "euro-area-yield-curves/aaa-rated-government-bonds-yield-curve/par-curve\n"
                "※ データは 2004年9月〜 のみです。"
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
