"""
既存の各国イールドカーブCSVを統一形式に変換する。

統一形式:
  - 1行目: ヘッダーのみ（Date + 残存期間を数値年で表した列名）
  - 1列目: Date、値は YYYY-MM-DD
  - 2列目以降: 列名は数値年（0.25, 0.5, 1, 2, 5, 10, 20, 30 など）、値は利回り

使い方:
  python normalize_csv_format.py japan [--in-place]
  python normalize_csv_format.py usa [--in-place]
  python normalize_csv_format.py uk [--in-place]
  python normalize_csv_format.py euro [--in-place]
  python normalize_csv_format.py all [--in-place]

  --in-place: 元ファイルを上書き。省略時は _normalized を付けた別名で保存。
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _parse_japanese_era_date(s: str) -> pd.Timestamp:
    """S49.9.24 や R7.12.30 を西暦 Timestamp に変換。"""
    if not isinstance(s, str) or not s:
        return pd.NaT
    s = s.strip()
    if len(s) < 2:
        return pd.NaT
    era = s[0]
    try:
        rest = s[1:]
        y_str, m_str, d_str = rest.split(".")
        era_year = int(y_str)
        month = int(m_str)
        day = int(d_str)
    except Exception:
        return pd.NaT
    if era == "S":
        year = 1925 + era_year
    elif era == "H":
        year = 1988 + era_year
    elif era == "R":
        year = 2018 + era_year
    else:
        return pd.NaT
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except Exception:
        return pd.NaT


def _usa_col_to_years(col: str) -> float:
    """米国列名（1 Mo, 10 Yr など）を年数に変換。"""
    col = col.strip()
    if "Mo" in col or "Month" in col:
        num = "".join(ch if (ch.isdigit() or ch == ".") else " " for ch in col)
        try:
            return float(num.split()[0]) / 12.0
        except Exception:
            return float("nan")
    if "Yr" in col or "Year" in col or "Years" in col:
        num = "".join(ch if (ch.isdigit() or ch == ".") else " " for ch in col)
        try:
            return float(num.split()[0])
        except Exception:
            return float("nan")
    return float("nan")


def _euro_col_to_years(col: str) -> float:
    """ユーロ列名（3M, 6M, 1Y, 10Y など）を年数に変換。"""
    col = str(col).strip().upper()
    if "M" in col and "Y" not in col:
        num = "".join(ch for ch in col if ch.isdigit() or ch == ".")
        try:
            return float(num or 0) / 12.0
        except ValueError:
            return float("nan")
    if "Y" in col:
        num = "".join(ch for ch in col if ch.isdigit() or ch == ".")
        try:
            return float(num or 0)
        except ValueError:
            return float("nan")
    try:
        return float(col)
    except (ValueError, TypeError):
        return float("nan")


def normalize_japan(in_place: bool) -> Path:
    df = pd.read_csv(DATA_DIR / "japan_yield_curve.csv", header=1)
    df = df.rename(columns={"基準日": "Date"})
    df["Date"] = df["Date"].map(_parse_japanese_era_date)
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    maturity_cols = [c for c in df.columns if c != "Date"]
    rename_map = {}
    for c in maturity_cols:
        num = "".join(ch for ch in c if ch.isdigit())
        if num:
            rename_map[c] = str(int(num)) if float(num) == int(float(num)) else str(float(num))
    df = df.rename(columns=rename_map)
    maturity_cols = [c for c in df.columns if c != "Date"]
    for c in maturity_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    cols_sorted = ["Date"] + sorted([c for c in maturity_cols], key=lambda x: float(x))
    df = df[cols_sorted]
    out = DATA_DIR / "japan_yield_curve.csv" if in_place else DATA_DIR / "japan_yield_curve_normalized.csv"
    df.to_csv(out, index=False)
    return out


def normalize_usa(in_place: bool) -> Path:
    df = pd.read_csv(DATA_DIR / "usa_yield_curve.csv")
    df = df.rename(columns={"Date": "Date"})
    maturity_cols = [c for c in df.columns if c != "Date"]
    rename_map = {}
    for c in maturity_cols:
        y = _usa_col_to_years(c)
        if pd.notna(y) and not (y != y):
            s = str(int(y)) if y == int(y) else str(y)
            rename_map[c] = s
    df = df.rename(columns=rename_map)
    maturity_cols = [c for c in df.columns if c != "Date"]
    for c in maturity_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    cols_sorted = ["Date"] + sorted([c for c in maturity_cols], key=lambda x: float(x))
    df = df[cols_sorted]
    out = DATA_DIR / "usa_yield_curve.csv" if in_place else DATA_DIR / "usa_yield_curve_normalized.csv"
    df.to_csv(out, index=False)
    return out


def normalize_uk(in_place: bool) -> Path:
    df = pd.read_csv(DATA_DIR / "uk_yield_curve.csv")
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    maturity_cols = [c for c in df.columns if c != "Date"]
    for c in maturity_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    try:
        cols_sorted = ["Date"] + sorted([c for c in maturity_cols], key=lambda x: float(x))
    except (ValueError, TypeError):
        cols_sorted = ["Date"] + maturity_cols
    df = df[cols_sorted]
    out = DATA_DIR / "uk_yield_curve.csv" if in_place else DATA_DIR / "uk_yield_curve_normalized.csv"
    df.to_csv(out, index=False)
    return out


def normalize_euro(in_place: bool) -> Path:
    df = pd.read_csv(DATA_DIR / "euro_yield_curve.csv")
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    maturity_cols = [c for c in df.columns if c != "Date"]
    rename_map = {}
    for c in maturity_cols:
        y = _euro_col_to_years(c)
        if pd.notna(y) and y == y:
            s = str(int(y)) if y == int(y) else str(y)
            rename_map[c] = s
    df = df.rename(columns=rename_map)
    maturity_cols = [c for c in df.columns if c != "Date"]
    for c in maturity_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    cols_sorted = ["Date"] + sorted([c for c in maturity_cols], key=lambda x: float(x))
    df = df[cols_sorted]
    out = DATA_DIR / "euro_yield_curve.csv" if in_place else DATA_DIR / "euro_yield_curve_normalized.csv"
    df.to_csv(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Yield curve CSV を統一形式に変換")
    parser.add_argument("country", choices=["japan", "usa", "uk", "euro", "all"], help="変換対象")
    parser.add_argument("--in-place", action="store_true", help="元ファイルを上書き")
    args = parser.parse_args()
    in_place = args.in_place

    runners = {
        "japan": normalize_japan,
        "usa": normalize_usa,
        "uk": normalize_uk,
        "euro": normalize_euro,
    }
    if args.country == "all":
        for name, fn in runners.items():
            try:
                path = fn(in_place)
                print(f"{name}: {path}")
            except FileNotFoundError as e:
                print(f"{name}: スキップ（ファイルなし） {e}")
            except Exception as e:
                print(f"{name}: エラー {e}")
                raise
    else:
        path = runners[args.country](in_place)
        print(f"保存: {path}")


if __name__ == "__main__":
    main()
