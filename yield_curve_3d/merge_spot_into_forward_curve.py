"""
Stooq で手動取得した現物価格（gold_spot_stooq.csv / silver_spot_stooq.csv）を
gold_forward_curve.csv / silver_forward_curve.csv の 0.0 列にマージする。
2023年以降の日付のみ反映する。

使い方:
  python merge_spot_into_forward_curve.py
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

GOLD_SPOT = DATA_DIR / "gold_spot_stooq.csv"
SILVER_SPOT = DATA_DIR / "silver_spot_stooq.csv"
GOLD_CURVE = DATA_DIR / "gold_forward_curve.csv"
SILVER_CURVE = DATA_DIR / "silver_forward_curve.csv"
CUTOFF = "2023-01-01"


def _round_sigfig(x: float, sigfig: int = 3) -> float:
    if pd.isna(x):
        return float("nan")
    if x == 0:
        return 0.0
    return float(f"{x:.{sigfig}g}")


def merge_spot_into_curve(curve_path: Path, spot_path: Path, name: str) -> None:
    if not spot_path.is_file():
        print(f"  {name}: 現物CSVが見つかりません: {spot_path}")
        return
    if not curve_path.is_file():
        print(f"  {name}: フォワードカーブが見つかりません: {curve_path}")
        return

    curve = pd.read_csv(curve_path)
    spot = pd.read_csv(spot_path)
    spot["Date"] = pd.to_datetime(spot["Date"]).dt.strftime("%Y-%m-%d")
    spot = spot[spot["Date"] >= CUTOFF][["Date", "Close"]].rename(columns={"Close": "0.0"})
    spot["0.0"] = spot["0.0"].apply(_round_sigfig)

    curve["Date"] = curve["Date"].astype(str)
    merged = curve.merge(spot, on="Date", how="left", suffixes=("", "_spot"))
    if "0.0_spot" in merged.columns:
        merged["0.0"] = merged["0.0_spot"].where(merged["0.0_spot"].notna(), merged["0.0"])
        merged = merged.drop(columns=["0.0_spot"])
    merged = merged[curve.columns]
    merged.to_csv(curve_path, index=False)
    filled = merged["Date"].ge(CUTOFF) & merged["0.0"].notna()
    print(f"  {name}: 0.0 を {filled.sum()} 行に反映しました。")


def main() -> None:
    print("フォワードカーブに現物価格（2023年以降）をマージしています...")
    merge_spot_into_curve(GOLD_CURVE, GOLD_SPOT, "ゴールド")
    merge_spot_into_curve(SILVER_CURVE, SILVER_SPOT, "シルバー")
    print("完了。")


if __name__ == "__main__":
    main()
