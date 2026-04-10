"""
silver_forward_rate_1m.csv を 2025年4月〜現在まで拡張する。

silver_forward_curve.csv の 0.083（約1ヶ月満期）を 1M としてマージし、
Rate_1M = (1M - Spot) / Spot * 100 で補完する。
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RATE_CSV = DATA_DIR / "silver_forward_rate_1m.csv"
CURVE_CSV = DATA_DIR / "silver_forward_curve.csv"
CUTOFF_DATE = "2025-04-01"


def main() -> None:
    if not RATE_CSV.exists():
        print(f"エラー: {RATE_CSV} が見つかりません")
        return
    if not CURVE_CSV.exists():
        print(f"エラー: {CURVE_CSV} が見つかりません")
        return

    df_rate = pd.read_csv(RATE_CSV)
    df_curve = pd.read_csv(CURVE_CSV)

    df_rate["Date"] = pd.to_datetime(df_rate["Date"], errors="coerce")
    df_curve["Date"] = pd.to_datetime(df_curve["Date"], errors="coerce")

    # 0.083 列を 1M として取得
    if "0.083" not in df_curve.columns:
        print("エラー: silver_forward_curve.csv に 0.083 列がありません")
        return

    curve_1m = df_curve[["Date", "0.083"]].rename(columns={"0.083": "1M_curve"})
    df_merged = df_rate.merge(curve_1m, on="Date", how="left")

    # 1M が空で 1M_curve がある場合に補完
    mask_fill_1m = df_merged["1M"].isna() & df_merged["1M_curve"].notna()
    df_merged.loc[mask_fill_1m, "1M"] = df_merged.loc[mask_fill_1m, "1M_curve"]

    # 2025年4月以降に限定して補完（ユーザー要望）
    mask_april = df_merged["Date"] >= pd.Timestamp(CUTOFF_DATE)
    mask_fill = mask_april & (df_merged["Rate_1M"].isna() | df_merged["1M"].isna())
    mask_fill = mask_fill & df_merged["Spot"].notna() & df_merged["1M"].notna()

    # Rate_1M を計算
    df_merged.loc[mask_fill, "Rate_1M"] = (
        (df_merged.loc[mask_fill, "1M"] - df_merged.loc[mask_fill, "Spot"])
        / df_merged.loc[mask_fill, "Spot"]
        * 100
    )

    # 既存の Rate_1M が空だが 1M と Spot がある場合も計算
    mask_calc = df_merged["Rate_1M"].isna() & df_merged["1M"].notna() & df_merged["Spot"].notna()
    df_merged.loc[mask_calc, "Rate_1M"] = (
        (df_merged.loc[mask_calc, "1M"] - df_merged.loc[mask_calc, "Spot"])
        / df_merged.loc[mask_calc, "Spot"]
        * 100
    )

    out = df_rate.copy()
    out["1M"] = df_merged["1M"]
    out["Rate_1M"] = df_merged["Rate_1M"]

    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(RATE_CSV, index=False)

    # 補完した行数をレポート
    n_filled = mask_fill.sum() + mask_calc.sum()
    n_april = mask_april.sum()
    n_with_rate = (out["Date"] >= CUTOFF_DATE) & out["Rate_1M"].notna()
    print(f"更新完了: {RATE_CSV}")
    print(f"  2025年4月以降の行数: {n_april}")
    print(f"  Rate_1M を補完した行数: {int(n_filled)}")
    print(f"  2025年4月以降で Rate_1M あり: {int(n_with_rate.sum())}")


if __name__ == "__main__":
    main()
