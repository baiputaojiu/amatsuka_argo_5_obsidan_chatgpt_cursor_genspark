"""
silver_forward_curve.csv を複製し、フロント月(列1)が欠損している行について、
その日の他の限月(2,3,...)のデータのみで線形補間して補完する。
- 他の日のデータは使わない
- 現物(0.0)は使わない
"""

import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
SRC = DATA_DIR / "silver_forward_curve.csv"
DST = DATA_DIR / "silver_forward_curve_filled.csv"


def main():
    df = pd.read_csv(SRC)
    # 補間に使う限月: 2, 3, ..., 24（0.0 と 1 は使わない）
    maturity_cols = [c for c in df.columns if c not in ("Date", "0.0", "1")]
    # 数値列に
    for c in maturity_cols + ["1"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    filled = 0
    for i in range(len(df)):
        if pd.notna(df.loc[i, "1"]):
            continue
        # その行の他限月で有効な (限月, 価格) を集める
        pairs = []
        for c in maturity_cols:
            if c not in df.columns:
                continue
            try:
                m = int(float(c))
            except (ValueError, TypeError):
                continue
            v = df.loc[i, c]
            if pd.notna(v):
                pairs.append((m, float(v)))
        if len(pairs) < 2:
            continue
        pairs.sort(key=lambda x: x[0])
        maturities = np.array([p[0] for p in pairs])
        values = np.array([p[1] for p in pairs])
        # 限月 1 の位置で線形補間（外挿含む）
        val_1 = np.interp(1, maturities, values)
        df.loc[i, "1"] = round(val_1, 3)
        filled += 1

    df.to_csv(DST, index=False)
    print(f"Read: {SRC}")
    print(f"Wrote: {DST} (filled {filled} rows for column 1)")


if __name__ == "__main__":
    main()
