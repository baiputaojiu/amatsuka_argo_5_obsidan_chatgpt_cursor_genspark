"""
限月データを yfinance で取得し、既存の silver / gold フォワードカーブにマージする。
あわせて日経（^N225 + NIY=F）を取得し nikkei_forward_curve.csv を上書きする。
- Silver/Gold: 新規日付だけ追加。既存の日付は変更しない（補完・手入力維持）。
- 日経: 全件取得して上書き。
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SINCE_YEAR = 2024


def _merge_curve(existing_path: Path, new_df: pd.DataFrame) -> tuple[pd.DataFrame | None, int]:
    """既存CSVはそのまま維持し、new_df のうち既存に無い日付の行だけを追加する。Returns (merged_df, num_added)."""
    if new_df is None or new_df.empty:
        return None, 0
    new_df = new_df.copy()
    new_df["Date"] = new_df["Date"].astype(str).str.strip()

    if not existing_path.exists():
        out = new_df.sort_values("Date").reset_index(drop=True)
        return out, len(out)

    existing = pd.read_csv(existing_path)
    if existing.empty or "Date" not in existing.columns:
        out = new_df.sort_values("Date").reset_index(drop=True)
        return out, len(out)

    existing["Date"] = existing["Date"].astype(str).str.strip()
    existing_dates = set(existing["Date"])
    # 既存に無い日付の行だけ追加
    append_rows = new_df[~new_df["Date"].isin(existing_dates)]
    num_added = len(append_rows)
    if append_rows.empty:
        return existing.sort_values("Date").reset_index(drop=True), 0

    # 列を揃えて結合（既存の列順を優先）
    append_rows = append_rows.reindex(columns=existing.columns)
    merged = pd.concat([existing, append_rows], ignore_index=True)
    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged, num_added


def run_update() -> tuple[bool, str]:
    """
    silver / gold の限月データを取得し、既存CSVにマージして保存する。
    Returns: (success, message)
    """
    try:
        from fetch_silver_data import _fetch_all as silver_fetch
        from fetch_gold_data import _fetch_all as gold_fetch
        from fetch_nikkei_data import fetch_nikkei_forward_curve as nikkei_fetch
    except ImportError as e:
        return False, f"インポートエラー: {e}"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    silver_path = DATA_DIR / "silver_forward_curve.csv"
    gold_path = DATA_DIR / "gold_forward_curve.csv"
    nikkei_path = DATA_DIR / "nikkei_forward_curve.csv"

    msg_parts = []

    # Silver
    try:
        new_silver = silver_fetch(since_year=SINCE_YEAR)
        if new_silver is not None and not new_silver.empty:
            merged_silver, n_silver = _merge_curve(silver_path, new_silver)
            if merged_silver is not None:
                merged_silver.to_csv(silver_path, index=False)
                msg_parts.append(f"シルバー: 追加 {n_silver} 行（既存は変更なし）")
        else:
            msg_parts.append("シルバー: 取得データなし")
    except Exception as e:
        msg_parts.append(f"シルバー: エラー - {e}")

    # Gold
    try:
        new_gold = gold_fetch(since_year=SINCE_YEAR)
        if new_gold is not None and not new_gold.empty:
            merged_gold, n_gold = _merge_curve(gold_path, new_gold)
            if merged_gold is not None:
                merged_gold.to_csv(gold_path, index=False)
                msg_parts.append(f"ゴールド: 追加 {n_gold} 行（既存は変更なし）")
        else:
            msg_parts.append("ゴールド: 取得データなし")
    except Exception as e:
        msg_parts.append(f"ゴールド: エラー - {e}")

    # Nikkei（全件取得で上書き）
    try:
        new_nikkei = nikkei_fetch(since_year=2023)
        if new_nikkei is not None and not new_nikkei.empty:
            new_nikkei.to_csv(nikkei_path, index=False)
            msg_parts.append(f"日経: 更新 {len(new_nikkei)} 行")
        else:
            msg_parts.append("日経: 取得データなし")
    except Exception as e:
        msg_parts.append(f"日経: エラー - {e}")

    return True, "更新完了 " + " / ".join(msg_parts)


if __name__ == "__main__":
    import sys
    ok, msg = run_update()
    print(msg)
    sys.exit(0 if ok else 1)
