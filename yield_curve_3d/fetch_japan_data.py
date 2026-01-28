"""
日本の国債金利データを財務省から取得し、japan_yield_curve.csv を更新する。

利用元: 財務省 国債金利情報
https://www.mof.go.jp/jgbs/reference/interest_rate/index.htm

使い方:
  python fetch_japan_data.py              # 財務省からダウンロードして保存
  python fetch_japan_data.py ファイル.csv  # 手動DLしたCSVを正規化して保存
"""

import sys
from io import StringIO
from pathlib import Path

import pandas as pd

# 財務省 過去金利データ（昭和49年〜）
MOF_CSV_URL = "https://www.mof.go.jp/jgbs/reference/interest_rate/data/jgbcm_all.csv"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_CSV = DATA_DIR / "japan_yield_curve.csv"

TITLE_ROW = "国債金利情報,,,,,,,,,,,,,,,(単位 : %)"


def _download_csv() -> bytes:
    """財務省のCSVをダウンロードする。"""
    try:
        import urllib.request
        req = urllib.request.Request(
            MOF_CSV_URL,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/csv,*/*",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as res:
            return res.read()
    except Exception as e:
        raise RuntimeError(
            f"ダウンロードに失敗しました: {MOF_CSV_URL}\n"
            "手動で上記URLから jgbcm_all.csv を保存し、\n"
            "  python fetch_japan_data.py 保存したファイル.csv\n"
            " で正規化して取り込めます。"
        ) from e


def _parse_and_normalize(csv_bytes: bytes) -> pd.DataFrame:
    """CSV を読み込み、app.py が期待する列名・順序に正規化する。"""
    for encoding in ("utf-8", "utf-8-sig", "cp932", "shift_jis"):
        try:
            text = csv_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("CSV の文字コードを判定できませんでした")

    # 1行目がタイトル（国債金利情報）の場合はスキップ
    first_line = text.split("\n")[0].strip()
    header_row = 1 if "国債金利" in first_line or "国債" in first_line[:20] else 0
    df = pd.read_csv(StringIO(text), encoding="utf-8", header=header_row)
    if df.empty:
        raise ValueError("CSV にデータがありません")

    # 先頭列を日付として 基準日 に統一
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "基準日"})
    df["基準日"] = df["基準日"].astype(str).str.strip()

    # 残存期間列: "1年物" → "1年" などに統一
    maturity_rename = {}
    for c in df.columns:
        if c == "基準日":
            continue
        stripped = c.replace("物", "").strip()
        if stripped and stripped != c:
            maturity_rename[c] = stripped
    if maturity_rename:
        df = df.rename(columns=maturity_rename)

    # 基準日以外はすべて残存期間列として残す（列を落とさない）
    date_col = [c for c in df.columns if c == "基準日"]
    other_cols = [c for c in df.columns if c != "基準日"]
    df = df[date_col + other_cols].copy()

    # 数値列は "-" 等を NaN に
    for c in df.columns:
        if c == "基準日":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) >= 2:
        local_path = Path(sys.argv[1])
        if not local_path.is_file():
            print(f"エラー: ファイルが見つかりません: {local_path}", file=sys.stderr)
            return 1
        print(f"ローカルファイルを読み込みます: {local_path}")
        csv_bytes = local_path.read_bytes()
    else:
        print("財務省の国債金利データを取得しています...")
        csv_bytes = _download_csv()

    print("読み込み・正規化しています...")
    df = _parse_and_normalize(csv_bytes)

    # app.py と同じ形式で保存: 1行目タイトル, 2行目ヘッダー, 3行目以降データ
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write(TITLE_ROW + "\n")
        df.to_csv(f, index=False, encoding="utf-8")

    print(f"保存しました: {OUTPUT_CSV}")
    print(f"行数: {len(df)} (基準日 {df['基準日'].iloc[0]} 〜 {df['基準日'].iloc[-1]})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)
