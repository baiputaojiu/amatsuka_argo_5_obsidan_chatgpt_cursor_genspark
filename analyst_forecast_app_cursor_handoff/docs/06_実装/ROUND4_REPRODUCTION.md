# Round4 バグ再現記録

## Bug A/B (Fix 02): P09 accept/correct ライフサイクル

**再現手順:**
1. P05→P08 を正常投入し `ACCEPTED` を得る（issuance生成済み）
2. P09 `accept` を再度投入 → `_insert_p08` が再呼出しされ UNIQUE制約違反
3. P09 `correct` を投入 → 旧issuanceが supersede されず重複アクティブ発生

**期待:** accept済みP08へのaccept は冪等、correct は旧supersede + 新generation生成

## Bug C (Fix 03): ソースオカレンス再利用

**再現手順:**
1. Source A に対しP05/P07を処理済みとする
2. 同一raw_bytesの Source B に対し再利用を試みる
3. P08の `_validate_references` が Source B 用の artifact を見つけられず reject

**期待:** artifact_applicability テーブルで再利用を追跡し、P08参照検証が通過

## Bug D (Fix 04): P09 reject/unresolved ステートマシン

**再現手順:**
1. P08 を `needs_review` にし P09 `reject` (retryable) を投入
2. RunSource の status が `RUN_PREPROCESS` に戻り、P05/P07が再実行される
3. terminal reject / unresolved → `REVIEW_AI_OUTPUT` 無限ループ

**期待:** retryable reject → `RUN_P08` (P05/P07保持), terminal → ループ停止

## Bug E (Fix 05): 不明時刻 & カットオフ

**再現手順:**
1. `made_at_source=unknown` で `made_at` に日時を設定した P08 を投入
2. Pydantic検証が通過してしまい、不正なissuanceが生成される
3. `knowledge_cutoff` が `made_at` を超過する P08 が検出されない

**期待:** unknown時は `made_at=null` 必須、cutoff > made_at → 検証エラー

## Bug F (Fix 06): バスケット共通日数

**再現手順:**
1. 2銘柄バスケットで共通取引日が1日のみのデータを用意
2. 評価関数が1ポイントで hit/miss を計算してしまう

**期待:** 共通日 < 2 → `unevaluable insufficient_common_dates`
