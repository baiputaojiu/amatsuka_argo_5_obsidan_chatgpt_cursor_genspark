# Round3 再現記録（修正前）

- 作成日：2026-07-21
- 対象HEAD：`e165a6c8142fc6085caa6e73fb2da6e303abe3e9`
- project root：`analyst_forecast_app_cursor_handoff/`（二重階層なし）
- 判定起点：`NEEDS_CHANGES_BEFORE_REAL_SAMPLE`

## Baseline（修正前）

| 検査 | 結果 |
|------|------|
| pytest unit | 83 passed, 3 deselected |
| ruff format / check | pass（Round2後） |
| mypy src | pass |
| 対象外として維持すべきもの | P11→P12、P12 disagree→P13、P07経路、mixed currency unevaluable、秘密非混入 |

## 主要5件の再現

### 1. 司会者segment + P08 target_confirmed → 正式化

| 項目 | 内容 |
|------|------|
| 入力 | P05 `speaker_candidate=司会者` accepted → P08 `speaker_attribution_status=target_confirmed` |
| 期待 | 正式 ForecastIssuance = 0 |
| 実結果 | p05=accepted, p08=accepted, **formal_issuances=1** |
| 原因候補 | `_is_formal_forecast` が `not_target`/`uncertain` 以外を許可。P05話者名とAnalyst照合なし |
| 受入ID | R3-001〜003、R2-013/014相当 |

### 2. legacy_unknown → 正式化

| 項目 | 内容 |
|------|------|
| 入力 | P08 `speaker_attribution_status=legacy_unknown` |
| 期待 | 正式 Issuance = 0（証拠保持可） |
| 実結果 | **formal_issuances=1** |
| 原因候補 | `_is_formal_forecast` が legacy_unknown を除外していない |
| 受入ID | R3-006 |

### 3. P09 correct の offset ずらしが正式化

| 項目 | 内容 |
|------|------|
| 入力 | 低確信度P08 → needs_review。P09 correct で quote offset を意図的にずらす |
| 期待 | reject / needs_review。正式化しない |
| 実結果 |（レビュー報告）initial=needs_review, P09 correct=accepted, formal=1 |
| 原因候補 | `corrected_payload` が `model_validate` のみ。通常P08の意味検証を再実行しない |
| 受入ID | R3-020〜022 |

### 4. 複数blogで2件目P08取り残し

| 項目 | 内容 |
|------|------|
| 入力 | blog source 2件、両方P07完了、1件目だけP08完了 |
| 期待 | next = EXTRACT_FORECASTS（2件目向け） |
| 実結果 | **next=RUN_P11**, still_needs_p08=True |
| 原因候補 | `workflow.py`: `pending_p08 and not context.issuances` — issuanceが1件あるとP08推奨を出さない |
| 受入ID | R3-026〜028 |

### 5. バスケットcacheが単一銘柄を汚染

| 項目 | 内容 |
|------|------|
| 入力 | AAA +20%、BBB -20%、50/50 basket → 0%。その後AAA単独 |
| 期待 | AAA単独 = +20% |
| 実結果 | basket return=0 が `AAA__*.csv` でcache。AAA単独も **return=0.000** |
| 原因候補 | `evaluation.py` が合成seriesを先頭銘柄 symbol で cache |
| 受入ID | R3-032〜034 |

## 追加確認

| 項目 | 実結果 |
|------|--------|
| P05 knowledge_cutoff が recorded_at より後 | **accepted**（未来cutoff未拒否）→ R3-011 |
| packaged FUTURE_ROADMAP | resources **3行** vs docs **287行** |
| packaged USER_MANUAL | resources **9行** vs docs **188行** |
| SYSTEM_SPECIFICATION / README | 同様に著しく短い |

## Remediation plan（Round3対応）

| 指示 | 対象問題 | 方針 |
|------|----------|------|
| 02 | 1, 2, 話者証拠 | Python verified attribution。正式化は verified target_confirmed のみ |
| 03 | future cutoff | source境界と共通検証を P05–P13 に適用 |
| 04 | P09迂回 | corrected_payload を通常 ingest validator と共有 |
| 05 | 複数source | source単位で pending_p08 を優先。issuancesがあってもP08未完了sourceを処理 |
| 06 | basket cache | basket identity を銘柄名と分離。CSV multi-symbol |
| 07 | Vault docs | resources/docs を repo docs 正本へ sync |
| 08 | migration/品質 | 0007+、wheel、品質ゲート記録 |
| 09 | 最終監査 | FINAL_REVIEW_ROUND3、受入マトリクス照合 |

## この工程での実装変更

なし（再現と計画のみ）。
