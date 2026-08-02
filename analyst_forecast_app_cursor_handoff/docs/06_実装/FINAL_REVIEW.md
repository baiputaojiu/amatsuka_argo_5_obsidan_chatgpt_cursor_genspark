# FINAL_REVIEW — Round2

## 判定

`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`

Round2（状態機械・媒体経路・AIレビュー・話者帰属・バスケット評価・migration）を実装し、内部unit縦断と品質ゲートが通過した。時期・程度・早期実現の完全採点、1/3/6/12か月観測、PNG、総合点などは未実装のため `FULL_MVP_READY` は使わない。

## 実行日時

2026-07-21

## 品質ゲート（実測）

| 検査 | 結果 |
|------|------|
| ruff format --check | pass（format実行後） |
| ruff check | pass |
| mypy src | pass（44 source files） |
| pytest tests/unit | **83 passed** |
| Alembic empty→head | pass（0006含む） |
| Alembic 0001→head | pass（migration互換テスト） |
| alembic check（metadata差分） | **0** |
| ネットワーク integration | external_blocked（通常suite除外） |
| wheel clean venv | 未実施（次作業候補） |

## シナリオ結果（証拠）

| ID | 内容 | 結果 | 証拠 |
|----|------|------|------|
| A | YouTube高確信度→CSV評価 | pass | `test_scenario_a_vertical_with_forecast`, `test_state_machine_r2` |
| B | 話者不明→P06経路 | pass（P06 Schema/取込み） | `test_p05_unknown_speaker_*`, Round2 P06実装 |
| C | 司会者予想を成績から除外 | pass | `test_host_segment_forecast_not_formalized` |
| D | blog→P07→P08 | pass | `test_blog_x_web_reach_p08_and_zero_forecast` |
| E | 予想0件 | pass | 同上（processed_no_forecast） |
| F | 第三者要約除外 | pass（statement_kindフィルタ） | `_is_formal_forecast` |
| G | P08低確信度→P09→解決 | pass | `test_p09_accept_resolves_needs_review_once` |
| H | P11/P12/P13状態機械 | pass | `test_state_machine_r2.py` |
| I | P12修正候補origin | pass（Schema/lock） | `recommended_candidate_origin` |
| J | 同一通貨basket | pass | `test_equal_weight_basket_and_mixed_currency` |
| K | mixed currency unevaluable | pass | 同上 |
| L | 前処理再利用 | pass | `test_safe_preprocess_reuse_across_runs` |
| M | 未評価component非隠蔽 | pass | `test_unevaluated_component_not_hidden_by_multi_as_of` |
| N | prompt/model設定 | pass（CLI/snapshot） | `config set-model`, runs snapshot |
| O | wheel縦断 | external_blocked / 未実施 | — |

## 既知制約

- `sources.raw_artifact_id` のDB FKはSQLite batch制約のため追加せず、modelもFKなしで一致。索引とアプリ層で担保。
- P05/P07のknowledge_cutoff厳密時刻検証はSQLite TZ表現の揺れを避けるため、発行後工程（P11+）を主に検証。
- wheelクリーン導入と実ネットワーク縦断は未実施。
- 総合点・ランキング・PNG・期間不明の自動観測は意図的未実装。

## 次の一作業

実VaultでYouTubeまたはblog原文1件を、ターミナルのNEXT_ACTIONSどおりに方向評価まで目視確認する。

## commit / push

未実施
