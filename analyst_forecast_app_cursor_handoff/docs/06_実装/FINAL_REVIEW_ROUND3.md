# Round3 最終監査（FINAL_REVIEW_ROUND3）

- 作成日：2026-07-21
- 対象作業tree：Round3 fixes 01–09 実装後（commit未実施）
- ベースHEAD：`e165a6c8142fc6085caa6e73fb2da6e303abe3e9`
- 判定：**READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE**

Round3の6重大バグ（話者誤認・未来cutoff・P09検証迂回・複数source取り残し・basket cache汚染・Vault docs要約）を閉じ、品質ゲートとwheel smoke・CHAT_HISTORY保持を確認した。総合点・PNG・1/3/6/12観測等は未実装のため `FULL_MVP_READY` は使わない。

## 品質ゲート（実測）

| 検査 | 結果 |
|------|------|
| pytest unit | **94 passed**, 3 deselected |
| ruff format --check | pass |
| ruff check | pass |
| mypy src | pass（48 source files） |
| packaged docs sync --check | pass |
| Alembic 0007 empty→head / 0005→head / 0006→head | pass |
| wheel build + clean venv install | pass |
| `analyst-forecast --help`（clean venv） | pass |
| `analyst-forecast init --vault-root`（clean venv） | pass |
| CHAT_HISTORY.pdf | 存在。SHA256=`F1D9567078A9E1F262C6E54B394D75EF3862A51F4EB6DABAE31AF0C356ADDD4A`。削除・改変なし |
| ネットワーク integration | external_blocked（通常suite除外。CSV/mockで代替） |

## シナリオ A–R（要約）

| ID | 内容 | 証拠 | 判定 |
|----|------|------|------|
| A | YouTube本人→評価 | `test_final_scenarios` / state machine | pass |
| B | 司会者+target_confirmed | `test_host_target_confirmed_claim_not_formalized` | pass（REJECT, formal=0） |
| C | legacy_unknown | `test_legacy_unknown_not_formalized` | pass |
| D/E | Web直接引用/要約 | P07 content_author+statement_speaker | pass（実装+既存フィルタ） |
| F | future cutoff | `test_p05_future_cutoff_rejected` + shared boundary | pass |
| G | P09不正offset | `test_p09_bad_offset_correct_rejected` | pass |
| H | P09正当correct再検証 | `_validate_p08` 共有経路 | pass |
| I | 複数source P08 | `test_multi_source_pending_p08_not_skipped` | pass |
| J–L | 追加source / 0件 / reuse | Round2 reuse + terminal states | pass |
| M | 対象解決状態機械 | `test_state_machine_r2` | pass |
| N | basket cache分離 | `test_basket_cache_does_not_pollute_single_symbol` | pass（0%→AAA +20%） |
| O–P | basket CSV / mixed | long-form CSV + mixed unevaluable | pass |
| Q | wheel | clean venv help/init | pass |
| R | Git衛生 | 秘密非追跡方針。CHAT_HISTORY保持 | pass |

## 受入マトリクス（主要）

| ID | 結果 |
|----|------|
| R3-001〜006 話者・正式化 | pass |
| R3-011 未来cutoff | pass |
| R3-020 P09再検証 | pass |
| R3-026 複数source | pass |
| R3-032/033 basket | pass |
| R3-041/043 Vault docs | pass |
| R3-046〜048 migration/品質 | pass |
| R3-049 wheel | pass |
| R3-051 CHAT_HISTORY | pass |

限定・部分項目（専用E2Eが薄いが実装あり）は D/E/H/O 等。致命経路のnegative testは閉じている。

## 6重大バグの修正要点

1. **Speaker**：`verify_forecast_attribution`。正式化は verified `target_confirmed` のみ。
2. **Knowledge cutoff**：`source_knowledge_boundary` を P05/P07/P06/P09 に適用。
3. **P09 correct**：corrected_payload を通常 `_validate_p08` と同一再検証。
4. **Multi-source**：`pending_p08` があれば issuances があっても EXTRACT。
5. **Basket cache**：`series_kind=basket` / `BASKET:{mapping_hash}`。
6. **Vault docs**：`scripts/sync_packaged_docs.py` で詳細版を resources へ同期。

## 意図的未実装（FULL MVP外）

総合点、複数アナリスト統合、上昇候補ランキング、時期・程度・早期実現の完全採点、1/3/6/12か月観測、PNG、完全自動収集、exe化。

## 次の一作業

実Vaultで原文1件を `source import` → AI取込み → CSV評価まで目視確認する。

## commit / push

未実施
