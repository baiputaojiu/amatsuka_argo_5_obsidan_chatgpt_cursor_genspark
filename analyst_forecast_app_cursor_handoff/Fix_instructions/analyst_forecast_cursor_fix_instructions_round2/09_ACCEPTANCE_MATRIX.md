# 第2次修正 — 受入マトリクス

Cursorの最終報告だけでなく、テスト名、DB状態、生成成果物、実行結果を対応付けて確認してください。

| ID | 受入事項 | 必須証拠 |
|---|---|---|
| R2-001 | P11後にP12を推奨する | 実ingestを使うworkflow test |
| R2-002 | P12 disagreed後にP13を推奨する | 実ingestを使うworkflow test |
| R2-003 | mapping未固定で市場評価へ進まない | negative test |
| R2-004 | blogがP07→P08を通る | blog縦断test |
| R2-005 | XがP07→P08を通る | X縦断test |
| R2-006 | WebがP07→P08を通る | Web縦断test |
| R2-007 | P08が正しいP05/P07だけを参照する | run/source/hash negative test |
| R2-008 | 同一rawの安全な処理済みartifact再利用 | positive/negative reuse test |
| R2-009 | P06で話者needs_reviewを解決できる | P05→P06 test |
| R2-010 | P09でP08 needs_reviewを解決できる | P08→P09 test |
| R2-011 | review後に元needs_reviewが未解決件数へ残らない | DB lineage/workflow test |
| R2-012 | review再取込みで二重forecastを作らない | idempotency test |
| R2-013 | 対象者本人segmentだけ正式成績へ入る | speaker attribution test |
| R2-014 | 司会者・第三者要約を正式成績へ入れない | negative attribution test |
| R2-015 | quote offsetとsegment範囲が一致する | offset/segment test |
| R2-016 | made_at、公開日時、knowledge cutoffを検証する | time leakage test |
| R2-017 | P12/P13のproposal-review-component関係を検証する | cross-component negative test |
| R2-018 | P12修正候補を採用できる | corrected candidate test |
| R2-019 | P13がP12修正候補を採用できる | adjudication test |
| R2-020 | 同一通貨basketをweight付き評価する | numeric basket test |
| R2-021 | mixed currencyを先頭銘柄で代用しない | unevaluable test |
| R2-022 | 単一銘柄方向評価の互換性を維持する | existing regression test |
| R2-023 | promptに予想認定・除外基準がある | resource content test |
| R2-024 | Vaultに詳細説明書とFUTURE_ROADMAPをseedする | wheel/init content test |
| R2-025 | 高性能モデル名を通常操作で設定できる | CLI/wizard test |
| R2-026 | ruff format/check、mypy、pytestがpass | command log |
| R2-027 | Alembic empty/0001/0005→headがpass | migration tests |
| R2-028 | alembic checkが差分0 | command log |
| R2-029 | wheel clean venvでhelp/init/run createがpass | package test |
| R2-030 | DB、raw、秘密値、実Vault pathをGit追跡しない | git/secret scan |
| R2-031 | CHAT_HISTORY.pdfを削除・改変しない | git diff |
| R2-032 | 実装済み・未実装・外部blockを正確に報告する | FINAL_REVIEW |

## 判定条件

`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`とするには、R2-001～032がすべてpassである必要があります。外部サービスだけに依存するネットワーク試験は理由付き`external_blocked`を許容しますが、内部Schema、workflow、SQLite、CSV、migration、packageのfailは許容しません。

`FULL_MVP_READY`は、上記に加えて正式仕様のMVP完成条件をすべて満たした場合だけ使用してください。時期・程度・早期実現、期間不明の1/3/6/12か月観測、PNG出力等が未実装なら、方向評価sliceの完成とフルMVP完成を区別してください。
