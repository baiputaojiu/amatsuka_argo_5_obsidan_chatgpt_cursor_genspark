# 修正受入マトリクス

Cursorの最終報告だけでなく、テスト名・成果物・実行結果を対応付けて確認する。

| ID | 受入事項 | 必須証拠 |
|---|---|---|
| FIX-001 | AI取込み後に実component IDが表示される | CLI test、NEXT_ACTIONS fixture |
| FIX-002 | 生成コマンドをそのまま実行できる | CLI縦断test |
| FIX-003 | 04_resultsへMarkdown／CSVが生成される | snapshot／内容test |
| FIX-004 | P08・P11・P12が別実行として保存される | DB test、監査log |
| FIX-005 | P12なしでverifiedにならない | negative test |
| FIX-006 | forecasts=[]を処理済みとして保存できる | Schema・workflow test |
| FIX-007 | symbolなしunresolvableを保存できる | Schema・評価test |
| FIX-008 | 最大3代理指標とレビューを表現できる | Schema・DB test |
| FIX-009 | needs_review解決後に先へ進む | workflow test |
| FIX-010 | 複数as_ofが未評価componentを隠さない | workflow regression test |
| FIX-011 | 最新成功が古いunevaluableより優先される | state regression test |
| FIX-012 | 同一bytes・別アナリストで所有者が混線しない | source identity test |
| FIX-013 | 同一bytesでも別URLの証拠を保持する | provenance test |
| FIX-014 | raw artifactを再処理しない | processing cache test |
| FIX-015 | 下落予想のMFE／MAEが方向対応する | numeric unit test |
| FIX-016 | rate limitとデータなしを区別する | provider mock test |
| FIX-017 | run_idとcomponent_idの案件不一致を拒否する | CLI negative test |
| FIX-018 | Vault内のdocsと中央promptsが空でない | init test |
| FIX-019 | 案件prompt snapshotがversion/hashを持つ | prompt generation test |
| FIX-020 | 対話wizardが既定6か月で動く | Typer interaction test |
| FIX-021 | 既存flag CLIが壊れていない | regression test |
| FIX-022 | 既存Schema／DBをmigrationできる | compatibility test |
| FIX-023 | raw・DB・秘密値がGit追跡されない | gitignore／secret scan |
| FIX-024 | 公開CHAT_HISTORY.pdfを今回削除しない | Git diff確認 |

## MVP再判定条件

`READY_FOR_REAL_SAMPLE`とするには、FIX-001～024がすべてpassまたは、外部サービスにだけ依存する項目が理由付き`external_blocked`であること。内部実装のfailを`external_blocked`として扱わない。

