# 07 — 文書同期と完了証拠

## 目的

実装・test・packaged resource・最終判定を同じ事実へ揃えます。「pytest総数がpass」「コードがある」だけで受入項目をPASSにせず、各IDを具体的な入力・DB assertion・commandへ対応付けます。

## Cursorへ渡す依頼文

```text
00～06の実装と実行結果を文書へ同期してください。commit・pushはしないでください。

必須文書:
- docs/06_実装/ROUND6_REPRODUCTION.md
- docs/06_実装/ROUND6_QUALITY_GATE.md
- docs/06_実装/FINAL_REVIEW_ROUND6.md
- docs/06_実装/IMPLEMENTATION_STATUS.md

必要に応じて次も更新してください:
- USER_MANUAL
- SYSTEM_SPECIFICATION
- DATA_MODEL
- WORKFLOW_DIAGRAMS
- METHODOLOGY
- TROUBLESHOOTING
- PROMPT_CATALOG

記載事項:
1. forecast operationのO/N集合、update/add/remove完全被覆、一対一対応、拒否reason code、transaction境界。
2. coverage_auditのfield一覧、各countとseries_hashの意味、single/basket・success/insufficientの保存例。
3. P09 2.1.0とlegacy 2.0.0のdecision別field契約、固定Schema/Pydantic/promptの責務。
4. data-bearing 0007 migrationのprojection hashと、DDL commit後失敗からのbackup復元。
5. wheel隔離条件と正式縦断の範囲。
6. READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICEとFULL_MVP_READYの違い。

ROUND6_QUALITY_GATEには、command、実行directory、return code、passed/failed/skipped/deselected件数、主要stdout要約、実行日時を記載してください。logを捏造せず、実行していないcommandはNOT_RUNとします。

FINAL_REVIEW_ROUND6ではR6-001～R6-050の各行へ、少なくとも次のいずれかを対応付けてください:
- test node IDとassert対象
- DB queryと期待値/実値
- CLI commandとreturn code/output
- Schema validator payloadとerror path
- 生成fileと内容/hash

単一の`pytest -q`合計を複数IDの唯一の証拠にしないでください。

判定規則:
- 1件でもFAIL、NOT_RUN、未検証、skip、xfail、証拠不足があればNOT_READY。
- live network integrationの既存deselectは、同じ内部処理をCSV/mockでpassし今回の項目を外していない場合だけ別記できる。
- R6-001～050が全件具体的証拠付きPASSの場合だけREADY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE。
- FULL_MVP_READYは使用しない。

文書のREADY表記は、全品質ゲートと08_FINAL_AUDIT完了後にだけ更新してください。途中ではNOT_READY/PENDINGのままにしてください。

FUTURE_ROADMAPの未実装項目と既存詳細を短縮・削除しないでください。
```

## 最終報告に含める内容

```text
判定:
対象base commit:
変更ファイル:
operation negative matrix:
coverage 4形状:
P09 contract matrix:
migration success / forced failure restore:
wheel isolated vertical:
pytest:
ruff format/check:
mypy:
docs/resource sync:
Alembic:
build:
git diff --check:
skip / xfail / deselected:
R6-001～R6-050:
意図的未実装:
次の一作業:
commit / push: 未実施
```

## 完了条件

- docsのSchema例を実際のvalidator testへ通している。
- implementation statusとfinal reviewが実行結果を超えていない。
- acceptance matrix全行が固有の証拠へ追跡可能である。
- repoとwheelの完全版docsが同期している。

