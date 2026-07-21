# AI作業ガイド

1. `NEXT_ACTIONS.md` の推奨行動だけを実行する。
2. 案件の `01_prompts` は中央テンプレートの snapshot であり、正本は workspace の `prompts/` です。
3. AI出力は Schema に従い、原文にない情報を補完しない。
4. 取込みは `analyst-forecast ai ingest <JSON>`。
5. 市場評価は Python が行い、AI は symbol を確定しない。
