# スタートアップガイド

1. Obsidian Vault の絶対パスを決める。
2. Vault 内に `30_Permanent/★アナリスト調査` を作る。
3. `analyst-forecast init --vault-root <そのパス>` を実行する。
4. 設定は `%LOCALAPPDATA%/AnalystForecast/config.yaml`（または指定 `--config`）。
5. APIキーは設定ファイルへ保存しない。
6. `analyst-forecast start` または `run create` で案件を作る。
7. 初回案件の `NEXT_ACTIONS.md` に従う。
