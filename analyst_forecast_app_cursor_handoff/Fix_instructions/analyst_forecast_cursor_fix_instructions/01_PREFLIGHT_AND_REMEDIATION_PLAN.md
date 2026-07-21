# 01 — 事前確認と修正計画

## コピペ用依頼文

```text
アナリスト予想検証アプリのレビュー指摘を修正するため、事前確認と実装計画を作成してください。まだ本体コードの機能変更は行わないでください。ただし文書の作成・更新は行って構いません。

対象アプリルートを特定し、次を読んでください。
- 00_Cursor引継ぎ/02_SPEC_AUTHORITY.md
- .cursor/rules/analyst-forecast-app.mdc
- docs/03_仕様書/METHODOLOGY.md
- docs/03_仕様書/DATA_MODEL.md
- docs/03_仕様書/SYSTEM_SPECIFICATION.md
- docs/03_仕様書/FOLDER_STRUCTURE.md
- docs/05_計画/DECISION_LOG.md
- docs/05_計画/MVP_PLAN.md
- docs/06_実装/IMPLEMENTATION_ASSUMPTIONS.md
- docs/06_実装/IMPLEMENTATION_PLAN.md
- docs/06_実装/IMPLEMENTATION_STATUS.md
- この修正指示セットの00_START_HERE.mdと02～07

最初にGit status、現在ブランチ、HEAD、対象アプリ外の差分を確認してください。ユーザーの既存変更を破壊しないでください。

以下を確定事項としてDECISION_LOG.mdへ追記してください。
- reference/CHAT_HISTORY.pdfが公開リポジトリに含まれることはユーザー確認済みで許容する。
- 同ファイルは今回の削除・履歴除去対象にしない。
- この許容は、将来の秘密情報、APIキー、個人パス、調査原文、SQLite等を公開してよいという一般許可ではない。

次のレビュー指摘を実コードとテストで再確認してください。
1. AI取込み後にcomponent_idが表示されず、NEXT_ACTIONS.mdもプレースホルダーのまま。
2. 市場評価後に04_resultsへ人間向け結果が作られない。
3. P08、P11、P12の出力が分離されず、独立レビューを実行記録で保証できない。
4. forecasts=[]と、代理指標を解決できないtargetをSchemaで表現できない。
5. needs_reviewファイル数と評価総数に基づくworkflow判定が、再レビュー・複数基準日で誤る。
6. raw hashの全体重複により、別アナリストのSOURCE所有者・保存場所が不整合になる。
7. Vault内のdocsとpromptsが空のまま。
8. 下落予想のMFE/MAEが方向対応していない。
9. yfinanceのrate limitと本当のデータなしを区別できない。
10. 対話式の案件作成が未実装。

docs/06_実装/REMEDIATION_PLAN.mdを作成し、02～07の順序に合わせて以下を記載してください。
- 問題と再現条件
- 関連する要件・決定ID
- DB／Schema migration
- 後方互換性
- 変更予定ファイル
- 必須テスト
- 完了条件
- ロールバック方法

既存のAI JSON Schema 1.0.0やSQLiteを既に利用している可能性を考慮し、破壊的な上書きではなく、新Schema versionとAlembic migrationを使用する計画にしてください。既存acceptedデータを可能な範囲で読める互換経路を残してください。

最後に、各指摘を reproduced / already_fixed / not_reproducible / intentional_future_scope のいずれかへ分類し、根拠を示してください。計画だけで終了し、02の実装にはまだ着手しないでください。
```

## 完了条件

- 公開PDFの許容が決定履歴へ記録される。
- 10項目の再現結果がある。
- migrationと互換性を含む修正計画がある。
- 本体コードはまだ変更されていない。

