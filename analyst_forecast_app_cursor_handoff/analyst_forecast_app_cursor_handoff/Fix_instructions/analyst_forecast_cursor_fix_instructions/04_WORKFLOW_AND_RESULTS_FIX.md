# 04 — 次行動・レビュー解決・人間向け結果の修正

## 目的

ユーザーがDBを直接調べず、ターミナルと`NEXT_ACTIONS.md`だけで次工程へ進めるようにする。

## コピペ用依頼文

```text
workflow状態管理、CLI表示、結果ファイル生成を修正してください。

現在の重大問題：
- ai ingest後にforecast_issuance_idとforecast_component_idを表示しない。
- NEXT_ACTIONS.mdが<component-id>のまま。
- 評価後に確認を指示する04_resultsが空。
- needs_reviewフォルダのファイル数だけでレビュー待ちを判定するため、修正版accepted後も待ち状態が残る。
- EvaluationRecord総数とcomponent総数を比較するため、同一componentの複数基準日評価が他componentの未評価を隠す。
- 過去のunevaluable評価が一件でも残ると、後の成功評価後も取得不能扱いになり得る。

workflowの正本をSQLiteの明示的状態にしてください。ファイルの存在数は診断材料には使えても、完了判定の正本にしないでください。

AI成果物またはworkflow taskについて、少なくとも次を追跡してください。
- pending / running / needs_review / resolved / accepted / rejected / superseded
- どの出力がどの出力を修正・置換したか
- review executionとresolution
- task dependency
- retry可能性と最終エラー

needs_reviewの元ファイルは削除せず監査用に残し、修正版や別AIレビューで解決したらDB状態をresolvedまたはsupersededへ変更してください。未解決件数だけを次行動へ反映してください。

市場評価の完了判定は、指定evaluation_as_ofごとに各componentの最新評価を1件選び、distinct component単位で判定してください。同じcomponentの過去snapshotを複数件として数えないでください。過去unevaluable後に新しい基準日または再試行で成功した場合、現在状態は最新結果を使用し、過去履歴は残してください。

CLIとNEXT_ACTIONSを次のように改善してください。
- ai ingest成功時にissuance ID、component ID、対象名、symbol、方向、期間、mapping状態を表または一覧表示する。
- 未評価componentごとに、実行可能な完全コマンドを生成する。
- CLIのrun_idとcomponent_idが同じ案件に属することを検証する。
- 複数候補がある場合、推奨1件と最大2件の代替案を維持する。
- AI作業では、使用する具体的prompt path、入力path、出力path、取込みコマンドを表示する。
- `status`だけで状態ファイルを再生成できる。

最低限、次の人間向け成果物を04_resultsへ生成してください。
- forecasts/all_forecasts.md
- tables/all_forecasts.csv
- evaluations/evaluations.md
- tables/evaluations.csv
- reports/vertical_mvp_summary.md

内容には原文へ戻れるsource_id、raw path、引用、issuance ID、component ID、対象、方向、期間、mapping、評価基準日、開始値、終了値、実際の変化率、状態、評価不能理由を含めてください。原文全文をレポートへ複製しないでください。

評価またはAI取込み成功後に結果ファイルと状態ファイルを原子的に再生成してください。MarkdownとCSVはSQLiteから生成し、手編集を正本にしないでください。

必須テスト：
- ai ingest出力とNEXT_ACTIONSに実component IDが現れる。
- 生成されたmarket evaluateコマンドをそのまま実行できる。
- 別案件のcomponent IDを指定すると拒否する。
- corrected output受理後、古いneeds_reviewファイルが残っていても待ち状態を抜ける。
- component Aを2基準日で評価しcomponent Bが未評価なら、Bを次行動に示す。
- 過去unevaluable、最新成功なら現在状態は成功側になる。
- 04_resultsの5成果物が生成され、内容がDBと一致する。
- 0 forecastの処理済み案件が無限にP08を要求しない。
- P05→P08→P11→P12→評価の各状態遷移。
- 中断後にstatusで正しい次工程へ復帰する。

既存の監査ファイルと履歴を消さず、migrationと後方互換性を保ってください。Ruff、mypy、pytest、Alembic checkを実行し、実装文書を更新してください。commit・pushは行わないでください。
```

