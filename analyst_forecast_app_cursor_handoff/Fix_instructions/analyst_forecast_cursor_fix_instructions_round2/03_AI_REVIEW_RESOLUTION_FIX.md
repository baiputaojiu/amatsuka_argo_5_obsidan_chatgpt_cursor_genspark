# 03 — AI再レビューとneeds_review解決経路

## 目的

低確信度または高重要度のP05/P08成果物を、DB直接操作や人間承認なしで、別AIレビューにより解決できるようにします。

## 確認済みの不具合

低確信度P08の後に高確信度の修正版P08を受理しても、元artifactの`needs_review`が残り、workflowが`REVIEW_AI_OUTPUT`から進みません。既存回帰テストはテストコードからDBを直接`resolved`へ書き換えています。

## コピペ用依頼文

```text
fix_instructions_round2/00_START_HERE.mdの共通拘束条件と、01～02の新しいpipelineに従い、AI再レビューの正式な解決経路を実装してください。人間承認を追加せず、commit・pushはしないでください。

P06「YouTube話者推定レビュー」とP09「予想抽出レビュー」を、独立したprompt execution、artifact、固定JSON Schemaとして実装してください。P07の著者・引用帰属が低確信度の場合に使用するレビュー種別も、P06を媒体共通化するか別promptとして一貫して設計してください。

レビュー成果物には最低限、次を持たせてください。

- schema version
- run_id、source_id
- reviewed_artifact_id
- reviewed artifactのoutput hashを示すinput_hash
- prompt execution ID／prompt ID／version／environment／model／executed_at
- decision: accept / correct / reject / unresolved
- findingsと根拠
- correct時の完全なcorrected payload、または曖昧さなく再検証できるversion付き修正表現
- knowledge_cutoff

レビューを取り込んだときの状態遷移を実装してください。

- accept: 元artifactをresolvedとし、review artifactとのlineageを保存する。P08なら予想を一度だけ正式テーブルへmaterializeする。
- correct: 修正版を元Schemaで完全検証し、immutableな修正artifactとして保存する。元artifactをsuperseded/resolvedとし、修正artifactだけを正式利用する。
- reject: 元artifactを正式予想に使わず、拒否理由とreview artifactを残す。
- unresolved: 判断不能を保持し、正式成績へ入れない。必要なら別AIでの再レビューを次行動として示す。

needs_reviewのP08を初回取込み時に正式forecast issuanceへ登録しない方針を維持してください。P05/P07の未確認segmentも、レビューを経ずにP08の正式上流artifactとして使えないようにしてください。

新しいレビューが元artifactと同じrun/source/input hashを参照していることをPythonで検証してください。レビューAI自身の自由記述だけで元artifactを解決扱いしないでください。reviewed_artifact_id、resolved_by_artifact_id、supersedes_artifact_idを監査可能に保存してください。

同じreview成果物の再取込みは冪等にし、forecastやsegmentを重複作成しないでください。needs_reviewディレクトリの古いファイルが残っていても、DBのlineageとresolution statusを正本にしてください。

必須テスト：

1. 低確信度P05→P06 accept→P08へ進む。
2. P06 correctで修正segmentだけが有効になる。
3. 低確信度P08→P09 accept→forecastが1回だけ作成される。
4. 高重要度P08→P09 correct→修正版forecastだけが作成される。
5. reject/unresolvedは正式成績へ入らない。
6. review後、元needs_review件数が0になり次工程へ進む。
7. 別run、別source、hash不一致のreviewを拒否する。
8. 同じreview再取込みで二重materializeしない。
9. staleなneeds_reviewファイルがあってもDB解決済みなら停止しない。
10. テスト内でresolution_statusを直接書き換えない。

P06/P09のCursor用・ChatGPT用prompt snapshotを生成し、使用モデルは設定された高性能モデルだけを表示してください。ruff format、ruff check、mypy、pytestを実行し、状態遷移図と実装状況を更新してください。
```

## 完了条件

- needs_reviewが実際のAIレビュー成果物で解決できる。
- 元artifactとレビュー・修正artifactのlineageを追跡できる。
- 人間承認やDB手修正を必要としない。
- 高重要度・低確信度を優先レビューする既存方針が維持される。
