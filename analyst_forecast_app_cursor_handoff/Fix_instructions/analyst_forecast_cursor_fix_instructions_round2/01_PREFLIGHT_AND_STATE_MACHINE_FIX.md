# 01 — 事前確認と状態遷移修正

## 目的

P11後にP11を繰り返し、P12不一致後にP13を飛ばして市場評価へ進む不具合を直します。DBに保存する状態とworkflowが解釈する状態を一つの状態機械として定義します。

## 確認済みの再現結果

```text
P08後             RUN_P11         正しい
P11後             RUN_P11         誤り。RUN_P12であるべき
P12 disagreed後   EVALUATE_MARKET 誤り。RUN_P13であるべき
```

## コピペ用依頼文

```text
fix_instructions_round2/00_START_HERE.mdの共通拘束条件に従い、事前確認とworkflow状態機械の修正を行ってください。

最初にgit status、現在のHEAD、pyproject.tomlの位置、既存migration head、既存テスト結果を確認してください。ユーザーの未コミット変更がある場合は上書きせず、重なる変更を報告してください。commit・pushはしないでください。

現在、AI pipelineが保存する状態名とapplication/workflow.pyが判定する状態名が一致していません。少なくとも次の遷移を正本として定義し、DB、Python、生成されるNEXT_ACTIONS.md、ターミナル表示で統一してください。

- P08受理直後・対象未解決: P11を推奨
- P11 proposed受理後: 同じcomponentにP12を推奨。P11を繰り返さない
- P11 unresolvable提案後: 独立確認のためP12を推奨
- P12 agreed後: mappingをlockし、市場評価へ進む
- P12 unresolved後: unresolvable mappingを確定し、推測値を使わず評価不能結果または結果確認へ進む
- P12 disagreed後: P13を推奨。市場評価へ進まない
- P13 verified後: mappingをlockし、市場評価へ進む
- P13 unresolvable後: unresolvableとして記録し、推測値を使わない
- mapping未固定、P12未完了、P13待ちのcomponentは市場評価コマンドを生成しない

状態名を定数、Enum、または単一の判定関数へ集約し、文字列の重複による再発を防いでください。既存legacy Schema 1.0.0のreview_pending等は後方互換として明示的に変換し、v2状態と混在させないでください。

複数componentがある場合、各componentの現在段階をDBから判定してください。先頭componentだけの件数や、全評価件数だけで案件全体を完了扱いしないでください。NEXT_ACTIONSには実在するrun ID、source ID、component ID、proposal artifact ID、review artifact IDのうち次処理に必要なものを表示してください。

必須回帰テストを追加してください。

1. P08取込み後、recommended_actionがRUN_P11。
2. P11 proposed取込み後、recommended_actionがRUN_P12。
3. P11 unresolvable取込み後もRUN_P12。
4. P12 agreed後、EVALUATE_MARKET。
5. P12 disagreed後、RUN_P13。
6. P13 verified後、EVALUATE_MARKET。
7. P12/P13 unresolvable後、P11や市場取得を無限反復しない。
8. mapping未固定componentのmarket evaluateは拒否される。
9. 複数componentのうち1件だけ完了しても、残りcomponentの正しい次工程を示す。
10. 生成されたcommand_or_promptに仮の<component-id>を残さない。

テストはDB状態をテスト側から直接書き換えて成功させず、実際のingest_ai_outputを経由してください。修正後、ruff format、ruff check、mypy、pytestを実行し、結果を報告してください。
```

## 完了条件

- P08→P11→P12→必要時P13→評価の推奨行動が自動遷移する。
- P11またはP12の無限反復がない。
- P13待ちを市場評価と誤認しない。
- 回帰テストが実際のAI成果物取込み経路を使用している。
