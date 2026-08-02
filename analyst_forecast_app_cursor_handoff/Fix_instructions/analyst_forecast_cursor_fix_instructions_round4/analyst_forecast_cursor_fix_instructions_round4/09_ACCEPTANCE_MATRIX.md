# 第4次修正 — 受入マトリクス

各項目に、test名、DB状態・constraint、生成成果物、CLI出力を対応付けてください。合計test数だけを証拠にしないでください。

| ID | 受入事項 | 必須証拠 |
|---|---|---|
| R4-001 | needs_review P08 + P09 acceptが正式予想を一度だけ作る | vertical accept test、件数assertion |
| R4-002 | accepted P08 + P09 acceptが例外・二重作成を起こさない | accepted re-review regression |
| R4-003 | accepted P08 + P09 correctが旧世代をsupersedeする | lifecycle DB query |
| R4-004 | 同一lineageのactive issuanceが最大1件 | constraint/transaction test |
| R4-005 | 同じP09再取込みが冪等である | ALREADY_IMPORTED＋件数assertion |
| R4-006 | 訂正前component/evaluationを履歴として保持する | history query |
| R4-007 | superseded/rejected/unresolvedを通常結果へ出さない | 04_results assertion |
| R4-008 | superseded componentをP11/P12/P13/評価へ進めない | negative component tests |
| R4-009 | active世代だけでforecast countとhit/missを計算する | summary numeric test |
| R4-010 | P09失敗時にtransactionが中途半端に適用されない | forced rollback test |
| R4-011 | 同一raw・同一条件の別URL sourceへ前処理を再利用できる | reuse vertical test |
| R4-012 | 再利用先sourceのP08がupstreamを正常参照できる | source 2 P08 accepted test |
| R4-013 | 再利用先からorigin artifact/rawまで追跡できる | lineage query |
| R4-014 | 再利用操作が冪等である | artifact/association count |
| R4-015 | raw/analyst/medium違いでは再利用しない | negative reuse matrix |
| R4-016 | schema/model/prompt互換性違いでは再利用しない | version negative tests |
| R4-017 | target boundaryを越えるcutoff artifactを再利用しない | cutoff reuse test |
| R4-018 | associationなしの別source artifact直指定を拒否する | invalid upstream test |
| R4-019 | workflowとP08が同じartifact applicabilityを判定する | shared-service/state test |
| R4-020 | P09 reject retryable後はRUN_P08を案内する | state transition test |
| R4-021 | retryable再抽出でaccepted P05/P07を再利用する | upstream ID assertion |
| R4-022 | P09 reject terminalが自動反復しない | terminal workflow test |
| R4-023 | P09 unresolvedが同じレビューを自動反復しない | no-loop workflow test |
| R4-024 | unresolved/rejected予想を正式集計しない | DB/report negative test |
| R4-025 | terminal sourceが他sourceの処理を止めない | two-source vertical test |
| R4-026 | 全source terminal・active 0件が有限完了する | completion test |
| R4-027 | NEXT_ACTIONSに正しいsource/artifact/component/reasonを出す | generated file assertion |
| R4-028 | unknown日時をnullとしてSchema/DBへ保存できる | schema＋DB test |
| R4-029 | unknown日時へ任意datetimeやtodayを入れない | negative/default test |
| R4-030 | unknown日時のP08を正式化・P11へ進めない | issuance/workflow test |
| R4-031 | P08 top-level knowledge_cutoffを固定Schemaで必須化する | fixed schema test |
| R4-032 | P08 cutoff > source boundary/made_atを拒否する | time leakage tests |
| R4-033 | P09 cutoff > reviewed made_atを拒否する | review time test |
| R4-034 | P09 corrected P08へ同じ時点検証を行う | corrected cutoff test |
| R4-035 | time evidence付きP09 correctでのみunknownを解決できる | positive correction test |
| R4-036 | legacy unknown timeをactiveへ昇格させない | migration/report test |
| R4-037 | 複数日basketの共通日1日をunevaluableにする | insufficient common dates test |
| R4-038 | 共通日不足でhit/miss/return/MFE/MAEを確定しない | evaluation field assertions |
| R4-039 | 共通日2日以上でweighted return/MFE/MAEが正しい | numeric basket test |
| R4-040 | 欠損instrumentを落とした部分basketを作らない | missing symbol test |
| R4-041 | coverage、input hash、weights、start/endを監査できる | evaluation audit query |
| R4-042 | basket修正後も単一symbol cacheを汚染しない | AAA/BBB regression |
| R4-043 | 0001/0005/0007/fixtureからheadへupgradeできる | migration command/tests |
| R4-044 | Alembic metadata差分0、FK check pass、件数保持 | command log＋assertions |
| R4-045 | repo/packaged docsとP08/P09 Schema/promptが同期する | sync/schema checks |
| R4-046 | clean wheelでhelp/init/source/P08/P09/docsが成立する | wheel vertical log |
| R4-047 | Ruff、mypy、pytest、diff checkがすべてpassする | command log |
| R4-048 | CHAT_HISTORY、秘密情報、生成物、未実装範囲を正確に扱う | hash/git scan/final review |

## READY判定条件

`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`は、R4-001～R4-048がすべてpassの場合だけ使用してください。

次はREADY判定の代替になりません。

- pytestの合計件数だけを示す。
- accepted済みP08ではなくneeds_review P08だけを試す。
- superseded行がDBに残ることを理由に通常結果への二重表示を許容する。
- workflowが再利用済みと表示するだけで、再利用先P08を実行しない。
- unresolvedを同じP09へ戻しながら「AIがいつか解決する」とする。
- unknown日時へ公開日・取得日・実行日を自動代入する。
- 1共通日から0%等を計算して数値が出たことを成功とする。
- network制約を理由にSQLite、CSV、state machine、wheelの内部failを隠す。

## FULL MVPとの区別

Round4を完了しても、総合点、複数アナリスト統合、現在上昇候補、時期・程度・早期実現の完全採点、期間不明の1/3/6/12か月観測、PNG等が未実装なら`FULL_MVP_READY`ではありません。

