# 02 — forecast_operationsの完全被覆・一対一対応

## 目的

`forecast_operations`が「存在する」だけで不正な訂正を受理する問題を修正します。reviewed P08とcorrected P08のforecast集合をoperationが過不足なく説明し、update部分を一対一対応にします。

## 正本となる集合契約

reviewed P08の全forecast_ref集合を`O`、corrected P08の全forecast_ref集合を`N`とします。

- `Uo`: update operationの`reviewed_forecast_ref`
- `Un`: update operationの`corrected_forecast_ref`
- `R`: remove operationの`reviewed_forecast_ref`
- `A`: add operationの`corrected_forecast_ref`

受理条件は次のすべてです。

```text
O = Uo ⊎ R
N = Un ⊎ A
|Uo| = |Un| = update operation数
各updateは旧1件 ↔ 新1件
```

`⊎`は重複のない直和です。旧ref・新refは各側で一度だけ使用でき、存在しないref、未申告ref、重複、多対一、一対多を許可しません。

## Cursorへ渡す依頼文

```text
00、01に従い、P09 correctのoperation validationと適用transactionを修正してください。commit・pushはしないでください。

1. validation順序
- reviewed artifact/source/run/input_hash/cutoff等の既存検証後、issuance/component/evidence/operation/review resolutionを変更する前にoperation集合を検証する。
- reviewed/corrected各payload内のforecast_ref自体がuniqueであることを先に確認する。
- operationsがある場合は、単一forecastでも集合契約を完全適用する。
- operationsがないlegacy入力は、旧1件・新1件の決定的updateだけ互換経路を許可してよい。旧または新が複数、add/removeを推定する必要がある、refが曖昧な場合は拒否する。
- 配列index、local DB row順、old_issuances[0]をidentityやfallbackに使わない。

2. action別field契約
- update: reviewed_forecast_refとcorrected_forecast_refを両方必須、各1件。reasonはtrim後非空。
- add: reviewed_forecast_refはnull/absent、corrected_forecast_refは必須。reasonはtrim後非空。
- remove: reviewed_forecast_refは必須、corrected_forecast_refはnull/absent。reasonはtrim後非空。
- actionに反するref、additional field、空白reasonを拒否する。

3. 推奨reason code
- duplicate_reviewed_forecast_ref
- duplicate_corrected_forecast_ref
- unknown_reviewed_forecast_ref
- unknown_corrected_forecast_ref
- duplicate_operation_reviewed_ref
- duplicate_operation_corrected_ref
- incomplete_reviewed_forecast_coverage
- incomplete_corrected_forecast_coverage
- invalid_forecast_operation_fields
- ambiguous_legacy_forecast_mapping

既存error体系へ合わせて名称を調整してよいが、testと文書で機械判定可能な安定codeにしてください。

4. 適用
- 全集合検証成功後だけ、各update/add/removeを対応するlineageへ適用する。
- updateは対応oldのlineageを引継ぎ、generation+1、pairwise supersedes/superseded_byとする。
- addは自分自身をlineage rootとするgeneration 1。
- removeは旧lineageを非activeにし、履歴を保持する。
- operationにないcorrected forecastを自動insertしない。operationにないoldをactiveのまま残さない。
- new component/evidenceは対応するnew issuanceへ所属させる。
- 全適用、operation監査行、P09 resolution、artifact classificationを1transactionにする。
- invalid payloadまたは途中例外では、active、issuance、component、evidence、operation、resolution、通常結果件数を変更しない。
- 同一P09再取込みはALREADY_IMPORTEDで冪等にする。

5. 監査
- accepted operationだけについて、review artifact ID、old/new forecast_ref、old/new issuance ID、action、reasonを復元可能にする。
- rejected invalid payloadを監査用import行として残す既存仕様がある場合は、その1行以外に副作用がないことを明記する。
- 04_results、summary、forecast count、NEXT_ACTIONS、評価はactive generationだけを見る。
```

## 必須negative test matrix

1. old A/B → new X、update A→X・B→X（多対一）。
2. old A → new X/Y、update A→X・A→Y（一対多）。
3. old A/B → new A2/B2/C、operationsがA→A2・add Cだけ（旧B、新B2未申告）。
4. 旧Bだけ未申告。
5. 新B2だけ未申告。
6. 存在しないold ref。
7. 存在しないnew ref。
8. reviewed payload内のforecast_ref重複。
9. corrected payload内のforecast_ref重複。
10. updateの片側null、add/removeの禁止側refあり、空白reason。
11. 複数forecastでoperationsなし。
12. operation配列は完全でも同一refが別actionで再使用される。

各caseでerror codeに加え、issuance/component/evidence/operation/review resolutionとactive ID集合がbefore/after同一であることをassertしてください。

## 必須positive test

1. old A/B → new A2/B2を逆順payloadでupdateし、別lineageとpairwise linkを保つ。
2. old A/B → new A2/Cを、A update、B remove、C addで処理し、A2/Cだけactiveにする。
3. ref名が同じA→Aの内容訂正を明示updateとして処理する。
4. legacy単一old→単一newの許可経路が決定的である。
5. 同一P09再取込みで全件数不変。
6. operation適用途中の強制例外でtransaction rollback。

## 完了条件

- `O = Uo ⊎ R`と`N = Un ⊎ A`をコードとtestで直接確認する。
- 多対一、未申告旧、未申告新を正式P09 ingestが拒否する。
- valid update/add/remove、reorder、idempotency、rollbackが成立する。
- Round5で改善したactive DB一意制約とinactive component guardが非回帰である。

