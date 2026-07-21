# 02 — 複数forecastの訂正lineageとactive一意保証

## 目的

1つのP08に複数の予想がある場合、P09 `correct`後の全新予想が最初の旧予想lineageへ束ねられる問題を修正します。配列位置ではなく明示的な対応関係で世代を作り、各論理予想lineageにactiveを最大1件だけ許可します。

## Cursorへ渡す依頼文

```text
00、01に従い、P09 correctionとForecastIssuance lineageを修正してください。commit・pushはしないでください。

現在のold_issuances[0]を全new issuanceへ適用する処理を廃止してください。P09 corrected_payloadのforecast配列順だけをidentityにしてはいけません。

訂正対応契約:

- reviewed P08のforecast_refを、そのartifact内の安定した論理参照として扱う。
- P09 decision=correctでは、旧forecast_refと訂正後forecast_refの対応を機械判定できるようにする。
- 推奨はP09 2.1.0へforecast_operationsを追加する方式:
  - action: update / add / remove
  - reviewed_forecast_ref: update/removeで必須、addではnull
  - corrected_forecast_ref: update/addで必須、removeではnull
  - reason: 必須
- updateは旧1件:新1件、addは新規lineage、removeは旧lineageのactive終了を意味する。
- 全旧forecast_refと全訂正後forecast_refがoperationで過不足なく一度だけ説明されなければrejectする。
- 同一refの重複、1旧ref→複数新ref、複数旧ref→1新ref、存在しないref、未申告の追加・削除をrejectする。
- ref名を変えない単純訂正も明示updateとして扱う。配列順変更は結果へ影響させない。
- backward compatibleなlegacy P09 correctに明示対応がない場合、単一旧・単一新だけは決定的に対応してよい。複数予想では曖昧な自動推定をせずneeds_review/rejectedにする。

DB lifecycle:

- update:
  - 旧lineage_root_idを新issuanceへ引き継ぐ。
  - new.generation = old.generation + 1。
  - new.supersedes_forecast_issuance_id = 対応するold ID。
  - old.superseded_by_issuance_id = 対応するnew ID。
  - oldはsuperseded、新だけactive。
- add:
  - 新issuance ID自身をlineage_root_idにし、generation=1、active。
- remove:
  - 旧issuanceをwithdrawn_by_correction等の非active状態にし、置換先はnullでよい。
  - 旧component/evidence/mapping/evaluationは履歴として保持する。
- non-formal、unknown time、attribution未確認のforecastはartifactには保持してもactive issuanceを作らない。operation検証とformal materializationを混同しない。

active一意保証:

- lineage_root_idがnullでないactive ForecastIssuanceは、同一lineage_root_idにつき最大1件。
- SQLiteのpartial unique index等、DBで保証できる制約を追加する。
- application transaction guardも併用し、意味の明確なerror codeを返す。
- 旧を非activeへ切り替え、新をactiveへ作る処理、review resolution、artifact保存を1transactionにする。
- unique制約違反や途中例外でold/new/reviewの一部だけが適用されないようrollback testを持つ。
- 同一P09の再取込みはALREADY_IMPORTEDで、issuance/component/evidence/operation行を増やさない。

lineage operationを独立tableへ保存するか、監査可能な列・JSONへ保存してください。少なくともreview artifact ID、old/new issuance ID、old/new forecast_ref、action、reasonをqueryで復元できる必要があります。

results、forecast count、target resolution、evaluation、NEXT_ACTIONSはactive generationだけを見る既存共通queryを維持してください。訂正で削除した予想、旧世代、legacy conflictを通常結果へ出さないでください。
```

## 必須テスト

1. 旧forecast A/Bを新A/Bへ訂正し、2つの異なるlineage rootを維持する。
2. old A→new A、old B→new Bの`supersedes`と`superseded_by`が一対一である。
3. corrected payloadの配列順をB/Aへ変えてもlineage対応が変わらない。
4. 同一lineageへactiveを2件insertしようとするとDB constraintで失敗する。
5. constraint失敗時に旧activeとreview状態がrollbackされる。
6. A更新、B削除、C追加を同じP09で処理し、A/Cだけがactiveになる。
7. addの新lineage rootは自分自身、generation=1である。
8. removeした旧lineageにactiveがなく、履歴は残る。
9. 1旧ref→2新ref、2旧ref→1新refをrejectする。
10. operationに存在しないrefをrejectする。
11. operationで説明されないforecast追加・削除をrejectする。
12. 複数forecastのlegacy P09 correctを配列indexで推定しない。
13. 同一P09再取込みで全件数が不変である。
14. active forecast count、04_results、NEXT_ACTIONSに旧・削除世代が出ない。
15. evidence/componentが対応する新issuanceへ正しく属する。

## 完了条件

- 複数forecast訂正が一対一lineageとして監査できる。
- 同一lineageのactive最大1件をDBとapplicationの両方で保証する。
- add/remove/reorder/idempotencyを含むnegative/positive testがある。

