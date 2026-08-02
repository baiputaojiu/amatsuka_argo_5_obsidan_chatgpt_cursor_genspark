# 02 — accepted済みP08の再レビューと正式予想lifecycle

## 目的

accepted済みP08へのP09 `accept`でtransactionが失敗し、`correct`で正式予想が二重化する問題を修正します。履歴は保持しつつ、workflow、結果、対象解決、市場評価が常に「有効な最新版」だけを見る構造にします。

## Cursorへ渡す依頼文

```text
00、01に従い、P08/P09とForecastIssuance lifecycleを修正してください。commit・pushはしないでください。

次の不変条件をDBとapplication serviceで保証してください。

1. 同じ論理予想lineageでactiveなForecastIssuanceは最大1件。
2. review artifact、訂正前issuance/component/evidence/evaluationは監査履歴として削除しない。
3. 結果、NEXT_ACTIONS、P11/P12/P13、市場評価、現在予想はactive issuance/componentだけを参照する。
4. 同じP09成果物の再取込みはALREADY_IMPORTEDとなり、DB件数・active状態を変えない。
5. P09適用は一transactionで行い、失敗時にclassified file、review resolution、issuanceが半端に残らない。

推奨データモデルはappend-only generationです。名称は既存規約へ合わせてよいですが、少なくとも次を表現してください。

- ForecastIssuanceのlineage/root IDまたはsupersedes_forecast_issuance_id
- revision/generation
- lifecycle status: active / superseded / rejected_or_withdrawn / review_unresolved_excluded等
- superseded_at、superseded_by、lifecycle reason、review artifact ID
- component/evidenceがどのissuance generationに属するか
- DB constraintまたはtransaction内guardによるactive 1件保証

P09 decisionごとの動作:

- accept + reviewed P08=needs_review:
  - 通常P08の意味検証を通過したpayloadを一度だけmaterializeする。
  - active issuanceを1件だけ作り、reviewed artifactをresolvedにする。

- accept + reviewed P08=accepted:
  - 既存accepted P08/issuanceを承認した記録だけを保存する。
  - _insert_p08を再実行してissuance/component/evidenceを増やさない。
  - unique violationやRuntimeErrorを起こさない。

- correct + reviewed P08=needs_review:
  - corrected_payloadを通常P08と同じ完全検証へ通す。
  - 有効なissuanceがまだなければ1件だけ作る。

- correct + reviewed P08=accepted:
  - 旧issuanceをsupersededにし、corrected payloadから新generationを作る。
  - 新generationだけをactiveにする。
  - 同一transaction内でactiveを切り替える。
  - 旧componentのmapping/evaluationは履歴として保持するが、active結果や次行動へ出さない。
  - 新componentは訂正内容から作り直し、必要なP11から進める。旧mappingを暗黙に流用しない。

- reject:
  - 具体的なnext stateは04に従うが、accepted済みissuanceをrejectする場合はactive結果から除外し、履歴は保持する。

- unresolved:
  - 具体的なnext stateは04に従う。accepted済み予想を再レビューして解決不能なら、信頼できるactive予想として集計し続けない。理由付きexcluded/blockedにする。

論理予想lineageの識別をAI出力の配列indexだけに依存させないでください。既存stable forecast_issuance_id、reviewed artifact lineage、または新しいroot IDを使い、複数forecastを含むP08で各forecastの対応が監査できるようにしてください。corrected payloadで予想の追加・削除がある場合も、旧世代と新世代の関係を追跡してください。

results.py、workflow.py、evaluation.py、target resolution query、summary countを横断確認し、superseded/rejected/unresolved-excludedを除外する共通query helperまたはrepository methodを使ってください。各所に異なるactive判定を重複実装しないでください。

既存legacy issuanceは自動でactiveにしてよい条件をmigrationで限定してください。同じlineageに複数候補がある場合は勝手に最新と推定せず、legacy_conflict/excludedとして報告してください。
```

## 必須テスト

1. needs_review P08 + P09 acceptでactive issuanceが1件だけ作られる。
2. accepted P08 + P09 acceptで例外がなく、issuance/component/evidence件数が増えない。
3. accepted P08 + P09 correctで旧1件がsuperseded、新1件だけがactiveになる。
4. correct再取込みでactive/total件数が変わらない。
5. 複数forecastを含むP08のうち1件訂正でもlineage対応が追跡できる。
6. 訂正で予想を削除した場合、削除対象がactive結果へ残らない。
7. 訂正で予想を追加した場合、新規lineageとして一度だけ作られる。
8. 旧component/evaluationはDBに残るが`04_results`へactive行として出ない。
9. superseded componentをP11/P12/P13/market evaluationへ指定すると拒否する。
10. NEXT_ACTIONSが旧component IDを案内しない。
11. active generationだけでforecast count、hit/missが計算される。
12. P09 transactionを意図的に失敗させてもactive世代とreview statusが中途半端に変わらない。
13. legacy複数候補を無根拠にactiveへ昇格させない。

## 完了条件

- accepted P08を再レビューしても失敗・二重化しない。
- 訂正履歴を失わず、有効な最新版だけが成績へ入る。
- active判定が結果、workflow、対象解決、市場評価で一致する。

