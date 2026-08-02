# 03 — P11/P12/P13/評価のactive component入口制御

## 目的

訂正前のsuperseded componentをP11へ進められ、P12/P13にも同じ検査がない問題を修正します。対象解決と評価のすべての入口が、同じactive component policyを使うようにします。

## Cursorへ渡す依頼文

```text
00～02に従い、P11/P12/P13/市場評価のcomponent参照検証を統一してください。commit・pushはしないでください。

共通のrequire_active_component_context相当のservice/repository methodを作り、少なくとも次を一度に確認してください。

- ForecastComponentが存在する。
- 親ForecastIssuanceが存在する。
- issuance.lifecycle_status == active。
- issuance.lineage_root_idが存在し、同一lineageにactiveが1件だけである。
- issuance.made_atが確定している。
- issuanceのrun、analyst、source、P08 artifactが要求contextと一致する。
- componentがそのactive issuanceに属する。
- legacy_conflict、superseded、rejected_or_withdrawn、review_unresolved_excluded、unknown-timeを許可しない。

拒否理由codeは共通化してください。推奨はinactive_forecast_componentとし、detailにlifecycle statusを持たせます。P11だけ別文言、評価だけValueErrorという分散を避けてください。

適用箇所:

1. P11 payload validation
2. P11 candidate materialization直前
3. P12 payload validation
4. P12 review/mapping materialization直前
5. P13 payload validation
6. P13 adjudication/mapping lock直前
7. evaluate_component
8. workflow/NEXT_ACTIONSのcomponent列挙
9. result/current forecast/summaryの対象列挙

P12では、P11 proposalを作った時点ではcomponentがactiveでも、P12取込み前にP09 correctでsupersededになる競合caseを拒否してください。P13も同様に、P12後・P13前のsupersedeを拒否してください。上流artifactが過去にacceptedだったことだけではactive性を満たしません。

validationとinsertの間で状態が変わるTOCTOUを避けてください。同じDB transaction/sessionで再検査し、active unique constraintと組み合わせてください。

拒否時:

- AI import/artifactの監査記録は既存方針に従ってrejectedとして残してよい。
- target resolution candidate、review、adjudication、target mapping、mapping lockを新規作成・更新しない。
- 旧componentの既存mapping/evaluationは履歴として保持する。
- active新componentへ旧mappingを暗黙にコピーしない。
- NEXT_ACTIONSで旧component IDを再案内しない。
```

## 必須テスト

1. superseded componentを指定したP11を`inactive_forecast_component`で拒否する。
2. 1でTargetResolutionCandidateRecordを増やさない。
3. P11 accepted後にcomponentをsupersedeし、そのP11を参照するP12を拒否する。
4. 3でreview/mapping/lockを増やさない。
5. P12 accepted後にcomponentをsupersedeし、そのchainのP13を拒否する。
6. 5でadjudication/mapping lockを増やさない。
7. rejected、unresolved-excluded、legacy-conflict、made_at nullも同じpolicyで拒否する。
8. activeな新世代componentのP11→P12→必要時P13は正常に通る。
9. superseded componentの市場評価を拒否する既存挙動を維持する。
10. workflowとNEXT_ACTIONSにinactive component IDが出ない。
11. 04_resultsの通常sectionにinactive componentが出ない。
12. validation後・materialization前の状態変更を模したtestで副作用なしにrollbackする。
13. P11/P12/P13/evaluationが同じpolicy helperを使用していることをtestまたは構造監査で示す。

## 完了条件

- inactive componentを対象解決・評価のどの段階からも進められない。
- 状態変更が途中に入ってもmapping等の副作用を残さない。
- active新世代だけが次行動へ出る。

