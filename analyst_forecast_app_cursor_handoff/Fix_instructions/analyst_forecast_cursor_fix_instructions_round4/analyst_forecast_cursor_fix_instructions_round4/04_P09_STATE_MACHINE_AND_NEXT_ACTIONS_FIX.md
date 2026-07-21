# 04 — P09 reject/unresolvedの有限状態機械と次行動

## 目的

P09 `reject`後に完了済みP05/P07を再案内し、`unresolved`後に同じP09を無期限反復する問題を修正します。preprocess状態とforecast extraction/review状態を分離し、sourceごとの次行動を決定的にします。

## Cursorへ渡す依頼文

```text
00～03に従い、RunSourceとAI artifactの状態機械、workflow、NEXT_ACTIONSを修正してください。commit・pushはしないでください。

現在はRunSource.processing_statusのneeds_review等がP05/P07とP08を十分に区別できず、P09 reject後にRUN_PREPROCESSへ戻ります。preprocessingとforecast extraction/reviewを別状態軸または明確な複合状態で表現してください。

最低限、sourceごとに次を区別できるようにしてください。

- preprocess_pending
- preprocess_needs_review
- preprocess_accepted
- p08_pending
- p08_needs_review
- p08_reextract_required_retryable
- p08_rejected_terminal
- p08_review_unresolved_terminal
- processed_no_forecast
- processed_no_formal_forecast
- processed_with_active_forecast
- processed_with_superseded_or_excluded_forecast_only

既存名称を使っても構いませんが、状態の意味、遷移元、遷移先、次行動をdocsとtestで固定してください。同じ文字列を前処理とP08レビューの両方の意味に使わないでください。

P09 decisionの遷移:

- accept:
  - 02のlifecycle規則に従う。
  - valid active forecastがあればsourceをprocessed_with_active_forecastへ進める。
  - 正式化0件ならprocessed_no_formal_forecast等のterminalへ進める。

- correct:
  - corrected payloadを完全検証してから02のactive世代へ切り替える。
  - validならterminal/next P11、invalidならP09 artifact自体をrejectし、元状態を壊さない。

- reject:
  - 「AIレビューがP08出力を不採用」と「原文前処理が不正」を混同しない。
  - acceptedなP05/P07 upstreamを維持する。
  - 理由が再抽出で直せるならp08_reextract_required_retryableとし、次行動はRUN_P08。新しいP08は同じaccepted upstreamを使う。
  - 原文から正式予想を決定不能、対象者発言でない等のterminal理由ならp08_rejected_terminalとして、そのsourceでP08/P09を自動反復しない。
  - retryable/terminalはPython側の限定enumまたは明示fieldで決め、自由文だけから推測しない。

- unresolved:
  - 同じreviewed_artifactへのP09を自動再案内しない。
  - p08_review_unresolved_terminalとして理由、review artifact、attempt/model/version、日時を保存する。
  - 正式予想へ入れず、accepted済み予想の再レビューなら02に従ってactive集計から除外する。
  - 将来、別モデル/新証拠で明示的に再開するCLIまたはapplication actionは用意してよいが、自動loopにしない。再開履歴を残す。

next-action priorityをsource単位で固定してください。

1. preprocess needs review → RUN_P06
2. preprocess pending → RUN_P05またはRUN_P07
3. P08 needs review → RUN_P09
4. P08 reextract required → RUN_P08
5. P08 pending → RUN_P08
6. 全sourceがterminalになった後だけactive componentのP11/P12/P13/market evaluation
7. active forecastが0件で全source terminalならCOMPLETE_NO_ACTIVE_FORECASTまたは同等の完了状態

複数sourceでは、1件のterminal unresolvedが他sourceの処理を永久に止めないようにしてください。NEXT_ACTIONSには、action IDだけでなくrun_id、source_id、reviewed/upstream artifact ID、component ID、理由、retryable/terminalを表示してください。
```

## 必須テスト

1. P09 reject retryable後の次行動がRUN_P08でありRUN_PREPROCESSではない。
2. retryable再抽出が既存accepted P05/P07を参照できる。
3. P09 reject terminal後に同じP08/P09を再案内しない。
4. P09 unresolved後に同じP09を再案内しない。
5. unresolved sourceが正式予想、P11、市場評価へ進まない。
6. accepted済みP08のunresolved再レビュー後、旧予想をactive結果へ残さない。
7. invalid corrected payloadで元P08/source状態を壊さない。
8. processed_no_forecastをP08再実行へ戻さない。
9. processed_no_formal_forecastをP08再実行へ戻さない。
10. source A unresolved terminal、source B pendingならsource Bの処理を案内する。
11. 全source terminal・active forecast 0件なら有限の完了状態になる。
12. 全source terminal・active forecastありならP11へ進む。
13. NEXT_ACTIONSに正しいsource/artifact/component IDとterminal理由が出る。
14. 明示的reopen操作を行わない限りterminal状態が変化しない。
15. reopenした場合はattempt lineageを保持し、同じartifactを無限再利用しない。

## 完了条件

- P09の全decisionが有限状態遷移を持つ。
- 完了済み前処理を不必要に再実行しない。
- terminal unresolvedを隠さず、他sourceの進行を止めない。

