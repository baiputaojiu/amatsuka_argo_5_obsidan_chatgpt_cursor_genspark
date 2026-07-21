# 06 — P09 reject区分のSchema・prompt契約

## 目的

`reject_terminal: bool = false`の省略が自動的にretryableを意味し、retryable理由も不要で、promptに選択規則がない問題を修正します。AIがreject時にretryableかterminalかを必ず明示し、Pythonが有限状態へ決定的に遷移できる契約にします。

## Cursorへ渡す依頼文

```text
00～05に従い、P09 rejectの固定Schema、Pydantic、prompt、legacy adapter、state machineを同期してください。commit・pushはしないでください。

新規P09 Schema versionを上げ、reject_terminal boolの代わりに次を推奨します。

- reject_disposition: retryable | terminal
- reject_reason: 非空文字列

条件:

- decision=rejectではreject_dispositionとreject_reasonを両方必須にする。
- decision!=rejectでは両fieldを禁止またはnull限定にする。
- missing dispositionをretryableへ暗黙defaultしない。
- 空白だけのreasonを拒否する。
- retryableは「P08を再抽出すれば、同じ受理済みP05/P07を使って修正可能」な場合だけ。
- terminalは「原文・帰属・時刻・証拠上、同じsourceから有効な正式予想を得られない」場合。
- unresolvedはrejectとは別であり、同じP09を無限反復せず既存terminal/excluded状態へ進む。

legacy compatibility:

- 既存P09 2.0.0のreject_terminal=true + reasonはterminalへ明示変換してよい。
- reject_terminal=false + reasonありをretryableへ変換する場合はlegacy adapterを明示し、変換記録を残す。
- dispositionもreasonもないlegacy rejectをretryableへ黙って昇格させない。legacy_reject_disposition_unknown等として再レビューまたは安全なterminal/excludedにする。
- 新規2.1.0 outputでは旧fieldを受理しない。

P09.md.j2へ次を明記してください。

- decision 4種の意味
- reject時の必須field
- retryableとterminalの判定基準
- 各1つ以上のJSON例
- reasonには再試行で何を直すか、またはterminal根拠を書く
- knowledge cutoffと市場結果不使用

固定JSON Schema、Pydantic generated schema、prompt catalog、USER_MANUAL、packaged promptを同期してください。if/thenまたはdiscriminated unionで条件必須を機械表現し、prompt文章だけに依存しないでください。

state transition:

- retryable → p08_reextract_required、accepted P05/P07を再利用、次行動EXTRACT_FORECASTS。
- terminal → p08_rejected_terminal、同sourceを自動反復しない。
- unresolved → p08_review_unresolved_terminal、同じreviewを再案内しない。
- 複数sourceではterminal sourceが他sourceを止めない。
```

## 必須テスト

1. 新Schemaでreject_disposition省略を拒否する。
2. 新Schemaでreject_reason省略・空白を拒否する。
3. retryable + reasonを受理し、EXTRACT_FORECASTSへ進む。
4. terminal + reasonを受理し、有限終端する。
5. decision=accept/correct/unresolvedへreject fieldを付けると拒否する。
6. 旧reject_terminal fieldを新Schemaで拒否する。
7. legacy true + reasonの変換結果と監査記録を確認する。
8. legacy false/no reasonをretryableへ黙って変換しない。
9. Pydantic generated schemaと固定Schemaのrequired/enum/conditionalが一致する。
10. repo P09 promptとpackaged P09 promptが一致する。
11. promptにreject_disposition、retryable、terminal、reject_reasonの説明と例がある。
12. retryable/terminal/unresolvedのNEXT_ACTIONSが有限である。
13. terminal sourceが別sourceのP08処理を止めない。

## 完了条件

- 新規rejectに区分・理由の省略余地がない。
- Schema、Pydantic、prompt、state machineが同じ契約を表す。
- legacy省略値を安全側で扱い、無根拠なretryable化をしない。

