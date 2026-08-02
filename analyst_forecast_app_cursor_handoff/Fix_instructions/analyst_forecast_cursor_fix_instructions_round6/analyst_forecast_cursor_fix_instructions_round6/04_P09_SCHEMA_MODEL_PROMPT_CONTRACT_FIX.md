# 04 — P09固定Schema・Pydantic・legacy adapter・promptの契約統一

## 目的

固定JSON Schema、Pydantic runtime、生成Schema、promptが別の入力契約を表す問題を修正します。AI出力の入口、application取込み、packaged resourceのどこで検査しても、同じpayloadを受理／拒否する状態にします。

## 正本契約表

### schema_version 2.1.0

| decision | 必須 | 禁止 |
|---|---|---|
| `reject` | `reject_disposition`=`retryable`または`terminal`、trim後非空`reject_reason` | `reject_terminal` |
| `accept` | 通常必須field | `reject_disposition`、`reject_reason`、`reject_terminal`、`forecast_operations` |
| `unresolved` | 通常必須field | `reject_disposition`、`reject_reason`、`reject_terminal`、`forecast_operations` |
| `correct` | `corrected_payload`。新規prompt出力では`forecast_operations`も必須 | 全reject field |

`correct`のlegacy互換としてoperation省略を許す場合は、02の旧1件→新1件だけに限定し、複数forecastでは固定Schemaまたはapplication validationで必ず拒否してください。

### schema_version 2.0.0 legacy import

| decision | 必須 | 禁止 |
|---|---|---|
| `reject` | `reject_terminal` boolean、trim後非空`reject_reason` | `reject_disposition` |
| `accept` / `correct` / `unresolved` | versionに応じた通常field | `reject_terminal`、`reject_disposition`、`reject_reason` |

- legacy `reject_terminal=true`はvalidation成功後に`terminal`、falseは`retryable`へ変換してよい。
- `reject_terminal`省略をfalse/retryableと推定しない。
- 新旧fieldが両方ある場合、一致していてもversion混在として拒否する。矛盾時だけ拒否する実装にしない。
- 既にDBへ保存されたhistorical payloadは削除・改変しない。新たなimportを安全に拒否する。

## Cursorへ渡す依頼文

```text
00～03に従い、P09の固定JSON Schema、Pydantic model/runtime validator、legacy adapter、prompt、packaged copyを統一してください。commit・pushはしないでください。

固定JSON Schema:
- Draft 2020-12としてcheck_schemaをpassさせる。
- schema_versionとdecisionのif/then/elseまたはoneOfで上記契約を表す。
- 2.1.0 rejectでdisposition/reasonをrequiredにする。
- reject_reasonとoperation reasonは空白だけを許可しないpatternまたは同等制約を持つ。
- reject以外にreject fieldが存在すること自体を拒否する。値nullでもfield混入を許可するかどうかを正本化し、推奨は存在禁止。
- ForecastCorrectionOperationのaction別required/null/禁止field、additionalProperties=falseを表す。
- duplicate ref、集合完全被覆はJSON Schemaだけでは困難なためapplication validatorの責務と明記する。

Pydantic/runtime:
- extra fieldをforbidする。
- 固定Schemaと同じversion/decision matrixをruntimeで拒否する。
- validatorがfieldを黙って補完して不正入力をvalidに変えない。
- legacy変換はvalidation成功後だけ行う。
- `model_json_schema()`にも主要required/enum/conditional契約が現れる設計を優先する。runtime validatorでしか表せない条件は、固定Schemaとの差と二重test理由を文書化する。
- validation errorはJSON pathと安定reason codeを返す。

Prompt:
- P09.md.j2へdecision別のfield規則と、retryable/terminalの選択基準を記載する。
- retryable reject、terminal reject、複数forecast correctの有効JSON例を各1件載せる。
- field省略例、新旧field混在例、未申告operationを有効例にしない。
- correct例は02の集合契約を満たす。
- prompt catalogとUSER_MANUALの例も同じ契約にする。

Package:
- repoとwheel内の固定Schema、prompt、catalogを同期する。
- sync scriptがP09 Schema/promptも検査対象に含むことを確認する。
```

## 必須契約matrix test

次の各payloadを、固定JSON Schemaの`Draft202012Validator.iter_errors`とPydantic runtimeの両方へ通し、期待が一致することをassertしてください。

1. 2.1 reject + disposition retryable + nonempty reason: valid。
2. 2.1 reject + disposition terminal + nonempty reason: valid。
3. 2.1 rejectでdispositionなし、reasonなし、各片方なし、空白reason: invalid。
4. 2.1 rejectにreject_terminal追加: invalid。
5. 2.1 accept/correct/unresolvedへ各reject fieldを1つずつ追加: invalid。
6. 2.0 reject + reject_terminal true/false + reason: valid、変換結果も確認。
7. 2.0 rejectでterminalなしまたはreasonなし: invalid。
8. 2.0 rejectにdispositionだけ、terminal+disposition両方（一致・矛盾）: invalid。
9. 2.0 accept/correct/unresolvedへreject field追加: invalid。
10. operationのaction別field欠落・禁止field・空白reason: invalid。
11. additional property: invalid。
12. 固定Schemaとpackaged Schemaのbyte/hash一致、repo/packaged prompt一致。

## workflow test

- valid retryable rejectは`p08_reextract_required`等の有限な再抽出状態へ進む。
- valid terminal rejectは終端状態へ進む。
- unresolvedは既存有限終端を維持する。
- invalid/legacy省略rejectは状態遷移やartifact世代を変更しない。
- 複数source runで1 sourceのterminal/unresolvedが他sourceを妨げない。

## 完了条件

- 固定Schema単体で、2.1 rejectの区分・理由省略を拒否する。
- Pydanticがreject以外へのreject fieldとversion混在を拒否する。
- promptの全例が固定Schema・Pydanticの両方でvalidである。
- repoとwheelの契約resourceが一致する。

