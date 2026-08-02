# 02 — 非YouTube媒体と安全な再利用の修正

## 目的

ブログ・X・Web案件でP05が存在しないのにP08がP05を必須とする行き止まりを解消します。また、同一rawを安全に再利用できる条件判定を、未使用helperではなく実workflowへ接続します。

## コピペ用依頼文

```text
fix_instructions_round2/00_START_HERE.mdの共通拘束条件と、01で統一した状態機械に従い、非YouTube媒体の前処理と安全なartifact再利用を実装してください。commit・pushはしないでください。

現状の問題は次のとおりです。

- P05はYouTube案件にだけ生成される。
- Schema 2.0.0のP08はp05_artifact_idを必須とする。
- P08 prompt catalogは常にprocessed/P05を入力としている。
- ブログ、X、Web案件にはP05 promptがなく、P08を正規経路で実行できない。

既存仕様にあるP07「テキスト原文整理」を実装してください。望ましい媒体別経路は次です。

- YouTube: raw → P05 話者・segment整理 → P08
- blog: raw → P07 著者・引用・本文整理 → P08
- X: raw → P07 投稿者・引用投稿・返信関係整理 → P08
- web: raw → P07 本人直接発言・記者要約・第三者引用の整理 → P08

P07用のversion付きPydantic model、固定JSON Schema、prompt template、catalog、DB artifact保存、案件snapshot、workflow、監査log、テストを追加してください。rawは一切変更しないでください。

P08の上流参照をP05専用から媒体共通へ変更してください。例えばupstream_artifact_idとupstream_prompt_idを持たせ、P05またはP07の受理済みartifactだけを許可します。既存P08 Schema 2.0.0のp05_artifact_idは後方互換で読めるようにし、新規Schema versionで置き換えてください。

必ず次を検証してください。

- upstream artifactのrun_id、source_id、prompt種別、classification、output hashがP08と一致する。
- YouTube sourceに無関係なP07、非YouTube sourceに無関係なP05を誤って参照しない。
- P07は本人執筆、本人の直接引用、第三者要約を区別する。
- 第三者要約だけの文は本人の正式予想として登録しない。
- P08 forecasts=[]を各媒体で処理済みとして保存できる。

raw artifact再利用helperを実workflowへ接続してください。現在のcan_reuse_processed_artifactがテストからしか呼ばれていない状態を解消します。再利用条件は最低限、raw hash、source occurrence、analyst、prompt ID、prompt version、model、上流artifact hash、話者条件が一致することです。

安全に再利用できる場合は、別案件からも実入力・artifact・監査履歴へ到達できるrun associationを作り、同じP05/P07を再実行させないでください。別アナリスト、別話者条件、prompt/model/version変更時は再利用しないでください。同じ予想表明を別runへ関連付ける場合、同一発言を新しい予想表明として二重カウントしないでください。

必須テスト：

1. blogだけの案件がP07→P08まで進む。
2. Xだけの案件がP07→P08まで進む。
3. webだけの案件がP07→P08まで進む。
4. YouTubeは従来どおりP05→P08。
5. 各媒体で予想0件を正常処理できる。
6. mixed-media案件でsourceごとにP05/P07を正しく選ぶ。
7. P08が別run、別source、未受理、hash不一致の上流artifactを拒否する。
8. 同一raw・同一analyst・同一処理条件の別runで安全な前処理artifactを再利用する。
9. 別analyst、別model、別prompt versionでは再利用しない。
10. 再利用後も両runの案件フォルダから実在する入力または参照manifestへ到達できる。

説明書、prompt catalog、workflow図、実装状況を新しい媒体経路に合わせて更新し、ruff format、ruff check、mypy、pytestを実行してください。
```

## 完了条件

- 選択可能な4媒体すべてに実行可能な前処理経路がある。
- P08がP05だけへ固定されていない。
- 安全な処理済みartifact再利用が実workflowで働く。
- 媒体・話者・案件の由来が混線しない。
