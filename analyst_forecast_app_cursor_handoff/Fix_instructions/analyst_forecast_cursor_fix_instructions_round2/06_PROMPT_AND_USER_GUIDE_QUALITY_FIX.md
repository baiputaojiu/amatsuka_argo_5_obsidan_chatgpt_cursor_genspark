# 06 — プロンプト内容・Vault文書・モデル設定の修正

## 目的

Schema名だけを示す短いプロンプトと、数行だけのVault説明書を、初回利用者とAIが単独で理解できる内容へ改善します。高性能モデル名を警告するだけで設定方法がない問題も直します。

## コピペ用依頼文

```text
fix_instructions_round2/00_START_HERE.mdの共通拘束条件と、01～05で確定したpipelineに従い、中央prompt template、案件snapshot、Vaultへseedする説明書、AIモデル設定を更新してください。commit・pushはしないでください。

現在のP05/P08/P11/P12/P13 templateは、目的、入力、Schema名と一文の依頼だけで、仕様で決めた判定基準をAIへ十分伝えていません。各promptを、単独で読んでも実行条件、入力、禁止事項、判断基準、出力Schema、確認手順が分かる内容にしてください。

特にP08には次の認定基準を明記してください。

- 明示的予想と方向性予想は正式候補。
- 投資推奨は正式候補として扱い、予想種別をrecommendationとして分離。
- 条件付き予想は条件と本命度を保存。
- 上昇・下落の複数シナリオを列挙しただけなら除外し、本人が本命、確率、優先順位を示した場合だけ正式候補。
- 現状分析、過去説明、単なる可能性、選好、第三者の見解は除外または非予想分類。
- 同じ主張を後日改めて表明した場合は新しいissuanceとして個別評価し、同一forecast groupへ関連付ける。
- 同時転載、リンク共有、切り抜きだけなら新issuanceにしない。
- 上昇後下落等はcomponentへ分割するが、予想件数自体を水増ししない。
- 発言者明示の期間、AI推定期間、system defaultを混同しない。
- 期間不明を発言者の1年予想へ変換しない。
- 原文にないsymbol、日付、数値、確率を補わない。

P05/P06には、話者交代、司会者の質問、相づち、他出演者、調査対象者の過去発言、動画メタデータ、knowledge cutoff、判断不能の扱いを具体的に書いてください。P07には、本人執筆、直接引用、記者要約、引用投稿、返信、転載の区別を書いてください。P09には否定誤読、条件欠落、本命誤認、時期・程度の捏造を重点確認するよう書いてください。

P11/P12/P13には、原文明示対象の優先順位、発言時点に存在した公式指数・ETF・固定basket、最大3候補、結果確認後の後知恵禁止、修正候補、unresolvable、別component参照禁止を明記してください。

各案件snapshotには次を実値で埋めてください。

- run ID、source IDまたはcomponent ID
- 必要なupstream artifact IDとinput hash
- 実在する入力ファイル
- 実在する出力先
- 固定Schemaのファイル名
- prompt versionとtemplate hash
- 設定された高性能モデル名

`<SOURCE_ID>`や`<COMPONENT_ID>`を残す必要がある段階では、どのコマンドで実IDを確認するかを書き、AIが推測しないようにしてください。

VaultへseedするREADME、AI_WORK_GUIDE、STARTUP_GUIDE、USER_MANUAL、SYSTEM_SPECIFICATION、PROMPT_CATALOG、PYTHON_TASK_CATALOG、TROUBLESHOOTING、DECISION_LOG、OPEN_QUESTIONS、FUTURE_ROADMAPを数行の要約だけにしないでください。repositoryの正式docsと整合する、後からAIが読んでも背景・処理・制約・未実装・次行動が分かる内容をpackage resourceへ同梱してください。

特にFUTURE_ROADMAPは、既存の詳細版にある背景、実現内容、導入条件、完了条件、見送り理由を維持してください。ループエンジニアリングをMVPへ導入せず、将来の大規模refactor・機能追加時の課題として残してください。

初期設定またはstart wizardでCursor/ChatGPTの高性能モデル名を入力・保存できるようにしてください。少なくともinit optionまたは対話設定を用意し、設定値をローカルconfigへ保存してください。Git管理ファイルやportable snapshotへ個人設定を漏らさないでください。モデル未設定時は警告だけでなく、具体的な設定コマンドまたは入力手順を表示してください。廉価モデルへの自動fallbackは追加しないでください。

必須テスト：

1. package resource版の主要文書が要約だけでなく必須章を含む。
2. init後のVault FUTURE_ROADMAPにループエンジニアリングの詳細説明がある。
3. init再実行でユーザー編集文書を無断上書きしない。
4. --update-docs時はbackup後に更新する。
5. P05/P07/P08/P11/P12/P13/P06/P09 snapshotが詳細基準とversion/hashを持つ。
6. snapshotの実ID・入力パスが案件と一致する。
7. 高性能モデル名を設定し、snapshotへ表示できる。
8. 未設定時に具体的な設定方法を表示する。
9. portable config、prompt、監査logへ実Vault絶対パスや秘密値を漏らさない。

古い説明書の「実装前」「完成版では」等、現状と矛盾する記載を修正してください。実装済みと未実装を明確に分けてください。ruff format、ruff check、mypy、pytestを実行してください。
```

## 完了条件

- AIがprompt単体で仕様に沿った判断を実行できる。
- Vault内の説明書が初回利用とCursorの「次に何をすべき？」に十分な内容を持つ。
- 詳細なFUTURE_ROADMAPがVaultにも保存される。
- 高性能モデル名を通常操作で設定できる。
