# レビュー指摘修正計画

## 1. 位置付け

- 作成日：2026-07-20
- 対象HEAD：`cc9e8db6a5efcd5e79db343480319dcd4c27c99c`
- 対象アプリ：`analyst_forecast_app_cursor_handoff/`
- 指示書：`Fix_instructions/analyst_forecast_cursor_fix_instructions/01_PREFLIGHT_AND_REMEDIATION_PLAN.md`
- 実装順：修正指示02、03、04、05、06、07
- 本文書作成時点では、本体コード、Schema、migration、テストの機能変更を行っていない。

本計画は現在の正式仕様を実現するための修正計画であり、評価方法や製品目的を新たに変更するものではない。実装中に正式仕様の変更が必要と判明した場合は、問題、変更案、影響範囲を提示し、仕様権限に従って承認後に反映する。

## 2. 事前確認

### Git

- 現在ブランチ：`fetch2_1_アナリスト調査の計画を追加`
- HEAD：基準コミット `cc9e8db` と一致する。
- 着手時の追跡済み差分：なし。
- 着手時の未追跡：今回ユーザーから渡された `Fix_instructions/` だけ。
- 対象アプリ外の差分：なし。
- `reference/CHAT_HISTORY.pdf` は存在し、今回の削除、変更、履歴除去対象にしない。

### 修正前ベースライン

- Python 3.12.10。
- 通常pytest：29件成功、integration 3件除外。
- 既存テストは単一の成功経路を主に確認しており、今回の10指摘を検出する回帰テストは不足している。

## 3. レビュー指摘の再現分類

### R-01 component IDが利用者へ渡らない

- 分類：`reproduced`
- 問題：`AiIngestResult` はissuance IDとcomponent IDを返すが、`cli/app.py` の `ai ingest` は分類とhashだけを表示する。`workflow.py` は市場評価コマンドへ実IDではなく `<component-id>` を書く。
- 再現条件：有効なP08 fixtureを取り込み、CLI出力と `NEXT_ACTIONS.md` を確認する。
- 根拠：`application/ai_ingestion.py::ingest_ai_output`、`cli/app.py::ingest_ai_command`、`application/workflow.py::_choose_action`。
- 関連：D-014、FR-18、T702、T704、FIX-001、FIX-002。

### R-02 04_resultsが空

- 分類：`reproduced`
- 問題：案件作成時に結果ディレクトリを作るだけで、市場評価後もMarkdown／CSVを生成しない。workflowは空の `04_results/` を確認対象として案内する。
- 再現条件：匿名fixtureを市場評価まで通し、`04_results/` 配下のファイル数を確認する。
- 根拠：`application/runs.py::_required_directories`、`application/evaluation.py`、`application/workflow.py::_choose_action`。
- 関連：FR-17、FR-19、FIX-003。
- 備考：既存の `IMPLEMENTATION_STATUS.md` でもレポート未実装と記録されていたが、正式要件未達の挙動自体は再現する。

### R-03 P08・P11・P12が独立工程ではない

- 分類：`reproduced`
- 問題：現行Schemaは `prompt_id=P08` の単一JSONに予想抽出、対象解決、`review_result` を同居させる。別のP11／P12 prompt executionがなくても文字列だけでverifiedにできる。
- 再現条件：現行1.0.0 fixture一つだけを取り込むと、P11／P12成果物なしでmappingがverifiedとして保存される。
- 根拠：`schemas/ai_output.py::PromptExecution`、`TargetMappingOutput.require_independent_review`、`PromptExecutionRecord.ai_import_id` のunique制約、`application/ai_ingestion.py::_insert_payload`。
- 関連：D-003、D-010、FR-10、FR-19、FIX-004、FIX-005、FIX-008。

### R-04 予想0件と対象解決不能を表現できない

- 分類：`reproduced`
- 問題：`forecasts` は `min_length=1`、targetの `symbol` と `currency` は必須文字列である。処理済みだが予想なし、またはdummy symbolを使わない `unresolvable` を保存できない。
- 再現条件：現行fixtureを `forecasts=[]` または `symbol=null` にして検証する。
- 根拠：`schemas/ai_output.py::ForecastExtractionOutput`、`TargetMappingOutput`、固定Schema 1.0.0。
- 関連：FR-06、FR-10、FR-13、FIX-006、FIX-007。

### R-05 workflowがファイル数と評価総数で誤判定する

- 分類：`reproduced`
- 問題：
  - `needs_review` の残存ファイル数を未解決件数として扱うため、修正版accepted後もレビュー待ちになる。
  - Evaluation総数とcomponent総数を比較するため、component Aを2基準日で評価すると未評価のcomponent Bを隠せる。
  - 過去のunevaluableを履歴全件から数えるため、後の成功後も取得不能状態が残る。
- 再現条件：低確信度版から修正版を取り込む、または複数component／複数as_ofを作る。
- 根拠：`application/workflow.py::refresh_workflow` と `_choose_action`。
- 関連：FR-13、FR-18、NFR-03、T707、FIX-009、FIX-010、FIX-011。

### R-06 raw artifactと情報源occurrenceが混同される

- 分類：`reproduced`
- 問題：`sources.raw_hash` がDB全体でuniqueであり、同一bytesは最初の `SourceRecord` を別アナリストにも再利用する。所有者、媒体、URL、保存場所が初回情報に固定され、後続案件のrawディレクトリは空になる。
- 再現条件：同一bytesを別アナリスト、別URL、別媒体、別案件へ取り込む。
- 根拠：`infrastructure/db/models.py::SourceRecord`、`application/raw_sources.py::import_raw_source`、`application/ai_ingestion.py::_validate_references_and_quotes`。
- 関連：FR-01、FR-04、FR-19、NFR-01、T102、T205、FIX-012、FIX-013、FIX-014。
- 備考：A-007のグローバル重複方針はbytes再処理防止を満たす一方、SOURCE所有者を混線させるため、artifactとoccurrenceへ分離する。

### R-07 初期化後のVault内docs／promptsが空

- 分類：`reproduced`
- 問題：`initialize_workspace` は `docs/` と `prompts/` を作るだけで、主要文書、AI_WORK_GUIDE、中央promptを配置しない。案件promptは `runs.py` の埋込み文字列から直接作られる。
- 再現条件：空の一時領域を `init` し、workspace内のdocs／promptsを列挙する。
- 根拠：`application/bootstrap.py::initialize_workspace`、`application/runs.py::_write_prompt_snapshots`。
- 関連：FR-03、NFR-02、D-014、FIX-018、FIX-019。

### R-08 下落予想のMFE／MAEが方向対応しない

- 分類：`reproduced`
- 問題：MFE／MAEを常に上昇予想と同じ式で計算する。下落予想では価格上昇をfavorable、価格下落を負のadverseとして保存してしまう。
- 再現条件：開始100、高値104、安値88、終了92の下落予想を評価する。
- 根拠：`application/evaluation.py::evaluate_component`。
- 関連：FR-12、T401、T402、FIX-015。

### R-09 yfinanceの取得失敗分類が不足

- 分類：`reproduced`
- 問題：空DataFrameを一律「指定期間データなし」とし、rate limit、network error、invalid symbol、true no dataを構造化して区別しない。retry、取得前cache利用、専用CSV案内もない。
- 再現条件：`yfinance.download` を空DataFrame、rate limit例外、接続例外でmockする。
- 根拠：`infrastructure/market/yfinance_provider.py::fetch`、`domain/market.py::MarketDataUnavailable`、`application/evaluation.py`。
- 関連：D-011、FR-11、NFR-04、OQ-007、FIX-016。

### R-10 対話式案件作成がない

- 分類：`reproduced`
- 問題：案件作成は必須flagを列挙する `run create` だけで、既定6か月、媒体複数選択、確認、取消、やり直しを行うwizardがない。
- 再現条件：CLI helpと `run --help` を確認する。
- 根拠：`cli/app.py`。
- 関連：UC-01、FR-01、FR-02、NFR-02、D-014、FIX-020、FIX-021。

### 分類集計

- `reproduced`：R-01～R-10の10件。
- `already_fixed`：なし。
- `not_reproducible`：なし。
- `intentional_future_scope`：なし。

一部には既存文書で未実装と明示されていた機能もあるが、今回の修正指示がMVP受入事項として明示しているため、`intentional_future_scope` ではなく `reproduced` とした。

## 4. migration・Schema・互換性の共通方針

### 4.1 破壊的な初期化をしない

- 既存 `0001_initial_mvp_schema` を編集しない。
- 新しいAlembic revisionを段階ごとに追加する。
- migration前に既存のSQLite backup機能でバックアップを作る。
- 既存ID、raw、AI出力、評価履歴、監査ファイルを削除しない。
- 空DBからのupgradeと、`0001` 適用済みDB fixtureからのupgradeを毎段階で試験する。

### 4.2 Schema version

- 現行 `forecast_extraction.schema.json` 1.0.0を変更して同名上書きしない。
- 分離後のP05、P08、P11、P12、P13にversion付きの個別Schemaを追加する。
- P08の責務と構造が破壊的に変わるため、新規成果物は暫定的に2.0.0系として計画する。実装時にSchema ID、version、互換adapterをテストで固定する。
- 1.0.0 JSONはlegacy adapterで読み込める経路を残す。
- 1.0.0のinline `review_result` は履歴として保存するが、新規ルール上の独立P12実行済みへ自動昇格させない。`legacy_inline_review` 等の監査属性を付け、必要ならP12待ちへする。

### 4.3 DB revision案

revision名は実装時に確定するが、責務を混ぜないため次の単位を予定する。

1. AI成果物、segment、対象提案・レビュー・裁定、artifact解決状態、prompt execution制約を追加するrevision。
2. raw artifactとsource occurrenceを分離し、既存SOURCEをbackfillするrevision。
3. workflow task／supersession／解決状態と、最新評価判定用indexを追加するrevision。
4. provider error code、retryable、cache検索条件等のnullable監査列を追加するrevision。

SQLiteの制約変更はAlembic batch migrationを使用する。各revisionはupgrade後の参照整合性、件数、ID、hashを確認し、`alembic check` を通す。

### 4.4 ロールバック原則

- 新規・空DBではAlembic downgradeも試験する。
- 実データDBで新Schemaのデータを書き込んだ後は、downgradeによる列・テーブル削除を通常運用の復旧手段にしない。
- 実運用のロールバックはmigration直前backupの復元と、旧アプリ／旧Schema資源の再配置を基本とする。
- ファイル生成だけの修正は、生成物がSQLiteから再生成可能であることを保証する。rawと監査成果物はロールバックで削除しない。

## 5. 修正指示02 — AI処理分離・Schema・独立レビュー

### 問題と再現条件

- 対象：R-03、R-04、およびP05未取込み。
- 現行P08 JSON一つでmappingをverifiedにできる。
- `forecasts=[]`、unknown speaker、symbolなしunresolvable、最大3候補を表現できない。

### 関連要件・決定

- D-003、D-004、D-010。
- FR-04～10、FR-13、FR-19。
- NFR-01、NFR-03。
- FIX-004～008、FIX-022。

### DB／Schema migration

- 共通AI成果物または同等の構造を追加し、独立artifact ID、run/source/forecast/component参照、prompt execution、input/output hash、Schema version、classification、resolution status、confidence、重要度、supersedes／resolved_byを保存する。
- P05 segmentを保存するテーブルを追加する。raw offset、raw text、normalized text、speaker candidate、confidence、根拠、review statusを持たせる。
- P11 proposal、1～3 candidate、P12 review、必要時P13 adjudicationを別レコードとして保存し、参照関係を外部キーで表す。
- `prompt_executions.ai_import_id` の単一unique前提を解除し、一つの案件・予想へ複数の独立実行を関連付ける。
- targetまたはcandidateはsymbol／currency nullableを許容し、unresolvable時は理由必須とする。
- 情報源処理結果へ `processed_no_forecast` を保存できる状態を追加する。
- P05、P08、P11、P12、P13の個別Schemaを追加する。P08はraw target labelと対象解決待ちまでに限定する。

### 後方互換性

- Schema 1.0.0を削除しない。
- legacy 1.0.0のP08＋inline mappingは、新DB構造へ変換して読めるようにする。
- 既存mapping、evaluation、PromptExecutionを削除・上書きしない。
- legacy inline reviewは独立P12とは区別して監査表示する。
- 修正指示03でSOURCE構造を分離しても、現行source ID参照を解決できるadapterを維持する。

### 変更予定ファイル

- `schemas/` のversion付きP05／P08／P11／P12／P13モデルとJSON Schema。
- `application/ai_ingestion.py` を成果物別use caseへ分割またはdispatch化。
- `application/runs.py` のprompt snapshot生成。
- `infrastructure/db/models.py`。
- Alembic新revision。
- `cli/app.py` の成果物別取込みコマンド。
- P05 processedファイルと監査log生成。
- AI pipeline用unit／compatibility fixture。

### 必須テスト

- 予想0件を正常取込みし、`processed_no_forecast` になる。
- unresolvable targetをdummy symbolなしで保存できる。
- P11単独ではverified／lockedにならない。
- P12がP11と別prompt executionでなければ拒否する。
- P11とP12一致時だけlockできる。
- 不一致はP13待ちになる。
- 最大3候補とweights／感度分析対象を保存できる。
- unknown speakerを対象者本人として登録しない。
- 低確信度または高重要度はAIレビュー待ちになる。
- knowledge_cutoffが発言日時より後なら拒否する。
- 対象解決入力に市場結果フィールドがあれば拒否または監査警告にする。
- 既存Schema 1.0.0 fixtureを互換経路で読める。
- 空DBと既存0001 DBのmigration試験。

### 完了条件

- P05→P08→P11→P12→必要時P13が別ID・別prompt executionとしてDBと監査logから追跡できる。
- Pythonが別P12実行を確認するまでmappingをverified／lockedにしない。
- 予想なし、対象解決不能、unknownを捏造なしで保存できる。
- 現行1.0.0データを破棄せず参照できる。

### ロールバック方法

- migration前DB backupを復元する。
- 新Schema資源を無効化し、legacy 1.0.0 readerを使用する。
- 新成果物ファイルは監査用に残し、旧正式テーブルへ逆流させない。

## 6. 修正指示03 — raw artifact・情報源occurrence・案件関連

### 問題と再現条件

- 対象：R-06。
- 同一bytesを別アナリストへ入れると、SOURCE所有者が初回アナリストのままになる。
- 同じbytesでも別URL・別媒体の証拠を独立保持できない。
- 重複案件のローカルrawフォルダが空なのにworkflowはそのglobを入力として示す。

### 関連要件・決定

- D-004。
- FR-01、FR-04、FR-19。
- NFR-01、NFR-03。
- T101、T102、T103、T205。
- FIX-012～014、FIX-022。

### DB／Schema migration

- `raw_artifacts` を追加し、artifact ID、SHA-256、canonical path、byte size、encoding、作成日時、versionを持たせる。bytes hashの一意性はここへ移す。
- 現行 `sources` はsource occurrence／evidenceとしてsource IDを維持し、analyst、媒体、URL、日時、証拠関係とraw artifact FKを持たせる。
- `sources.raw_hash` のunique制約を解除し、既存値はlegacy監査列として当面保持する。
- `run_sources` はoccurrenceを参照し、案件内の具体的入力manifestまたはcopy pathを保存する。
- 既存SOURCEをhashごとにartifactへbackfillし、source IDを変えずにartifact FKを設定する。
- 既存run-sourceでrunとsourceのanalystが異なる場合は、新occurrenceを作り、記録済みのobserved URL、媒体、日時を移す。AI importとforecast evidenceはrun scopeを用いて新occurrenceへ付け替え、移行対応表を監査用に残す。
- 現行DBに保存されていない二つ目以降のURL等は推測して復元しない。

### 後方互換性

- forecast evidenceが参照する既存source IDを可能な限り維持する。
- 分割が必要な既存run associationだけ新source IDを発行し、旧IDとの移行対応を残す。
- Schema 1.0.0の `source_id` はoccurrence IDとして解釈する。
- `raw_hash`／`raw_file_path` を読むlegacyコードには移行期間の互換propertyまたはqueryを用意する。
- Windows symlinkは必須にせず、manifestまたは変更禁止copyを使用する。

### 変更予定ファイル

- `infrastructure/db/models.py`。
- Alembic新revisionとbackfill処理。
- `application/raw_sources.py`。
- `application/ai_ingestion.py` のraw解決。
- `application/workflow.py` とprompt入力path生成。
- artifact／occurrence repositoryまたは小さなservice。
- source identity fixtureとmigration fixture。

### 必須テスト

- 同一bytes・同一案件の再取込みが二重処理されない。
- 同一bytes・同一アナリスト・別案件の双方から具体的入力へ到達できる。
- 同一bytes・別アナリストでoccurrence所有者が混線しない。
- 同一bytes・別URLで両URLと日時を保持する。
- 同文・別媒体で媒体証拠を保持する。
- raw改変時にhash mismatchで拒否する。
- duplicate sourceのNEXT_ACTIONSが存在する具体的入力pathを示す。
- 処理version、prompt/model、対象話者、出力hashが一致する場合だけprocessed結果を再利用する。
- migration後も既存SOURCE、AI import、forecast evidenceを追跡できる。

### 完了条件

- bytes再処理防止と、証拠occurrenceの独立保持が両立する。
- どの案件でも表示された入力pathまたはmanifestからrawへ到達できる。
- 別アナリストの話者推定を無条件再利用しない。

### ロールバック方法

- migration直前backupを復元する。
- 新artifact storageを削除せず監査用に保持し、旧source path readerへ戻す。
- 案件内に作ったraw copy／manifestはraw証拠として削除しない。

## 7. 修正指示04 — workflow・次行動・人間向け結果

### 問題と再現条件

- 対象：R-01、R-02、R-05。
- AI取込み後に実IDと実行可能コマンドが出ない。
- file countとEvaluation総数により現在状態を誤る。
- SQLiteに結果があっても `04_results` が空。

### 関連要件・決定

- D-014。
- FR-13、FR-17、FR-18、FR-19。
- NFR-02、NFR-03、NFR-04。
- T601～607、T701～707。
- FIX-001～003、FIX-009～011、FIX-017。

### DB／Schema migration

- AI成果物またはworkflow taskに `pending / running / needs_review / resolved / accepted / rejected / superseded`、supersedes、resolved_by、retryability、last errorを保存する。
- task dependencyと、runごとの推奨順位をSQLite正本として保存する。
- 既存needs_review／rejectedファイルは削除せず、初回同期時にDB状態へ登録する。
- 評価判定用にcomponent、as_of、method version、作成日時のindexを確認・追加する。
- 04_resultsはDBから再生成する派生物のため、専用正式テーブルは追加しない。

### 後方互換性

- 既存監査ファイルを残し、解決済み／supersededをDB側で表す。
- 既存Evaluation履歴は変更せず、現在状態を求めるqueryだけを最新・distinct component単位へ変更する。
- 現行 `status.yaml`、`WORKFLOW_STATE.json`、`NEXT_ACTIONS.md` の主要キーを維持し、必要な詳細を追加する。

### 変更予定ファイル

- `application/workflow.py`。
- 新しい `application/results.py` または同等のrenderer。
- `cli/app.py`。
- `infrastructure/db/models.py` とAlembic revision。
- Markdown／CSV atomic writer。
- workflow、CLI、results、compatibility tests。

### 必須テスト

- AI ingestのCLI出力とNEXT_ACTIONSに実issuance／component IDが出る。
- 生成されたmarket evaluateコマンドをそのまま実行できる。
- run IDと別案件component IDの組合せを拒否する。
- 修正版accepted後、古いneeds_reviewファイルを残したまま先へ進む。
- component Aの複数as_ofが未評価component Bを隠さない。
- 過去unevaluable後の最新成功を現在状態に使う。
- 予想0件の処理済みsourceがP08を無限に要求しない。
- P05→P08→P11→P12→評価の各遷移。
- 中断後の `status` で同じ状態へ復帰する。
- 次の5成果物を生成し、DB内容と一致する。
  - `forecasts/all_forecasts.md`
  - `tables/all_forecasts.csv`
  - `evaluations/evaluations.md`
  - `tables/evaluations.csv`
  - `reports/vertical_mvp_summary.md`

### 完了条件

- 利用者がDBを直接調べず、CLIとNEXT_ACTIONSだけで次工程を実行できる。
- workflow完了判定の正本がSQLite状態と最新evaluation queryになる。
- 評価またはAI取込み後に、人間向け成果物と状態4ファイルが原子的に再生成される。

### ロールバック方法

- workflow migration前backupを復元する。
- 結果ファイルはDB正本から旧版・新版どちらでも再生成可能にする。
- 既存監査ファイルとEvaluation履歴は削除しない。

## 8. 修正指示05 — 市場評価・方向・provider失敗

### 問題と再現条件

- 対象：R-08、R-09、および複数componentの親状態上書き。
- 下落予想のMFE／MAEが上昇用の式になる。
- 一component評価のたびに親issuance状態を最後のcomponent状態で上書きする。
- yfinance空応答を一律no dataとして扱う。

### 関連要件・決定

- D-011、D-012。
- FR-11、FR-12、FR-13、FR-19。
- NFR-03、NFR-04。
- T301～305、T401、T402、T411、T412。
- FIX-015、FIX-016。

### DB／評価method migration

- 既存 `direction-v1.0.0` のMFE／MAE意味は変更しない。
- 方向対応MFE／MAEを新しいmethod version、暫定 `direction-v2.0.0` として保存する。既存v1評価は履歴として残し、必要時に再評価する。
- 下落予想は `(start-period_low)/start` をfavorable正値、`(period_high-start)/start` をadverse正値とする。上昇予想の既存符号規約はversion文書で明示する。
- component評価状態と親issuance集約状態を分離する。親状態を集約できない段階では誤って完了へ上書きしない。
- provider error code、retryable、attempt count、原メッセージ、cache利用有無をnullable監査列または専用audit recordへ追加する。
- 既存Evaluation schemaへnullable列を追加する場合、旧行はunknownとして保持する。

### 後方互換性

- v1評価を上書き・再解釈しない。
- Evaluationのunique keyはmethod versionを含むため、新versionを並存させる。
- `MarketDataUnavailable` を構造化しても、既存の日本語message取得を維持する。
- 既存market cacheはmetadataが十分な場合だけ再利用し、不足時に推測してcache hitにしない。

### 変更予定ファイル

- `application/evaluation.py`。
- `domain/market.py`。
- `infrastructure/market/yfinance_provider.py`、CSV／FRED provider。
- cache reader／provider audit。
- 必要なDB modelとAlembic revision。
- 方向、品質、provider mock、cache tests。

### 必須テスト

- 上昇・下落・flatのhit／miss。
- 上昇・下落のversion別MFE／MAE。
- active、expired、not_started、unevaluable。
- 同一componentの複数as_of履歴。
- 複数componentで親状態を誤上書きしない。
- rate limit、network error、invalid symbol、true no dataをmockで区別する。
- 小さい固定上限のretry後は推測せず終了する。
- 有効cache hit時にnetwork providerを呼ばない。
- CSV fallbackで同じ方向計算を再現する。
- adjusted OHLCの基準、休場日、範囲外、重複、欠損、0以下を検査する。
- 株式分割をまたぐ保存fixtureを通常pytestで回帰する。

### 完了条件

- 下落予想のfavorable／adverseが予想方向に対応し、method versionから意味を追跡できる。
- 複合予想の親状態を一componentで誤確定しない。
- rate limitとno dataを区別し、retryまたはCSV代替を具体的に案内する。
- 取得不能値を推測しない。

### ロールバック方法

- 新method versionの利用を停止し、v1表示へ戻す。新評価行は履歴として保持する。
- DB列追加がある場合は実DBでdowngradeせず、migration前backupを復元する。
- provider retry／cacheを無効化しても、CSVと従来のunevaluable経路を維持する。

## 9. 修正指示06 — Obsidian資源・設定・対話CLI

### 問題と再現条件

- 対象：R-07、R-10。
- `--vault-root` がVaultそのものかworkspaceか不明確。
- init後のdocs／中央promptsが空。
- 案件prompt本文が `runs.py` にハードコードされ、version／hashを記録しない。
- flag方式以外の開始導線がない。

### 関連要件・決定

- D-002、D-003、D-014。
- UC-01、FR-01～03、FR-18、FR-19。
- NFR-02、NFR-05。
- OQ-002、OQ-004、OQ-010。
- FIX-018～021、FIX-023。

### DB／設定migration

- DB migrationは原則不要。
- 新設定は `obsidian_vault_path` と安全な `workspace_relative_path` を持ち、`workspace_root` を算出する。
- 現行 `vault_root` をlegacy入力として読み、新形式へ正規化する。既存configを無条件上書きしない。
- `workspace_relative_path` の絶対パス、drive指定、`..` によるVault外脱出を拒否する。
- prompt snapshotのversion／hashはrequest、監査log、必要なら既存PromptExecutionへ保存する。

### 後方互換性

- `init --vault-root` と `run create` のflag方式を維持する。
- 既存configの `vault_root` を読める。
- init再実行はユーザー編集済み文書を上書きしない。更新はbackup＋明示的 `--update-docs` 等に限定する。
- package resourceはwheel／将来exeから `importlib.resources` 等で読める構造にする。

### 変更予定ファイル

- `application/settings.py`、`application/bootstrap.py`。
- versioned `resources/docs/`、`resources/prompts/`。
- `application/runs.py` のprompt埋込みをresource rendererへ移行。
- `cli/app.py` と対話wizard use case。
- README、スタートアップガイド、取扱説明書。
- package data設定。
- init、path、resource、wizard、secret scan tests。

### 必須テスト

- Windows形式、空白、日本語、`★` を含むパス。
- legacy `vault_root` configの読込み。
- path traversalと絶対relative pathの拒否。
- initで主要docs、AI_WORK_GUIDE、中央promptを配置する。
- init再実行でユーザー編集を上書きしない。
- 案件prompt snapshotが中央version／hashを持つ。
- wizardの既定過去6か月、基準日今日、媒体複数選択、確認、取消、やり直し。
- 既存flag CLI回帰。
- 個人絶対パス、APIキー、秘密値がprompt／監査log／Git管理資源へ漏れない。

### 完了条件

- 初回利用者がVault内文書と対話CLIだけで案件を作成できる。
- 中央promptが正本、案件promptがversion付きsnapshotになる。
- 既存config、flag CLI、ユーザー編集文書を壊さない。
- package resourceがwheelから読める。

### ロールバック方法

- config変更前backupを復元し、legacy `vault_root` readerを使用する。
- package資源の旧versionを再配置する。
- Vault内のユーザー文書は削除・上書きしない。
- 新 `start` コマンドを無効化しても既存flag方式を維持する。

## 10. 修正指示07 — 最終横断監査

### 問題と再現条件

- 各段階が単独テストだけ通り、P05から評価・結果まで接続されない危険がある。
- migration、wheel、秘密情報、公開PDFの扱いを横断確認する必要がある。

### 関連要件・決定

- D-001～D-016。
- FR-01～19、NFR-01～05の今回実装対象。
- FIX-001～024。

### DB／Schema migration

- 新規migrationは原則作らず、02～06で追加した全revisionを空DBと既存0001 DBから検査する。
- Schema 1.0.0互換fixtureと新分離Schema fixtureを両方通す。
- 監査で不具合を発見した場合だけ、原因を修正する最小revisionを追加する。

### 後方互換性

- 公開許容済み `reference/CHAT_HISTORY.pdf` を削除・変更しない。
- 既存raw、accepted、needs_review、evaluation履歴を残す。
- 旧config、flag CLI、Schema 1.0.0を回帰確認する。
- wheelをクリーンvenvへ導入し、開発リポジトリ相対pathへ依存していないことを確認する。

### 変更予定ファイル

- 原則 `docs/06_実装/FINAL_REVIEW.md` と状態文書だけ。
- 失敗がある場合のみ、該当責務のコード、migration、テストを最小変更する。

### 必須テスト

- 指示書07のシナリオA～G。
- Ruff format check、Ruff lint、mypy、通常pytest。
- 空DBからAlembic head。
- 既存0001 DB fixtureからAlembic head。
- `alembic check`。
- wheel build、クリーンvenv install、init、CLI help。
- ネットワーク試験は成功、skip、provider unavailableを分離報告する。
- Git追跡・secret scan・PDF無変更・raw不変・人間承認非必須・市場情報非逆流・総合点なし・ループ基盤なしの監査。

### 完了条件

- `FINAL_REVIEW.md` に各項目を `pass / fail / not_applicable / external_blocked` で記録する。
- 内部実装failをexternal blockedへ偽装しない。
- `READY_FOR_REAL_SAMPLE`、`READY_WITH_LIMITATIONS`、`NOT_READY` のいずれかを根拠付きで判定する。

### ロールバック方法

- 監査文書だけなら追記を取り消せる。
- 監査中の修正は該当段階のbackup／rollback方針に従う。
- commit、push、履歴変更は行わない。

## 11. 段階間の依存関係

1. 02でAI成果物と独立レビューの正本を作る。
2. 03でそのsource参照をartifact／occurrence構造へ安全に移す。
3. 04でAI成果物、source、評価を使うworkflowと結果生成を直す。
4. 05で評価methodとprovider状態を正しくする。
5. 06で資源配布と利用者導線を完成する。
6. 07で新旧DB、Schema、CLIを横断監査する。

前段のテスト、migration、文書更新が成功するまで次段へ進まない。各段階で本体コード変更前に失敗するテストを追加し、実装後にformatter、lint、mypy、pytest、必要なAlembic検査を実行する。

## 12. 修正指示 03～07 実装ステータス（2026-07-20 追記）

| 指示 | 状態 | 主な成果 |
|---|---|---|
| 03 | 完了 | Alembic 0003、raw_artifacts、occurrence 分離、再利用条件 |
| 04 | 完了 | Alembic 0004、workflow_tasks、04_results 生成、実IDコマンド |
| 05 | 完了 | direction-v2.0.0、provider 分類、Alembic 0005 |
| 06 | 完了 | resources seed、wizard start、path 正規化 |
| 07 | 完了 | FINAL_REVIEW.md、READY_WITH_LIMITATIONS |

詳細は `FINAL_REVIEW.md` と更新済み `IMPLEMENTATION_STATUS.md` を参照。
