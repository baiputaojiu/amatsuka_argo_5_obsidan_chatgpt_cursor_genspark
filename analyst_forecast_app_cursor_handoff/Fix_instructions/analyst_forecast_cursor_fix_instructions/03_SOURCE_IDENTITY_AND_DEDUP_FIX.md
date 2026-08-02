# 03 — 原文同一性・情報源・案件間重複の修正

## 目的

同じbytesを再処理しない方針を維持しながら、別アナリスト、別URL、別媒体、別案件の証拠情報を失わないようにする。

## コピペ用依頼文

```text
raw原文の同一性とSOURCEの意味を分離してください。現在はraw_hashが同じだけで既存SourceRecordを全案件・全アナリストへ再利用するため、SOURCEのanalyst_idと案件のanalyst_idが不一致になり、別案件のrawフォルダも空になります。

次の概念を分離する最小設計を実装してください。

A. immutable raw artifact
- bytesのSHA-256で一意
- canonical保存パス
- byte size、encoding、作成日時
- raw bytesは上書き禁止

B. source occurrence / source evidence
- analyst_id
- medium、URL、external_source_id、title、publisher/channel
- recorded_at、published_at、retrieved_at、evidence_level
- source_relation、original_source_id
- raw artifactへの参照
- 同じbytesでも別URL・別媒体・別アナリストなら別の証拠レコードを保持可能

C. run association
- run_idとsource occurrenceの関連
- 案件内でAIが読む実パスまたは参照manifest

既存SourceRecordをmigrationでどう扱うかはREMEDIATION_PLANに従い、既存データを消さないでください。テーブルを分ける場合は、既存SOURCEをsource occurrenceとして移行し、raw hash/pathをraw artifactへ移してください。

重複bytesを別案件へ取り込んだときも、案件フォルダ内に次のいずれかを必ず作ってください。
- 同一bytesの変更禁止コピーとartifact参照metadata
- または、Cursorとアプリが確実に解決できる参照manifest

Windowsで不安定なsymlinkを必須にしないでください。保存容量より操作の分かりやすさを優先してよいですが、再処理判定はartifact hashと処理versionで行い、物理コピー数で判断しないでください。

同じraw artifactに対するP05/P07等の処理済み結果を再利用する場合、処理version、prompt/model、対象話者、出力hashが一致する場合だけ再利用してください。対象アナリストが異なる場合、話者推定結果を無条件に再利用しないでください。

workflowと生成プロンプトの入力一覧には、存在する具体的ファイルまたは参照先を出してください。現在案件の02_sources/*/rawが空なのに登録済み扱いする状態を禁止してください。

必須テスト：
- 同一bytesを同一案件へ再取込みしても二重処理しない。
- 同一bytes・同一アナリスト・別案件で、両案件から実入力へ到達できる。
- 同一bytes・別アナリストで、source occurrenceの所有者が混線しない。
- 同一bytes・別URLで両方のURLと日時を保持する。
- 別媒体で同文でも媒体証拠を保持する。
- raw artifactを変更した場合はhash mismatchで拒否する。
- duplicate sourceでもNEXT_ACTIONSが存在する具体的入力を示す。
- migration後も既存SOURCEとforecast evidenceを追跡できる。

DB取込み前バックアップ、transaction、冪等性を維持してください。Ruff、mypy、pytest、Alembic upgrade/checkを実行し、実装文書を更新してください。commit・pushは行わないでください。
```

