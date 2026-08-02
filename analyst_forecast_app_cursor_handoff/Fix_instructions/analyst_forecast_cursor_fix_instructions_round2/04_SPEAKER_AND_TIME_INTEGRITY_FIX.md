# 04 — 話者帰属・引用・発言日時の整合性

## 目的

YouTubeの司会者や別出演者の予想を分析対象者本人の成績へ混入させないようにします。また、発言時点より後の情報や不正な日時が対象解決へ流れることを防ぎます。

## コピペ用依頼文

```text
fix_instructions_round2/00_START_HERE.mdの共通拘束条件と、02～03のP05/P07/P06/P09経路に従い、話者帰属、引用segment、発言日時、knowledge cutoffの整合性を実装してください。commit・pushはしないでください。

現状、P05には話者segmentがありますが、P08のforecastはP05 segment IDや発言者を持たず、原文offsetだけで登録できます。このため、司会者や別出演者の発言を分析対象者の予想として保存できてしまいます。

P08の各forecast/evidenceを、上流P05またはP07の具体的segmentへ関連付けてください。最低限、次をSchemaとDBで表現してください。

- upstream segment refs
- speaker/author candidate
- speaker attribution status: target_confirmed / uncertain / not_target
- attribution confidenceと根拠
- direct statement / direct quote / third-party summaryの区別
- 原文offsetとsegment範囲

分析対象者のcanonical nameとaliasesを使い、segmentのspeaker/authorが対象者本人と確認できることを検証してください。target_confirmedだけを正式予想・正式成績へ入れてください。uncertainはAI再レビューへ回し、not_targetは証拠として保持しても正式成績へ入れないでください。

引用offsetがraw全文と一致するだけでなく、申告された上流segmentのoffset範囲内にあることを検証してください。複数segmentにまたがる引用は、参照segmentをすべて列挙し、順序と範囲を検証してください。

発言時刻・公開時刻・推定時刻の由来を分けてください。

- source recorded_at: 実際の収録・発言日時が分かる場合
- source published_at: 公開日時
- forecast made_at: 予想が発言された日時
- publicly_available_at: 一般に利用可能になった日時
- made_at_source: explicit / source_metadata / context_inferred / unknown
- knowledge_cutoff: AIが対象解決等に使用した情報の上限

少なくとも次をPythonで検証してください。

- made_at <= publicly_available_at。ただし同時公開なら同値を許容。
- recorded_atがある場合、made_atはその収録文脈と矛盾しない。
- recorded_atがなく本人執筆記事・投稿の場合、made_atは原則published_atを基準とする。
- P11/P12/P13のknowledge_cutoff <= forecast made_at。
- P05/P07/P06/P09のknowledge_cutoffも、当該処理が参照を許された発言時点を越えない。
- 日時不明を勝手な確定日で埋めない。unknownまたはinferredとして保存する。
- YouTubeの公開日時と実発言日時を同じ列へ潰さない。

発言時点以前の他媒体情報を話者推定へ使うことは許可しますが、使用した根拠URL・日時・knowledge cutoffを保存し、後発情報を使わないでください。市場評価結果は一切入力に含めないでください。

必須テスト：

1. 対象者本人segmentの予想は登録できる。
2. 司会者segmentの予想は正式成績へ入らない。
3. unknown speakerはP06へ回る。
4. Web記事の記者要約を本人の直接予想として登録しない。
5. 本人の直接引用は引用範囲を保持して登録できる。
6. raw offsetは一致していても別speaker segmentを参照したP08を拒否する。
7. made_at > publicly_available_atを拒否する。
8. knowledge_cutoff > made_atをP05/P07/P11/P12/P13で拒否する。
9. 後日公開された録画でrecorded_atとpublished_atを別々に保持する。
10. 日時不明をsystem defaultで発言者明示値に変換しない。

Schema変更にはmigrationと後方互換を用意してください。既存予想でsegment linkがないものはlegacy_unknown等とし、本人確認済みと捏造しないでください。ruff format、ruff check、mypy、pytestを実行し、データモデルと方法論文書を更新してください。
```

## 完了条件

- 予想から原文、segment、話者、分析対象者へ追跡できる。
- 別話者の発言を対象者の的中率へ混入させない。
- 発言日時・公開日時・推定日時を分離して保存する。
- 後発情報の逆流をSchemaとPython検証の両方で防ぐ。
