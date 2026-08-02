# 04 — P06/P09修正版の完全再検証と冪等性

## 目的

P09 correctのcorrected_payloadが、通常P08なら拒否される不正な引用offsetや参照関係を迂回して正式化される問題を修正します。P06/P09のaccept、correct、reject、unresolvedをsource workflowと一貫させます。

## Cursorへ渡す依頼文

    00_START_HERE.md、01～03に従い、P06/P09のreviewとcorrected payload取込みを修正してください。commit・pushはしないでください。

    現状はcorrected_payloadへPydantic model_validateだけを行い、通常取込み時のreference、raw hash、quote offset、segment、speaker、time、knowledge cutoff検証を再実行していません。この迂回をなくしてください。

    corrected payloadは、元promptの通常取込みと同じ順序ですべて検証してください。

    1. prompt別固定JSON Schema
    2. Pydantic
    3. run/source/prompt/upstream artifactの参照整合性
    4. input hash
    5. raw hashとquote offset
    6. segment linkとspeaker attribution
    7. made_at、publicly_available_at、knowledge cutoff
    8. review対象artifactとのlineage
    9. semantic review条件

    検証ロジックを二重実装せず、通常取込みとcorrect取込みが同じvalidatorを呼ぶ構造にしてください。ただしreviewの中からreviewを無限再帰しないguardを設けてください。

    corrected payloadには次を強制してください。

    - 元artifactと同じprompt family。
    - 同じrun_idとsource_id。
    - P08なら同じ正当なP05/P07 upstream sourceを参照。
    - 元source raw hashと一致。
    - 修正した項目以外もSchemaに適合。
    - 後発情報を使わない。

    immutable artifact:

    - corrected payload自身のcanonical bytesを別ファイルへ保存する。
    - corrected artifactのoutput_hashと保存ファイルbytesのhashを一致させる。
    - corrected artifactのclassified_file_pathをP06/P09 review JSONのpathで代用しない。
    - reviewed_artifact_id、resolved_by_artifact_id、supersedes_artifact_idを追跡できる。
    - review artifactとcorrected artifactを別recordとして保持する。

    materialization:

    - needs_review P08はreview前にForecastIssuanceを作らない。
    - P09 acceptは元P08を一度だけmaterializeする。
    - P09 correctはcorrected P08だけを一度だけmaterializeする。
    - すでにmaterialize済みのaccepted P08をreviewしても重複ForecastIssuanceを作らない。必要ならreview対象をneeds_reviewに限定するか、DB unique keyと既存確認を併用する。
    - ai_artifact_idとlocal_ref等の安定した一意制約を設け、同じ意味の再取込みで重複しない。

    decisionとworkflow:

    - accept: 元内容を確認済みにする。ただし元segmentがunknownならunknownをtarget_confirmedへ変えない。
    - correct: 完全検証済みcorrected artifactだけを有効にする。
    - reject: 正式利用せず、sourceを無限にP05/P07へ戻さない。再試行可能かterminalかをreasonとともに明示する。
    - unresolved: 正式利用せず、別AI reviewが必要な状態として残す。無限反復を避けるretry countまたは明示的次行動を持つ。
    - staleなneeds_reviewファイルが残ってもDBのresolutionを正本にする。

## 必須テスト

1. quote offsetを1文字ずらしたP09 corrected P08をrejectする。
2. 別runのcorrected payloadをrejectする。
3. 別sourceのcorrected payloadをrejectする。
4. 不正upstream artifactのcorrected payloadをrejectする。
5. 未来knowledge cutoffを含むcorrected payloadをrejectする。
6. host segmentをtarget_confirmedへ変えただけのcorrected payloadを正式化しない。
7. 正当なP09 correctはcorrected forecastを1件だけ作る。
8. 正当なP06 correctはcorrected segmentだけを有効にする。
9. 同じreviewの再取込みはALREADY_IMPORTEDで重複しない。
10. accepted P08へのreviewでも二重ForecastIssuanceを作らない。
11. reject後に元needs_reviewが未解決件数へ残らない。
12. unresolved後は正式成績に入らず、次のAI review行動が示される。
13. corrected artifact fileのhash、DB output_hash、payloadが一致する。
14. testからDBのresolution_statusを直接書き換えずに全状態を作る。

## 完了条件

- 通常取込みで拒否されるpayloadはreview correct経路でも拒否される。
- review、元artifact、corrected artifact、正式forecastのlineageを追跡できる。
- 同一成果物、再review、再実行で件数が増殖しない。

