# 02 — 話者帰属と原文証拠の完全性

## 目的

司会者、記者、別出演者、第三者要約、由来不明のlegacy予想を、分析対象者本人の正式予想・成績・現在予想へ混入させないようにします。P08の自己申告ではなく、上流segment、分析対象者master、引用offsetをPythonで照合します。

## Cursorへ渡す依頼文

    00_START_HERE.mdと01の再現結果に従い、話者・著者・引用元の検証を修正してください。commit・pushはしないでください。

    基本原則:

    - AIが出したtarget_confirmedは「申告値」であり、それだけで正式化しない。
    - 正式ForecastIssuanceへ入れられるのは、Python検証後のverified attributionがtarget_confirmedであるものだけ。
    - uncertain、not_target、legacy_unknown、segment linkなしは原文証拠として保持してよいが、正式成績、現在予想、対象解決へ渡さない。
    - legacy recordを自動でtarget_confirmedへ昇格させない。

    データモデルを、少なくとも次を追跡できるようにしてください。

    - sourceのcontent authorまたはchannel owner
    - segmentで実際に発言・執筆・直接引用されたperson
    - P08が申告したspeaker candidateとstatus
    - Pythonが検証したverified attribution statusと検証理由
    - ForecastEvidenceから上流SegmentRecordへのstable foreign keyまたは同等の参照
    - forecast issuanceにspeaker candidate、status、confidence、basis、statement_kind、made_at_source

    Web記事では「記事の著者」と「直接引用の発言者」を同じfieldへ潰さないでください。例えば記者が記事を書き、分析対象者の発言を直接引用した場合、content_author=記者、statement_speaker=分析対象者として表現できる必要があります。第三者による要約は直接引用と区別してください。

    分析対象者照合:

    - AnalystRecordのcanonical_nameとaliasesを正本にする。
    - Unicode NFKC、前後空白、連続空白、大小文字等の決定的normalizationを行う。
    - normalized exact matchだけを自動確認に使う。曖昧一致を本人確認済みにしない。
    - 通常操作でaliasを登録・表示できるCLIまたはwizard経路を用意する。
    - alias変更履歴または少なくとも監査日時を残す。

    P08検証:

    - 正式候補はnon-empty upstream segment referenceを必須にする。
    - 各evidenceがどのsegmentに属するかを明示する。
    - evidence quoteがraw offsetと一致し、参照segmentのoffset union内にあることを確認する。
    - 複数segmentにまたがる引用は、全evidenceが「各segmentすべての内側」にあることを要求せず、順序付きsegmentのunionで正しく検証する。
    - 参照segmentが同じupstream artifact/sourceに属することを確認する。
    - predictionを支えるsegmentのstatement speakerがcanonical nameまたはaliasと一致しなければtarget_confirmedを拒否する。
    - upstream speakerが司会者なのにP08だけが対象者本人と申告した入力をrejectまたはneeds_reviewにする。正式化してはいけない。
    - upstream自体がunknownなら、先にP06 correctまたは別AIの解決が必要。P08の自己申告で迂回させない。

    P08内に予想らしい記述があっても、すべてnot_target、third_party_summary、legacy_unknown等で正式化0件なら、source処理をprocessed_no_formal_forecast等のterminal stateにしてください。P08を無限再要求しないでください。

    migrationでは既存forecastを削除しないでください。既存行にsegment linkがない場合はlegacy_unknown/excludedとして扱い、現在予想や評価集計から外してください。

## 必須テスト

1. 対象者本人segmentとcanonical name一致で正式化できる。
2. alias一致でも正式化できる。
3. 司会者segmentをP08がtarget_confirmedと偽装しても正式化されない。
4. 別出演者segmentを正式化しない。
5. uncertainを正式化しない。
6. legacy_unknownを正式化しない。
7. segment refなしを正式化しない。
8. raw offsetは正しいが別speaker segmentを参照した入力を拒否する。
9. 記者記事内の対象者本人の直接引用を、記者の予想ではなく対象者の予想として追跡できる。
10. 記者による対象者見解の要約だけなら正式化しない。
11. 複数segmentにまたがる正当な引用を受理できる。
12. 複数segmentの順序・範囲が不正なら拒否する。
13. 正式化0件sourceがterminal stateとなりP08を再要求しない。
14. legacy DB upgrade後も既存データを失わず、未確認行が正式集計に入らない。

## 完了条件

- 正式予想からraw、evidence、segment、statement speaker、分析対象者masterまで追跡できる。
- Python検証を通らないAI自己申告が的中率へ入らない。
- Webの著者と引用発言者を区別できる。

