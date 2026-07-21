# 09 — Round3最終監査

## 目的

個別unit testの合計ではなく、実際のCLI、AI ingest、SQLite、workflow、CSV評価、wheelをつないだ縦断シナリオで、Round3の修正を独立確認します。

## Cursorへ渡す依頼文

    00_START_HERE.mdから08までの実装を、自己申告ではなく実行証拠で最終監査してください。commit・pushはしないでください。

    次のシナリオA～Rを、可能な限りpublic APIまたはCLIから実行してください。内部DBをtestから直接書き換えて目的状態を作らないでください。

    A. YouTube本人発言
    raw import → P05本人segment → P08 target confirmed → P11 → P12 agreed → CSV単一銘柄方向評価 → 04_resultsまで完走する。

    B. 司会者偽装
    P05は司会者segment、P08は対象者本人target_confirmedと申告する。正式ForecastIssuanceを作らない。

    C. legacy unknown
    segment linkなし、legacy_unknownの予想を保持できるが、正式成績、現在予想、対象解決に入れない。

    D. Web直接引用
    記者記事内の分析対象者の直接引用を、content authorとstatement speakerを分けて本人予想として追跡する。

    E. 第三者要約
    記者による要約だけを本人予想へ入れない。

    F. 時点逆流
    P05、P07、P06、P09、P11、P12 correction、P13の各future cutoffを拒否する。

    G. P09不正修正
    quote offsetをずらしたcorrected payloadを拒否し、forecastを作らない。

    H. P09正当修正
    正しいcorrected payloadを一度だけ正式化し、再取込みでも重複しない。

    I. 複数source
    blog 2件をP07まで処理し、1件目P08後に2件目P08が次行動となる。両方terminal後だけP11へ進む。

    J. 追加source
    評価済みrunへsourceを追加し、新sourceだけを処理する。既存forecast/evaluationを再作成しない。

    K. 0件・除外のみ
    予想0件と全件not_targetを別terminal stateで保持し、P08を無限反復しない。

    L. 安全なartifact再利用
    同一raw・同一条件の別run/別occurrenceで前処理を再利用し、P08参照を壊さない。条件違いでは再利用しない。

    M. 対象解決状態機械
    P11後はP12、P12 disagreed後はP13、mapping固定前は市場評価へ進まない。

    N. basket cache分離
    AAA +20%、BBB -20%の50/50 basketを0%評価後、AAA単独を+20%で評価する。

    O. basket CSV
    long形式CSVとmock providerでbasket return、MFE、MAEが一致する。

    P. 異常market mapping
    mixed currency、provider currency不一致、欠損symbol、共通日不足を理由付きunevaluableにする。

    Q. wheel
    clean venvのwheelからhelp、init、model設定、run create、source importを実行し、完全版docsと全promptをVaultへ生成する。

    R. Git・秘密情報
    CHAT_HISTORY.pdfを維持し、DB、raw、cache、secret、実Vault path、build生成物を追跡しない。

    各シナリオについて、入力、実行commandまたはtest名、期待結果、実結果、DB確認値、生成file、判定をdocs/06_実装/FINAL_REVIEW_ROUND3.mdへ記載してください。

    10_ACCEPTANCE_MATRIXの全項目をtest名、DB列、生成成果物へ対応付けてください。「コードがある」だけでpassにせず、negative caseは実際に拒否された証拠を示してください。

## 判定規則

- A～Rまたは受入マトリクスに内部fail、未実装、未検証があればNOT_READY。
- networkだけが外部制約で、同じ内部処理のmock/CSVがpassしている場合だけexternal_blockedを許容。
- 全Round3項目pass時だけREADY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE。
- 総合点、PNG、1/3/6/12か月観測等が未実装ならFULL_MVP_READYを使用しない。

## 最終報告形式

    判定:
    対象commit:
    pytest:
    ruff format/check:
    mypy:
    Alembic:
    wheel clean venv:
    scenarios A-R:
    acceptance R3-001以降:
    external_blocked:
    意図的未実装:
    次の一作業:
    commit / push: 未実施

