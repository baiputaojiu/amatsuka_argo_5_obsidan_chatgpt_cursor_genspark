# 第3次修正 — 受入マトリクス

最終報告だけでなく、test名、実DB状態、生成成果物、CLI出力を対応付けてください。

| ID | 受入事項 | 必須証拠 |
|---|---|---|
| R3-001 | P08自己申告だけでtarget confirmedにしない | host偽装negative test |
| R3-002 | canonical name一致の本人segmentだけ正式化する | positive speaker test |
| R3-003 | 登録alias一致を本人として検証できる | alias CLI＋positive test |
| R3-004 | uncertain/not_target/legacy_unknownを正式集計しない | DB/report negative test |
| R3-005 | segment linkなし予想を正式化しない | negative test |
| R3-006 | ForecastEvidenceから上流segmentへ追跡できる | FK/lineage query |
| R3-007 | Webのcontent authorとstatement speakerを分離する | direct quote test |
| R3-008 | 第三者要約を本人予想へ入れない | summary negative test |
| R3-009 | 複数segment引用をunionと順序で検証する | positive/negative offset test |
| R3-010 | 正式予想0件sourceをterminalにする | workflow test |
| R3-011 | P05 future cutoffを拒否する | time leakage test |
| R3-012 | P07 future cutoffを拒否する | time leakage test |
| R3-013 | P06/P09 future cutoffを拒否する | review time test |
| R3-014 | P08 made/public/source timeを検証する | source metadata test |
| R3-015 | 日時不明をsystem dateで確定しない | unknown time test |
| R3-016 | P11 candidateのcutoff/existed_atを検証する | future candidate test |
| R3-017 | P12 corrected candidateのcutoff/existed_atを検証する | correction time test |
| R3-018 | P13がfuture correctionを採用できない | adjudication negative test |
| R3-019 | 意味判断promptへ市場結果を渡さない | prompt input boundary test |
| R3-020 | P09 corrected payloadへ通常と同じ完全検証を行う | invalid quote correction test |
| R3-021 | corrected payloadの別run/source/upstreamを拒否する | reference negative tests |
| R3-022 | corrected artifact fileとDB hashが一致する | file/hash audit |
| R3-023 | P09 valid correctを一度だけmaterializeする | idempotency test |
| R3-024 | accepted P08のreviewでも二重forecastを作らない | re-review test |
| R3-025 | reject/unresolvedのworkflowが無限反復しない | state transition test |
| R3-026 | 複数sourceの全P08をP11前に処理する | two-source vertical test |
| R3-027 | 完了runへの追加sourceを差分処理する | reopen run test |
| R3-028 | 差分処理で既存forecast/evaluationを再作成しない | count assertion |
| R3-029 | 別occurrenceへの安全なartifact再利用 | lineage reuse test |
| R3-030 | analyst/medium/model/version違いで再利用しない | negative reuse matrix |
| R3-031 | NEXT_ACTIONSにpending source/component IDを出す | generated file assertion |
| R3-032 | raw seriesとbasket seriesを別identityでcacheする | DB/cache test |
| R3-033 | basket cacheが単一symbol評価を汚染しない | AAA/BBB regression |
| R3-034 | provider symbol/currency不一致を検出する | provider negative test |
| R3-035 | duplicate symbol、weight不正を拒否する | mapping validation test |
| R3-036 | missing/common-date不足をunevaluableにする | market negative test |
| R3-037 | mixed currencyを先頭銘柄へ縮退しない | unevaluable test |
| R3-038 | basketの全input hash/weight/common ruleを追跡する | audit query |
| R3-039 | long形式CSVで複数symbolを評価できる | CSV basket test |
| R3-040 | CSV basketとmock providerが数値一致する | numeric comparison |
| R3-041 | repo docsとpackaged docsのdriftを検出する | sync --check |
| R3-042 | wheelから完全版Vault docsをseedする | clean wheel content test |
| R3-043 | Vault FUTURE_ROADMAPが詳細説明を維持する | required-section test |
| R3-044 | promptがspeaker/time/review新規則と整合する | resource content test |
| R3-045 | 設定済み高性能modelだけをpromptへ表示する | CLI/snapshot test |
| R3-046 | empty/0001/0005/0006 DBからheadへupgradeできる | migration tests |
| R3-047 | Alembic metadata差分0、FK check pass | command log |
| R3-048 | ruff format/check、mypy、pytestがpass | command log |
| R3-049 | clean wheelでhelp/init/run/source importがpass | package vertical test |
| R3-050 | DB/raw/cache/secret/build生成物をGit追跡しない | git/secret scan |
| R3-051 | CHAT_HISTORY.pdfをe165a6cから変更しない | blob hash/diff |
| R3-052 | 実装済み・未実装・external blockを正確に報告する | FINAL_REVIEW_ROUND3 |

## READY判定条件

READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICEはR3-001～052がすべてpassの場合だけ使用してください。

次はREADY判定の代替になりません。

- pytestの合計件数だけを示す。
- happy pathだけを通す。
- AIが正しく出力すると仮定する。
- 不正入力をSchemaではなくprompt文だけで防ぐ。
- repo docsが詳しいためVault要約版を許容する。
- network制約を理由にSQLite、CSV、wheelの内部failを隠す。

## FULL MVPとの区別

Round3を完了しても、総合点、複数アナリスト統合、現在上昇候補、時期・程度・早期実現の完全採点、期間不明の1/3/6/12か月観測、PNG等が未実装ならFULL_MVP_READYではありません。

