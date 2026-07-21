# 01 — 事前確認と不具合再現

## 目的

Round3で直す5つのデータ破損経路を、修正前に自分の環境で再現し、偶然テストが通るだけの修正を防ぎます。この工程では原因を記録し、無関係な実装変更はしません。

## Cursorへ渡す依頼文

    00_START_HERE.mdの拘束条件に従い、Round3のpreflightを実施してください。commit・pushはしないでください。

    project rootはrepository直下のanalyst_forecast_app_cursor_handoff/です。pyproject.toml、src、tests、docsが同じ階層にあることを確認し、二重階層を作らないでください。

    review commit e165a6cの実装を読み、次の5件を修正前のテストまたは一時probeで再現してください。一時probeを正式testへ残す場合は、修正後の期待値へ変更してください。

    1. P05 segmentのspeaker_candidateが「司会者」でも、P08がtarget_confirmedと自己申告すると正式ForecastIssuanceが作成される。
    2. P08のspeaker_attribution_statusがlegacy_unknownでも正式ForecastIssuanceが作成される。
    3. P09 correctのcorrected_payloadで原文quote offsetをずらしても、reviewがacceptedとなり正式ForecastIssuanceが作成される。
    4. 同じrunにblog sourceを2件入れ、両方P07まで完了後、1件目だけP08を完了すると、2件目がP08待ちなのにnext actionがRUN_P11になる。
    5. AAAが+20%、BBBが-20%の50/50 basketを評価すると、0%の合成seriesがAAA名でcacheされ、その後AAAを読むと0%が返る。

    さらに次を確認してください。

    - P05のknowledge_cutoffをsource recorded_atより後にしても受理されること。
    - packaged resources/docsのREADME、USER_MANUAL、SYSTEM_SPECIFICATION、FUTURE_ROADMAPがrepo docsより著しく短いこと。
    - P08のupstream_segment_refsを空にできるか。
    - not_targetだけを含むprocessed_with_forecastsの後、workflowがP08を無限に再要求しないか。
    - CsvMarketDataProviderが複数symbolを一つのCSVから識別できるか。

    再現結果をdocs/06_実装/ROUND3_REPRODUCTION.mdへ、入力条件、期待結果、実結果、原因候補、関連受入IDを含めて記録してください。既存83件のtest、ruff format/check、mypyのbaselineも記録してください。

    既存0001～0006 migrationは編集しないでください。DB、raw、一時VaultをGitへ追加しないでください。

## 必須成果物

- docs/06_実装/ROUND3_REPRODUCTION.md
- 修正前にfailする、またはxfail理由が明記された最小回帰test
- Round3の修正対象と対象外を対応付けた短いremediation plan

## 完了条件

- 5つの主要不具合をコード上だけでなく実行結果でも確認している。
- baselineの合格項目を壊してはならないことが明記されている。
- 問題を再現できない場合、pass扱いにせず環境・入力・観測値を報告して停止する。

