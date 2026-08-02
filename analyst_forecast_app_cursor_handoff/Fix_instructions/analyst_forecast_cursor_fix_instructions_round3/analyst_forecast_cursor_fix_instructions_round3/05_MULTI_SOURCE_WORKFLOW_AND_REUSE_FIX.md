# 05 — 複数情報源のsource単位workflowと安全な再利用

## 目的

一つのsourceから予想が作成された後、同じrun内の別sourceがP08未実行のままP11へ進む問題を直します。追加情報源だけを差分処理し、全sourceの状態をターミナルとNEXT_ACTIONSで把握できるようにします。

## Cursorへ渡す依頼文

    00_START_HERE.md、01～04に従い、workflowをsource単位で修正してください。commit・pushはしないでください。

    現在の「pending_p08 and not context.issuances」というrun全体条件を廃止してください。1件でもpending P08 sourceがあるなら、別sourceにforecast issuanceが存在していても、そのpending sourceのP08を次行動として提示する必要があります。

    source processing stateを明確にしてください。名称は既存互換を考えて決めてよいですが、少なくとも次を区別してください。

    - raw imported
    - P05/P07 pending
    - preprocessing needs AI review
    - preprocessing accepted
    - P08 pending
    - P08 needs AI review
    - processed with formal forecasts
    - processed with no forecast
    - processed with only excluded/non-target statements
    - rejected retryable
    - rejected terminal
    - unresolved AI review

    推奨actionの優先順序:

    1. 参照整合性またはDB破損blocker
    2. 未解決P06/P09 review
    3. 未処理P05/P07 source
    4. 前処理済みだがP08未処理の全source
    5. 正式componentのP11
    6. P12
    7. 必要時P13
    8. 市場評価
    9. 結果確認

    同順位が複数ある場合は、source_id、component_id、入力path、必要promptを表示し、毎回同じ決定順になるようにしてください。件数だけでなくpending ID一覧をWORKFLOW_STATE.jsonとNEXT_ACTIONS.mdへ出してください。

    追加情報源:

    - 完了済みrunへ新しいsourceをimportしたら、そのsourceだけP05/P07→P08を処理する状態へ戻る。
    - 既存source、既存forecast、既存evaluationを再処理しない。
    - 新sourceのP08が完了するまでREVIEW_RESULTSへ戻さない。
    - selected_mediaが複数でも媒体別promptを正しく選ぶ。

    再利用:

    - 同一raw hash、同一analysis target、同一medium、同一prompt version、同一model、同一speaker条件の前処理artifactだけを再利用する。
    - 別source occurrenceでsource_idが異なる場合、P08のsource参照が破綻しないよう、origin artifactへのlineageを持つimmutable associationまたはderived artifactを作る。
    - source_idが異なるartifactを同一sourceだと偽装しない。
    - 元artifact fileをコピーしてhashを変えたことにしない。
    - reuse manifestにorigin run/source/artifact、raw hash、prompt version、model、reuse reasonを保存する。
    - analyst、medium、raw hash、model、prompt versionのいずれかが異なれば再利用しない。

    0件処理:

    - P08に予想0件ならprocessed_no_forecastとしてterminalにする。
    - 予想候補はあるが全件not_target、third_party、legacy_unknownならprocessed_no_formal_forecastとしてterminalにする。
    - terminal sourceを同じP08へ無限に戻さない。

## 必須テスト

1. blog 2件をP07まで処理し、1件だけP08後も次が残り1件のP08になる。
2. YouTubeとblog混在runでP05/P07/P08をsourceごとに完了できる。
3. source Aにforecastがあってもsource BのP08が取り残されない。
4. source Aがno forecastでもsource BのP08を処理できる。
5. 全件not_target sourceがterminalとなり無限反復しない。
6. 完了runへsource追加後、そのsourceだけ差分処理する。
7. 追加source処理で既存forecast/evaluation件数が増殖しない。
8. 同一raw・同一条件を別runで安全に再利用できる。
9. 同一rawだが別URL/source occurrenceでもP08参照が壊れない。
10. 別analyst、別medium、別model、別prompt versionでは再利用しない。
11. unresolved/rejectedの次行動とretry可否がterminal stateに応じて正しい。
12. WORKFLOW_STATEとNEXT_ACTIONSにpending source IDとpromptが表示される。

## 完了条件

- run内の全sourceがterminalになるまで対象解決へ進まない。
- 新しく追加した情報源だけを再処理する。
- 再利用してもrun/source/artifact lineageが壊れない。

