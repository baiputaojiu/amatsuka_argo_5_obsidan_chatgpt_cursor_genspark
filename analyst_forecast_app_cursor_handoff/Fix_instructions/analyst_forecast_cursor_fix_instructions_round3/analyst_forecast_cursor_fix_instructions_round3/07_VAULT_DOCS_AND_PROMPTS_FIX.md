# 07 — Obsidian Vault説明書とpromptの正本化

## 目的

repository内docsは詳細なのに、wheelからVaultへseedされるresources/docsが3～9行の要約である問題を直します。後日ユーザーやAIがVaultだけを読んでも、操作、仕様、将来課題、制約を理解できる状態にします。

## Cursorへ渡す依頼文

    00_START_HERE.md、01～06の最終仕様に従い、packaged docsとpromptを更新してください。commit・pushはしないでください。

    現状のsrc/analyst_forecast/resources/docsにある数行の要約を、詳細説明書として扱わないでください。analyst-forecast init後のObsidian Vaultへ、少なくとも次の完全版を配置してください。

    - README
    - STARTUP_GUIDE
    - USER_MANUAL
    - SYSTEM_SPECIFICATION
    - FOLDER_STRUCTURE
    - WORKFLOW_DIAGRAMS
    - DATA_MODEL
    - METHODOLOGY
    - PROMPT_CATALOG
    - PYTHON_TASK_CATALOG
    - TROUBLESHOOTING
    - DECISION_LOG
    - OPEN_QUESTIONS
    - FUTURE_ROADMAP
    - AI_WORK_GUIDE

    同じ文書をrepo docsとresources/docsへ手作業で二重更新し続けない仕組みを用意してください。推奨は次です。

    - repo docsを編集上の正本にする。
    - scripts/sync_packaged_docs.py等でpackaged resourcesへ決定的に同期する。
    - --checkで差分があれば非0終了する。
    - build前または品質ゲートで--checkを必須にする。
    - runtimeでrepository相対pathへ依存せず、wheel内resourceだけでinitできるようにする。

    FUTURE_ROADMAPは短い箇条書きへ戻さないでください。各将来課題に、背景、実現したいこと、必要なデータ・変更、導入条件、完了条件、リスク、初期版で見送る理由を残してください。ループエンジニアリングは初期版では不採用、将来の大規模refactorや機能追加で検討する方針を明記してください。

    USER_MANUALとSTARTUP_GUIDEには、AIとPythonの役割を初心者向けに区別して説明してください。

    - AI: 話者・意味・予想・対象候補・レビュー
    - Python: ID、hash、Schema、DB、状態機械、市場値計算、集計、cache、出力
    - USER: raw準備、prompt実行、出力JSON取込み、結果確認

    02～06で変更した、alias、speaker verification、knowledge cutoff、P06/P09、source terminal state、basket CSV、cache分離を、操作例とトラブル時の次行動へ反映してください。

    prompt更新:

    - P05/P07はcontent authorとstatement speakerを区別する。
    - P08はnon-empty segment/evidence linkを出力する。
    - AIのtarget_confirmedはPython検証前のclaimであると明記する。
    - P06/P09 correctは完全corrected payloadを出す。
    - P11/P12/P13はknowledge cutoffと後知恵禁止を明記する。
    - 市場結果、04_results、market cacheを意味判断promptへ渡さない。
    - 使用モデル欄は通常操作で設定した高性能モデル名を表示し、未設定なら実行前に明確に停止・案内する。

## 必須テスト

1. sync packaged docs --checkがpassする。
2. wheelからinitしたVaultに上記文書がすべて存在する。
3. Vault FUTURE_ROADMAPに各将来課題の説明が残る。
4. Vault USER_MANUALだけでAI/Python/USERの役割を理解できる。
5. SYSTEM_SPECIFICATIONにfolder構造とworkflow図または参照がある。
6. promptに新しいspeaker/time/review rulesが含まれる。
7. prompt snapshotに設定したmodel名が入る。
8. 未設定modelを「高性能モデル」と曖昧表示したまま意味処理を開始しない。
9. update-docs時はユーザー編集をbackupしてから更新する。
10. wheel実行がrepo docsの存在へ依存しない。

## 完了条件

- 実Vaultに数行の要約ではなく完全版説明書が生成される。
- repo docsとwheel同梱docsのdriftを自動検出できる。
- 後日AIへVaultを読ませれば、目的、手順、制約、未実装、将来課題を復元できる。

