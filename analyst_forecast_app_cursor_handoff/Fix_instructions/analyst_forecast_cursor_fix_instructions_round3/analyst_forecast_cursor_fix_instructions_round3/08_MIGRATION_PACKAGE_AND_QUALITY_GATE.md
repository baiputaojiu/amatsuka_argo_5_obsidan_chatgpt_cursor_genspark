# 08 — Migration、package、品質ゲート

## 目的

02～07の変更を既存DBとwheel配布へ安全に反映し、unit testだけでは見えないmetadata drift、package resource欠落、実データ混入を防ぎます。

## Cursorへ渡す依頼文

    00_START_HERE.md、01～07の変更を統合し、migration、package、品質ゲートを実行してください。commit・pushはしないでください。

    migration:

    - 既存0001～0006を原則編集しない。
    - 必要な変更は0007以降の新revisionにする。
    - SQLiteではrender_as_batchを使い、upgrade中のdata lossを防ぐ。
    - upgrade前backupを維持する。
    - legacy speaker/segment不明データをtarget_confirmedへ変換しない。
    - legacy market seriesをraw扱いする条件を明記し、不明ならlegacy_unknown等で区別する。
    - foreign key、unique constraint、index、nullable変更がSQLAlchemy metadataと一致するようにする。
    - downgradeでデータ喪失が避けられない場合は、制約とbackup前提をmigration docstringへ明記する。

    migration試験:

    - empty database → head
    - 0001 → head
    - 0005 → head
    - 0006 → head
    - Round2 fixture data入りDB → head
    - headでalembic check差分0
    - upgrade後のforeign_key_check
    - legacy forecast、source、evaluationの件数保持

    package:

    - sdistとwheelをbuildする。
    - repositoryのsrcをPYTHONPATHへ入れず、clean venvへwheelをinstallする。
    - clean venvでanalyst-forecast --helpを実行する。
    - Obsidian Vaultをinitする。
    - model名を通常CLIで設定する。
    - run createをYouTube、blogの各条件で実行する。
    - raw source 1件をimportし、NEXT_ACTIONSまで確認する。
    - packaged Schema、prompt、詳細docs、Alembic migrationがwheelに含まれることを確認する。
    - wheelからinitしたVaultだけを読み、repo checkoutなしで説明書が成立することを確認する。

    品質ゲート:

    - ruff format --check .
    - ruff check .
    - mypy src
    - pytest。integrationは通常除外理由を明示。
    - git diff --check
    - packaged docs sync --check
    - JSON Schema全件のparseとPydanticとの主要enum整合
    - 実行probeは正式testへ移し、一時fileへ依存しない。

    Git衛生:

    - reference/CHAT_HISTORY.pdfはe165a6cから変更しない。
    - .env.example以外の.env、secret、token、API keyを追跡しない。
    - SQLite、raw、market cache、backup、AI出力、実Vault絶対pathを追跡しない。
    - build、dist、egg-info、venv、pytest cacheを追跡しない。
    - repository内の既存ObsidianユーザーnoteをRound3作業で変更しない。

    外部network試験が環境制約で実行できない場合はexternal_blockedとして、実行command、error分類、代替mock/CSV試験を記録してください。内部Schema、SQLite、CSV、package、wheelのfailをexternal_blockedで隠さないでください。

## 必須成果物

- 新Alembic revision
- migration compatibility tests
- wheel clean venv縦断testまたは再現可能script
- docs/06_実装/ROUND3_QUALITY_GATE.md
- 追跡対象外fileの検査結果

## 完了条件

- 全内部品質ゲートがpassする。
- clean wheelからVault、docs、prompt、DB、runを生成できる。
- 既存DBを失わずheadへupgradeでき、metadata差分が0である。

