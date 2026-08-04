# Claude Codeレビュー依頼：クラシックOutlook直接D&D実装計画

作成日：2026-08-04
依頼先：Claude Code
レビュー対象：実装前の計画書

---

## 1. Claude Codeへ渡す依頼文

OneDrive年度別業務フォルダ向け「保存先レコメンダー」へ、クラシックOutlookからメールと添付ファイルを直接D&Dする機能の実装計画をレビューしてください。

今回は**実装前レビューだけ**が目的です。コードや文書の編集、依存導入、テストデータ作成、コミット、push、ブランチ作成、Outlook・OneDrive上の操作は行わないでください。リポジトリと公開資料の読み取り、read-onlyのGit・検索コマンドは利用して構いません。

### 対象リポジトリとコミット

| 項目 | 値 |
|---|---|
| リポジトリ | `https://github.com/baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark` |
| ブランチ | `codex/outlook-direct-dnd-plan` |
| 比較元コミット | `46563f9` |
| レビュー対象コミット | `23107f5` |
| 対象プロジェクト | `onedrive_destination_recommender/` |
| レビュー対象文書 | `obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_Outlook直接D&D.md` |

```bash
git fetch origin
git show 23107f5:"obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_Outlook直接D&D.md"
git diff 46563f9..23107f5 -- "obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_Outlook直接D&D.md"
```

別リポジトリの`baiputaojiu/myproject`ではありません。

### 先に読む資料

次の順で確認してください。

1. `obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_Outlook直接D&D.md` — レビュー対象
2. `onedrive_destination_recommender/README.md` — 現在の操作・安全制約・確認状況
3. `onedrive_destination_recommender/src/onedrive_destination_recommender/app.py` — 現行D&DとGUI接続
4. `onedrive_destination_recommender/src/onedrive_destination_recommender/session.py` — 入力規則と状態更新
5. `onedrive_destination_recommender/src/onedrive_destination_recommender/msg_reader.py` — MSG解析と現行の書込み禁止境界
6. `onedrive_destination_recommender/src/onedrive_destination_recommender/document_reader.py` — Office・PDF本文抽出
7. `onedrive_destination_recommender/tests/unit/test_msg_reader.py`
8. `onedrive_destination_recommender/tests/unit/test_session.py`
9. `onedrive_destination_recommender/tests/integration/test_gui.py`
10. `obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_ローカルファイルD&D.md` — 既存Explorer D&Dの設計・実測
11. `obsidan vault/30_Permanent/★業務/ファイル管理/要件定義_OneDrive保存先レコメンダー.md` — 現行要件

要件定義書にはOutlook直接D&Dを対象外とした過去の記録があります。今回の計画は、その対象外項目を新しい小改修として明示的に解除するものです。その他の安全制約は維持します。

### ユーザーと合意済みの前提

次はレビューで変更提案の対象にせず、実装方法が満たしているかを確認してください。

- 対象はクラシックOutlook for Windows
- Outlookのメール本体と添付ファイルの両方を直接D&Dできるようにする
- メール1通をドロップした場合、メールとその添付ファイル群を1案件として扱う
- 添付ファイルは0件でも1件以上でもよい
- 今回のメールは1通ずつ。将来は複数メールD&Dへ拡張したい
- 添付ファイルだけの単一・複数D&Dも受け付ける
- 新しいOutlook for Windowsは今回の対象外

### 変更しない安全制約

- OneDrive上のファイル・フォルダを変更、移動、削除、作成しない
- D&Dだけで候補確定、クリップボード更新、Audit追記を行わない
- メール本文、添付本文、検索語全文を設定・カタログ・Audit・ログへ残さない
- 外部サービスやAIへ自動送信しない
- 新しい入力が失敗したら直前の入力、検索語、候補、確定パスを維持する
- ファイル選択、Explorer D&D、Explorer Preview、確定・コピーを回帰させない

## 2. 最優先のレビュー観点

### 2.1 TkDND方式の実現性

計画では既存の`tkinterdnd2==0.6.2`が公開する次のタイプを明示登録します。

```python
DND_FILES
FileGroupDescriptorW  # "FileGroupDescriptorW - FileContents"
FileGroupDescriptor   # "FileGroupDescriptor - FileContents"
```

クラシックOutlookの仮想ファイルをTkDNDが一時領域へ実体化し、イベントへパスを返す前提です。次を確認してください。

- `tkinterdnd2==0.6.2`同梱のTkDND 2.10.1で、このタイプ対がdrop targetとして実際に有効か
- `drop_target_register()`へ3タイプを同時登録する呼び方が正しいか
- `set_dropfile_tempdir()`をstaging切替ごとに再設定できるか
- Unicode版とANSI版の順序に意味または副作用があるか
- 現行`DND_FILES`のExplorer D&Dと競合しないか
- 現行方式で不足する場合、Task 1のGo/No-Goで止める条件が十分か

可能ならMicrosoftのShell仕様、TkDNDの公式ソースまたはマニュアル、`tkinterdnd2`の現行ソースを根拠にしてください。推測は推測と明記してください。

### 2.2 一時領域の安全性

`DropWorkspaceManager`は、受理済みの`current`と次の`staging`を別ディレクトリとして所有します。次を重点確認してください。

- 成功時、失敗時、ファイル選択への置換、手動検索復帰、root破棄の全経路が閉じているか
- セッション更新と`accept_staging()`の順序で、新旧どちらかの入力が壊れる窓がないか
- 空イベント、ランタイム未読込、TkDND初期化失敗でも一時実体が残らないか
- 外部ローカルファイルを誤削除できないAPI境界になっているか
- TkDNDが同名ファイルをstaging内で上書きする可能性への対処が必要か
- 正常終了だけを対象にし、異常終了後の古いtemp自動掃除を今回入れない判断が妥当か

### 2.3 MSG添付の保存境界

現行`msg_reader.py`は、添付ファイル名を読むだけで、添付実体を保存しません。現行テストも`open()`、`.write()`、`logging`等の書込み・診断経路を禁止しています。

今回、対応文書本文を読むために、次の1経路だけを許可する計画です。

```text
Outlook Attachment.SaveAsFile
  → TemporaryDirectory(prefix="odr-msg-attachments-")
  → build_document_terms
  → 関数終了時に削除
```

次を確認してください。

- `probe_msg_access()`では保存せず、`build_msg_search_terms()`だけが保存する境界が明確か
- 対応拡張子だけを保存し、未対応添付は名前だけ使う方針が妥当か
- basename化と添付index接頭辞で、パストラバーサル・予約名・同名衝突を十分防げるか
- `Attachment.SaveAsFile()`の部分失敗をメール全体の失敗にしない実装が可能か
- 添付本文由来語を補助検索語だけへ入れる判断が既存ランキングと整合するか
- 生本文と一時パスが公開型、InputState、Audit、Codex相談プロンプトへ漏れないか
- `MsgSearchTerms`へ追加する3件の集計値が最小か。より単純で矛盾しない型があるか

### 2.4 将来の複数メール対応

今回は複数MSGを拒否し、`build_msg_search_terms()`を1通分の独立境界として保ちます。将来はこの関数をメールごとに反復する想定です。

- 今回の型と責務で複数通へ自然に拡張できるか
- 将来対応のためだけの不要な抽象化が混入していないか
- 複数通対応時に問題になる一時領域、検索語量、重複、状態表示の論点を、今回のBlockerと将来課題に分けてください

### 2.5 TDD計画と受け入れ条件

- 各Taskが失敗先行テスト→最小実装→合格確認の順になっているか
- テストが実装詳細ではなく利用者に見える挙動と安全性を固定しているか
- 実Outlookでしか確認できない事項と自動化できる事項の分離が妥当か
- 添付なし、添付あり、添付だけ、部分失敗、複数メール拒否、既存Explorer回帰が揃っているか
- テストコマンド、pytest marker、ファイルパス、型名、関数名に誤りがないか
- 受け入れ条件13件に、削れる重複または重大な不足がないか

## 3. レビュー方針

- 文体、命名の好み、将来あり得るだけの問題は指摘しないでください。
- 実装開始前に直す価値のある、再現可能な欠陥・矛盾・未定義・安全性問題を優先してください。
- 指摘には、該当する計画書の節または行、影響、発生条件、具体的修正案を含めてください。
- 現行コードを読まずに計画書だけから推測しないでください。
- 計画にない新規依存、Outlookアドイン、独自Win32 OLE実装を安易に追加しないでください。必要と判断する場合は、既存方式が成立しない根拠と追加コストを示してください。
- 「将来複数メールに対応するから」という理由だけで、今回キュー、並列処理、複数案件UI、永続モデルを追加しないでください。
- 資料から確認できる事実、合理的な推論、提案を区別してください。

## 4. 希望する回答形式

### 4.1 総評

- 実装着手可／条件付きで着手可／計画修正後に再レビュー、のいずれか
- 最大の技術リスク3件
- 計画が過剰または不足している箇所

### 4.2 指摘一覧

重大度順に、次の表で示してください。

| ID | 重大度 | 計画書の節・行 | 指摘 | 影響・発生条件 | 具体的修正案 |
|---|---|---|---|---|---|

重大度：

- Blocker：方式が成立しない、データ損失・情報漏えい、実装開始不能
- High：主要受け入れ条件違反または重大な回帰
- Medium：特定条件での誤動作、テスト不足、計画の重要な曖昧さ
- Low：実装前に直す価値はあるが主要動作を妨げない

指摘がない重大度は「なし」と明記してください。

### 4.3 要件・Task対応表

受け入れ条件13件について、実装Taskとテストが存在するかを確認してください。

| 受け入れ条件 | 実装Task | 自動テスト | 手動確認 | 判定 |
|---|---|---|---|---|

### 4.4 計画書へ反映する修正文

採用を推奨する変更について、そのまま計画書へ反映できる具体的な日本語またはコード断片を提示してください。

### 4.5 実装開始前チェックリスト

未解決事項だけを、優先順に最大10件で示してください。未解決事項がなければ「なし」としてください。

## 5. 最後に

レビュー結果だけを返してください。コードやファイルは変更しないでください。

---

## Claude Codeへの渡し方

1. 上記「Claude Codeへ渡す依頼文」をClaude Codeへ貼り付ける。
2. Claude Codeにリポジトリ、ブランチ、比較元コミット、レビュー対象コミットを直接参照させる。
3. 実業務メールや添付ファイルは渡さない。
4. GitHubへアクセスできない場合だけ、本計画書と§1の参照資料をローカルから読ませる。
5. 回答を受領したら、Blocker、High、Mediumの順に採否を判断し、本計画書へ反映する。
