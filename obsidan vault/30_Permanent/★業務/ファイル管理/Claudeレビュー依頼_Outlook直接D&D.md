# Claude Code再レビュー依頼：Outlook直接D&DとGitHub履歴削除の実装計画

作成日：2026-08-04
依頼先：Claude Code
レビュー対象：実装前の計画書2件

---

## 1. Claude Codeへ渡す依頼文

OneDrive年度別業務フォルダ向け「保存先レコメンダー」について、次の実装前計画2件を再レビューしてください。

1. クラシックOutlookからメール・添付ファイルを直接D&Dし、送信者表示名を推薦へ利用する計画
2. PublicのGitHubリポジトリから私用フォルダ資料を全履歴ごと除去する計画

今回は**計画レビューだけ**が目的です。コード・文書・Git状態・GitHub設定を変更しないでください。依存導入、テストデータ作成、commit、push、force-push、branch作成、worktree作成、Outlook・OneDrive操作も禁止します。リポジトリ、依存ソース、Git履歴はread-onlyコマンドで確認して構いません。

### 対象リポジトリとGit範囲

| 項目 | 値 |
|---|---|
| リポジトリ | `https://github.com/baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark` |
| ブランチ | `codex/outlook-direct-dnd-plan` |
| 比較元 | `42a69f4` |
| 計画コミット | `05cfc1a` |
| 対象プロジェクト | `onedrive_destination_recommender/` |
| Outlook計画 | `obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_Outlook直接D&D.md` |
| 履歴削除計画 | `obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_GitHub公開情報の履歴削除.md` |

```bash
git fetch origin
git show 05cfc1a:"obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_Outlook直接D&D.md"
git show 05cfc1a:"obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_GitHub公開情報の履歴削除.md"
git diff --stat 42a69f4..05cfc1a
git diff 42a69f4..05cfc1a -- \
  "obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_Outlook直接D&D.md" \
  "obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_GitHub公開情報の履歴削除.md"
```

別リポジトリの`baiputaojiu/myproject`ではありません。

### 機密資料の取り扱い

履歴削除計画§2.1が削除対象として定義する私用資料は、レビューのために開く必要がありません。次を守ってください。

- 削除対象資料の本文を読まない。
- 実フォルダ名、階層、件数、説明を回答へ引用しない。
- 削除対象の存在確認は`git ls-files`、`git log --name-status`、`git filter-repo`仕様の検討に限定する。
- 実メール、添付、ローカルruntime dataを探索しない。
- 計画書に含まれる合成名だけを例示に使用する。

### 先に読む資料

1. `実装計画_Outlook直接D&D.md` — 再レビュー対象A
2. `実装計画_GitHub公開情報の履歴削除.md` — レビュー対象B
3. 初回レビュー結果として依頼者から渡すテキスト — 指摘IDの原文
4. `onedrive_destination_recommender/README.md`
5. `onedrive_destination_recommender/src/onedrive_destination_recommender/app.py`
6. `onedrive_destination_recommender/src/onedrive_destination_recommender/session.py`
7. `onedrive_destination_recommender/src/onedrive_destination_recommender/msg_reader.py`
8. `onedrive_destination_recommender/src/onedrive_destination_recommender/terms.py`
9. `onedrive_destination_recommender/src/onedrive_destination_recommender/ranking.py`
10. 関連する`tests/unit/`と`tests/integration/test_gui.py`
11. `.gitignore`

私用フォルダ地図、生成詳細ページ、生成目次は読まないでください。

## 2. ユーザーと合意済みの前提

次は要件として固定し、変更提案ではなく計画が満たすかを確認してください。

### 2.1 Outlook入力

- 対象はクラシックOutlook for Windows
- メール本体と添付ファイルの両方を直接D&Dできる
- メール1通とその内部添付全体を1案件として扱う
- 添付ファイルは0件でも1件以上でもよい
- 今回はメール1通ずつ。将来は複数メールD&Dへ拡張したい
- 添付ファイルだけの単一・複数D&Dも受け付ける
- 新しいOutlook for Windowsは対象外

### 2.2 送信者

- 送信者情報を今回から推薦へ利用する
- Outlookの送信者表示名を使う
- 同じ表示名の別人をメールアドレスで区別しない
- `山田 太郎`、`山田　太郎`、`山田太郎`のような空白差を同一送信者として扱う
- 送信者を取得できなくても他の情報で処理を継続する
- 送信者表示名をAudit、設定、カタログ、ログへ保存しない

### 2.3 GitHub公開履歴

- リポジトリはPublicのまま維持する
- 私用フォルダ構造・詳細資料は最新版だけでなく全branch/tagの履歴から除去する
- ローカル資料は削除せず未追跡・ignore状態で保持する
- 現在のユーザー未コミット変更を保持する
- 履歴書き換えはレビュー承認後に実施する
- force-push直前に影響範囲と検証結果を提示し、ユーザーの明示承認をもう一度得る

## 3. 初回Outlook計画レビューの反映確認

初回レビューは「計画修正後に再レビュー」でした。指摘はB-1、B-2、H-1、H-2、M-1〜M-7、L-1〜L-3です。計画§8の反映表だけを信頼せず、本文・Task・コード例・受け入れ条件が実際に修正されているか確認してください。

特に次を再検証してください。

1. Tcl辞書補正後に`DND_FILES`だけを登録する方式が、TkDND 2.10.1の`generic::supported_type`、`GetWindowCommonTypes`、ネイティブ`GetData()`の動作と整合するか。
2. `root.tk.eval()`失敗時にOutlookだけを無効化し、Explorer D&Dを維持できる接続になっているか。
3. DragEnter時実体化に対する`discard_unreferenced(keep)`が、正しいイベントパス、外部パス、ディレクトリ、同名上書きを安全に扱えるか。
4. `accept_staging()`失敗後も新stagingを返す契約が実装可能で、セッションが一時パスを参照したまま壊れる窓がないか。
5. `PureWindowsPath.name`と`resolve().is_relative_to()`の順序、Windows予約名、空basename、末尾空白・ピリオドへの対策が十分か。
6. 保存前target、文書解析後parsed、メールwarning、文書warningの定義が矛盾しないか。
7. 一時MSG・添付の寿命案内が実際のCodex相談フローと一致するか。
8. 公開動作のテストが私有関数や実装詳細だけを固定していないか。

## 4. 送信者推薦設計のレビュー

計画は送信者名を一般主検索語へ混ぜず、空白を除去した`sender_key`と`PreparedFolder.sender_key_path`を専用照合し、主検索語一致数へ最大1件加算します。

次を確認してください。

- `NFKC + casefold + Unicode空白除去`が合意済みの空白差同一を満たすか
- 同じ表示名の別人を区別しない要件にメールアドレス非使用が一貫しているか
- 相対パス全体へのsubstring照合で階層境界をまたぐ誤一致が起きないか
- 送信者一致を主一致1件とする重みが、既存の年度優先、主一致数、補助一致、折り畳みと矛盾しないか
- 送信者一致だけの候補を表示する判断が要件から明確か
- 検索語編集後も自動送信者キーを維持するUXが既存の本文補助語と整合するか
- `sender_key`または生表示名がCandidate、InputState、Audit、Codex相談、例外、ログへ漏れないか
- 将来の複数メール対応のための抽象化が過剰でないか

曖昧な仕様があれば、実装前にユーザーへ確認すべき質問として明記してください。

## 5. GitHub履歴削除計画のレビュー

この計画は強い破壊的操作を含みます。コマンドを実行せず、次を厳密に確認してください。

### 5.1 削除範囲

- `--paths-from-file`の`literal:`と`glob:`が非ASCII・空白パスで意図どおり機能するか
- `--replace-text`のregex構文とnegative lookbehindが`git-filter-repo>=2.47`で機能するか
- 実フォルダ語句の置換が一般コード・テスト・`.gitignore`を壊す可能性を検知できるか
- rename・移動された過去パスの発見方法が十分か
- heads/tags以外の参照、PR refs、fork、cached view、LFS objectの扱いが正確か

### 5.2 作業順序とデータ保護

- 現在のdirty worktreeを変更せず、使い捨てmirrorで書き換える方式が安全か
- `git rm --cached`後もローカル資料を確実に保持できるか
- ユーザー変更の外部バックアップとSHA-256検証が十分か
- 準備push後にremote refs基準を取り直す順序が正しいか
- `git-filter-repo`が`origin`を削除する場合の復元手順が安全か
- bare mirrorで`filter-repo` metadataの場所を`git rev-parse --git-path`から得る方針が正しいか
- commit mapを使う対象外tree比較が、誤削除・誤置換を検知できるか
- default branch以外を含む全refへ再混入防止が行き渡るか
- force-pushの部分成功時に追加操作をせず停止する方針が妥当か
- fresh clone移行前に旧cloneを隔離する手順が十分か

### 5.3 実行可能性

現在の環境では、計画作成時点の`gh auth status`がinvalid tokenでした。計画はこれを実行前Blockerとして停止する方針です。ほかに必要なGitHub権限、branch protection、GitHub Support条件、共同利用者調整があれば指摘してください。

GitHubが非機密データのcache削除を拒否し得る点と、第三者cloneを削除できない点を踏まえ、「完全削除」という表現が過大でないかも確認してください。

## 6. テスト計画と基準値

計画改訂後、プログラム本体を変更せず次を確認済みです。

- 単体テスト：`146 passed, 7 deselected`
- 結合テスト：`6 passed, 1 skipped, 146 deselected`
- skip理由：`ODR_TEST_MSG_PATH`未設定
- `ruff check src tests`：合格
- `ruff format --check src tests`：25 files already formatted

各Taskについて次を確認してください。

- 失敗先行テスト、失敗理由、最小実装、合格確認、コミットの順序があるか
- テスト名、対象ファイル、公開インターフェース、コマンドに誤りがないか
- 実Outlookでしか確認できない項目と自動テストが分離されているか
- Outlook計画の受け入れ条件16件、履歴削除計画の受け入れ条件10件がTaskへ対応するか
- 重複、実装不能な例、重大な未検証経路がないか

## 7. レビュー方針

- 資料から確認できる事実、合理的な推論、提案を区別してください。
- 現行コードと依存ソースを読まずに計画書だけから推測しないでください。
- 文体、好み、今回の要件に影響しない将来論は指摘しないでください。
- 実装開始前またはforce-push前に直す価値のある欠陥、矛盾、未定義、安全性問題を優先してください。
- 指摘には計画書名、節または行、影響、発生条件、具体的修正案を含めてください。
- 計画にないOutlookアドイン、独自Win32 OLE実装、別Gitホスティングへの移行を安易に追加しないでください。
- 履歴削除について「バックアップがあるから安全」とせず、remote refs、対象外tree、部分push、再混入を個別に評価してください。

## 8. 希望する回答形式

### 8.1 総評

計画ごとに次のいずれかを判定してください。

- 実装／実行着手可
- 条件付きで着手可
- 計画修正後に再レビュー

各計画について最大のリスク3件を示してください。

### 8.2 指摘一覧

| ID | 対象計画 | 重大度 | 節・行 | 指摘 | 影響・発生条件 | 具体的修正案 |
|---|---|---|---|---|---|---|

重大度：

- Blocker：方式不成立、情報漏えい・データ損失、実行開始不能
- High：主要受け入れ条件違反、重大な回帰、復旧困難
- Medium：特定条件の誤動作、重要な曖昧さ、テスト不足
- Low：実装前に直す価値はあるが主要動作を妨げない

指摘がない重大度は「なし」と明記してください。

### 8.3 初回指摘対応表

B-1〜L-3の各IDについて、`解消／一部解消／未解消／別問題を導入`を判定し、根拠となる節を示してください。

### 8.4 要件・Task対応表

Outlook 16件と履歴削除10件の各受け入れ条件について、Task、自動検証、手動検証、判定を示してください。

### 8.5 計画書へ反映する修正文

採用を推奨する変更は、そのまま貼り付けられる日本語、型定義、テスト例、PowerShellコマンドとして提示してください。

### 8.6 開始前チェックリスト

未解決事項だけを計画ごとに優先順で最大10件示してください。なければ「なし」としてください。

## 9. 最後に

レビュー結果だけを返してください。コード、計画書、Git、GitHub、Outlook、OneDriveを変更しないでください。

---

## Claude Codeへの渡し方

1. 本文書の「Claude Codeへ渡す依頼文」をClaude Codeへ渡す。
2. 初回レビュー結果の原文も同時に渡す。
3. Claude Codeにブランチと計画コミット`05cfc1a`をread-onlyで参照させる。
4. 実業務メール、添付、私用フォルダ資料は渡さない。
5. 回答受領後、Blocker、High、Medium、Lowの順で採否を判断する。
6. レビュー承認前に実装または履歴書き換えへ進まない。
