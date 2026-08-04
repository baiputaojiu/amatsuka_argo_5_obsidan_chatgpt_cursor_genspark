# Claude Code第3回レビュー依頼：Outlook直接D&DとGitHub履歴削除の実装計画

作成日：2026-08-04
依頼先：Claude Code
レビュー対象：第2回レビュー反映後の計画書2件

---

## 1. Claude Codeへ渡す依頼文

OneDrive年度別業務フォルダ向け「保存先レコメンダー」について、次の実装前計画2件を第3回レビューしてください。

1. クラシックOutlookからメール・添付ファイルを直接D&Dし、送信者表示名を推薦へ利用する計画
2. PublicのGitHubリポジトリから私用フォルダ資料を全履歴ごと除去する計画

今回は**計画レビューだけ**が目的です。コード、文書、Git、GitHub、Outlook、OneDriveを変更しないでください。依存導入、テストデータ作成、commit、push、force-push、branch作成、worktree作成も禁止します。リポジトリ、依存ソース、Git履歴はread-onlyコマンドで確認して構いません。

### 対象リポジトリと差分

| 項目 | 値 |
|---|---|
| リポジトリ | `https://github.com/baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark` |
| ブランチ | `codex/outlook-direct-dnd-plan` |
| 今回の比較元 | `13dc50c` |
| レビュー対象 | `origin/codex/outlook-direct-dnd-plan`の最新tip |
| 対象プロジェクト | `onedrive_destination_recommender/` |
| Outlook計画 | `obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_Outlook直接D&D.md` |
| 履歴削除計画 | `obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_GitHub公開情報の履歴削除.md` |

```bash
git fetch origin
git rev-parse origin/codex/outlook-direct-dnd-plan
git diff --stat 13dc50c..origin/codex/outlook-direct-dnd-plan
git diff 13dc50c..origin/codex/outlook-direct-dnd-plan -- \
  "obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_Outlook直接D&D.md" \
  "obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_GitHub公開情報の履歴削除.md"
```

別リポジトリの`baiputaojiu/myproject`ではありません。

### 機密資料の取り扱い

履歴削除計画が削除対象として定義する私用資料は、レビューのために開く必要がありません。

- 削除対象資料の本文を読まない。
- 実フォルダ名、階層、件数、説明を回答へ引用しない。
- 削除対象の存在確認は追跡状態、履歴上のパス、`git-filter-repo`仕様の検討に限定する。
- 実メール、添付、ローカルruntime dataを探索しない。
- 計画書に含まれる合成名だけを例示に使用する。

## 2. ユーザーが確定した要件

次は変更提案の対象ではありません。計画が矛盾なく実装できるかだけを確認してください。

### 2.1 Outlook入力

- 対象はクラシックOutlook for Windows。新しいOutlookは対象外。
- メール1通と、その内部添付0件以上を1案件として扱う。
- Outlook添付だけの単一・複数D&Dも受け付ける。
- 今回はメール1通ずつ。将来は複数メールへ拡張するが、複数案件UIやキューは今回作らない。

### 2.2 送信者推薦

- Outlookの送信者表示名を今回から推薦へ利用する。
- 同じ表示名の別人をメールアドレスで区別しない。
- Unicode空白差を同一送信者として扱う。
- 送信者を取得できなくても処理を継続する。
- 送信者表示名と送信者キーをAudit、設定、カタログ、ログへ保存しない。
- **一般の主検索語が0件一致でも、送信者だけが一致すれば主照合1件の候補として表示する。**
- 「送信者だけでは候補を出さない」への変更は行わない。候補数が増えることは受容し、既存の候補件数上限と祖先・兄弟折り畳みを適用する。

### 2.3 GitHub公開履歴

- リポジトリはPublicのまま維持する。
- 実際のフォルダ名、命名例、階層、詳細資料は最新版と全branch/tagの履歴から除去する。
- 年度単位で探索する一般機能は公開対象として残す。
- 設定例とテストデータに実名がある場合は、動作する合成名へ置換する。
- ローカル資料と現在のユーザー未コミット変更を保持する。
- 履歴書き換えはレビュー承認後に行い、force-push直前に改めてユーザーの明示承認を得る。

## 3. 第2回レビューからの修正

### 3.1 Outlook計画

| ID | 対応 |
|---|---|
| NA-1 | 不採用。送信者単独候補を許可する要件をGlobal Constraints、ランキング、テスト、受け入れ条件へ明記 |
| NA-2 | `open(`を含む4項目の安全制約テストへ戻した |
| NA-3 | 生送信者名を持たない`Candidate.sender_matched`を追加し、UIは固定文言`送信者一致`だけを表示する計画にした |
| NA-4 | DragEnter孤児は次のD&D開始時または終了時までに削除すると明記した |
| NA-5 | `inspect.getsource()`への変更をやめ、現行の`Path.read_text()`を維持した |
| NA-6 | 不正確なテスト行番号を削除した |
| 受け入れ条件4 | Outlook添付だけの複数D&Dを1案件にする自動テストをTask 5へ追加した |

特に次を確認してください。

1. 一般主検索語一致数と`int(sender_matched)`を合算した後に0件除外する順序で、送信者単独候補が確実に残るか。
2. `primary_match_count == 1`、`matched_primary_terms == ()`、`sender_matched is True`という表現に内部矛盾がないか。
3. 候補表へ固定文言だけを表示し、生の送信者表示名・キーを永続化しない境界が十分か。
4. 添付だけの複数D&D、DragEnter孤児、安全制約4項目のテストが実装可能か。
5. 初回レビューB-1〜L-3の修正が今回の差分で後退していないか。

### 3.2 GitHub履歴削除計画

| ID | 対応 |
|---|---|
| NB-1 | 実際の年度フォルダ名・例は非公開、一般的な年度探索機能は公開と確定。設定例・テストを合成名へ変更する準備Taskを追加 |
| NB-2 | `allowed-changed-paths.txt`の最新版存続ファイルを、履歴書き換え前に保存した期待ファイルとSHA-256比較するStepを追加 |
| NB-3 | 全コミットメッセージを検査し、該当時だけ`--replace-message`を追加する分岐を追加 |
| NB-4 | `git ls-remote origin`で全ref種別を棚卸しする。pushはGitHub公式手順どおり`--force --mirror`を維持し、`refs/pull/*`だけを想定内失敗として記録する |
| NB-5 | `--sensitive-data-removal`実行直前の再fetchと、準備push後のremote凍結・再比較を明記 |
| NB-6 | `private-verification-patterns.txt`と`git grep -a -F -f`による全heads/tags検査を具体化 |
| NB-7 | originを自動復元せず、保持されているoriginの存在とURL一致だけを確認する方式へ修正 |
| NB-8 | `glob:`が`/`を跨ぐ可能性を確認し、想定外一致時は各パスの`literal:`列挙へ切り替えるStepを追加 |

NB-4はレビュー提案を一部変更しています。GitHub公式手順は`git push --force --mirror origin`を案内し、`refs/pull/*`の失敗を想定しています。そのためheads/tagsだけの個別pushへ変更せず、全refの事前棚卸し、pull ref失敗の記録、pull以外の失敗時停止で対応しています。この判断が妥当か検証してください。

特に次を確認してください。

1. 合成名化した最新版の設定例・テストが履歴書き換え後も期待SHA-256と一致する検証で、許可変更パスの穴を塞げるか。
2. `private-verification-patterns.txt`をregex置換manifestと分離し、`git grep -a -F -f`で検証する方式に漏れがないか。
3. コミットメッセージ検査と`--replace-message`の条件分岐が正しいか。
4. `--sensitive-data-removal`の再fetch前に全remote refsを比較する順序が正しいか。
5. `refs/pull/*`以外のpush失敗時に追加操作をせず停止する方針が安全か。
6. 公開状態を維持しながら、最新版・履歴・ローカル資料・dirty worktreeを分離して扱えているか。

## 4. Cursorでの実装運用

Outlook機能の実装はClaude Codeではなく、Windows上のCursor Composerを使う予定です。ユーザーの通常作業ツリーから分離した実装用worktreeをCursorで開き、Agents Window / Agent Tabsで複数エージェントを横並びまたはグリッド表示します。現在どの役割が動いているかをユーザーが眺められる運用であり、全エージェントが同時に動く必要はありません。

計画は`superpowers:subagent-driven-development`に基づき、次の役割を分離します。

1. Coordinator：計画と進捗ledgerを管理するread-only担当
2. TaskごとのImplementer：TDDで実装・テスト・commitする唯一の書き込み担当
3. Task Reviewer：commit差分の仕様適合と品質をread-onlyで判定
4. Final Reviewer：全Task後のブランチ全体レビュー

全タブは同じ実装用worktreeのcommit済み状態を参照し、複数のImplementerを同時に走らせません。重要指摘は元Implementerへ戻し、修正とscoped re-reviewを最大5周行います。クラシックOutlook実機確認はローカルWindowsで実施し、LinuxのBackground Agentへ委ねません。

この運用について、次だけ確認してください。

- 各Taskが別Implementerへ渡せるだけのInterfaceと入力情報を持つか。
- Task単位commit後のread-onlyレビューで、次Taskへ進むゲートを作れるか。
- 共有作業ツリーで書き込み担当を1つに限定すれば、複数Agent Tabsを開いたままでも競合を避けられるか。
- 進捗ledger、Task review、fix loop、final reviewが計画に十分明記されているか。

## 5. 基準値

計画改訂前のプログラム本体で確認済みです。

- 単体テスト：`146 passed, 7 deselected`
- 結合テスト：`6 passed, 1 skipped, 146 deselected`
- skip理由：`ODR_TEST_MSG_PATH`未設定
- `ruff check src tests`：合格
- `ruff format --check src tests`：25 files already formatted

今回のcommitは計画書とレビュー依頼文だけを変更する想定です。プログラム本体の再実行が不要という判断も含め、差分を確認してください。

## 6. 希望する回答形式

### 6.1 総評

計画ごとに次のいずれかを判定してください。

- 実装／実行着手可
- 条件付きで着手可
- 計画修正後に再レビュー

### 6.2 指摘一覧

| ID | 対象計画 | 重大度 | 節・行 | 指摘 | 影響・発生条件 | 具体的修正案 |
|---|---|---|---|---|---|---|

重大度はBlocker、High、Medium、Lowを使用し、指摘がない重大度は「なし」としてください。

### 6.3 第2回指摘対応表

NA-1〜NA-6、NB-1〜NB-8、添付だけの複数D&Dテストについて、`解消／要件として不採用／一部解消／未解消／別問題を導入`を判定してください。

### 6.4 要件・Task対応表

Outlookの受け入れ条件16件と履歴削除の受け入れ条件10件について、Task、自動検証、手動検証、判定を示してください。

### 6.5 開始前チェックリスト

未解決事項だけを計画ごとに優先順で最大10件示してください。なければ「なし」としてください。

## 7. 最後に

レビュー結果だけを返してください。コード、計画書、Git、GitHub、Outlook、OneDriveを変更しないでください。私用資料の内容・実名・件数を回答へ転載しないでください。

---

## Claude Codeへの渡し方

1. 本文書の「Claude Codeへ渡す依頼文」をClaude Codeへ渡す。
2. 第2回レビュー結果の原文も同時に渡す。
3. ブランチ`codex/outlook-direct-dnd-plan`の最新tipをread-onlyで参照させる。
4. 実業務メール、添付、私用フォルダ資料は渡さない。
5. 回答受領後、Blocker、High、Medium、Lowの順で採否を判断する。
6. レビュー承認前に実装または履歴書き換えへ進まない。
