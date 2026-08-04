# Claude Code第5回レビュー依頼：GitHub公開履歴削除計画

作成日：2026-08-04
依頼先：Claude Code
レビュー対象：第4回レビューのND-1・ND-3反映後の履歴削除計画

---

## 1. Claude Codeへ渡す依頼文

OneDrive年度別業務フォルダ向け「保存先レコメンダー」に関連する、Public GitHubリポジトリの履歴削除計画を第5回レビューしてください。

第4回レビューでOutlook直接D&D計画は「実装着手可」と判定済みで、今回その計画は変更していません。今回のレビュー対象は、履歴削除計画に残ったND-1（High）とND-3（Low）の反映だけです。

今回は**計画レビューだけ**が目的です。コード、文書、Git、GitHub、Outlook、OneDriveを変更しないでください。依存導入、テストデータ作成、commit、push、force-push、branch作成、worktree作成も禁止します。リポジトリ、依存ソース、Git履歴はread-onlyコマンドで確認して構いません。

### 対象リポジトリと差分

| 項目 | 値 |
|---|---|
| リポジトリ | `https://github.com/baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark` |
| ブランチ | `codex/outlook-direct-dnd-plan` |
| 今回の比較元 | `d0332f9` |
| レビュー対象 | `origin/codex/outlook-direct-dnd-plan`の最新tip |
| 履歴削除計画 | `obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_GitHub公開情報の履歴削除.md` |

```bash
git fetch origin
git rev-parse origin/codex/outlook-direct-dnd-plan
git diff --stat d0332f9..origin/codex/outlook-direct-dnd-plan
git diff d0332f9..origin/codex/outlook-direct-dnd-plan -- \
  "obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_GitHub公開情報の履歴削除.md"
```

別リポジトリの`baiputaojiu/myproject`ではありません。

### 機密資料の取り扱い

- 削除対象資料の本文を読まない。
- 実フォルダ名、階層、件数、説明を回答へ引用しない。
- 削除対象の存在確認は追跡状態、履歴上のパス、`git-filter-repo`仕様の検討に限定する。
- 実メール、添付、ローカルruntime dataを探索しない。
- 計画書に含まれる合成名だけを例示に使用する。

## 2. ユーザーが確定した境界

次は変更提案の対象ではありません。計画内で矛盾なく実行・検証できるかだけを確認してください。

- リポジトリはPublicのまま維持する。
- 私用フォルダ構造・詳細資料は最新版と全branch/tag履歴から除去する。
- ローカル資料と現在のユーザー未コミット変更を保持する。
- 履歴書き換えとforce-pushは、計画承認とforce-push直前の明示承認後に限る。
- **`.gitignore`全体を、最新版および全commitの`private-verification-patterns.txt`による固定文字列検査から除外してよい。**
- `.gitignore`の代替検証は、filter-repo実行前に作った期待ファイルと、書き換え後の各head/tag tipとのSHA-256比較とする。
- 全commitの過去`.gitignore`本文を固定文字列検査しないことは、ユーザーが受容した境界である。完全一致行だけを例外にする代替案への変更提案は不要。
- 既定ブランチ準備用worktreeは、ignore済みローカル資料を保護するため自動削除しない。旧cloneと同じネットワーク操作禁止の隔離対象として記録する。

## 3. 第4回レビューからの修正

### 3.1 ND-1（High）

第4回レビューでは、§3で追加する`.gitignore`行が`private-verification-patterns.txt`のリテラルを含むため、従来の`git grep -a -F -f`が必ず一致し、準備検査と全commit検査が必ず失敗すると指摘されました。

次の修正を行いました。

1. §2.2へ、`.gitignore`を固定文字列検査から除外する理由と、期待SHA-256比較で担保する境界を追加。
2. Task 3 Step 4の最新版検査を次のコマンドへ具体化。

```powershell
$cleanupLatestHits = git grep -a -n -F -f $cleanupVerificationPatterns -- . ':(exclude).gitignore'
if ($LASTEXITCODE -gt 1) { throw "git grep failed for the prepared worktree." }
if ($cleanupLatestHits) { throw "Residual private text in the prepared worktree." }
```

3. Task 5 Step 2の全commit検査を次のコマンドへ変更。

```powershell
$cleanupAllCommits = git rev-list --all
if (-not $cleanupAllCommits) { throw "No commits were found for verification." }
foreach ($cleanupCommit in $cleanupAllCommits) {
    $cleanupHits = git grep -a -n -F -f $cleanupVerificationPatterns $cleanupCommit -- . ':(exclude).gitignore'
    if ($LASTEXITCODE -gt 1) { throw "git grep failed for commit $cleanupCommit" }
    if ($cleanupHits) { throw "Residual private text in commit $cleanupCommit" }
}
```

4. Task 5 Step 2・3-2に、各head/tag tipの`.gitignore`を期待SHA-256と比較し、追加・欠落・変更を許可しないことを明記。

### 3.2 ND-3（Low）

既定ブランチ準備用worktreeと`codex/privacy-default-prep`の後処理が未定義だったため、Task 7 Step 4へ次を追加しました。

- worktreeの絶対パス、旧commit SHA、作業ツリー状態を実行記録へ保存する。
- ignore済みローカル資料が残り得るため自動削除しない。
- 旧cloneと同じネットワーク操作禁止の隔離対象として扱う。
- 将来削除するときは、バックアップと期待ファイルのSHA-256を再確認し、絶対パスを提示して別途明示承認を得る。

### 3.3 対応不要とした指摘

- ND-2はOutlook候補表の固定文言に関するLowで、第4回レビュー自身が「対応不要」と判定している。
- Outlook直接D&D計画は第4回レビューで実装着手可となっており、今回変更していない。

## 4. 重点確認事項

1. PowerShell 7、通常worktree、bare mirror cloneの各環境で、`git grep <commit> -- . ':(exclude).gitignore'`が意図どおり動作するか。
2. Task 3 Step 4とTask 5 Step 2の両方で、`.gitignore`由来の必然的な一致を除外しつつ、grep自体のエラーを`$LASTEXITCODE -gt 1`で検出できるか。
3. 各head/tag tipの`.gitignore`を期待SHA-256と比較するTask 5 Step 3-2が、ユーザー承認済みの除外境界と矛盾しないか。
4. worktreeを自動削除せず、絶対パス・旧SHA・状態を記録して旧cloneとともに隔離する手順が、ローカル資料の保持と旧履歴の再混入防止を両立するか。
5. ND-1・ND-3の修正により、NC-1〜NC-5、NB-1〜NB-8への既存対応が後退していないか。
6. `gh auth status`失敗を実行前ストップゲートとして残していることに矛盾がないか。

## 5. 基準値と今回の変更範囲

第4回レビューで、Outlook計画は実装着手可、履歴削除計画はND-1修正を条件として着手可と判定されています。

今回の変更はMarkdown計画書とレビュー依頼文だけです。`onedrive_destination_recommender/`配下を変更していないため、次の既存基準値は再実行していません。

- 単体テスト：`146 passed, 7 deselected`
- 結合テスト：`6 passed, 1 skipped, 146 deselected`
- skip理由：`ODR_TEST_MSG_PATH`未設定
- `ruff check src tests`：合格
- `ruff format --check src tests`：25 files already formatted

## 6. 希望する回答形式

### 6.1 総評

履歴削除計画を次のいずれかで判定してください。

- 実行着手可
- 条件付きで着手可
- 計画修正後に再レビュー

### 6.2 第4回指摘対応表

| ID | 判定 | 根拠 |
|---|---|---|
| ND-1 | 解消／一部解消／未解消／別問題を導入 |  |
| ND-3 | 解消／一部解消／未解消／別問題を導入 |  |

### 6.3 新規指摘一覧

| ID | 重大度 | 節・行 | 指摘 | 影響・発生条件 | 具体的修正案 |
|---|---|---|---|---|---|

重大度はBlocker、High、Medium、Lowを使用し、指摘がない重大度は「なし」としてください。

### 6.4 開始前チェックリスト

未解決事項だけを優先順で最大10件示してください。なければ「なし」としてください。

## 7. 最後に

レビュー結果だけを返してください。コード、計画書、Git、GitHub、Outlook、OneDriveを変更しないでください。私用資料の内容・実名・件数を回答へ転載しないでください。

---

## Claude Codeへの渡し方

1. 本文書の「Claude Codeへ渡す依頼文」をClaude Codeへ渡す。
2. 第4回レビュー結果の原文も同時に渡す。
3. ブランチ`codex/outlook-direct-dnd-plan`の最新tipをread-onlyで参照させる。
4. 実業務メール、添付、私用フォルダ資料は渡さない。
5. 回答受領後、Blocker、High、Medium、Lowの順で採否を判断する。
6. レビュー承認前に履歴書き換えへ進まない。
