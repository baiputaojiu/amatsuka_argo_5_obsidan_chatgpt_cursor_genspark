# Claude Code第6回レビュー依頼：GitHub公開履歴削除計画

作成日：2026-08-04
依頼先：Claude Code
レビュー対象：第5回レビューのNE-1〜NE-3反映後の履歴削除計画

---

## 1. Claude Codeへ渡す依頼文

OneDrive年度別業務フォルダ向け「保存先レコメンダー」に関連する、Public GitHubリポジトリの履歴削除計画を第6回レビューしてください。

第5回レビューでND-1とND-3は解消済みと判定され、NE-1（Medium）、NE-2（Low）、NE-3（Low）が新たに指摘されました。今回のレビュー対象は、この3件の反映だけです。Outlook直接D&D計画は第4回レビューで実装着手可となっており、変更していません。

今回は**計画レビューだけ**が目的です。コード、文書、Git、GitHub、Outlook、OneDriveを変更しないでください。依存導入、テストデータ作成、commit、push、force-push、branch作成、worktree作成も禁止します。リポジトリ、依存ソース、Git履歴はread-onlyコマンドで確認して構いません。

### 対象リポジトリと差分

| 項目 | 値 |
|---|---|
| リポジトリ | `https://github.com/baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark` |
| ブランチ | `codex/outlook-direct-dnd-plan` |
| 今回の比較元 | `6370494` |
| レビュー対象 | `origin/codex/outlook-direct-dnd-plan`の最新tip |
| 履歴削除計画 | `obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_GitHub公開情報の履歴削除.md` |

```bash
git fetch origin
git rev-parse origin/codex/outlook-direct-dnd-plan
git diff --stat 6370494..origin/codex/outlook-direct-dnd-plan
git diff 6370494..origin/codex/outlook-direct-dnd-plan -- \
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

- リポジトリはPublicのまま維持する。
- 私用フォルダ構造・詳細資料は最新版と全branch/tag履歴から除去する。
- ローカル資料と現在のユーザー未コミット変更を保持する。
- ルート直下の`.gitignore`は、再混入防止のため削除対象のリポジトリ内パスとファイル名を公開状態で保持する。
- ルート直下の`.gitignore` 1ファイルだけを固定文字列検査から除外し、各head/tag tipの期待SHA-256一致で検証する。
- ネストした`.gitignore`は固定文字列検査の対象に残す。
- 既定ブランチ準備用worktreeは自動削除せず、旧cloneと同じネットワーク操作禁止の隔離対象として記録する。
- 履歴書き換えとforce-pushは、計画承認とforce-push直前の明示承認後に限る。

## 3. 第5回レビューからの修正

### 3.1 NE-1（Medium）：受け入れ条件と公開残存情報

次の3箇所を同じ境界へ統一しました。

1. §6 受け入れ条件3に、ルート直下の`.gitignore`が再混入防止規則として削除対象パスを意図的に保持する例外と、各head/tag tipの期待SHA-256一致を合格条件として追加。
2. §7へ、ルート直下の`.gitignore`に残る削除対象のリポジトリ内パス・ファイル名は公開され続け、実際の業務フォルダ構造・詳細内容とは別の残存情報であることを追加。
3. Task 8 Step 3の最終記録項目へ、ルート直下の`.gitignore`へ意図的に残す公開パス・名称を追加。

実際の業務フォルダ構造、詳細資料の本文、階層一覧を公開する変更ではありません。

### 3.2 NE-2（Low）：既定ブランチ準備worktreeの検査対象

Task 3 Step 6へ、現在の作業ディレクトリに依存しない完全な検査コマンドを追加しました。

```powershell
$cleanupDefaultLatestHits = git -C $cleanupDefaultWorktree grep -a -n -F -f $cleanupVerificationPatterns -- . ':(exclude).gitignore'
if ($LASTEXITCODE -gt 1) { throw "git grep failed for the default-branch preparation worktree." }
if ($cleanupDefaultLatestHits) { throw "Residual private text in the default-branch preparation worktree." }
```

### 3.3 NE-3（Low）：`.gitignore`除外範囲の表現

§2.2、Task 3 Step 4、Task 5 Step 2の表現を、実測済みのpathspec挙動へ合わせました。

- `:(exclude).gitignore`で除外するのはルート直下の`.gitignore` 1ファイルだけ。
- ネストした`.gitignore`は固定文字列検査の対象に残す。
- 期待SHA-256比較で担保する対象もルート直下の`.gitignore`と明記。

## 4. 重点確認事項

1. 受け入れ条件3、§7、Task 8 Step 3が、ルート直下の`.gitignore`へ公開状態で残るリポジトリ内パス・ファイル名を一貫して扱っているか。
2. 実際の業務フォルダ構造・詳細内容と、再混入防止のため残すリポジトリ内パス・ファイル名の境界が明確か。
3. `git -C $cleanupDefaultWorktree grep ...`が、既定ブランチ準備worktreeを確実に検査し、exit code 0・1・128を既存契約どおり扱えるか。
4. `:(exclude).gitignore`の説明が、ルート直下だけを除外し、ネストした`.gitignore`を検査対象に残す実挙動と一致するか。
5. NE-1〜NE-3の修正により、ND-1・ND-3、NC-1〜NC-5、NB-1〜NB-8への既存対応が後退していないか。
6. `gh auth status`失敗が、実行開始条件として唯一残る外部環境ゲートであるという理解に漏れがないか。

## 5. 基準値と今回の変更範囲

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

### 6.2 第5回指摘対応表

| ID | 判定 | 根拠 |
|---|---|---|
| NE-1 | 解消／一部解消／未解消／別問題を導入 |  |
| NE-2 | 解消／一部解消／未解消／別問題を導入 |  |
| NE-3 | 解消／一部解消／未解消／別問題を導入 |  |

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
2. 第5回レビュー結果の原文も同時に渡す。
3. ブランチ`codex/outlook-direct-dnd-plan`の最新tipをread-onlyで参照させる。
4. 実業務メール、添付、私用フォルダ資料は渡さない。
5. 回答受領後、Blocker、High、Medium、Lowの順で採否を判断する。
6. レビュー承認前に履歴書き換えへ進まない。
