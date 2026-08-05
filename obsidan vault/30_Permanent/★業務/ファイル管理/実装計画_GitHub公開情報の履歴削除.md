# 実装計画：GitHub公開履歴からの私用フォルダ資料削除

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to execute this plan with stop gates. Use `superpowers:verification-before-completion` before the force-push and before declaring the cleanup complete.

**Goal:** PublicのGitHubリポジトリから私用フォルダ構造・詳細資料を全ブランチ・タグの履歴ごと除去し、ローカル資料は保持したまま再混入を防止する。

**Architecture:** 対象機能ブランチの作業ツリーと`origin/<default>`基点の分離worktreeで、削除対象を未追跡化してignoreし、存続文書から具体的な構造情報を除去する準備コミットを別々に作る。履歴書き換えは作業ツリーではなく新しい使い捨てmirror cloneに対して`git-filter-repo>=2.47`で行い、削除パスと非公開語句を全refから除去する。ローカル検証、リモート無変更確認、ユーザーの最終承認を通過した場合だけ`git push --force --mirror`を実行し、旧cloneは再利用せず隔離する。

**Tech Stack:** Git、GitHub、GitHub CLI、`git-filter-repo>=2.47`、PowerShell 7。

## Global Constraints

- リポジトリはPublicのまま維持し、Privateへ変更しない。
- フォルダ構造一覧、生成詳細ページ、生成目次、生成スクリプトを全履歴から除去する。
- 存続文書に含まれる実際のフォルダ名・階層・詳細内容・削除対象への参照も全履歴から除去または一般表現へ置換する。
- 「全階層を検索する」等、実構造を明かさない一般要件とプログラム本体は残す。
- 実際の年度フォルダ名、命名例、階層断片は非公開情報として除去する。年度単位で探索する一般機能は公開対象として残し、設定例とテストデータは実名ではない合成名へ置換する。
- ローカルの私用資料は削除せず、元の配置で未追跡・ignore状態として保持する。
- 現在の未コミット変更を上書き、破棄、履歴書き換え対象へ混入させない。
- force-push前に全remote refsのスナップショットを保存し、開始後のリモート更新を検知した場合は中止する。
- 対象外パス、コミット作者、コミット日時、一般文書、プログラムコードは変更しない。
- 既定ブランチへ機能ブランチ全体をfast-forwardまたはmergeしない。既定ブランチには、分離worktreeで作ったプライバシー準備差分だけを通常pushする。
- ブランチ保護を自動解除しない。保護によりpushできない場合は停止してユーザー判断を求める。
- fork、第三者clone、GitHubのPR参照・キャッシュはローカル履歴書き換えだけでは消去できないものとして別確認する。
- 履歴書き換え後、旧cloneから`pull`や`push`を行わない。新しいclean cloneを正本とする。

---

## 1. 文書情報

| 項目 | 値 |
|---|---|
| 対象リポジトリ | `baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark` |
| 公開状態 | Publicを維持 |
| 作成日 | 2026-08-04 |
| 状態 | Claude Code第6回レビュー承認済み・実行環境ゲート確認中 |
| 履歴書き換え方式 | `git-filter-repo --sensitive-data-removal --invert-paths` |
| リモート更新方式 | 最終承認後の`git push --force --mirror origin` |

本計画はClaude Code第6回レビューで実行着手可と判定された。§4と§9の実行環境ゲートを満たすまでは、追跡解除、`.gitignore`変更、履歴書き換え、force-push、GitHub設定変更を行わない。

## 2. 削除範囲

### 2.1 全履歴から削除するパス

実行時にリポジトリ外の`private-removal-paths.txt`へ次の規則をUTF-8で保存する。

```text
literal:private/local-structure.txt
glob:private/local-detail-*.md
literal:private/local-file-management.md
literal:private/local-page-generator.py
```

`glob:`の`*`は`/`を跨いで一致し得るため、Task 1で全履歴の一致パスを列挙する。想定ディレクトリ直下以外へ1件でも一致する場合は、この`glob:`を使用せず、確認済みの各パスを`literal:`で個別列挙する。

同じ資料が過去に別パス・別名で存在した場合は、`git log --all --name-status --follow`と`git filter-repo --analyze`の結果から旧パスを同ファイルへ追加する。旧パスが0件である場合は、その確認結果を実行記録へ残す。

### 2.2 全履歴から置換する内容

リポジトリ外の`private-replacements.txt`へ、次の2種類を記載する。

1. 削除対象ファイル名・参照名を一般表現へ置換する規則。
2. 存続文書に現れる実フォルダ名・階層断片・詳細説明を`***REMOVED***`へ置換する規則。

固定規則は、行頭`/`を持つ`.gitignore`を壊さないよう、リポジトリ相対パスの直前が`/`ではない文書参照だけを対象にする。

```text
regex:(?<!/)private/local-structure\.txt==>***REMOVED***
regex:(?<!/)private/local-file-management\.md==>***REMOVED***
regex:(?<!/)private/local-page-generator\.py==>***REMOVED***
regex:(?<!/)private/local-detail-[^\r\n`/]+\.md==>***REMOVED***
```

実フォルダ名の規則は、削除対象資料と存続文書の共通語をローカルで抽出して作る。ファイル自体はリポジトリ外に置き、内容を画面出力、Audit、コミットへ含めない。短い一般語、年度表現、製品名等まで過剰置換しないよう、各規則を`git grep`で事前確認する。書き換え後の`.gitignore`を期待ファイルとbyte比較し、固定規則または実フォルダ名規則がignore行を変更していないことをforce-push前に確認する。

§3の再混入防止規則は、`private-verification-patterns.txt`に含まれる削除対象パスをリポジトリルート直下の`.gitignore`内へ意図的に保持する。固定文字列検査は行頭`/`の有無を区別できないため、最新版検査と全commit検査ではルート直下の`.gitignore` 1ファイルだけをパススペック`:(exclude).gitignore`で除外する。ネストした`.gitignore`は検査対象のまま残す。この除外はユーザーが明示的に承認した境界であり、ルート直下の`.gitignore`の正しさは既定・対象機能ブランチを含む各head/tag tipについて、filter-repo実行前に生成した期待ファイルとのSHA-256比較で担保する。

### 2.2.1 存続コード・テスト内の実名

実フォルダ名・階層断片が存続する設定例、テスト、一般文書は、一律に`***REMOVED***`へ置換しない。既定ブランチと対象機能ブランチの最新版は、それぞれ専用の準備コミットで次の規則に従って一般化し、その結果を履歴上の期待内容とする。その他のbranch/tag tipは、Task 1で人が承認した置換規則から機械生成した期待内容と比較し、実行可能性は要求しない。

1. 設定例とテストデータは、実際の年度フォルダ名や階層を示さない合成名へ置換し、テストの期待値も同じ合成名へ更新する。
2. 一般文書は、実名、件数、具体階層、実例だけを一般表現へ置換し、年度単位の探索、推薦、回帰テストという一般要件は残す。
3. 変更した存続パスを`allowed-changed-paths.txt`へ列挙し、準備コミットの期待ファイルをリポジトリ外へ保存してSHA-256を記録する。
4. 既定ブランチと対象機能ブランチの各準備コミットでアプリの全自動テストとruffを通してから通常pushする。`***REMOVED***`を設定値やテスト入力として使用しない。

### 2.3 最新版から削除する参照

準備コミットでは存続文書を手動で読み、実構造を前提とするリンク、件数、例示、詳細説明を一般化する。一般的な機能要件、テスト要件、設計判断は残す。

## 3. 最新版の再混入防止

`.gitignore`へ次を追加する。

```gitignore
# Private folder-map artifacts must remain local and untracked.
/private/local-structure.txt
/private/local-details/*.md
/private/local-file-management.md
/private/local-page-generator.py
```

`git rm --cached`を使用し、作業ツリーの実ファイルは残す。`git add -A`、`git commit -a`は使用しない。

## 4. 実行前ストップゲート

次のいずれかに該当する場合は履歴書き換えへ進まない。

- `gh auth status`が失敗する。
- `git filter-repo --version`が2.47未満、またはコマンドが存在しない。
- open PR、fork、ブランチ保護、共同利用者の有無を確認できない。
- 現在の未コミット変更をリポジトリ外へバックアップできない。
- 削除パスまたは置換語句の一覧に未確認項目がある。
- 許可変更パスの期待ファイルとSHA-256をリポジトリ外へ保存できない。
- mirror cloneの検証で対象外ファイルのtree差分が発生する。
- force-push直前のremote refsが開始時スナップショットと異なる。
- ユーザーからforce-push直前の明示承認を得ていない。

## 5. 実行タスク

### Task 1: 対象と外部状態の棚卸し

**Files:**
- Read: Git refs、PR、fork、branch protection、追跡ファイル一覧
- Create outside repository: `remote-refs-before.txt`
- Create outside repository: `remote-refs-force-baseline.txt`
- Create outside repository: `private-removal-paths.txt`
- Create outside repository: `private-replacements.txt`
- Create outside repository: `private-verification-patterns.txt`
- Create outside repository: `allowed-changed-paths.txt`
- Create outside repository: `expected-allowed-blobs.tsv`と期待ファイル群

- [ ] **Step 1: ツールと認証を確認する**

```powershell
git --version
git filter-repo --version
gh --version
gh auth status
```

Expected: 全コマンド成功。`git-filter-repo`は2.47以上。

- [ ] **Step 2: GitHub上の影響範囲を確認する**

```powershell
gh repo view baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark --json visibility,defaultBranchRef,url
gh pr list --state open --limit 100
gh api repos/baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark --jq '{forks_count,open_issues_count}'
gh api repos/baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark/branches --paginate --jq '.[] | {name,protected}'
git ls-remote origin |
    ForEach-Object { ($_ -split "`t")[1] } |
    Where-Object { $_ -like 'refs/*' } |
    Group-Object { ($_ -split '/')[1] } |
    Select-Object Count, Name
```

Expected: `visibility`は`PUBLIC`。open PR、fork、protected branchに加え、`refs/heads/*`、`refs/tags/*`、`refs/pull/*`、その他refの実数を実行記録へ保存する。

- [ ] **Step 3: 準備開始前のremote refsをリポジトリ外へ保存する**

```powershell
git ls-remote origin | Set-Content -LiteralPath $cleanupRemoteRefs -Encoding utf8
```

`$cleanupRemoteRefs`はタスク専用一時ディレクトリ内の絶対パスとし、リポジトリ内へ作らない。この一覧は監査用であり、Task 5の同時更新判定には準備push後の別一覧を使う。

- [ ] **Step 4: 全履歴の対象パスを棚卸しする**

```powershell
git -c core.quotepath=false log --all --name-status -- "obsidan vault/30_Permanent/★業務/ファイル管理"
git filter-repo --analyze
```

分析結果から改名・移動履歴を確認し、§2.1のmanifestを確定する。分析出力はコミットしない。

詳細ページの`glob:`が想定ディレクトリ直下以外へ一致しないことも確認する。該当があれば`glob:`を廃止し、確認済みパスを`literal:`で列挙する。

コミットメッセージは`--replace-text`の対象外なので、`private-verification-patterns.txt`の各リテラルを全コミットメッセージへ照合する。

```powershell
$cleanupPrivateLiterals = Get-Content -LiteralPath $cleanupVerificationPatterns | Where-Object { $_.Length -gt 0 }
if (-not $cleanupPrivateLiterals) { throw "Verification patterns are empty." }
$cleanupMessageText = git log --all --format='%H%n%B%n--END--'
$cleanupMessageHits = $cleanupMessageText | Select-String -SimpleMatch -Pattern $cleanupPrivateLiterals
```

Expected: 0件。1件以上なら`--replace-message $cleanupReplacements`をTask 4 Step 3へ追加し、書き換え後にも同じ検査を行う。

- [ ] **Step 5: 存続文書の具体情報を棚卸しする**

削除資料の名称と階層断片を存続ファイルへ照合し、`private-replacements.txt`と、書き換え後検証用のリテラルだけを持つ`private-verification-patterns.txt`を確定する。設定例・テスト・一般文書を§2.2.1に分類し、置換前後のヒット件数を記録して一般語の誤置換が0件であることを人手確認する。最新版で手動編集する一般化・合成名化も、その他refの期待内容を機械生成できる決定的な置換規則としてmanifestへ表現し、適用順を固定する。パス削除、内容置換、合成名化、`.gitignore`追加によりblob変更を許可するパスだけを`allowed-changed-paths.txt`へ列挙する。

### Task 2: ローカル資料と未コミット変更の保護

**Files:**
- Copy outside repository: 私用資料4分類
- Copy outside repository: `obsidan vault/.obsidian/plugins/editing-toolbar/data.json`
- Copy outside repository: `obsidan vault/.obsidian/workspace.json`
- Create outside repository: `working-tree-status.txt`

- [ ] **Step 1: 状態を記録する**

```powershell
git status --short --branch | Set-Content -LiteralPath $cleanupStatusPath -Encoding utf8
git diff --binary | Set-Content -LiteralPath $cleanupPatchPath -Encoding utf8
git ls-files --others --exclude-standard | Set-Content -LiteralPath $cleanupUntrackedPath -Encoding utf8
```

- [ ] **Step 2: 対象資料とユーザー変更をリポジトリ外へコピーする**

`Copy-Item -LiteralPath`を使い、タスク専用バックアップディレクトリへ1ファイルずつコピーする。コピー元とコピー先のSHA-256を`Get-FileHash`で比較する。

- [ ] **Step 3: バックアップ検証を記録する**

Expected: 全対象でコピー元とコピー先のSHA-256が一致する。不一致が1件でもあれば停止する。

### Task 3: 既定・対象機能ブランチ最新版の追跡解除と文書一般化

**Files:**
- Modify: `.gitignore`
- Modify: 実構造を参照する存続文書、設定例、テスト
- Untrack, keep on disk: §2.1の4分類
- Create outside repository: 既定ブランチ専用worktree

- [ ] **Step 1: `.gitignore`の失敗先行確認を行う**

```powershell
git check-ignore -v -- "private/local-structure.txt"
```

Expected before edit: ignore規則がないため非0終了。

- [ ] **Step 2: §3のignore規則を追加する**

- [ ] **Step 3: 私用資料をディスクに残して追跡だけ解除する**

```powershell
git rm --cached -- "private/local-structure.txt"
git rm --cached -- "private/local-file-management.md"
git rm --cached -- "private/local-page-generator.py"
$cleanupTrackedDetails = git -c core.quotepath=false ls-files -- "private/local-detail-*.md"
foreach ($cleanupTrackedDetail in $cleanupTrackedDetails) {
    git rm --cached -- $cleanupTrackedDetail
}
```

PowerShellまたはGitのglob解釈に依存せず、実行時は`git ls-files`で得た各パスを個別に`git rm --cached -- <exact-path>`へ渡す。

- [ ] **Step 4: 存続ファイルから実構造情報を一般化する**

一般文書は実名・具体階層・詳細だけを一般表現へ変更する。設定例とテストは実際の命名を示さない合成名へ変更し、期待値も同時に更新する。変更後、ルート直下の`.gitignore`を除く最新版の全追跡ファイルで`private-verification-patterns.txt`の全リテラルが0件であり、`***REMOVED***`が設定値またはテスト入力へ入っていないことを確認する。

```powershell
$cleanupLatestHits = git grep -a -n -F -f $cleanupVerificationPatterns -- . ':(exclude).gitignore'
if ($LASTEXITCODE -gt 1) { throw "git grep failed for the prepared worktree." }
if ($cleanupLatestHits) { throw "Residual private text in the prepared worktree." }
```

ルート直下の`.gitignore`だけをこのリテラル検査から全体除外し、Step 5・6で保存する期待ファイルとのSHA-256一致で検証する。ネストした`.gitignore`は除外しない。

変更した存続ファイルと`.gitignore`は、`allowed-changed-paths.txt`に含まれることを確認してから、`git add -- <exact-path>`で1パスずつstageする。`git add -A`、`git commit -a`は使わない。私用資料のindex削除はStep 3の`git rm --cached`でstage済みの状態を維持する。

- [ ] **Step 5: 最新treeと許可変更パスの期待値を検証する**

```powershell
git check-ignore -v --no-index -- "private/local-structure.txt"
git diff --cached --name-status
git diff --cached --check
```

アプリの全自動テストとruffも実行する。Expected: 私用資料は削除予定としてstageされるがローカルには存在し、合成名へ変更した設定例・テストを含めて検査が成功し、無関係なユーザー変更はstageされない。

対象機能ブランチの準備コミット予定内容を一時worktreeへ展開し、`allowed-changed-paths.txt`の各存続ファイルをリポジトリ外の`expected-allowed/feature/`へコピーする。`Get-FileHash -Algorithm SHA256`で`expected-allowed-blobs.tsv`へ`ref<TAB>path<TAB>sha256`を記録する。`.gitignore`は必ず含め、期待ファイルは履歴書き換え後の出力から作らない。

- [ ] **Step 6: 既定ブランチ専用worktreeでプライバシー準備差分だけを作る**

既定ブランチ名とpush前SHAをGitHubから取得し、対象機能ブランチとは別のリポジトリ外worktreeを`origin/<default>`から作る。既存の同名ローカルブランチまたはworktreeがある場合は自動削除・再利用せず停止する。

```powershell
$cleanupDefaultBranch = gh repo view baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark --json defaultBranchRef --jq '.defaultBranchRef.name'
if (-not $cleanupDefaultBranch) { throw "Default branch was not resolved." }
$cleanupDefaultRef = "refs/heads/$cleanupDefaultBranch"
$cleanupDefaultBeforeLine = git ls-remote origin $cleanupDefaultRef | Select-Object -First 1
if (-not $cleanupDefaultBeforeLine) { throw "Default branch ref was not found." }
$cleanupDefaultBefore = ($cleanupDefaultBeforeLine -split "`t")[0]
git worktree add -b codex/privacy-default-prep $cleanupDefaultWorktree "origin/$cleanupDefaultBranch"
```

このworktree内で、その既定ブランチに存在する対象だけへStep 1〜4と同じignore、追跡解除、一般化、合成名化、1パスずつのstageを適用する。対象機能ブランチの準備commitをcherry-pickせず、機能実装・計画・その他のcommitを既定ブランチへ持ち込まない。`allowed-changed-paths.txt`以外の差分が0件であること、私用資料がworktreeの外に保存済みであること、全自動テストとruffが成功することを確認する。既定ブランチの期待ファイルを`expected-allowed/default/`へ保存し、SHA-256を`expected-allowed-blobs.tsv`へ記録する。

Step 4のリテラル検査は現在の作業ディレクトリへ依存させず、次のとおり既定ブランチ準備worktreeを`git -C`で明示して実行する。

```powershell
$cleanupDefaultLatestHits = git -C $cleanupDefaultWorktree grep -a -n -F -f $cleanupVerificationPatterns -- . ':(exclude).gitignore'
if ($LASTEXITCODE -gt 1) { throw "git grep failed for the default-branch preparation worktree." }
if ($cleanupDefaultLatestHits) { throw "Residual private text in the default-branch preparation worktree." }
```

- [ ] **Step 7: 2つの準備差分と検査結果を提示し、通常pushの承認を得る**

対象機能ブランチと既定ブランチ専用worktreeについて、基点SHA、変更パス、削除・一般化内容、対象外差分0件、テスト・ruff結果を提示する。既定ブランチを含む2回の通常pushについてユーザーの明示承認を得るまでcommitとpushへ進まない。

- [ ] **Step 8: 2つの準備コミットを作成して通常pushする**

```powershell
git commit -m "privacy: 私用フォルダ資料を公開対象から除外"
git push origin codex/outlook-direct-dnd-plan

git -C $cleanupDefaultWorktree commit -m "privacy: 私用フォルダ資料を公開対象から除外"
$cleanupDefaultNowLine = git ls-remote origin $cleanupDefaultRef | Select-Object -First 1
$cleanupDefaultNow = ($cleanupDefaultNowLine -split "`t")[0]
if ($cleanupDefaultNow -ne $cleanupDefaultBefore) { throw "Default branch moved during preparation." }
git -C $cleanupDefaultWorktree push origin "HEAD:$cleanupDefaultRef"
```

これらのpushは既定ブランチと対象機能ブランチの最新版を先に非公開化する準備であり、過去履歴はまだ残る。既定ブランチへのpushはremote SHA不変を再確認したfast-forwardだけを許し、ブランチ保護または競合で失敗した場合は停止する。force-pushはTask 6まで行わない。

- [ ] **Step 9: 両方の準備push後の全remote refsをforce-push基準として保存する**

```powershell
git ls-remote origin | Set-Content -LiteralPath $cleanupRemoteRefsForceBaseline -Encoding utf8
```

以後、Task 6完了まで通常pushを含む全リモート更新を凍結する。

### Task 4: 使い捨てmirror cloneの履歴書き換え

**Files:**
- Create outside repository: fresh mirror clone
- Consume outside repository: `private-removal-paths.txt`、`private-replacements.txt`

- [ ] **Step 1: 新しい空ディレクトリへmirror cloneする**

```powershell
git clone --mirror https://github.com/baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark.git $cleanupMirrorPath
```

- [ ] **Step 2: mirror内で分析し、ref数・object数・対象外tree基準を記録する**

```powershell
git show-ref
git count-objects -vH
git filter-repo --analyze
```

各head/tagのtip SHAごとに`git ls-tree -r`を実行し、`allowed-changed-paths.txt`に含まれるパスを除いた`mode type object path`をリポジトリ外の`before-trees/<ref>.txt`へ保存する。ref名と旧tip SHAの対応も保存する。

許可変更パスは除外するだけで終わらせない。各head/tag tipの許可変更パスをリポジトリ外へ展開し、Task 1 Step 5で人が1回だけ承認した規則を定義順に適用する決定的な変換helperで、filter-repo実行前に期待ファイルを`expected-allowed/<ref>/`へ機械生成する。helperとmanifestはリポジトリ外に置き、`literal:`、`glob:`、`regex:`ごとの合成fixtureで期待どおりのbyte置換になることを検証する。人手確認の対象は規則、適用順、fixture、既定・対象機能ブランチの準備差分であり、各refの変換差分を個別に目視しない。

変更不要なblobは展開内容をそのまま期待ファイルとする。既定ブランチと対象機能ブランチのtipはTask 3で保存した各期待ファイルを正本とし、その他のbranch/tag tipは上記の機械変換結果を正本とする。`expected-allowed-blobs.tsv`へ`ref<TAB>path<TAB>sha256`を保存する。期待ファイルとSHA-256はfilter-repo実行後の出力から作らない。

- [ ] **Step 3: パス除去と内容置換を1回の書き換えで実行する**

```powershell
git filter-repo --sensitive-data-removal --invert-paths --paths-from-file $cleanupRemovalPaths --replace-text $cleanupReplacements
```

Task 1のコミットメッセージ検査が1件以上の場合だけ、同じコマンドへ`--replace-message $cleanupReplacements`を追加する。`--sensitive-data-removal`は実行直前にoriginから全refを再fetchするため、Task 3 Step 9以降のリモート凍結が前提である。実行直前にも`git ls-remote origin`をforce baselineと比較し、差分があればmirrorを破棄して最初からやり直す。

- [ ] **Step 4: `origin`を検証する**

```powershell
$cleanupExpectedRemote = "https://github.com/baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark.git"
if ((git remote get-url origin) -ne $cleanupExpectedRemote) {
    throw "Unexpected origin URL after history rewrite."
}
```

`--sensitive-data-removal`ではoriginが保持されることを前提とし、存在とURL一致だけを確認する。originが無い、またはURLが異なる場合は自動追加・変更せず停止する。

- [ ] **Step 5: `changed-refs`とfirst changed commitを保存する**

`git rev-parse --git-path filter-repo/changed-refs`で得たファイル、first changed commit、LFS orphan報告の有無をリポジトリ外の実行記録へコピーする。bare mirrorと通常cloneの`.git`配置差をハードコードしない。

### Task 5: force-push前の完全性検証

- [ ] **Step 1: 削除パスが全refから消えたことを確認する**

```powershell
git rev-list --objects --all | Select-String -Pattern 'private-structure|/private-detail-|private-file-management|private-page-generator'
```

Expected: 0件。

- [ ] **Step 2: 置換対象が全refから消えたことを確認する**

`private-verification-patterns.txt`はregex式ではなく、消えるべきリテラルを1行1件で持つ。バイナリ判定による検査漏れを避けるため`-a`で全blobをテキストとして扱い、tip treeだけでなく`git rev-list --all`が返す全到達可能commitを検査する。§3の再混入防止規則を意図的に保持するルート直下の`.gitignore` 1ファイルだけは、ユーザー承認済みの例外として全commitのリテラル検査から除外する。ネストした`.gitignore`は検査対象に残す。

```powershell
$cleanupAllCommits = git rev-list --all
if (-not $cleanupAllCommits) { throw "No commits were found for verification." }
foreach ($cleanupCommit in $cleanupAllCommits) {
    $cleanupHits = git grep -a -n -F -f $cleanupVerificationPatterns $cleanupCommit -- . ':(exclude).gitignore'
    if ($LASTEXITCODE -gt 1) { throw "git grep failed for commit $cleanupCommit" }
    if ($cleanupHits) { throw "Residual private text in commit $cleanupCommit" }
}
```

`***REMOVED***`の出現は許容する。ルート直下の`.gitignore`は各head/tag tipの期待SHA-256比較をTask 5 Step 3-2で必須とし、期待側に存在しない追加・欠落・変更を許可しない。Task 1でコミットメッセージに該当があった場合は、書き換え後の`git log --all --format='%B'`にも同じリテラル検査を行う。

- [ ] **Step 3: 対象外treeの一致を機械比較する**

commit mapの実パスを`git rev-parse --git-path filter-repo/commit-map`で取得し、旧tip SHAに対応する新tip SHAを求める。書き換え前に保存した`before-trees/<ref>.txt`と、新tipの`git ls-tree -r`から`allowed-changed-paths.txt`を除いた結果を比較する。差分が1件でもあればforce-pushしない。

- [ ] **Step 3-2: 許可変更パスを期待ファイルと比較する**

書き換え後の各head/tag tipを一時領域へ展開し、`allowed-changed-paths.txt`の各存続ファイルについて`Get-FileHash -Algorithm SHA256`を実行する。Task 4で確定した`expected-allowed-blobs.tsv`の`ref/path`ごとのSHA-256と1件ずつ比較し、default branchの`.gitignore`を含む全件が一致することを確認する。期待側に存在しない追加パス、期待側から欠落したパス、hash不一致が1件でもあればforce-pushしない。

- [ ] **Step 4: アプリの基準検査をclean work cloneで行う**

mirrorから別のclean work cloneを作り、次を実行する。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest -m integration
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m ruff format --check src tests
```

依存環境を作れない場合はコードtree一致検証を必須とし、テスト未実行理由を明記する。

- [ ] **Step 5: force-push直前にremote無変更を確認する**

```powershell
git ls-remote origin | Set-Content -LiteralPath $cleanupRemoteRefsNow -Encoding utf8
Compare-Object (Get-Content -LiteralPath $cleanupRemoteRefsForceBaseline) (Get-Content -LiteralPath $cleanupRemoteRefsNow)
```

Expected: 0件。差分があれば停止してmirror cloneからやり直す。

- [ ] **Step 6: 検証結果と影響ref数を提示し、ユーザーの最終承認を得る**

提示項目は変更branch数、tag数、PR影響数、fork数、first changed commit、検証結果、復旧方法とする。

### Task 6: 全refのforce-push

- [ ] **Step 1: 承認後にmirrorをforce-pushする**

```powershell
git push --force --mirror origin
```

GitHub公式手順どおりmirror pushを維持する。`refs/pull/*`はread-onlyのため失敗し得るので、該当refと影響PR数を記録してTask 8のSupport判断へ渡す。`refs/pull/*`以外で1件でも失敗した場合は停止し、追加pushを行わず部分成功状態を報告する。

- [ ] **Step 2: remote refsを再取得して予定値と比較する**

```powershell
git ls-remote --heads --tags origin
```

- [ ] **Step 3: GitHub上のdefault branchと主要branchを確認する**

`gh api`で各tip SHAを取得し、mirrorの対応SHAと一致することを確認する。

### Task 7: clean cloneへの移行と再混入防止

- [ ] **Step 1: 新しいディレクトリへfresh cloneする**

旧cloneは削除せず、ネットワーク操作を行わない隔離対象とする。

- [ ] **Step 2: 私用資料と未コミット変更を復元する**

私用資料はignore対象の元パスへ戻す。ユーザー変更はバックアップと新cloneの対象ファイルを比較し、競合がない場合だけ復元する。

- [ ] **Step 3: 再混入防止を確認する**

```powershell
git status --short
git check-ignore -v --no-index -- "private/local-structure.txt"
git log --all -- "private/local-structure.txt"
```

Expected: 私用資料はstatusへ現れず、履歴は0件。復元したユーザー変更だけが意図どおり表示される。

- [ ] **Step 4: 旧cloneを再利用しない旨を記録する**

共同利用者がいる場合は全員へ再cloneを依頼する。旧履歴をmergeまたはpushしない。

既定ブランチ準備用の`$cleanupDefaultWorktree`とローカルブランチ`codex/privacy-default-prep`について、絶対パス、旧commit SHA、作業ツリー状態を実行記録へ保存する。このworktreeにはignore済みのローカル資料が残り得るため自動削除せず、旧cloneと同じネットワーク操作禁止の隔離対象として扱う。将来削除する場合は、バックアップと期待ファイルのSHA-256一致を再確認し、削除対象の絶対パスを提示して別途ユーザーの明示承認を得る。

### Task 8: GitHub残存参照の確認

- [ ] **Step 1: forkとPR参照を再確認する**

forkが存在する場合は所有者へ削除または履歴書き換えを依頼する。影響PRが存在する場合はPR番号を記録する。

- [ ] **Step 2: GitHub Support対象を判断する**

cached view、PR ref、LFS orphanが残り、GitHubが機密情報と判断し得る場合はSupportへowner/repository、影響PR数、first changed commit、LFS報告を添えて依頼する。Supportが非機密データとして対応しない場合は、その結果を残存リスクとして記録する。

- [ ] **Step 3: 最終結果を記録する**

削除対象0件、remote SHA一致、fork/PR/cache状態、clean clone移行、再混入防止、ルート直下の`.gitignore`へ意図的に残す公開パス・名称、未解決リスクを事実ベースで記録する。

## 6. 受け入れ条件

1. リポジトリはPublicのまま維持される。
2. §2.1の対象パスが全remote heads/tagsから消える。
3. 存続ファイルの実フォルダ名・命名例・階層断片・詳細説明が全remote heads/tagsから消える。ただしルート直下の`.gitignore`は、§3の再混入防止規則として削除対象パスを意図的に保持する。この1ファイルはリテラル検査の対象外とし、各head/tag tipの期待SHA-256一致をもって合格とする。
4. 一般的な要件、実装計画、プログラム、テストは`allowed-changed-paths.txt`に列挙した一般化・合成名化以外の差分なく残り、既定・対象機能ブランチ最新版とその他のbranch/tag tipの存続ファイルは各期待SHA-256と一致する。
5. ローカル私用資料は削除されず、未追跡・ignore状態で利用できる。
6. 現在のユーザー未コミット変更が保持される。
7. force-push前後のremote refsと書き換え予定refsが一致する。
8. clean cloneで削除対象の`git log --all -- <path>`が0件になる。
9. 古いcloneからの再混入防止手順が記録される。
10. fork、PR ref、cache、第三者cloneの残存可能性が確認・記録される。

## 7. ロールバックと限界

- force-push前は使い捨てmirrorを破棄すればリモートへ影響しない。
- force-push後の復旧には、リポジトリ外へ保存した開始時ref一覧と隔離した旧cloneを使用する。復旧自体も全refのforce-pushになるため、別途明示承認を必要とする。
- 第三者clone内のデータは削除できない。
- forkの履歴はfork所有者の対応が必要である。
- GitHubのPR参照とcached viewはSupport対応が必要な場合があり、非機密データでは対応されない可能性がある。
- 履歴書き換え後はコミットSHA、署名、PR差分、SHA依存のリンクや自動化が変わる。
- ルート直下の`.gitignore`は再混入防止のため削除対象のリポジトリ内パスとファイル名を保持し続ける。これらは公開され続けるため、実際の業務フォルダ構造・詳細内容とは別の残存情報としてTask 8 Step 3の最終記録へ含める。

## 8. Claude Codeレビュー

### 8.1 第3回レビュー指摘の反映

| 指摘 | 判定・反映先 |
|---|---|
| NC-1 既定ブランチ最新版が非公開化されない | Blockerとして採用。ただし対象機能ブランチ全体は既定ブランチへ進めず、`origin/<default>`基点の分離worktreeでプライバシー準備差分だけを作り、remote SHA不変確認後に通常pushする。Global Constraints、§2.2.1、Task 3へ反映 |
| NC-3 tip treeしか残存語句を検査しない | 採用。Task 5 Step 2を`git rev-list --all`の全到達可能commit検査へ変更 |
| NC-4 全refの期待差分を人が確認する運用負荷 | 採用。人は規則・適用順・合成fixture・2つの最新版準備差分を確認し、その他refの期待内容はfilter-repo実行前に機械生成する。Task 4 Step 2へ反映 |
| NC-5 `HEAD`行がref集計へ混入する | 採用。Task 1 Step 2で`refs/*`だけに絞ってから集計する |

### 8.2 第4回レビュー指摘の反映

| 指摘 | 判定・反映先 |
|---|---|
| ND-1 `.gitignore`が検証リテラルへ必ず一致する | Highとして採用。ユーザー判断によりルート直下の`.gitignore` 1ファイルを最新版・全commitの固定文字列検査から除外し、各head/tag tipの期待SHA-256比較で担保する。§2.2、Task 3 Step 4、Task 5 Step 2・3-2へ反映 |
| ND-3 一時worktreeとブランチの後処理が未定義 | 採用。ignore済みローカル資料を保護するため自動削除せず、絶対パス・旧SHA・状態を記録して旧cloneとともに隔離する。削除はSHA再確認と別途承認を必須とする。Task 7 Step 4へ反映 |

### 8.3 第5回レビュー指摘の反映

| 指摘 | 判定・反映先 |
|---|---|
| NE-1 受け入れ条件と`.gitignore`例外が不一致 | Mediumとして採用。受け入れ条件3へ例外と合格条件を追加し、§7とTask 8 Step 3へ公開状態で残るパス・名称を明記 |
| NE-2 既定ブランチworktreeの検査対象が曖昧 | 採用。Task 3 Step 6へ`git -C $cleanupDefaultWorktree grep ...`の完全なコマンドと停止条件を追加 |
| NE-3 `.gitignore`除外範囲の表現が広すぎる | 採用。ルート直下の1ファイルだけを除外し、ネストした`.gitignore`は検査対象に残す実挙動へ§2.2、Task 3、Task 5の表現を統一 |

### 8.4 第6回レビュー結果

| 項目 | 判定・反映先 |
|---|---|
| 総評 | 実行着手可。NE-1〜NE-3は解消済みで、新規Blocker・High・Mediumなし |
| NF-1 Task 3 Step 4の除外対象表現 | Lowとして採用。「ルート直下の`.gitignore`を除く」へ修正し、コード例と後続説明へ統一 |
| NF-2 §8.2の旧表現 | Lowとして採用。ND-1の記録を「ルート直下の`.gitignore` 1ファイル」へ修正 |
| 実行環境 | PowerShell 7.5.5とGitHub CLI 2.96.0を確認済み。`gh auth status`の再認証と`git-filter-repo>=2.47`の導入が未完了 |

## 9. 実行開始条件

- Claude CodeレビューのBlockerとHighを解消している。
- `gh auth status`が成功する。
- `git-filter-repo>=2.47`を利用できる。
- private path manifestとreplacement manifestをリポジトリ外で確定している。
- バックアップのSHA-256一致を確認している。
- force-push直前に改めてユーザーの明示承認を得る手順が合意されている。

## 10. 参考資料

- GitHub Docs: [Removing sensitive data from a repository](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- git-filter-repo: [Official documentation](https://github.com/newren/git-filter-repo/blob/master/Documentation/git-filter-repo.txt)
