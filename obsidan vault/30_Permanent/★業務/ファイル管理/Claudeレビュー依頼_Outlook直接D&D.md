# Claude Code第4回レビュー依頼：Outlook直接D&DとGitHub履歴削除の実装計画

作成日：2026-08-04
依頼先：Claude Code
レビュー対象：第3回レビュー反映後の計画書2件

---

## 1. Claude Codeへ渡す依頼文

OneDrive年度別業務フォルダ向け「保存先レコメンダー」について、次の実装前計画2件を第4回レビューしてください。

1. クラシックOutlookからメール・添付ファイルを直接D&Dし、送信者表示名を推薦へ利用する計画
2. PublicのGitHubリポジトリから私用フォルダ資料を全履歴ごと除去する計画

今回は**計画レビューだけ**が目的です。コード、文書、Git、GitHub、Outlook、OneDriveを変更しないでください。依存導入、テストデータ作成、commit、push、force-push、branch作成、worktree作成も禁止します。リポジトリ、依存ソース、Git履歴はread-onlyコマンドで確認して構いません。

### 対象リポジトリと差分

| 項目 | 値 |
|---|---|
| リポジトリ | `https://github.com/baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark` |
| ブランチ | `codex/outlook-direct-dnd-plan` |
| 今回の比較元 | `2d51406` |
| レビュー対象 | `origin/codex/outlook-direct-dnd-plan`の最新tip |
| 対象プロジェクト | `onedrive_destination_recommender/` |
| Outlook計画 | `obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_Outlook直接D&D.md` |
| 履歴削除計画 | `obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_GitHub公開情報の履歴削除.md` |

```bash
git fetch origin
git rev-parse origin/codex/outlook-direct-dnd-plan
git diff --stat 2d51406..origin/codex/outlook-direct-dnd-plan
git diff 2d51406..origin/codex/outlook-direct-dnd-plan -- \
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

### 2.1 Outlook入力と送信者推薦

- 対象はクラシックOutlook for Windows。新しいOutlookは対象外。
- メール1通と、その内部添付0件以上を1案件として扱う。
- Outlook添付だけの単一・複数D&Dも受け付ける。
- 今回はメール1通ずつ。将来は複数メールへ拡張するが、複数案件UIやキューは今回作らない。
- Outlookの送信者表示名を今回から推薦へ利用する。
- 同じ表示名の別人をメールアドレスで区別しない。
- Unicode空白差を同一送信者として扱う。
- 送信者を取得できなくても処理を継続する。
- 送信者表示名と送信者キーをAudit、設定、カタログ、ログへ保存しない。
- **一般の主検索語が0件一致でも、送信者だけが一致すれば主照合1件の候補として表示する。**
- 「送信者だけでは候補を出さない」への変更は行わない。候補数が増えることは受容し、既存の候補件数上限と祖先・兄弟折り畳みを適用する。

### 2.2 GitHub公開履歴

- リポジトリはPublicのまま維持する。
- 実際のフォルダ名、命名例、階層、詳細資料は最新版と全branch/tagの履歴から除去する。
- 年度単位で探索する一般機能は公開対象として残す。
- 設定例とテストデータに実名がある場合は、動作する合成名へ置換する。
- ローカル資料と現在のユーザー未コミット変更を保持する。
- 履歴書き換えはレビュー承認後に行い、既定ブランチへの通常push前とforce-push直前に、それぞれユーザーの明示承認を得る。

## 3. 第3回レビューからの修正

### 3.1 Outlook計画

| ID | 判定・対応 |
|---|---|
| NC-2 | 採用。兄弟折り畳みで生成する親Candidateへ`sender_matched`を継承する。ただし`first.sender_matched`ではなく、兄弟全体の`any(sibling.sender_matched for sibling in siblings)`を使う |
| NC-6 | 不採用。`Candidate.sender_matched`は既定値なしの必須フィールドを維持する。Candidateの構築点は`_make_candidate()`だけであり、必須にする方が新しい構築経路の伝播漏れを検出しやすい |
| Cursor運用 | Task 1のクラシックOutlook実機Go/No-GoをCoordinator ledgerへ`pending/passed/failed`、実施者、日時、観測結果とともに記録し、`passed`までTask Reviewerが完了扱いにしないゲートを追加 |

NC-2では、同じ主一致数の兄弟に送信者一致候補と一般主検索語だけの候補が混在し得るため、入力順に依存する`first.sender_matched`より論理和を採用しました。次を重点確認してください。

1. `_make_candidate(..., inherited_sender_matched: bool | None = None)`と兄弟全体の論理和で、折り畳み後も固定表示`送信者一致`の理由を失わないか。
2. `Candidate.sender_matched`を必須のままにする判断に、現行のCandidate構築経路を踏まえた問題がないか。
3. 既存の`_fold_descendants()`でも送信者一致理由の伝播が必要か。それとも今回の兄弟折り畳み対応だけで要件を満たすか。
4. 実機Go/No-Goのledger記録がCursorのTask完了ゲートとして十分か。
5. 第2回までのNA-1〜NA-6、初回B-1〜L-3への対応が今回の修正で後退していないか。

### 3.2 GitHub履歴削除計画

| ID | 判定・対応 |
|---|---|
| NC-1 | Blockerとして採用。ただし第3回レビュー時点で機能ブランチは既定ブランチより19 commit先行していたため、ブランチ全体をfast-forwardしない。`origin/<default>`基点の分離worktreeでプライバシー準備差分だけを作り、全検査とremote SHA不変確認後に既定ブランチへ通常pushする |
| NC-3 | 採用。tip treeの検査をやめ、`git rev-list --all`が返す全到達可能commitへ`git grep -a -F -f`を実行する |
| NC-4 | 採用。人は置換規則、適用順、合成fixture、既定・対象機能ブランチの準備差分を1回確認する。その他branch/tag tipの期待ファイルは、承認済み規則からfilter-repo実行前に機械生成する |
| NC-5 | 採用。`git ls-remote`のref種別集計前に`Where-Object { $_ -like 'refs/*' }`を入れ、`HEAD`行を除外する |

NC-1の修正は、第3回レビュー時点の機能計画など19 commitを`main`へ公開することを避けつつ、既定ブランチ最新版にも合成名化・一般化・ignore・追跡解除を先に適用するためのものです。既定ブランチへのpushはプライバシー準備差分だけのfast-forwardとし、ブランチ保護、競合、remote更新があれば停止します。

次を重点確認してください。

1. `origin/<default>`基点の専用worktreeで同じプライバシー規則を適用し、準備差分だけを`HEAD:refs/heads/<default>`へ通常pushする手順がNC-1を安全に解消するか。
2. 既定ブランチと対象機能ブランチの最新版だけを実行可能な合成名へ準備し、その他branch/tag tipは承認済み規則による機械期待値と比較する境界が妥当か。
3. 全到達可能commitを1件ずつ`git grep`することで、tipから消えた古いblobの残存を検出できるか。
4. filter-repo実行前に独立した変換helperで期待内容を作り、合成fixtureとSHA-256で検証する方式が実装可能か。置換manifestと同じ誤りを見逃す循環検証になっていないか。
5. `refs/*`フィルタ後の集計で、remote ref種別の棚卸しに不足がないか。
6. 第2回までのNB-1〜NB-8への対応が今回の修正で後退していないか。

## 4. Cursorでの実装運用

Outlook機能の実装はClaude Codeではなく、Windows上のCursor Composerを使う予定です。ユーザーの通常作業ツリーから分離した実装用worktreeをCursorで開き、Agents Window / Agent Tabsで複数エージェントを横並びまたはグリッド表示します。現在どの役割が動いているかをユーザーが眺められる運用であり、全エージェントが同時に動く必要はありません。

計画は`superpowers:subagent-driven-development`に基づき、次の役割を分離します。

1. Coordinator：計画と進捗ledgerを管理するread-only担当
2. TaskごとのImplementer：TDDで実装・テスト・commitする唯一の書き込み担当
3. Task Reviewer：commit差分の仕様適合と品質をread-onlyで判定
4. Final Reviewer：全Task後のブランチ全体レビュー

全タブは同じ実装用worktreeのcommit済み状態を参照し、複数のImplementerを同時に走らせません。重要指摘は元Implementerへ戻し、修正とscoped re-reviewを最大5周行います。クラシックOutlook実機確認はローカルWindowsで実施し、LinuxのBackground Agentへ委ねません。

次を確認してください。

- 各Taskが別Implementerへ渡せるだけのInterfaceと入力情報を持つか。
- Task単位commit後のread-onlyレビューで、次Taskへ進むゲートを作れるか。
- 共有作業ツリーで書き込み担当を1つに限定すれば、複数Agent Tabsを開いたままでも競合を避けられるか。
- 進捗ledger、Task review、fix loop、final review、実機Go/No-Goが計画に十分明記されているか。

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

### 6.3 第3回指摘対応表

NC-1〜NC-6とCursor実機ゲートについて、`解消／要件として不採用／一部解消／未解消／別問題を導入`を判定してください。不採用のNC-6は、理由の妥当性も判定してください。

### 6.4 要件・Task対応表

Outlookの受け入れ条件16件と履歴削除の受け入れ条件10件について、Task、自動検証、手動検証、判定を示してください。

### 6.5 開始前チェックリスト

未解決事項だけを計画ごとに優先順で最大10件示してください。なければ「なし」としてください。

## 7. 最後に

レビュー結果だけを返してください。コード、計画書、Git、GitHub、Outlook、OneDriveを変更しないでください。私用資料の内容・実名・件数を回答へ転載しないでください。

---

## Claude Codeへの渡し方

1. 本文書の「Claude Codeへ渡す依頼文」をClaude Codeへ渡す。
2. 第3回レビュー結果の原文も同時に渡す。
3. ブランチ`codex/outlook-direct-dnd-plan`の最新tipをread-onlyで参照させる。
4. 実業務メール、添付、私用フォルダ資料は渡さない。
5. 回答受領後、Blocker、High、Medium、Lowの順で採否を判断する。
6. レビュー承認前に実装または履歴書き換えへ進まない。
