# 08 — Round4最終監査

## 目的

個別unit testの合計ではなく、CLI/application API、AI ingest、SQLite、workflow、結果、CSV評価、wheelをつないだ縦断シナリオで、Round4の修正とRound3合格項目の非回帰を確認します。

## Cursorへ渡す依頼文

```text
00_START_HERE.mdから07までを、自己申告ではなく実行証拠で最終監査してください。commit・pushはしないでください。

次のシナリオA～Rを、可能な限りpublic application APIまたはCLIから実行してください。目的状態を作るためにtestからDB列を直接書き換えないでください。

A. accepted P08のaccept再レビュー
P05/P07 → accepted P08 → active issuance 1件 → P09 accept。例外なし、issuance/component/evidence増加なし、active 1件、P11へ進む。

B. accepted P08のcorrect再レビュー
active issuance 1件 → valid P09 correct。旧世代をsuperseded、新世代だけactiveにし、04_resultsとNEXT_ACTIONSに旧componentを出さない。

C. correct冪等性
Bと同じP09を再取込みし、ALREADY_IMPORTED、DB件数・active状態・成果物が変化しない。

D. accepted P08のreject/unresolved
reject retryableは旧activeを除外してRUN_P08、unresolvedはterminal/excluded。同じP09を無限再案内しない。

E. needs_review P08の4decision
accept、correct、reject retryable、unresolvedを別fixtureで実行し、それぞれ有限の期待状態へ進む。

F. 別URL再利用
同一raw、同一analyst/medium、別URLのsource 2へP05/P07を再利用し、source 2のP08をacceptedにする。originまでlineage queryで追跡する。

G. 不正再利用
raw、analyst、medium、model/schema、cutoffのいずれかが違うcaseを拒否し、別source artifact IDの直指定も拒否する。

H. 複数source
source A unresolved terminal、source B P08 pendingで、Bを先に案内する。全source terminal後だけP11またはCOMPLETE_NO_ACTIVE_FORECASTへ進む。

I. unknown time
made_at_source=unknown、made_at=nullのP08を保持するが正式化しない。任意datetime入りunknownを拒否し、today/retrieved_atへ補完しない。

J. unknown timeの正当な訂正
P09 correctでtime evidence付きmade_atを確定し、P08通常検証を通して一度だけ正式化する。

K. future cutoff
P08 cutoff > made_at、P09 cutoff > reviewed forecast made_at、corrected P08 cutoff > corrected made_atをそれぞれ拒否する。

L. basket 1 common date
複数日・複数symbol・共通日1日をinsufficient_common_datesでunevaluableにし、hit/miss/return/MFE/MAEを確定しない。

M. basket 2 common dates
AAA +20%、BBB -20%の50/50 basketを0%として正しく評価し、coverage auditを保存する。その後AAA単独を+20%で評価する。

N. active結果
訂正前後、reject、unresolved、legacy conflictを含むDBから04_resultsを再生成し、active世代だけが通常forecast countと最新hit/missに入る。

O. Round3非回帰
司会者/legacy_unknown非正式化、不正P09 offset拒否、2 source P08優先、mixed currency unevaluable、basket cache分離、Vault完全版docsを再確認する。

P. Migration
empty/0001/0005/0007/Round3 fixtureからhead、alembic check差分0、foreign_key_check、件数保持を確認する。

Q. Wheel
clean venvのwheelからhelp、init、model設定、run create、source import、代表P08/P09、完全版docsとNEXT_ACTIONS生成まで実行する。

R. Git・秘密情報
CHAT_HISTORY.pdfを維持し、DB、raw、cache、secret、実Vault path、build生成物を追跡しない。

各シナリオについて、入力、command/test名、期待結果、実結果、DB query値、生成file、判定をdocs/06_実装/FINAL_REVIEW_ROUND4.mdへ記載してください。

09_ACCEPTANCE_MATRIXの全項目をtest名、DB constraint/列、生成成果物、CLI出力へ対応付けてください。「コードがある」「pytest全体がpass」だけで個別項目をpassにしないでください。negative caseは実際の拒否理由codeを示してください。
```

## 判定規則

- A～Rまたは受入マトリクスに内部fail、未実装、未検証があれば`NOT_READY`。
- networkだけが外部制約で、同じ内部処理のmock/CSVがpassしている場合だけ`external_blocked`を許容。
- 全Round4項目pass時だけ`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`。
- 総合点、PNG、1/3/6/12か月観測等が未実装なら`FULL_MVP_READY`を使用しない。

## 最終報告形式

```text
判定:
対象base commit:
変更ファイル数:
pytest:
ruff format/check:
mypy:
docs sync:
Alembic:
wheel clean venv:
scenarios A-R:
acceptance R4-001～R4-048:
external_blocked:
意図的未実装:
次の一作業:
commit / push: 未実施
```

