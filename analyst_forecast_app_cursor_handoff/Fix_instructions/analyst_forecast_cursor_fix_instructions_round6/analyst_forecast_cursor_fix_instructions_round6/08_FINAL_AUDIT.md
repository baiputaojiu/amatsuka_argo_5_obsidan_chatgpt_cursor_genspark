# 08 — Round6最終監査

## 目的

個別unit testの合計ではなく、不正P09、正式訂正、DB世代、coverage、Schema、migration、wheel、結果出力をつないで、Round6の修正とRound5までの改善を独立に再確認します。

## Cursorへ渡す依頼文

```text
00_START_HERE.mdから07までを、自己申告ではなく実行証拠で最終監査してください。commit・pushはしないでください。

migration historical fixture以外は、目的状態を作るためにDB lifecycle列を直接書き換えず、public application API、CLIまたは正式AI ingest経路を使ってください。

A. 多対一operation拒否
- accepted P08にA/Bを作る。
- corrected XへA→X、B→Xのupdateを申告する。
- 正式P09 ingestがduplicate/ambiguous corrected refの安定codeで拒否する。
- issuance/component/evidence/operation/resolution/active ID集合がbefore/after不変。

B. 未申告operation拒否
- old A/B、new A2/B2/C、operations A→A2・add Cだけを投入する。
- 未申告old Bとnew B2を両方検出して拒否する。
- oldだけ欠落、newだけ欠落、unknown ref、ref重複もparameterized caseで拒否する。
- activeがA2/old B/B2/Cへ増殖しない。

C. valid operation縦断
- old A/Bをnew A2/B2へ、corrected payloadを逆順にしてupdateする。
- old A/B→new A2/CをA update、B remove、C addで処理する。
- lineage root、generation、supersedes/superseded_by、component/evidence所属、active件数、operation監査行を確認する。
- 同一P09再取込みはALREADY_IMPORTEDで全件数不変。
- 適用途中の強制例外はold active/review状態へrollbackする。

D. active世代と下流非回帰
- correction後の旧componentをP11へ渡して拒否する。
- P11/P12後にsupersedeしたstale P12/P13、旧component評価を拒否する。
- 04_results、summary、forecast count、NEXT_ACTIONSにactive A2/Cだけが出る。
- DBの同一lineage active一意制約をnegative insertで再確認する。

E. coverage 4形状
- single 1日: unevaluable、全評価値null、完全audit。
- single 2日: 正しいreturn/MFE/MAE、完全audit。
- basket共通1日: unevaluable、common_date_count=1、全instrument audit。
- basket共通2日: 正しいweighted数値、weights/mapping hash/intersection/audit。
- input first/last、in-range、unique、duplicate、invalid、dropped、series hashを全形状で確認する。

F. coverage不正入力
- out-of-range、duplicate、NaN、非正価格を含むseriesを使う。
- 件数とreasonをauditへ残し、黙って2日を選んでhit/missにしない。
- input順変更でhash不変、値変更でhash変化。
- basket後の単一symbol cache非汚染。

G. P09契約matrix
- fixed Draft 2020-12 SchemaとPydantic runtimeへ04の12分類を通す。
- 2.1 reject区分/理由省略、reject以外のreject field、2.0 field混在を拒否する。
- valid 2.0 boolをvalidation後だけ有限なdispositionへ変換する。
- repo/packaged Schema/prompt例の一致と、全prompt例validを確認する。

H. P09 workflow
- valid retryable、terminal、unresolvedを別source fixtureで処理する。
- retryableは有限な再抽出、terminal/unresolvedは有限終端へ進む。
- invalid legacy rejectは状態を変えない。
- 1 sourceの終端が他sourceのNEXT_ACTIONを妨げない。

I. data-bearing migration
- 0007へ2 issuance、2 components、evidence、mapping、evaluation、snapshot等をseedする。
- headへ上げ、全PK集合、全legacy projection hash、sentinel、FK、integrity、revisionを確認する。
- 0009→head、head→head、empty/0001/0005経路もpass。

J. migration失敗復元
- backup後に実ALTER TABLE、sentinel UPDATE、commitしてから強制例外。
- callerはfailureを受け取り、元revision/Schema/PK/hash/sentinelへ復元され、marker columnがない。

K. clean wheel
- repo外cwd、PYTHONPATHなしの新venvへwheelのみinstallし、import pathがsite-packagesであることを確認する。
- help/init/run/source/P05/P07/P08/P09、A/B valid correction、多対一negative、旧P11拒否、single 1/2日coverage、data-bearing migration、results/NEXT_ACTIONSを実行する。
- installed Schema/prompt/docs/migrationsを確認する。
- testにskip/xfail/importorskipがない。

L. Round3～5非回帰
- 別URL source occurrenceへの再利用、unknown日時の非正式化、P09 corrected cutoff、basket mixed currency、active component guard、migration active indexを再確認する。
- 市場結果がAI意味判断promptへ入らないこと、raw/historyが保持されることを確認する。

M. 品質・Git
- 06の全品質commandを実行しreturn code 0。
- base 88864c2からのgit diff --checkがpass。
- CHAT_HISTORY hash維持、secret/DB/raw/cache/backup/build生成物を追跡しない。
- required internal testにskip/xfail/importorskipがない。

各シナリオについて、入力、command/test node、期待、実結果、DB query、error path/code、生成file、判定をdocs/06_実装/FINAL_REVIEW_ROUND6.mdへ記載してください。

09_ACCEPTANCE_MATRIXの全項目を個別証拠へ対応付けてください。「実装済み」「pytest全体pass」「以前のRoundでpass」だけでは今回のPASS証拠にしないでください。
```

## 判定規則

- A～MまたはR6-001～R6-050に内部fail、未実装、NOT_RUN、skip、xfail、証拠不足があれば`NOT_READY`。
- live networkだけが外部制約で、同じ内部経路をlocal fixture/CSV/mockでpassしている場合だけ`external_blocked`として分離できる。今回の受入項目自体をexternal_blockedにしない。
- 全項目PASS時だけ`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`。
- `FULL_MVP_READY`は使用しない。

