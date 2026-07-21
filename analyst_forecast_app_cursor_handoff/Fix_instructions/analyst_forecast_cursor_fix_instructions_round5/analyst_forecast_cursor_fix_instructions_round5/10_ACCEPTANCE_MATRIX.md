# 第5次修正 — 受入マトリクス

各項目へ、test名、DB状態・constraint、生成成果物、CLI出力のいずれか具体的な証拠を対応付けてください。合計test数やコード行の存在だけを証拠にしないでください。

| ID | 受入事項 | 必須証拠 |
|---|---|---|
| R5-001 | issuance/component/evidence/evaluation入り0007 DBをheadへupgradeできる | data-bearing historical fixture test |
| R5-002 | migration前後で全主要row count・ID・主要値を保持する | before/after query・hash assertion |
| R5-003 | migration後のintegrity checkとforeign key checkがpassする | PRAGMA実行結果 |
| R5-004 | component/evidence等のFKが再作成後の正しい親tableを参照する | foreign_key_list assertion |
| R5-005 | 既適用0009 DBとhead DBのupgrade/re-runが成立する | 0009→head・head→head test |
| R5-006 | migration途中失敗時に元revision・Schema・データを復元する | forced failure rollback/restore test |
| R5-007 | legacy conflictを無根拠にactiveへ昇格させない | migration lifecycle query |
| R5-008 | empty/0001/0005/0007経路とAlembic metadata差分0が成立する | migration matrix＋alembic check |
| R5-009 | 複数forecast訂正で旧A/Bが新A/Bへ別lineageとして対応する | 2-forecast vertical DB query |
| R5-010 | 各oldのsupersedes/superseded_byが対応newを一対一で指す | pairwise lineage assertion |
| R5-011 | 同一lineageのactiveはDB制約上最大1件である | partial unique index＋negative insert |
| R5-012 | P09世代切替失敗時にold/new/reviewが原子的にrollbackする | forced transaction failure |
| R5-013 | corrected payloadの配列順をidentityに使用しない | reorder regression |
| R5-014 | correctionのaddが新lineage generation 1を作る | add operation query |
| R5-015 | correctionのremoveがactiveを消し履歴を保持する | remove operation query/report |
| R5-016 | 重複・曖昧・未申告のforecast対応を拒否する | operation negative matrix |
| R5-017 | 同一P09 correct再取込みが全件数・active状態を変えない | ALREADY_IMPORTED＋count assertion |
| R5-018 | 通常結果・forecast countはactive世代だけを含む | 04_results/summary numeric test |
| R5-019 | superseded componentのP11を共通reason codeで拒否する | P11 negative ingest |
| R5-020 | 拒否P11がcandidate/mapping等の副作用を作らない | before/after DB counts |
| R5-021 | P11後にsupersedeされたcomponentのP12を拒否する | stale P12 vertical test |
| R5-022 | P12後にsupersedeされたcomponentのP13を拒否する | stale P13 vertical test |
| R5-023 | inactive componentの市場評価を拒否する | evaluation negative test |
| R5-024 | P11/P12/P13/評価が同じactive policyを使用する | shared policy test/structural evidence |
| R5-025 | NEXT_ACTIONSがinactive componentを案内しない | generated file assertion |
| R5-026 | 旧component/mapping/evaluationを監査履歴として保持する | history query |
| R5-027 | P09 cutoffが訂正後made_atより後ならsource境界内でも拒否する | unknown→known future cutoff test |
| R5-028 | P09 cutoffと訂正後made_atの同値を許可し+1µsを拒否する | exact-boundary tests |
| R5-029 | 複数訂正forecastの最も早いmade_atを境界にする | multi-forecast time test |
| R5-030 | corrected P08自身のcutoffも各corrected made_at以前にする | corrected payload cutoff test |
| R5-031 | source allowed boundaryとforecast made_atの早い方を越えない | dual-boundary test |
| R5-032 | 正当なunknown→known訂正だけを一度正式化する | positive correction＋count |
| R5-033 | 時点検証を通常P08/P09 correctで共有し市場結果を入力しない | shared validator＋prompt input audit |
| R5-034 | 単一symbol複数日予想を1取引日から評価しない | insufficient trading dates test |
| R5-035 | coverage不足でhit/miss/price/return/MFE/MAEをnullにする | evaluation field assertions |
| R5-036 | basket共通日1日でcommon_date_count=1を保存する | unevaluable basket DB query |
| R5-037 | unevaluable時にもrequested/effective period・銘柄別件数・hash・reasonを監査できる | coverage_audit assertion |
| R5-038 | 成功時の単一/basketにも完全なcoverage auditを保存する | successful evaluation audit |
| R5-039 | 単一symbol 2取引日で正しいreturn/MFE/MAEを計算する | numeric single-symbol test |
| R5-040 | basket共通2取引日で正しいweighted return/MFE/MAEを計算する | numeric basket test |
| R5-041 | 同日予想を暗黙の複数日methodで評価しない | method-version/unsupported test |
| R5-042 | provider間coverage一致とbasket後の単一symbol cache非汚染を維持する | CSV/mock・AAA regression |
| R5-043 | 新規P09 rejectでdispositionを必須にする | fixed/Pydantic schema negative test |
| R5-044 | retryable/terminalの両方で非空reasonを必須にする | validation matrix |
| R5-045 | reject以外へのreject fieldを拒否しlegacy省略を黙ってretryable化しない | conditional/legacy tests |
| R5-046 | P09固定Schema・Pydantic・repo/packaged promptが同じ契約を表す | schema/prompt sync tests |
| R5-047 | retryable/terminal/unresolvedが有限で複数sourceを妨げない | workflow vertical tests |
| R5-048 | Ruff format/check、mypy、pytest、diff checkがすべてpassする | 全command log |
| R5-049 | packaged migration・Schema・prompt・docsを含むclean wheel縦断がpassする | wheel install/vertical log |
| R5-050 | internal wheel/migration/Schema試験をskip・xfailで隠さない | pytest collection/result audit |
| R5-051 | docs、IMPLEMENTATION_STATUS、FINAL_REVIEWが実装・未実装・判定を正確に記載する | docs内容と実行証拠の照合 |
| R5-052 | CHAT_HISTORY、秘密情報、実DB/raw/cache、build生成物を正しく保護する | hash・git status・secret scan |

## READY判定条件

`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`は、R5-001～R5-052がすべてpassの場合だけ使用してください。

次はREADY判定の代替になりません。

- analyst/runしか入っていないhistorical DBでmigrationを試す。
- `PRAGMA foreign_keys=OFF`のままupgrade成功とする。
- 単一forecastのP09 correctだけで複数forecast対応をpassにする。
- active最大1件をapplicationの`if`だけに任せ、DB制約を持たない。
- evaluate_componentだけを守り、P11/P12/P13を未検査にする。
- corrected P08 cutoffだけを確認し、P09自身のcutoffを確認しない。
- basketだけ2日条件にし、単一銘柄1日をhit/missにする。
- unevaluableにしたことだけで、coverage auditがnullでもpassにする。
- bool省略をretryableと見なし続ける。
- pytest合計件数だけでformat、diff、wheel、migrationをpass扱いする。
- internal wheel testをbuild不能時にskip/xfailする。

## FULL MVPとの区別

Round5を完了しても、総合点、複数アナリスト統合、現在上昇候補、時期・程度・早期実現の完全採点、期間不明の1/3/6/12か月観測、PNG等が未実装なら`FULL_MVP_READY`ではありません。

