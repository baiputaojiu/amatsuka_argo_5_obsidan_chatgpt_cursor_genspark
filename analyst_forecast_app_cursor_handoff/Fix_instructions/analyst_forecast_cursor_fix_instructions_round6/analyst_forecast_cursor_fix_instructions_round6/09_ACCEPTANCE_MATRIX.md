# 第6次修正 — 受入マトリクス

各項目へ、固有のtest node、DB query/constraint、validator payload/error path、CLI output、生成成果物のいずれか具体的な証拠を対応付けてください。合計test数やコード行の存在だけを証拠にしないでください。

| ID | 受入事項 | 必須証拠 |
|---|---|---|
| R6-001 | Round5独立レビューの多対一、未申告、coverage、Schema、wheel、migration証拠不足を同じ入力で再現・fixture化する | ROUND6_REPRODUCTION＋正式test node |
| R6-002 | reviewed P08とcorrected P08の各forecast_refがpayload内でuniqueである | old/new duplicate ref negative tests |
| R6-003 | reviewed集合`O`がupdate-oldとremove-oldの重複なし和に完全一致する | set equality assertion |
| R6-004 | corrected集合`N`がupdate-newとadd-newの重複なし和に完全一致する | set equality assertion |
| R6-005 | updateが旧1件↔新1件で、多対一・一対多を拒否する | A/B→X、A→X/Y formal ingest tests |
| R6-006 | update/add/removeのref nullability・禁止field・非空reasonを強制する | action contract matrix |
| R6-007 | 存在しないreviewed/corrected refを安定reason codeで拒否する | unknown old/new tests |
| R6-008 | operationで説明されない旧forecastを拒否する | missing-old test |
| R6-009 | operationで説明されない新forecastを拒否する | missing-new test |
| R6-010 | operation間で同一old/new refを再使用できない | cross-action duplicate tests |
| R6-011 | invalid operationを世代変更前に拒否し、issuance/component/evidence/operation/resolutionへ副作用を作らない | before/after ID・count・active assertions |
| R6-012 | valid A/B updateが配列順に依存せず異なるlineageとpairwise linkを維持する | reorder vertical DB query |
| R6-013 | valid update/remove/addで期待するlineageだけがactiveになり履歴が残る | mixed operation vertical |
| R6-014 | new component/evidenceが対応new issuanceへ所属する | FK ownership assertions |
| R6-015 | operation適用途中の例外でold active、review、全子行がrollbackする | forced transaction failure |
| R6-016 | 同一P09再取込みがALREADY_IMPORTEDで全件数・active状態不変となる | idempotency before/after |
| R6-017 | DB active一意制約、inactive P11/P12/P13/評価拒否、active-only結果が非回帰である | constraint＋downstream vertical |
| R6-018 | coverage auditがsingle/basket、success/insufficientで同じversioned構造を持つ | four-shape DB JSON assertions |
| R6-019 | 各instrumentにsymbol、currency、basket weightを保存する | instrument audit assertion |
| R6-020 | requested/effective period、evaluation_as_of、method、series kindを保存する | top-level audit assertion |
| R6-021 | input first/last date、input/in-range/unique valid件数を保存する | range fixture assertion |
| R6-022 | duplicate、invalid、out-of-range、総dropped件数を意味どおり保存する | malformed series matrix |
| R6-023 | series hashがcanonical inputで再現可能、順序不変、値変更で変化する | hash determinism tests |
| R6-024 | single 1取引日をunevaluableとし全評価値null、完全auditを保存する | one-day EvaluationRecord query |
| R6-025 | single 2取引日で正しいreturn/MFE/MAEと完全auditを保存する | numeric positive test |
| R6-026 | basket共通1日/2日でcommon_date_count、銘柄別audit、数値判定が正しい | basket negative/positive tests |
| R6-027 | invalid rowを黙って落として評価せず、cache/provider間coverageと単一cache非汚染を維持する | invalid＋cache regression |
| R6-028 | 固定P09 Draft 2020-12 Schemaが2.1 rejectのdisposition/reason省略を拒否する | direct iter_errors paths |
| R6-029 | 固定Schemaがreject以外へのreject fieldと2.1のreject_terminalを拒否する | fixed Schema matrix |
| R6-030 | Pydantic runtimeがR6-028/029と同じpayloadを同じ方向で判定する | dual-validator parameterized test |
| R6-031 | legacy 2.0 rejectは明示reject_terminalと非空reasonなしでは拒否される | legacy missing-field tests |
| R6-032 | valid legacy boolはvalidation後だけretryable/terminalへ変換される | adapter positive assertions |
| R6-033 | legacyの新旧field混在を、一致・矛盾の両方で拒否する | mixed-field negative tests |
| R6-034 | fixed Schema/Pydanticがcorrected_payloadとoperation action別field契約を表す | correct/action Schema tests |
| R6-035 | reject_reasonとoperation reasonがtrim後非空である | whitespace negative tests |
| R6-036 | P09 promptがretryable/terminal/valid correct規則とSchema-valid例を持つ | prompt example validation |
| R6-037 | repo/generated/packaged Schema・promptが同期し、retryable/terminal/unresolvedの有限状態が非回帰である | hashes＋workflow vertical |
| R6-038 | 2 issuance/2 component/evidence/evaluation/snapshot等を含む0007 DBをheadへupgradeできる | data-bearing migration test |
| R6-039 | migration前後で全PK集合、全legacy projection hash、sentinel値を保持する | per-table snapshot assertions |
| R6-040 | migration後FK/integrity/revision/backfillが正しく、head再実行が冪等である | PRAGMA＋head→head |
| R6-041 | backup後に実DDLとdata UPDATEをcommitしてから強制失敗させる | injected partial DDL test log |
| R6-042 | 強制失敗後に元revision/Schema/PK/hash/sentinelへ復元しmarker列が残らない | restore assertions＋failure propagation |
| R6-043 | empty/0001/0005/0007/0009/head経路、Alembic差分0、active unique indexが成立する | migration matrix＋constraint |
| R6-044 | `build`がdev setupで導入され、wheel/migration/Schema testにskip/xfail/importorskipがない | pyproject＋source scan＋pytest -ra |
| R6-045 | clean wheelのimport pathがrepo srcでなく隔離venv site-packagesを指す | installed `__file__` assertion |
| R6-046 | wheelからrun/source/P05/P07、valid複数P08/P09と多対一・未申告negativeを通す | installed vertical log＋DB query |
| R6-047 | wheelから旧P11拒否、single 1/2日coverage、data-bearing migration、active resultsを通す | installed vertical assertions |
| R6-048 | wheel内migrations、P09 Schema/prompt、完全版docsがrepo正本と一致する | wheel contents＋hash assertions |
| R6-049 | Ruff format/check、Ruff lint、mypy、pytest、docs sync、Alembic、build、baseからのgit diff checkが全てpassする | 個別command・return code log |
| R6-050 | FINAL_REVIEW/IMPLEMENTATION_STATUSが証拠を超えず、CHAT_HISTORY・秘密情報・DB/raw/cache/backup/build生成物を保護し、commit/pushしていない | docs照合＋hash＋git status/secret scan |

## READY判定条件

`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`は、R6-001～R6-050がすべて具体的証拠付きPASSの場合だけ使用してください。

次はPASSの代替になりません。

- `forecast_operations`が1件以上あることだけを確認する。
- old refだけ、またはnew refだけを検査し、集合の完全一致を確認しない。
- application適用後に不正を検知し、部分的なissuanceを残す。
- coverageをunevaluableにしただけで、DB audit keyが欠落している。
- provider requestを監査証拠とし、保存済みcoverage_auditを確認しない。
- Pydantic runtimeだけで拒否し、固定JSON Schemaを直接検証しない。
- legacy欠落fieldを既定retryableへ変換する。
- helpとdocsだけのwheel testで正式縦断を代替する。
- `build`がないときwheel testをskipする。
- migration前に例外を発生させて失敗復元testとする。
- migration前後のrow countだけを比較する。
- `git diff --check`、format、buildのfailをpytest合計で上書きする。
- required testのskip/xfailを「環境依存」としてREADY扱いする。

## FULL MVPとの区別

Round6を完了しても、総合点、複数アナリスト統合、現在上昇候補、時期・程度・早期実現の完全採点、期間不明の1/3/6/12か月観測、PNG等が未実装なら`FULL_MVP_READY`ではありません。

