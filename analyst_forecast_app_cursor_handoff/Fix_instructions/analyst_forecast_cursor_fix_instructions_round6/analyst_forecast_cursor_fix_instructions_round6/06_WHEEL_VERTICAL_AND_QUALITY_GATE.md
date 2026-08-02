# 06 — clean wheel正式縦断と品質ゲート

## 目的

`build`未導入をskipするtestと、help/docsだけのwheel probeを廃止します。clean環境へwheelのみをinstallし、Round6のpositive/negative経路、coverage、data-bearing migration、packaged resourcesを実行します。

## Cursorへ渡す依頼文

```text
00～05を統合し、wheel test、package data、dev依存、品質ゲートを修正してください。commit・pushはしないでください。

1. build依存とskip禁止
- `python -m build`を実行するための`build`をpyproject.tomlのdev依存へ、他dev toolと同様に再現可能な形で追加する。
- tests/unit/test_round4_wheel.pyのpytest.importorskip("build")を削除する。
- build/import/install失敗はassert failureにする。skip、skipif、importorskip、xfail、成功扱いreturnへ変換しない。
- Round6 required testが参照するcommand/packageはdev setupだけで揃うようにする。

2. clean wheel隔離条件
- sdistとwheelをbuildする。
- temp directoryへ新規venvを作り、作成したwheelを通常installする。editable installは禁止。
- subprocessのcwdをrepository外のtemp directoryにし、PYTHONPATH/PYTHONHOME等のrepo参照を除去する。
- installed processで`analyst_forecast.__file__`を表示し、repoの`src/`ではなくvenv site-packages配下であることをassertする。
- test helperがrepoのPython moduleをimportしてbusiness logicを代行しない。fixture JSON/CSVの生成はtest側でよいが、操作はinstalled CLIまたはinstalled public APIから行う。

3. wheel正式縦断
- `--help`、`init --vault-root`、model設定、run作成、local raw/source importを実行する。
- YouTube相当とblog相当の2 sourceを、network不要のlocal fixtureで登録する。
- P05/P07、forecast A/Bを含むP08、P09 correctを正式ingestする。
- valid A→A2、B→B2をpassさせ、pairwise lineage、active 2件、結果件数をinstalled packageのSQLiteで確認する。
- old A/B→new Xの多対一P09と、未申告old/new P09を拒否し、DB件数不変を確認する。
- correct後の旧componentをP11へ渡し、inactive_forecast_component等で拒否する。
- CSV/mock providerの1取引日をunevaluableにし、DB coverage_auditの必須instrument keysを確認する。2取引日のpositive controlも行う。
- installed packageのmigration resourceを使ってdata-bearing 0007 fixtureをheadへ上げ、PK/値/FKを確認する。
- NEXT_ACTIONSと04_resultsを再生成し、active世代だけを含むことを確認する。

4. package resource
- wheelに全Alembic revision、固定P09 Schema、P09 prompt、完全版docsを含める。
- installed固定Schemaでreject field省略negative testを行う。
- repoとinstalled Schema/prompt/docsのhashまたは内容一致を確認する。
- `scripts/sync_packaged_docs.py --check`の対象外resourceがあれば、別testまたはscript拡張で同期を保証する。

5. test構造
- wheel build/installを毎unit caseで繰り返さず、1つの非skip縦断またはsession fixtureで安全に再利用してよい。
- 失敗時にstdout/stderr、command、return codeをassert messageへ含める。secretや実pathは出さない。
- Windows/Linuxのvenv python path差を既存helperで吸収する。
- network不要・実API key不要にする。
```

## 必須品質ゲート

formatter実行後に、次をすべて再実行してください。

```text
python -m ruff format .
python -m ruff format --check .
python -m ruff check .
python -m mypy src/analyst_forecast --ignore-missing-imports
python -m pytest -q
python -m pytest <Round6とwheel/migration/Schemaの必須test paths> -q -ra
python scripts/sync_packaged_docs.py --check
python -m alembic upgrade head
python -m alembic check
python -m build
```

repository rootから次も実行してください。

```text
git diff --check 88864c289750f8323c27b6e3f2c09fd70a79923d -- analyst_forecast_app_cursor_handoff
git status --short
```

さらにrequired internal testについて次を監査してください。

```text
rg -n "importorskip|pytest\.skip|pytest\.mark\.skip|skipif|pytest\.xfail|pytest\.mark\.xfail" tests
```

既存live network integrationの正当なmarkerは理由付きで区別してください。今回のwheel/migration/Schema/operation/coverage testに上記が1件でもあればFAILです。

## Git衛生

- Round5で発生した空白error14件を修正し、base commitからの全Round6差分で`git diff --check`をpassさせる。
- `reference/CHAT_HISTORY.pdf`のbase hashを維持する。
- `.env`、secret、token、実DB、backup、WAL/SHM、raw、market cache、AI実出力、実Vault pathを追跡しない。
- `dist/`、`build/`、`*.egg-info`、venv、pytest/mypy/ruff cacheを追跡しない。
- 既存の無関係なユーザー変更をformat対象として巻き込まない。

## 完了条件

- clean wheelがrepo srcへfallbackせず、operation positive/negative、旧P11拒否、coverage、migration、resultsを通す。
- `build`不足でskipする経路がない。
- 必須品質ゲートがすべてreturn code 0。
- required internal testにskip/xfail/importorskipがない。

