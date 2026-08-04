# 実装計画：クラシックOutlookからのメール・添付ファイル直接D&D

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development` while implementing each task. Use `superpowers:verification-before-completion` before declaring a task or the whole feature complete.

**Goal:** クラシックOutlookからメール1通または添付ファイル群をOneDrive保存先レコメンダーへ直接D&Dし、メール情報と添付情報を1案件として保存先候補の判定に利用できるようにする。

**Architecture:** 既存の`tkinterdnd2`へOutlookの仮想ファイル形式を明示登録し、D&Dされたメールまたは添付をアプリ専用の一時領域へ実体化する。メールは1通を1つの処理単位とし、件名・本文・添付ファイル名・対応文書の本文から検索語を生成して既存ランキングへ1回だけ渡す。Outlook由来の一時ファイル管理、MSG解析、セッション統合、GUI接続を分離し、将来は同じメール単位の解析を複数通へ反復できるようにする。

**Tech Stack:** Windows、Python 3.12、Tkinter、`tkinterdnd2==0.6.2`（同梱TkDND 2.10.1）、`pywin32==311`、pytest、ruff。

## Global Constraints

- 対象はクラシックOutlook for Windowsとし、新しいOutlook for Windowsは今回の対応対象外とする。
- 今回受け付けるメールは1通とする。複数メールの同時D&Dは明示的に拒否する。
- メールの添付ファイル数は0件以上とする。添付なしメールも正常な1案件として処理する。
- メール1通をドロップした場合、件名・本文・添付ファイル名・対応する添付文書本文を1案件へ統合する。
- Outlookの添付ファイルだけを1件以上ドロップした場合は、既存の通常ファイル案件として処理する。
- 対応する添付本文は`.xlsx`、`.xlsm`、`.docx`、`.pptx`、`.pdf`に限定する。その他はファイル名だけを利用する。
- 新規の実行時依存を追加しない。既存の`tkinterdnd2`と`pywin32`だけを利用する。
- Outlook由来の実体はアプリ専用の一時領域だけへ保存し、入力置換・手動検索への復帰・アプリ終了時に削除する。
- MSG内部から本文抽出用に保存した添付実体は、検索語生成の完了時に削除する。
- OneDrive上のファイル・フォルダを作成、変更、移動、削除しない。
- D&D操作だけで候補を確定せず、クリップボードとAuditを変更しない。
- メール本文、添付本文、検索語全文を`settings.json`、`catalog.json`、`audit.jsonl`、ログへ保存しない。
- 新しい入力の処理に失敗した場合、直前の入力、検索語、候補、確定パスを維持し、失敗分の一時ファイルだけを削除する。
- 将来の複数メール対応は、今回作る「メール1通の解析関数」を複数回呼ぶ方式とし、今回は複数案件管理や新画面を実装しない。

---

## 1. 文書情報

| 項目 | 値 |
|---|---|
| 対象 | OneDrive業務フォルダ向け「保存先レコメンダー」 |
| 位置付け | MVP 1運用後の小改修 |
| 作成日 | 2026-08-04 |
| 計画ブランチ | `codex/outlook-direct-dnd-plan` |
| 基準コミット | `46563f9` |
| 対象プロジェクト | `onedrive_destination_recommender/` |
| 状態 | 実装前・Claude Codeレビュー待ち |

本計画のレビューが完了するまで、プログラム本体、テスト、依存関係、README、要件定義書を変更しない。

## 2. 合意済みの利用仕様

### 2.1 受け付ける操作

1. クラシックOutlookのメール一覧からメール1通を入力一覧へD&Dする。
2. クラシックOutlookの閲覧画面から添付ファイル1件以上を入力一覧へD&Dする。
3. 既存どおり、デスクトップまたはExplorerからローカルファイルをD&Dする。
4. 既存どおり、「ファイルを選択」からローカルファイルまたは保存済みMSGを選ぶ。

メールをドロップしたとき、添付は必須としない。

- 添付0件：メールファイル名、件名、本文で判定する。
- 添付1件以上：上記に全添付ファイル名と、読めた対応文書本文を加えて判定する。
- 添付本文を読めない：メール情報と添付ファイル名だけで処理を続ける。

### 2.2 今回拒否する操作

- メール2通以上の同時D&D
- MSG 1件と、別に選択した通常ファイルを同じ入力操作で混在させること
- 現在の案件へのファイル追記
- フォルダ、存在しないパス、複数MSG
- 新しいOutlook for Windowsからの直接D&Dについての動作保証

メール本体に内包されている添付は「MSGと通常ファイルの混在」には数えない。アプリがメールを開いて添付情報を読む処理は、1件のメール案件の内部処理である。

### 2.3 表示

入力欄の常設案内は、次の趣旨へ変更する。

```text
Outlookのメール・添付ファイル、またはExplorerのファイルをここへドロップできます
```

解析状態の代表文言：

- `メール解析完了（添付なし）`
- `メール解析完了（添付2件／本文解析対象なし）`
- `メール解析完了／添付本文を利用：2/3件`
- `メールの一部を利用できませんでした。読めた情報で処理を続けます。`

## 3. 技術方式

### 3.1 Outlook仮想ファイルの受け取り

Explorerの実ファイルは`CF_HDROP`で絶対パスを渡す。一方、クラシックOutlookのメールと添付ファイルは、実在パスではなく次の組で仮想ファイルを渡す。

- `FileGroupDescriptorW`または`FileGroupDescriptor`：ファイル名等の記述
- `FileContents`：各仮想ファイルの内容

MicrosoftのShell仕様では、この組がメール等の非ファイルシステムオブジェクトをファイルとして転送する標準方式である。

現在のアプリは`DND_FILES`だけを登録している。`tkinterdnd2`には次のWindows専用タイプが公開されており、TkDNDが仮想ファイルを指定一時ディレクトリへ実体化して、保存したパスをドロップイベントへ返す。

```python
from tkinterdnd2 import (
    DND_FILES,
    FileGroupDescriptor,
    FileGroupDescriptorW,
    TkinterDnD,
)

TkinterDnD.require(root)
input_list.set_dropfile_tempdir(str(workspaces.staging_directory))
input_list.drop_target_register(
    DND_FILES,
    FileGroupDescriptorW,
    FileGroupDescriptor,
)
```

`DND_FILES`を残すため、既存のExplorer D&D経路は変更しない。Unicode版を先に登録し、日本語ファイル名を優先する。ANSI版は古い送出元への退避として残す。

### 3.2 一時領域のライフサイクル

新規`outlook_drop.py`に`DropWorkspaceManager`を置く。1つのアプリ実行につき`TemporaryDirectory(prefix="odr-outlook-drop-")`を1つ所有し、その配下に次を作る。

```text
odr-outlook-drop-<random>/
├─ current-<random>/   # 現在受理済みのOutlook仮想ファイル
└─ staging-<random>/   # 次のドロップを受ける領域
```

ディレクトリ名は実装時にUUID等で一意化する。`staging`を`current`へrenameするとイベントが返したパスが無効になるため、ディレクトリはrenameせず、マネージャーが参照を昇格する。

公開インターフェースは、`staging_directory: Path`プロパティ、`accept_staging() -> Path`、`reject_staging() -> Path`、`clear_current() -> None`、`close() -> None`の5つに限定する。`accept_staging()`は旧currentを削除してstagingをcurrentへ昇格し、`reject_staging()`はstagingだけを削除する。両メソッドは空の次stagingを作り、そのパスを返す。`close()`はアプリが所有する一時ルート全体を削除し、複数回呼んでも成功する。

`accept_staging()`と`reject_staging()`の戻り値は、新しく作った`staging_directory`とする。GUIはその値を`set_dropfile_tempdir()`へ再設定する。

外部のローカルパスは一時ルート配下にないため、マネージャーは削除しない。一時ルート以外を削除対象にしないことを単体テストで固定する。

### 3.3 メール1通と添付の統合

`msg_reader.build_msg_search_terms()`を「メール1通の解析境界」として維持する。この関数が`TemporaryDirectory(prefix="odr-msg-attachments-")`を開き、そのパスを内部関数`_read_msg_content(msg_path, attachment_directory: Path | None = None)`へ渡す。件名・本文・添付ファイル名を取得した同じOutlook COMセッション内で、対応文書だけを`Attachment.SaveAsFile()`で保存する。`probe_msg_access()`は`attachment_directory`を渡さず、従来どおり添付を保存しない。

保存対象は`document_reader.is_supported_document()`が真になる添付だけとする。画像、CAD、圧縮ファイル、旧Office形式等は保存せず、ファイル名だけを主検索語へ使う。

一時保存名はOutlookから渡された文字列をそのままパスへ連結しない。区切り文字を正規化してbasenameだけを取り、添付の1始まりindexを接頭辞にして同名衝突を避ける。

```text
元の添付名                 一時保存名
図面.pdf                   0001_図面.pdf
..\..\設備仕様.docx       0002_設備仕様.docx
図面.pdf（同名2件目）       0003_図面.pdf
```

内部型`_MsgContent`に、関数スコープの一時領域だけを指す`saved_document_paths: tuple[Path, ...]`を追加する。`build_document_terms()`へこのパス群をまとめて渡し、戻った補助検索語をメール本文由来の補助検索語へ順序を保って重複排除しながら連結する。`build_msg_search_terms()`を抜けると`TemporaryDirectory`が添付実体を削除する。`saved_document_paths`は公開API、`InputState`、Audit、Codex相談用プロンプトへ渡さない。

`MsgSearchTerms`へ、画面状態を組み立てるための集計値を追加する。

```python
@dataclass(frozen=True, slots=True)
class MsgSearchTerms:
    primary_terms: tuple[str, ...]
    auxiliary_terms: tuple[str, ...]
    fully_parsed: bool
    body_available: bool
    attachment_count: int
    attachment_document_parsed_count: int
    attachment_document_target_count: int
    warning: str | None
```

生の件名、本文、添付本文は公開型へ追加しない。この戻り値1個がメール1通分を表すため、将来の複数メール対応では`build_msg_search_terms()`をメールパスごとに呼び、検索語と件数を集約できる。今回は`session.select_files()`の複数MSG拒否を維持する。

### 3.4 検索語の統合規則

| 入力情報 | 種別 | 備考 |
|---|---|---|
| 仮想MSGのファイル名 | 主検索語 | 既存どおり |
| 件名 | 主検索語 | 既存どおり |
| 全添付ファイル名 | 主検索語 | 既存どおり。検索対象外の一般名は既存規則で除外 |
| メール本文 | 補助検索語 | 既存の`clean_msg_body`を使用 |
| 対応添付文書本文 | 補助検索語 | 既存の`build_document_terms`を使用 |

主検索語一致0件の候補を除外する既存規則は変更しない。添付本文を主検索語へ混ぜないため、文書本文だけを根拠に候補集合が膨張しない。

## 4. 変更ファイルと責務

| ファイル | 変更内容 |
|---|---|
| `onedrive_destination_recommender/src/onedrive_destination_recommender/outlook_drop.py` | 新規。一時領域の所有、staging/currentの昇格・破棄・終了処理 |
| `onedrive_destination_recommender/src/onedrive_destination_recommender/app.py` | Outlook仮想形式の登録、一時領域の成功・失敗接続、終了時cleanup、案内文 |
| `onedrive_destination_recommender/src/onedrive_destination_recommender/msg_reader.py` | 対応添付の一時保存、添付本文検索語の統合、解析件数 |
| `onedrive_destination_recommender/src/onedrive_destination_recommender/session.py` | 添付0件・添付解析件数に応じた状態文言。複数MSG拒否は維持 |
| `onedrive_destination_recommender/tests/unit/test_outlook_drop.py` | 新規。一時領域ライフサイクルと削除境界 |
| `onedrive_destination_recommender/tests/unit/test_msg_reader.py` | 添付なし、保存対象、本文統合、部分失敗、パス安全性、即時削除 |
| `onedrive_destination_recommender/tests/unit/test_session.py` | 新しい`MsgSearchTerms`と状態文言、複数MSG拒否回帰 |
| `onedrive_destination_recommender/tests/integration/test_gui.py` | 3種類のD&D登録、成功・失敗時のworkspace処理、既存Explorer D&D回帰 |
| `onedrive_destination_recommender/README.md` | 操作方法、対象Outlook、添付任意、一時ファイル、対象外 |
| `obsidan vault/30_Permanent/★業務/ファイル管理/要件定義_OneDrive保存先レコメンダー.md` | §6、§15、§20、§22、§23へ本機能と受け入れ条件を反映 |
| 本計画書 | 各Stepの実装・検証結果を追記 |

`pyproject.toml`は変更しない。`tkinterdnd2==0.6.2`と`pywin32==311`は既にWindows限定の実行時依存である。

## 5. 実装タスク

### Task 1: Outlook仮想D&D形式と一時領域

**Files:**
- Create: `onedrive_destination_recommender/src/onedrive_destination_recommender/outlook_drop.py`
- Create: `onedrive_destination_recommender/tests/unit/test_outlook_drop.py`
- Modify: `onedrive_destination_recommender/tests/integration/test_gui.py`
- Modify: `onedrive_destination_recommender/src/onedrive_destination_recommender/app.py:514`

**Interfaces:**
- Consumes: `pathlib.Path`、`tempfile.TemporaryDirectory`、TkDNDの`set_dropfile_tempdir()`
- Produces: `DropWorkspaceManager.staging_directory`、`accept_staging()`、`reject_staging()`、`clear_current()`、`close()`

- [ ] **Step 1: 一時領域の失敗先行テストを書く**

```python
def test_accept_replaces_only_owned_current_workspace(tmp_path: Path) -> None:
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"outside")
    manager = DropWorkspaceManager(base_directory=tmp_path / "owned")
    first_staging = manager.staging_directory
    (first_staging / "first.msg").write_bytes(b"first")

    next_staging = manager.accept_staging()
    assert (first_staging / "first.msg").is_file()
    assert next_staging != first_staging

    (next_staging / "second.msg").write_bytes(b"second")
    manager.accept_staging()

    assert not first_staging.exists()
    assert outside.read_bytes() == b"outside"
```

`reject_staging()`、`clear_current()`、`close()`の冪等性についても、それぞれ1挙動ずつ独立したテストを書く。

- [ ] **Step 2: テストが期待どおり失敗することを確認する**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_outlook_drop.py -v
```

Expected: `ModuleNotFoundError: onedrive_destination_recommender.outlook_drop`。

- [ ] **Step 3: 最小の`DropWorkspaceManager`を実装する**

削除前に、対象を`Path.resolve()`し、所有する一時ルート配下であることを`Path.is_relative_to()`で検証する。外部パスを受け取る削除APIは作らない。

- [ ] **Step 4: GUIへ3種類の形式を登録する失敗先行テストを書く**

既存の`test_dnd_initialization_failure_keeps_file_selection_available`を維持し、正常初期化時に`drop_target_register`へ次の3タイプが渡ることを確認する。

```python
assert registered_types == (
    "DND_Files",
    "FileGroupDescriptorW - FileContents",
    "FileGroupDescriptor - FileContents",
)
```

- [ ] **Step 5: テスト失敗を確認してから、D&D登録を最小実装する**

`_initialize_dnd()`で`DropWorkspaceManager`を作成し、`set_dropfile_tempdir()`を呼んでから3タイプを登録する。importまたはTkDND初期化に失敗した場合は、workspaceを閉じ、既存どおりファイル選択を残す。

- [ ] **Step 6: Task 1のテストと既存GUIテストを実行する**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_outlook_drop.py -v
.\.venv\Scripts\python.exe -m pytest tests\integration\test_gui.py -v -m integration
```

Expected: 対象テストがすべてPASS。`ODR_TEST_MSG_PATH`未指定による既存の実MSGテストskipだけ許容する。

- [ ] **Step 7: クラシックOutlookとのGo/No-Go確認を行う**

機密情報を含まないダミーメール1通とダミー添付1件を実際にD&Dし、TkDNDが`.msg`と添付をstagingへ実体化して正しいパスをイベントへ返すことを確認する。失敗した場合はTask 2へ進まず、独自OLE実装を追加せずに結果を本計画書へ記録して再レビューを依頼する。

- [ ] **Step 8: コミットする**

```powershell
git add onedrive_destination_recommender/src/onedrive_destination_recommender/outlook_drop.py onedrive_destination_recommender/src/onedrive_destination_recommender/app.py onedrive_destination_recommender/tests/unit/test_outlook_drop.py onedrive_destination_recommender/tests/integration/test_gui.py
git commit -m "feat: Outlook仮想ファイルのD&D受け取りを追加"
```

### Task 2: メール添付の本文抽出

**Files:**
- Modify: `onedrive_destination_recommender/src/onedrive_destination_recommender/msg_reader.py:38`
- Modify: `onedrive_destination_recommender/tests/unit/test_msg_reader.py:19`

**Interfaces:**
- Consumes: `Attachment.FileName`、`Attachment.SaveAsFile(path)`、`document_reader.is_supported_document()`、`build_document_terms()`
- Produces: 集計値を追加した`MsgSearchTerms`。生本文と添付実体は返さない

- [ ] **Step 1: 添付なしメールの失敗先行テストを書く**

```python
def test_build_msg_terms_accepts_message_without_attachments(tmp_path, monkeypatch):
    msg_path = tmp_path / "案件.msg"
    msg_path.write_bytes(b"")
    content = msg_reader._MsgContent(
        subject="設備更新",
        body="秋田 工場",
        attachment_file_names=(),
        attachment_count=0,
        subject_available=True,
        body_available=True,
        attachments_available=True,
    )
    monkeypatch.setattr(msg_reader, "_read_msg_content", lambda _path: content)

    result = msg_reader.build_msg_search_terms(msg_path)

    assert result.attachment_count == 0
    assert result.attachment_document_target_count == 0
    assert result.attachment_document_parsed_count == 0
```

- [ ] **Step 2: 対応添付本文の統合と即時削除の失敗先行テストを書く**

偽`Attachment.SaveAsFile()`は`.docx`等の合成ファイルを指定パスへ保存する。`build_document_terms()`を通して添付本文由来語が`auxiliary_terms`へ追加され、関数終了後に`odr-msg-attachments-*`が残らないことを確認する。

- [ ] **Step 3: 保存対象の限定とパス安全性の失敗先行テストを書く**

次を別テストで固定する。

- `.png`は`SaveAsFile()`を呼ばず、名前だけを主検索語へ入れる
- `..\..\設備仕様.docx`が一時ルート外へ保存されない
- 同名添付2件が異なる一時名で保存される
- 対応文書1件の保存失敗時も、メール本文と他の添付で処理を続ける

- [ ] **Step 4: 失敗理由を確認する**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_msg_reader.py -v
```

Expected: 新しい`MsgSearchTerms`フィールドと添付保存処理が未実装のためFAIL。既存テストはPASSを維持する。

- [ ] **Step 5: 最小実装する**

現在の`_read_attachment_names()`を、同じCOMセッションで添付名と対応文書を扱える内部処理へ置き換える。`SaveAsFile`以外の汎用書込みAPIを`msg_reader.py`へ追加しない。

既存の`test_msg_modules_have_no_file_or_log_write_path`は、次の安全制約へ変更する。

```python
assert ".write(" not in source
assert "write_text(" not in source
assert "open(" not in source
assert "logging" not in source
assert source.count(".SaveAsFile(") == 1
assert "TemporaryDirectory(" in source
```

これは従来制約の無条件解除ではなく、Outlook COM添付を関数スコープの一時領域へ保存する1経路だけを許可する変更である。

- [ ] **Step 6: Task 2のテストを通す**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_msg_reader.py tests\unit\test_document_reader.py -v
```

Expected: すべてPASS。一時添付実体がテスト終了後に残らない。

- [ ] **Step 7: コミットする**

```powershell
git add onedrive_destination_recommender/src/onedrive_destination_recommender/msg_reader.py onedrive_destination_recommender/tests/unit/test_msg_reader.py
git commit -m "feat: メールと添付文書を一案件として解析"
```

### Task 3: セッション状態と表示文言

**Files:**
- Modify: `onedrive_destination_recommender/src/onedrive_destination_recommender/session.py:121`
- Modify: `onedrive_destination_recommender/tests/unit/test_session.py:103`

**Interfaces:**
- Consumes: 拡張した`MsgSearchTerms`
- Produces: `InputState.msg_status`の添付件数別文言。候補・Auditインターフェースは不変

- [ ] **Step 1: 状態文言の失敗先行テストを書く**

```python
@pytest.mark.parametrize(
    ("attachment_count", "parsed", "target", "expected"),
    [
        (0, 0, 0, "メール解析完了（添付なし）"),
        (2, 0, 0, "メール解析完了（添付2件／本文解析対象なし）"),
        (3, 2, 3, "メール解析完了／添付本文を利用：2/3件"),
    ],
)
def test_msg_status_describes_optional_attachments(
    attachment_count: int,
    parsed: int,
    target: int,
    expected: str,
) -> None:
    result = MsgSearchTerms(
        primary_terms=("案件",),
        auxiliary_terms=("設備",),
        fully_parsed=True,
        body_available=True,
        attachment_count=attachment_count,
        attachment_document_parsed_count=parsed,
        attachment_document_target_count=target,
        warning=None,
    )

    assert _msg_status(result) == expected
```

テスト本体では各値を持つ`MsgSearchTerms`を生成し、`select_files()`後の`state.msg_status`を比較する。`warning`がある場合は、件数表示より警告を優先する既存方針を固定する。

- [ ] **Step 2: テスト失敗を確認する**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_session.py -v
```

Expected: 既存の固定値`MSG解析完了`が返るため、新規ケースがFAIL。

- [ ] **Step 3: `_msg_status()`を最小変更する**

候補計算、`InputState.file_paths`、Audit入力名、Codex相談用パスは変更しない。メールの仮想MSGパスはcurrent workspaceに残るため、既存の相談用添付パスとして利用できる。

- [ ] **Step 4: 複数MSG拒否と添付なし回帰を確認する**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_session.py -v
```

Expected: すべてPASS。

- [ ] **Step 5: コミットする**

```powershell
git add onedrive_destination_recommender/src/onedrive_destination_recommender/session.py onedrive_destination_recommender/tests/unit/test_session.py
git commit -m "feat: メール添付の解析状態を表示"
```

### Task 4: GUIの成功・失敗トランザクション

**Files:**
- Modify: `onedrive_destination_recommender/src/onedrive_destination_recommender/app.py:538`
- Modify: `onedrive_destination_recommender/tests/integration/test_gui.py:352`

**Interfaces:**
- Consumes: `DropWorkspaceManager`と既存`RecommenderSession.select_files()`
- Produces: 成功時だけworkspaceと画面状態を置換するD&D・ファイル選択・手動検索の共通動作

- [ ] **Step 1: 成功・失敗時workspace処理の失敗先行テストを書く**

GUIテストで次を固定する。

- D&D成功：`accept_staging()`が1回呼ばれ、新stagingがTkDNDへ設定される
- D&D失敗：`reject_staging()`が1回呼ばれ、直前のセッション状態と確定パスを維持する
- ファイル選択成功：前のOutlook current workspaceを削除する
- 「手動検索に戻す」：`clear_current()`を呼ぶ
- root破棄：`close()`を1回呼び、2回目も例外にならない
- ランタイム未読込または空のドロップ：新規stagingを破棄し、仮想ファイルを残さない

- [ ] **Step 2: テスト失敗を確認する**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\integration\test_gui.py -v -m integration
```

Expected: workspaceの昇格・破棄接続が未実装のため新規テストだけFAIL。

- [ ] **Step 3: `_accept_files()`へ成功・失敗後処理を接続する**

セッション処理が成功するまで`accept_staging()`を呼ばない。既知例外と予期しない例外の双方で`reject_staging()`を呼ぶ。workspace操作の失敗は元入力を壊さず、利用者へ一時ファイル整理失敗の一般警告を出す。

終了処理は`root.bind("<Destroy>", self._on_root_destroy, add="+")`で登録し、`event.widget is self.root`の場合だけ実行する。子widgetの破棄ではcleanupしない。

- [ ] **Step 4: 常設案内と処理中・完了状態を更新する**

D&D受付時は`Outlook入力を処理しています。`、成功時は`ファイルから検索語を生成しました。`を表示する。新しい進捗画面、スレッド、キャンセル機構は追加しない。

- [ ] **Step 5: GUIテストを通す**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\integration\test_gui.py -v -m integration
```

Expected: 既存Explorer D&Dを含めすべてPASS。実MSGパス未指定のテストだけ既存どおりskip可。

- [ ] **Step 6: コミットする**

```powershell
git add onedrive_destination_recommender/src/onedrive_destination_recommender/app.py onedrive_destination_recommender/tests/integration/test_gui.py
git commit -m "feat: Outlook D&Dの入力状態を安全に置換"
```

### Task 5: 安全性回帰と文書更新

**Files:**
- Modify: `onedrive_destination_recommender/tests/unit/test_no_network.py`
- Modify: `onedrive_destination_recommender/tests/unit/test_audit.py`
- Modify: `onedrive_destination_recommender/README.md`
- Modify: `obsidan vault/30_Permanent/★業務/ファイル管理/要件定義_OneDrive保存先レコメンダー.md`
- Modify: `obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_Outlook直接D&D.md`

**Interfaces:**
- Consumes: 完成した機能と既存受け入れ基準
- Produces: 自動検査、実Outlook手動結果、利用者向け手順、要件定義の追跡可能性

- [ ] **Step 1: Audit・ネットワーク・一時ファイルの安全性テストを先に追加する**

合成メール本文と添付文書へ固有識別子を埋め、候補確定後も`catalog.json`と`audit.jsonl`に識別子がなく、Auditは入力ファイル名だけを持つことを確認する。パッケージのネットワークclient import禁止と遅延import許可リストは変更しない。

- [ ] **Step 2: 全自動テストと静的検査を実行する**

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest -m integration
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m ruff format --check src tests
```

Expected: 全テストPASS。`ODR_TEST_MSG_PATH`未指定による既存の実MSGテストskipだけ許容する。

- [ ] **Step 3: クラシックOutlook実機確認を行う**

実業務の件名、本文、添付名、絶対パスを記録へ残さず、次を確認する。

1. 添付なしメール1通
2. 対応文書添付ありメール1通
3. 未対応添付だけを持つメール1通
4. 対応文書と未対応添付が混在するメール1通
5. Outlook添付ファイルだけの単一・複数D&D
6. 日本語、空白、全角・半角括弧、同名を含む添付
7. メール2通の同時D&D拒否と直前状態維持
8. 入力置換・手動検索復帰・アプリ終了後の一時ルート削除
9. Explorer・デスクトップD&D、ファイル選択、Explorer Preview、確定・コピーの回帰

- [ ] **Step 4: READMEと要件定義書を更新する**

対象がクラシックOutlookであること、添付は任意であること、メール1通制限、添付だけのD&D、一時ファイルの扱い、失敗時のファイル選択フォールバック、新しいOutlookが対象外であることを明記する。

- [ ] **Step 5: 本計画書へ実測結果を追記する**

自動テスト件数、skip理由、ruff結果、実Outlook確認9項目、未確認項目を事実ベースで記録する。合格していない項目を完了扱いにしない。

- [ ] **Step 6: 最終コミットする**

```powershell
git add onedrive_destination_recommender/tests/unit/test_no_network.py onedrive_destination_recommender/tests/unit/test_audit.py onedrive_destination_recommender/README.md "obsidan vault/30_Permanent/★業務/ファイル管理/要件定義_OneDrive保存先レコメンダー.md" "obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_Outlook直接D&D.md"
git commit -m "docs: Outlook直接D&Dの検証結果を記録"
```

## 6. 受け入れ条件

1. クラシックOutlookのメール1通を入力欄へ直接D&Dできる。
2. 添付なしメールを、メールファイル名・件名・本文による1案件として正常処理できる。
3. 添付ありメールを、メール情報・全添付名・読めた対応文書本文による1案件として処理できる。
4. Outlookの添付ファイルだけを単一・複数でD&Dし、既存の通常ファイル規則で処理できる。
5. 日本語、空白、括弧、同名を含む仮想ファイルを内容と順序を壊さず受け取れる。
6. メール2通以上を全件拒否し、直前の入力、検索語、候補、確定パスを維持できる。
7. メール本文または一部添付を読めない場合も、読めた情報で推薦を継続し、状態を表示できる。
8. D&D初期化に失敗してもアプリを終了せず、既存のファイル選択を利用できる。
9. 失敗したD&Dの一時実体だけを削除し、受理済みの現在入力を維持できる。
10. 入力置換、手動検索復帰、アプリ終了時に、アプリが所有するOutlook一時実体を削除できる。
11. D&DだけではOneDrive、Audit、クリップボード、確定パスを変更しない。
12. メール本文、添付本文、検索語全文がランタイムファイルとログへ残らない。
13. 既存のExplorer D&D、保存済みMSG、文書本文抽出、候補表示、Explorer Preview、確定・コピーが回帰しない。

## 7. 主なリスクと対策

| リスク | 対策 |
|---|---|
| 現行TkDNDバイナリで仮想ファイルの実体化が動かない | 実装の最初に3タイプの登録とクラシックOutlook実D&Dを確認する。失敗時は独自OLE実装へ自動的に拡大せず、計画を再レビューする |
| 仮想ファイル名によるパストラバーサル | basename化、index接頭辞、所有一時ルート配下検証を行う |
| 同名添付の上書き | Outlookの1始まりindexを一時名へ付与する |
| 大きい添付でUIが停止する | 今回は既存の同期文書抽出を維持し、実測で操作不能な停止が出た場合だけ別計画で非同期化を検討する |
| 添付保存失敗でメール全体が失敗する | 対応添付を個別に失敗扱いとし、メール本文・添付名・成功した添付で継続する |
| 一時実体が残る | staging/currentを所有する専用マネージャー、成功・失敗・手動復帰・終了のテストを用意する |
| 異常終了時に一時ファイルが残る | 正常終了cleanupを必須とする。異常終了後の古い一時ディレクトリ掃除は、安全な所有判定を別途設計する必要があるため今回は自動削除しない |
| 新しいOutlookでも動くと誤解される | UI案内、README、要件定義へ「クラシックOutlook」を明記する |
| 将来拡張のための過剰抽象化 | 1通の解析関数だけを独立させ、複数案件モデル、キュー、並列処理、追加画面は作らない |

## 8. Claude Codeレビューで確認してほしい点

1. `tkinterdnd2`の`FileGroupDescriptorW - FileContents`を明示登録する方式が、現行0.6.2／TkDND 2.10.1で妥当か。
2. `DropWorkspaceManager`のstaging/current方式に、受理済み入力の誤削除または失敗分の残存につながる穴がないか。
3. `msg_reader`の従来の「書込み経路なし」制約を、`SaveAsFile` 1経路だけへ狭く緩和する判断が妥当か。
4. 対応文書だけを一時保存し、その他の添付は名前だけ使う方式がユーザー要件を満たすか。
5. 添付0件、部分失敗、Outlook COM利用不可の状態文言と継続方針に矛盾がないか。
6. 将来の複数メール対応を妨げず、今回の実装を過剰にしない境界になっているか。
7. 自動テストと実Outlook手動確認に不足または重複がないか。

## 9. 実装開始条件

- Claude Codeレビューの重大指摘を解消し、採用しない指摘は理由を記録している。
- クラシックOutlook、Python 3.12、`tkinterdnd2==0.6.2`の実行環境を利用できる。
- 実装前の`pytest`、Windows GUI結合テスト、ruffの基準値を記録している。
- 本計画のGlobal Constraintsと受け入れ条件に未解決の矛盾がない。

## 10. 参考資料

- Microsoft Learn: [Shell Clipboard Formats](https://learn.microsoft.com/en-us/windows/win32/shell/clipboard)
- Microsoft Learn: [IDropTarget interface](https://learn.microsoft.com/en-us/windows/win32/api/oleidl/nn-oleidl-idroptarget)
- Microsoft Support: [What version of Outlook do I have?](https://support.microsoft.com/en-us/office/what-version-of-outlook-do-i-have-b3a9568c-edb5-42b9-9825-d48d82b2257c)
- PyPI: [tkinterdnd2](https://pypi.org/project/tkinterdnd2/)
