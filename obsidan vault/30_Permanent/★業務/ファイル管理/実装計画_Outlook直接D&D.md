# 実装計画：クラシックOutlookからのメール・添付ファイル直接D&D

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task, `superpowers:test-driven-development` for each behavior change, and `superpowers:verification-before-completion` before each completion claim.

**Goal:** クラシックOutlookからメール1通または添付ファイル群を保存先レコメンダーへ直接D&Dし、送信者表示名を含むメール情報と任意の添付情報を1案件として保存先候補の判定に利用できるようにする。

**Architecture:** TkDND 2.10.1のネイティブ仮想ファイル機能を使うが、同梱Tclで無効なOutlook形式の対応表を起動時に補正し、登録形式は既存の`DND_FILES`へ統一する。Outlook由来の実体はstaging/currentを持つ専用マネージャーだけが所有し、DragEnterで先行生成された未参照ファイルも削除する。メール1通の解析境界は、送信者表示名・件名・本文・添付名・対応添付本文をメモリ上の検索情報へ変換し、送信者は空白を除去した専用キーとして既存ランキングへ1件分の主照合を追加する。

**Tech Stack:** Windows、Python 3.12、Tkinter、`tkinterdnd2==0.6.2`（同梱TkDND 2.10.1）、`pywin32==311`、pytest、ruff。

## Global Constraints

- 対象はクラシックOutlook for Windowsとし、新しいOutlook for Windowsは対象外とする。
- 今回受け付けるメールは1通とする。複数メールの同時D&Dは明示的に拒否する。
- メールの添付ファイル数は0件以上とし、添付なしメールも正常な1案件として処理する。
- メール1通をドロップした場合、送信者表示名、件名、本文、添付ファイル名、対応する添付文書本文を1案件へ統合する。
- Outlookの添付ファイルだけを1件以上ドロップした場合は、既存の通常ファイル案件として処理する。
- 送信者はOutlookの表示名を使用し、メールアドレスによる同名人物の区別は行わない。
- 送信者照合ではUnicode空白を除去し、`山田 太郎`と`山田太郎`を同一キーとして扱う。
- 送信者を今回から推薦へ利用し、候補フォルダとの一致は主検索語1件相当として順位付けする。
- 送信者を取得できない場合も、他の読めたメール情報だけで処理を継続する。
- 対応する添付本文は`.xlsx`、`.xlsm`、`.docx`、`.pptx`、`.pdf`に限定する。その他はファイル名だけを利用する。
- 新規の実行時依存を追加しない。既存の`tkinterdnd2`と`pywin32`だけを利用する。
- Outlook由来の実体はアプリ専用の一時領域だけへ保存し、入力置換・手動検索への復帰・アプリ終了時に削除する。
- MSG内部から本文抽出用に保存した添付実体は、検索語生成の完了時に削除する。
- OneDrive上のファイル・フォルダを作成、変更、移動、削除しない。
- D&D操作だけで候補を確定せず、クリップボードとAuditを変更しない。
- メール本文、添付本文、送信者表示名、検索語全文を`settings.json`、`catalog.json`、`audit.jsonl`、ログへ保存しない。
- 新しい入力の処理に失敗した場合、直前の入力、検索語、候補、確定パスを維持し、失敗分の一時ファイルだけを削除する。
- 将来の複数メール対応は、今回作る「メール1通の解析関数」をメールごとに呼ぶ方式とし、今回は複数案件管理、キュー、並列処理、新画面を実装しない。

---

## 1. 文書情報

| 項目 | 値 |
|---|---|
| 対象 | OneDrive業務フォルダ向け「保存先レコメンダー」 |
| 位置付け | MVP 1運用後の小改修 |
| 作成日 | 2026-08-04 |
| 改訂日 | 2026-08-04 |
| 計画ブランチ | `codex/outlook-direct-dnd-plan` |
| 初回レビュー対象 | `23107f5` |
| 初回レビュー判定 | 計画修正後に再レビュー |
| 対象プロジェクト | `onedrive_destination_recommender/` |
| 状態 | 実装前・Claude Code再レビュー待ち |

本計画の再レビューが完了するまで、プログラム本体、テスト、依存関係、README、要件定義書は変更しない。

### 1.1 計画改訂時の基準値

2026-08-04にプログラム無変更の状態で確認した。

- 単体テスト：`146 passed, 7 deselected`
- 結合テスト：`6 passed, 1 skipped, 146 deselected`
- skip理由：`ODR_TEST_MSG_PATH`未設定
- `ruff check src tests`：合格
- `ruff format --check src tests`：25 files already formatted
- サンドボックス内ではpytest一時ディレクトリのACLによりsetup errorとなるため、通常権限と専用`--basetemp`で再実行して上記結果を得た。

## 2. 合意済みの利用仕様

### 2.1 受け付ける操作

1. クラシックOutlookのメール一覧からメール1通を入力一覧へD&Dする。
2. クラシックOutlookの閲覧画面から添付ファイル1件以上を入力一覧へD&Dする。
3. 既存どおり、デスクトップまたはExplorerからローカルファイルをD&Dする。
4. 既存どおり、「ファイルを選択」からローカルファイルまたは保存済みMSGを選ぶ。

メールをドロップしたとき、添付は必須としない。

- 添付0件：仮想MSGのファイル名、送信者表示名、件名、本文で判定する。
- 添付1件以上：上記に全添付ファイル名と、読めた対応文書本文を加えて判定する。
- 添付本文を読めない：メール情報と添付ファイル名だけで処理を続ける。
- 送信者を読めない：件名、本文、添付情報だけで処理を続ける。

### 2.2 送信者の扱い

- 取得元はOutlook `MailItem.SenderName`とする。
- `SenderEmailAddress`、`SenderEmailType`、ExchangeのSMTPアドレスは取得・照合に使用しない。
- `unicodedata.normalize("NFKC", value).casefold()`後、`character.isspace()`が真の文字をすべて除去した値を`sender_key`とする。
- 同じ表示名から同じ`sender_key`が得られる人物は同一送信者として扱う。
- `sender_key`はメモリ上の現在入力だけが保持し、入力置換または手動検索復帰で破棄する。
- 送信者一致は主照合1件相当とし、本文由来の補助照合より先に順位へ影響させる。

### 2.3 今回拒否する操作

- メール2通以上の同時D&D
- MSG 1件と、別に選択した通常ファイルを同じ入力操作で混在させること
- 現在の案件へのファイル追記
- フォルダ、存在しないパス、複数MSG
- 新しいOutlook for Windowsからの直接D&Dについての動作保証

メール本体に内包される添付は「MSGと通常ファイルの混在」には数えない。

### 2.4 表示

入力欄の常設案内は次の趣旨へ変更する。

```text
Outlookのメール・添付ファイル、またはExplorerのファイルをここへドロップできます
```

解析状態の代表文言：

- `メール解析完了（添付なし）`
- `メール解析完了（添付2件／本文解析対象なし）`
- `メール解析完了／添付本文を利用：2/3件`
- `メールの一部を利用できませんでした。読めた情報で処理を続けます。`

Codex相談の添付案内には次を追加する。

```text
Outlookからドロップしたファイルは一時領域にあります。次の入力を行う前に添付してください。
```

## 3. 技術方式

### 3.1 Outlook仮想ファイルの受け取り

クラシックOutlookは`FileGroupDescriptorW`または`FileGroupDescriptor`と`FileContents`の組で仮想ファイルを渡す。`tkinterdnd2==0.6.2`が公開する定数値は`"FileGroupDescriptorW - FileContents"`等だが、ドラッグ元が提示する型名は素の`FileGroupDescriptorW`であるため、この2定数は登録に使用しない。

また、同梱TkDND 2.10.1の`tkdnd_windows.tcl`では、この2型から`DND_Files`への対応がコメントアウトされている。ネイティブDLLには仮想ファイル実体化処理があるため、`TkinterDnD.require(root)`直後にTcl辞書だけを補正する。

```python
_TKDND_OUTLOOK_TYPES = (
    "namespace eval ::tkdnd::generic {"
    " dict set _platform2tkdnd FileGroupDescriptorW DND_Files;"
    " dict set _platform2tkdnd FileGroupDescriptor DND_Files;"
    " dict set _tkdnd2platform DND_Files"
    " {CF_HDROP FileGroupDescriptorW FileGroupDescriptor}"
    "}"
)

TkinterDnD.require(root)
root.tk.eval(_TKDND_OUTLOOK_TYPES)
input_list.set_dropfile_tempdir(str(workspaces.staging_directory))
input_list.drop_target_register(DND_FILES)
input_list.dnd_bind("<<Drop>>", on_drop)
```

登録は`DND_FILES`1つとし、その展開先に`CF_HDROP`を残す。Tcl補正が失敗した場合はOutlook仮想D&Dを無効として警告し、既存のExplorer D&Dとファイル選択は利用可能な状態を維持する。

### 3.2 一時領域のライフサイクル

新規`outlook_drop.py`に`DropWorkspaceManager`を置く。1つのアプリ実行につき`TemporaryDirectory(prefix="odr-outlook-drop-")`を1つ所有し、その配下に受理済み`current`と次のドロップ用`staging`を作る。

公開インターフェース：

```python
class DropWorkspaceManager:
    @property
    def staging_directory(self) -> Path: ...
    def discard_unreferenced(self, keep: Iterable[Path]) -> None: ...
    def accept_staging(self) -> Path: ...
    def reject_staging(self) -> Path: ...
    def clear_current(self) -> None: ...
    def close(self) -> None: ...
```

TkDNDは仮想ファイルをDrop時ではなくDragEnter時に実体化する。したがって、ドロップせず通過しただけのファイルがstagingへ残り得る。D&D由来の`_accept_files()`入口で、イベントが返した絶対パス集合を`discard_unreferenced()`へ渡し、staging直下のうち集合に含まれないエントリを削除する。

削除条件：

1. 対象を`resolve()`する。
2. 対象が所有一時ルート配下であることを`is_relative_to()`で確認する。
3. `keep`側もstaging配下の実在パスだけを採用する。
4. 外部ローカルファイルを引数から削除できるAPIは作らない。

`accept_staging()`と`reject_staging()`は成功・失敗にかかわらず空の次stagingを作成し、そのパスを返す。旧current削除や昇格に失敗しても、GUIが`set_dropfile_tempdir()`を再設定できる新stagingを確保する。`close()`は一時ルート全体を削除し、複数回呼んでも成功する。

### 3.3 メール1通、送信者、添付の統合

`msg_reader.build_msg_search_terms()`をメール1通の解析境界として維持する。内部型と公開型を次へ拡張する。

```python
@dataclass(frozen=True, slots=True)
class _MsgContent:
    subject: str
    sender_name: str
    body: str
    attachment_file_names: tuple[str, ...]
    attachment_count: int
    subject_available: bool
    sender_available: bool
    body_available: bool
    attachments_available: bool
    saved_document_paths: tuple[Path, ...] = ()

@dataclass(frozen=True, slots=True)
class MsgSearchTerms:
    primary_terms: tuple[str, ...]
    auxiliary_terms: tuple[str, ...]
    sender_key: str | None
    fully_parsed: bool
    body_available: bool
    attachment_count: int
    attachment_document_parsed_count: int
    attachment_document_target_count: int
    warning: str | None
```

`_read_msg_content(msg_path, attachment_directory: Path | None = None)`は`SenderName`を件名等と同じ読み取り境界で取得する。`probe_msg_access()`は`attachment_directory`を渡さず、添付保存も検索語生成も行わない。

`build_msg_search_terms()`だけが`TemporaryDirectory(prefix="odr-msg-attachments-")`を作り、対応文書を`Attachment.SaveAsFile()`で保存する。保存対象は`document_reader.is_supported_document()`が真の添付に限定する。

保存名は`PureWindowsPath(file_name).name`を使ってWindows形式の区切りを除去し、1始まりindexを接頭辞へ付ける。保存直前に`destination.resolve().is_relative_to(attachment_directory.resolve())`を確認する。

```python
safe_name = PureWindowsPath(str(file_name)).name
destination = attachment_directory / f"{index:04d}_{safe_name}"
if not destination.resolve().is_relative_to(attachment_directory.resolve()):
    raise ValueError("添付の保存先が一時領域外です。")
attachment.SaveAsFile(str(destination))
```

`attachment_document_target_count`は保存開始前に確定した「対応拡張子を持つ添付名の件数」とする。`attachment_document_parsed_count`は保存に成功したパスを`build_document_terms()`へ渡した結果の`parsed_count`とする。これにより保存失敗分も母数へ残る。

`DocumentSearchTerms.warning`は`MsgSearchTerms.warning`へ合流しない。添付本文の部分失敗は`parsed/target`件数で表し、`warning`は送信者・件名・本文・添付一覧の取得失敗だけを表す。

送信者キーは次で作る。

```python
def normalize_sender_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if not character.isspace())
```

空文字になった場合は`None`とする。生の送信者表示名は公開型、Audit、ログへ渡さない。

### 3.4 送信者を使うランキング

一般の検索語と送信者照合を混ぜて空白除去を全パスへ適用すると、階層区切りをまたぐ誤一致が増える。そのため送信者専用キーを分離する。

```python
@dataclass(frozen=True, slots=True)
class PreparedFolder:
    year: YearScope
    relative_path: str
    absolute_path: str
    normalized_path: str
    sender_key_path: str

def rank_candidates(
    prepared_folders: Iterable[PreparedFolder],
    settings: Settings,
    primary_terms: Iterable[str],
    auxiliary_terms: Iterable[str] = (),
    *,
    sender_key: str | None = None,
    limit: int | None = None,
) -> tuple[Candidate, ...]: ...
```

`sender_key_path`は相対パスをNFKC・casefoldした後、Unicode空白だけを除去する。`sender_key`が`sender_key_path`に含まれた場合、主検索語一致数へ1を加える。通常主検索語と補助検索語の正規化、年度優先、祖先・兄弟折り畳みは変更しない。

`InputState`へ`sender_key: str | None`を追加し、MSG入力中だけ保持する。`select_files()`、カタログ再読込、検索語編集後の再ランキングでは同じキーを渡し、`reset_manual()`または次の入力で破棄する。送信者は本文と同様の自動情報であり、検索欄へ生表示しない。

### 3.5 検索情報の統合規則

| 入力情報 | 種別 | 備考 |
|---|---|---|
| 仮想MSGのファイル名 | 主検索語 | 既存どおり |
| 件名 | 主検索語 | 既存どおり |
| 送信者表示名 | 送信者主照合 | 空白除去した専用キーで最大1件加点 |
| 全添付ファイル名 | 主検索語 | 既存どおり。一般的なインライン画像名は除外 |
| メール本文 | 補助検索語 | 既存の`clean_msg_body`を使用 |
| 対応添付文書本文 | 補助検索語 | 既存の`build_document_terms`を使用 |

主検索語一致0件の候補を除外する既存規則は、送信者一致を主一致1件として含めた後に適用する。

## 4. 変更ファイルと責務

| ファイル | 変更内容 |
|---|---|
| `onedrive_destination_recommender/src/onedrive_destination_recommender/outlook_drop.py` | 新規。Tcl辞書補正、一時領域の所有、未参照staging削除、昇格・破棄 |
| `onedrive_destination_recommender/src/onedrive_destination_recommender/app.py` | `DND_FILES`登録、D&Dトランザクション、終了cleanup、案内文 |
| `onedrive_destination_recommender/src/onedrive_destination_recommender/msg_reader.py` | `SenderName`取得、送信者キー、対応添付の一時保存、解析件数 |
| `onedrive_destination_recommender/src/onedrive_destination_recommender/terms.py` | `normalize_sender_key()`と送信者用パス正規化 |
| `onedrive_destination_recommender/src/onedrive_destination_recommender/ranking.py` | `sender_key`の主照合1件加点 |
| `onedrive_destination_recommender/src/onedrive_destination_recommender/session.py` | 送信者キーのライフサイクル、添付件数別状態文言 |
| `tests/unit/test_outlook_drop.py` | 新規。一時領域、DragEnter孤児、削除境界、失敗後staging |
| `tests/unit/test_msg_reader.py` | 送信者、添付任意、保存対象、件数、警告、パス安全性、即時削除 |
| `tests/unit/test_terms.py` | 送信者キーの空白・NFKC正規化 |
| `tests/unit/test_ranking.py` | 送信者一致、空白差、同名非区別、既存順位回帰 |
| `tests/unit/test_session.py` | 送信者キー引き渡し、表示文言、複数MSG拒否 |
| `tests/integration/test_gui.py` | Tcl補正、`DND_FILES`登録、成功・失敗処理、Explorer回帰 |
| `README.md` | 操作方法、対象Outlook、一時ファイルの寿命、送信者利用、対象外 |
| `要件定義_OneDrive保存先レコメンダー.md` | 新機能、送信者、受け入れ条件を反映 |

`pyproject.toml`は変更しない。

## 5. 実装タスク

### Task 1: TkDND補正と一時領域

**Files:**
- Create: `onedrive_destination_recommender/src/onedrive_destination_recommender/outlook_drop.py`
- Create: `onedrive_destination_recommender/tests/unit/test_outlook_drop.py`
- Modify: `onedrive_destination_recommender/src/onedrive_destination_recommender/app.py:514`
- Modify: `onedrive_destination_recommender/tests/integration/test_gui.py`

**Interfaces:**
- Produces: `patch_outlook_drop_types(root) -> None`
- Produces: `DropWorkspaceManager`の6インターフェース

- [ ] **Step 1: 一時領域と未参照削除の失敗先行テストを書く**

```python
def test_discard_unreferenced_keeps_only_event_paths(tmp_path: Path) -> None:
    manager = DropWorkspaceManager(base_directory=tmp_path / "owned")
    kept = manager.staging_directory / "kept.msg"
    orphan = manager.staging_directory / "hover-only.msg"
    kept.write_bytes(b"kept")
    orphan.write_bytes(b"orphan")

    manager.discard_unreferenced([kept])

    assert kept.is_file()
    assert not orphan.exists()
```

`accept_staging()`、`reject_staging()`、`clear_current()`、`close()`、外部パス非削除、`accept_staging()`失敗後も新stagingを返すケースを独立テストにする。

- [ ] **Step 2: 対象テストが未実装理由で失敗することを確認する**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_outlook_drop.py -v
```

Expected: `ModuleNotFoundError`または未定義インターフェースによるFAIL。

- [ ] **Step 3: `DropWorkspaceManager`を最小実装して単体テストを通す**

- [ ] **Step 4: Tcl補正と`DND_FILES`単独登録の失敗先行GUIテストを書く**

```python
assert registered_types == ("DND_Files",)
assert "dict set _platform2tkdnd FileGroupDescriptorW DND_Files" in evaluated_scripts
assert "dict set _tkdnd2platform DND_Files" in evaluated_scripts
```

- [ ] **Step 5: Tcl補正、初期化失敗フォールバック、次staging再設定を実装する**

- [ ] **Step 6: Task 1の自動テストを通す**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_outlook_drop.py -v
.\.venv\Scripts\python.exe -m pytest tests\integration\test_gui.py -v -m integration
```

- [ ] **Step 7: クラシックOutlookでGo/No-Go確認する**

機密情報を含まないダミーメール1通とダミー添付1件をD&Dする。Dropイベント発火、返却パス、DragEnter後に離脱した孤児の次回削除を確認する。失敗時はTask 2へ進まず、独自OLE実装を追加せず再レビューする。

- [ ] **Step 8: コミットする**

```powershell
git add onedrive_destination_recommender/src/onedrive_destination_recommender/outlook_drop.py onedrive_destination_recommender/src/onedrive_destination_recommender/app.py onedrive_destination_recommender/tests/unit/test_outlook_drop.py onedrive_destination_recommender/tests/integration/test_gui.py
git commit -m "feat: Outlook仮想ファイルのD&D受け取りを追加"
```

### Task 2: 送信者と任意添付のメール解析

**Files:**
- Modify: `onedrive_destination_recommender/src/onedrive_destination_recommender/msg_reader.py:39`
- Modify: `onedrive_destination_recommender/tests/unit/test_msg_reader.py:29`

**Interfaces:**
- Consumes: `MailItem.SenderName`、`Attachment.SaveAsFile()`、`build_document_terms()`
- Produces: 拡張した`MsgSearchTerms`。生本文・生送信者名・一時パスは返さない

- [ ] **Step 1: 添付なしと送信者取得の失敗先行テストを書く**

```python
content = msg_reader._MsgContent(
    subject="設備更新",
    sender_name="山田 太郎",
    body="確認をお願いします",
    attachment_file_names=(),
    attachment_count=0,
    subject_available=True,
    sender_available=True,
    body_available=True,
    attachments_available=True,
)
monkeypatch.setattr(
    msg_reader,
    "_read_msg_content",
    lambda _path, attachment_directory=None: content,
)
result = msg_reader.build_msg_search_terms(msg_path)
assert result.sender_key == "山田太郎"
assert result.attachment_count == 0
```

送信者取得失敗時に`sender_key is None`で他情報が残るケースも分けて書く。

- [ ] **Step 2: 添付本文統合、母数、即時削除の失敗先行テストを書く**

対応添付3件のうち保存失敗1件・解析成功2件で`parsed=2`、`target=3`となり、`warning`へ文書警告を混ぜないことを固定する。

- [ ] **Step 3: Windows形式の保存名と同名衝突の失敗先行テストを書く**

`PureWindowsPath`で`..\..\設備仕様.docx`をbasename化し、同名2件へ異なるindexが付き、一時ルート外へ保存されないことを確認する。

- [ ] **Step 4: 安全制約テストを分離する**

```python
for module in (msg_reader, terms_module):
    source = inspect.getsource(module)
    assert ".write(" not in source
    assert "write_text(" not in source
    assert "logging" not in source

msg_source = inspect.getsource(msg_reader)
assert msg_source.count(".SaveAsFile(") == 1
assert "TemporaryDirectory(" in msg_source
```

- [ ] **Step 5: テスト失敗を確認する**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_msg_reader.py -v
```

- [ ] **Step 6: 最小実装してMSG・文書テストを通す**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_msg_reader.py tests\unit\test_document_reader.py -v
```

- [ ] **Step 7: コミットする**

```powershell
git add onedrive_destination_recommender/src/onedrive_destination_recommender/msg_reader.py onedrive_destination_recommender/tests/unit/test_msg_reader.py
git commit -m "feat: メール送信者と任意添付を解析"
```

### Task 3: 送信者名による推薦

**Files:**
- Modify: `onedrive_destination_recommender/src/onedrive_destination_recommender/terms.py:86`
- Modify: `onedrive_destination_recommender/src/onedrive_destination_recommender/ranking.py:35`
- Modify: `onedrive_destination_recommender/src/onedrive_destination_recommender/session.py:44`
- Modify: `onedrive_destination_recommender/tests/unit/test_terms.py`
- Modify: `onedrive_destination_recommender/tests/unit/test_ranking.py`
- Modify: `onedrive_destination_recommender/tests/unit/test_session.py`

**Interfaces:**
- Produces: `normalize_sender_key(value: str) -> str`
- Produces: `rank_candidates(..., sender_key: str | None = None)`
- Produces: `InputState.sender_key: str | None`

- [ ] **Step 1: 空白差の正規化テストを書く**

```python
assert normalize_sender_key("山田 太郎") == "山田太郎"
assert normalize_sender_key("山田\u3000太郎") == "山田太郎"
assert normalize_sender_key("山田太郎") == "山田太郎"
```

- [ ] **Step 2: 送信者一致が主照合1件になるランキングテストを書く**

`山田 太郎`フォルダと`山田太郎`フォルダの双方が同じ`sender_key`で一致し、メールアドレス差を入力しないこと、送信者不在では既存順位と同一であることを固定する。

- [ ] **Step 3: セッションの送信者キー保持・破棄テストを書く**

MSG選択、検索語編集、カタログ再読込ではキーが維持され、通常ファイル選択と`reset_manual()`では`None`になることを確認する。

- [ ] **Step 4: 失敗を確認する**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_terms.py tests\unit\test_ranking.py tests\unit\test_session.py -v
```

- [ ] **Step 5: 送信者専用照合を最小実装する**

一般主検索語の正規化は変更せず、送信者キーだけを`sender_key_path`へ照合する。

- [ ] **Step 6: 対象テストを通す**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_terms.py tests\unit\test_ranking.py tests\unit\test_session.py -v
```

- [ ] **Step 7: コミットする**

```powershell
git add onedrive_destination_recommender/src/onedrive_destination_recommender/terms.py onedrive_destination_recommender/src/onedrive_destination_recommender/ranking.py onedrive_destination_recommender/src/onedrive_destination_recommender/session.py onedrive_destination_recommender/tests/unit/test_terms.py onedrive_destination_recommender/tests/unit/test_ranking.py onedrive_destination_recommender/tests/unit/test_session.py
git commit -m "feat: メール送信者を保存先推薦へ利用"
```

### Task 4: メール解析状態と一時パス案内

**Files:**
- Modify: `onedrive_destination_recommender/src/onedrive_destination_recommender/session.py:121`
- Modify: `onedrive_destination_recommender/src/onedrive_destination_recommender/codex_prompt.py`
- Modify: `onedrive_destination_recommender/tests/unit/test_session.py`
- Modify: `onedrive_destination_recommender/tests/unit/test_codex_prompt.py`

**Interfaces:**
- Consumes: 拡張した`MsgSearchTerms`
- Produces: `select_files()`後の利用者向け状態と相談案内

- [ ] **Step 1: `select_files()`経由の状態文言テストを書く**

添付0件、解析対象0件、`2/3件`、メール項目の部分失敗をパラメータ化し、私有関数を直接呼ばず`state.msg_status`を比較する。

- [ ] **Step 2: 一時パスの寿命を伝える相談文テストを書く**

Outlook current配下の入力だけに「次の入力前に添付」の案内が含まれ、通常ローカルファイルでは既存文言を維持する。

- [ ] **Step 3: 失敗を確認し、文言を最小実装する**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_session.py tests\unit\test_codex_prompt.py -v
```

- [ ] **Step 4: 対象テストを通す**

- [ ] **Step 5: コミットする**

```powershell
git add onedrive_destination_recommender/src/onedrive_destination_recommender/session.py onedrive_destination_recommender/src/onedrive_destination_recommender/codex_prompt.py onedrive_destination_recommender/tests/unit/test_session.py onedrive_destination_recommender/tests/unit/test_codex_prompt.py
git commit -m "feat: Outlook入力の解析状態と寿命を表示"
```

### Task 5: GUIの成功・失敗トランザクション

**Files:**
- Modify: `onedrive_destination_recommender/src/onedrive_destination_recommender/app.py:544`
- Modify: `onedrive_destination_recommender/tests/integration/test_gui.py:352`

**Interfaces:**
- Consumes: `DropWorkspaceManager`と`RecommenderSession.select_files()`
- Produces: 成功時だけ状態を置換し、常に次stagingをTkDNDへ設定するGUI動作

- [ ] **Step 1: GUIトランザクションの失敗先行テストを書く**

- D&D入口でイベントにないDragEnter孤児を削除する。
- D&D成功時だけ`accept_staging()`を呼ぶ。
- D&D失敗時は`reject_staging()`を呼び、直前状態を維持する。
- `accept_staging()`自体が失敗しても新stagingを取得・再設定する。
- ファイル選択成功で旧Outlook currentを削除する。
- 手動検索復帰で`clear_current()`を呼ぶ。
- root破棄で`close()`を冪等に呼ぶ。

- [ ] **Step 2: GUIテストの失敗を確認する**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\integration\test_gui.py -v -m integration
```

- [ ] **Step 3: `_accept_files()`へ処理を接続する**

セッション処理成功まで`accept_staging()`を呼ばない。workspace操作失敗は直前入力を壊さず、一般警告を表示する。

- [ ] **Step 4: 案内と処理中状態を更新する**

新しいスレッド、キャンセル機構、進捗画面は追加しない。

- [ ] **Step 5: GUIテストを通す**

- [ ] **Step 6: コミットする**

```powershell
git add onedrive_destination_recommender/src/onedrive_destination_recommender/app.py onedrive_destination_recommender/tests/integration/test_gui.py
git commit -m "feat: Outlook D&Dの入力状態を安全に置換"
```

### Task 6: 安全性回帰、文書、実機確認

**Files:**
- Modify: `onedrive_destination_recommender/tests/unit/test_no_network.py`
- Modify: `onedrive_destination_recommender/tests/unit/test_audit.py`
- Modify: `onedrive_destination_recommender/README.md`
- Modify: `obsidan vault/30_Permanent/★業務/ファイル管理/要件定義_OneDrive保存先レコメンダー.md`
- Modify: 本計画書

- [ ] **Step 1: 送信者・本文・添付本文が永続化されないテストを書く**

固有識別子を合成入力へ入れ、処理後の`catalog.json`、`audit.jsonl`、ログ、相談文に生送信者表示名・本文・添付本文が存在しないことを確認する。相談文には入力パスだけを許可する。

- [ ] **Step 2: 全自動テストと静的検査を実行する**

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest -m integration
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m ruff format --check src tests
```

Expected: 全テストPASS。`ODR_TEST_MSG_PATH`未指定による実MSGテストskipだけ許容する。

- [ ] **Step 3: クラシックOutlook実機確認を行う**

1. 添付なしメール1通
2. 対応文書添付ありメール1通
3. 未対応添付だけを持つメール1通
4. 対応文書と未対応添付が混在するメール1通
5. Outlook添付だけの単一・複数D&D
6. 日本語、空白、括弧、同名を含む添付
7. 空白あり・なしの送信者表示名と送信者フォルダ照合
8. 送信者取得失敗時の継続
9. メール2通拒否と直前状態維持
10. DragEnter後離脱、入力置換、手動復帰、終了後の一時ルート削除
11. Explorer D&D、ファイル選択、Preview、確定・コピーの回帰

- [ ] **Step 4: READMEと要件定義書を更新する**

クラシックOutlook限定、添付任意、メール1通制限、送信者表示名の推薦利用、同名非区別、空白正規化、一時実体の存在期間、新しいOutlook対象外を明記する。

- [ ] **Step 5: 検証結果を本計画書へ記録する**

実行コマンド、PASS件数、skip理由、ruff結果、実機11項目の成否を記録し、未確認項目を完了扱いにしない。

- [ ] **Step 6: 最終コミットする**

```powershell
git add onedrive_destination_recommender/tests/unit/test_no_network.py onedrive_destination_recommender/tests/unit/test_audit.py onedrive_destination_recommender/README.md "obsidan vault/30_Permanent/★業務/ファイル管理/要件定義_OneDrive保存先レコメンダー.md" "obsidan vault/30_Permanent/★業務/ファイル管理/実装計画_Outlook直接D&D.md"
git commit -m "docs: Outlook直接D&Dの検証結果を記録"
```

## 6. 受け入れ条件

1. クラシックOutlookのメール1通を入力欄へ直接D&Dできる。
2. 添付なしメールを、送信者表示名を含むメール情報による1案件として処理できる。
3. 添付ありメールを、メール情報・全添付名・読めた対応文書本文による1案件として処理できる。
4. Outlook添付だけを単一・複数でD&Dし、既存通常ファイル規則で処理できる。
5. `山田 太郎`と`山田太郎`を同じ送信者キーとして候補へ主照合1件を加算できる。
6. 同じ送信者表示名の別人をメールアドレスで区別しない。
7. 送信者を取得できなくても他のメール情報で推薦を継続できる。
8. 日本語、空白、括弧、同名を含む仮想ファイルを内容と順序を壊さず受け取れる。
9. メール2通以上を拒否し、直前の入力、検索語、候補、確定パスを維持できる。
10. メール項目または一部添付を読めない場合も、読めた情報で推薦を継続し、状態を表示できる。
11. D&D初期化に失敗してもアプリを終了せず、Explorer D&Dとファイル選択を利用できる。
12. DragEnterで生成された未参照実体と、失敗したD&Dの実体だけを削除できる。
13. 入力置換、手動復帰、終了時に所有するOutlook一時実体を削除できる。
14. D&DだけではOneDrive、Audit、クリップボード、確定パスを変更しない。
15. メール本文、添付本文、送信者表示名、検索語全文がランタイムファイルとログへ残らない。
16. 既存のExplorer D&D、保存済みMSG、文書本文抽出、候補表示、Preview、確定・コピーが回帰しない。

## 7. 主なリスクと対策

| リスク | 対策 |
|---|---|
| Tcl補正がTkDND内部実装へ依存する | 対象バージョンを固定し、補正文字列と実Outlook Go/No-Goを最初に検証する |
| DragEnterだけで仮想実体が残る | 次のD&D入口でイベント未参照のstaging直下を所有確認後に削除する |
| 仮想ファイル名によるパストラバーサル | `PureWindowsPath.name`、index接頭辞、所有ルート配下検証を行う |
| `accept_staging()`失敗後に次の入力が混在する | 失敗時も空の新stagingを必ず作りTkDNDへ再設定する |
| 対応添付の保存失敗が母数から消える | targetを保存前、parsedを文書解析後に確定する |
| 送信者空白正規化が一般検索へ誤一致を増やす | 一般検索語と分離した`sender_key_path`だけで照合する |
| 送信者情報が永続化される | 生表示名を公開型へ出さず、キーもInputStateの現在入力だけに保持する |
| 一時MSG・添付を相談時に見失う | 次の入力前に添付する案内とREADMEの寿命説明を追加する |
| 異常終了時に一時ファイルが残る | 正常終了cleanupを必須とし、起動時の古いtemp掃除は別計画とする |
| 将来拡張の過剰抽象化 | メール1通の解析境界だけを独立させ、複数案件UIやキューを作らない |

## 8. 初回レビュー指摘の反映表

| 指摘 | 反映先 |
|---|---|
| B-1/B-2 Tclマッピングと型名 | §3.1、Task 1 |
| H-1 DragEnter実体化 | §3.2、Task 1、Task 5 |
| H-2 安全制約テスト | Task 2 Step 4 |
| M-1 引数・既定値 | §3.3、Task 2 Step 1 |
| M-2 Windows basename | §3.3、Task 2 Step 3 |
| M-3 target/parsed定義 | §3.3、Task 2 Step 2 |
| M-4 warning規則 | §3.3、Task 4 |
| M-5 失敗後staging | §3.2、Task 1、Task 5 |
| M-6 一時パス寿命 | §2.4、Task 4、Task 6 |
| M-7 公開動作テスト | Task 4 Step 1 |
| L-1 集計値重複 | 3整数を維持し、単一構築点と整合テストでずれを防止 |
| L-2 app.py対象行 | Task 5で`app.py:544`へ修正 |
| L-3 一時MSGの存在 | Global Constraints、Task 6、README更新 |

## 9. 再レビューで確認してほしい点

1. Tcl辞書補正と`DND_FILES`単独登録が同梱TkDNDで成立するか。
2. `discard_unreferenced()`がDragEnter孤児だけを安全に削除できるか。
3. `accept_staging()`失敗後も新stagingを確保する契約に矛盾がないか。
4. sender_keyを一般検索語から分離し、主照合1件として加算する方式が既存順位と整合するか。
5. 同名非区別、空白差同一、メールアドレス不使用という要件を満たすか。
6. target/parsed、文書warning、メールwarningの境界が部分失敗を正しく表すか。
7. 生送信者名と一時パスを永続化しない境界に漏れがないか。
8. 16件の受け入れ条件に不足・重複がないか。

## 10. 実装開始条件

- Claude Code再レビューのBlockerとHighを解消し、採用しない指摘は理由を記録している。
- クラシックOutlook、Python 3.12、`tkinterdnd2==0.6.2`の実行環境を利用できる。
- 実装前のpytest、Windows GUI結合テスト、ruffの基準値を記録している。
- Global Constraintsと受け入れ条件に未解決の矛盾がない。

## 11. 参考資料

- Microsoft Learn: [Shell Clipboard Formats](https://learn.microsoft.com/en-us/windows/win32/shell/clipboard)
- Microsoft Learn: [MailItem.SenderName property](https://learn.microsoft.com/en-us/office/vba/api/outlook.mailitem.sendername)
- Microsoft Learn: [IDropTarget interface](https://learn.microsoft.com/en-us/windows/win32/api/oleidl/nn-oleidl-idroptarget)
- PyPI: [tkinterdnd2](https://pypi.org/project/tkinterdnd2/)
