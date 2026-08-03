import ast
from pathlib import Path

import onedrive_destination_recommender

_NETWORK_MODULES = frozenset(
    {
        "asyncio",
        "ftplib",
        "http",
        "httpx",
        "imaplib",
        "poplib",
        "requests",
        "smtplib",
        "socket",
        "socketserver",
        "ssl",
        "telnetlib",
        "urllib",
        "urllib3",
        "webbrowser",
        "xmlrpc",
    }
)


def _package_sources() -> list[Path]:
    package_root = Path(onedrive_destination_recommender.__file__).parent
    return sorted(package_root.glob("*.py"))


def _imported_roots(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_module_imports_a_network_client() -> None:
    sources = _package_sources()

    assert sources, "package sources were not found"
    for source in sources:
        offending = _imported_roots(source) & _NETWORK_MODULES
        assert not offending, f"{source.name} imports {sorted(offending)}"


def test_delayed_dependency_names_are_limited_to_the_declared_parsers() -> None:
    from onedrive_destination_recommender import document_reader, msg_reader

    delayed: set[str] = set()
    for module, loaders in (
        (document_reader, ("_load_openpyxl", "_load_pypdf")),
        (msg_reader, ("_load_win32com_client",)),
    ):
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in {
                "import_module",
                "_load_optional_module",
            }:
                delayed.update(
                    argument.value
                    for argument in node.args
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                )
        for loader in loaders:
            assert hasattr(module, loader)

    assert delayed == {"openpyxl", "pypdf", "win32com.client"}
