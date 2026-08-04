import logging

from onedrive_destination_recommender.app import create_main_window

_QUIET_DEPENDENCY_LOGGERS = ("pypdf", "openpyxl")


def _silence_dependency_diagnostics() -> None:
    """Keep third-party parser diagnostics, which may quote file bytes, off the console."""
    for name in _QUIET_DEPENDENCY_LOGGERS:
        dependency_logger = logging.getLogger(name)
        dependency_logger.addHandler(logging.NullHandler())
        dependency_logger.propagate = False


def main() -> None:
    """Run the desktop application."""
    _silence_dependency_diagnostics()
    root = create_main_window()
    root.mainloop()
