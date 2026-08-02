from onedrive_destination_recommender.app import create_main_window


def main() -> None:
    """Run the desktop application."""
    root = create_main_window()
    root.mainloop()
