import os
import sys

# Ensure UTF-8 output on Windows terminal so emojis and Unicode symbols don't crash
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def log(message: str) -> None:
    """Prints debug log messages in grey when DEBUG=True in .env."""
    debug_mode = os.getenv("DEBUG", "False").strip().lower()
    if debug_mode in ("true", "1", "yes"):
        try:
            print(f"\033[90m{message}\033[0m")
        except UnicodeEncodeError:
            print(f"\033[90m{message.encode('ascii', 'replace').decode('ascii')}\033[0m")
