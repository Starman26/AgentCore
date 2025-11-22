from Settings.tools import _summarize_all_chats
from datetime import datetime

if __name__ == "__main__":
    print("Starting daily chat summary process")

    stats = _summarize_all_chats()

    if stats.get('failed', 0) > 0:
        print("\nSome sessions failed.")
        exit(1)
    else:
        print("\nProcess completed")
        exit(0)