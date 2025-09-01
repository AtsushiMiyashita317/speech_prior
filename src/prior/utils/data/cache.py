import os
import shutil
import threading


def _async_copy(src, dst):
    def _copy():
        try:
            shutil.copy2(src, dst)
        except Exception:
            with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)
    t = threading.Thread(target=_copy)
    t.start()

# ------------------------------------------------------------
# Disk LRU Cache (simple)
# ------------------------------------------------------------
class DiskLRU:
    """Very simple disk LRU cache for files.

    Files are identified by original absolute path; cached path is under cache_dir/aa/bb/<sha>.
    Eviction is by total size (approximate, bytes).
    """

    def __init__(self, base_dir: str, cache_dir: str, budget_bytes: int = 0) -> None:
        self.base_dir = base_dir
        self.cache_dir = cache_dir
        self.budget = int(budget_bytes) if cache_dir else 0
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def ensure(self, relpath: str) -> str:
        """Ensure file is present in cache. Return cached absolute path (or original if disabled)."""
        abspath = os.path.join(self.base_dir, relpath)
        if not self.budget or not self.cache_dir:
            return abspath
        dst = os.path.join(self.cache_dir, relpath)
        if os.path.exists(dst):
            return dst

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        _async_copy(abspath, dst)

        return abspath
