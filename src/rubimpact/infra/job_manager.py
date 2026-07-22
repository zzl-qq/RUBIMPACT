"""Job folder management (framework S9.2)."""
import shutil
from pathlib import Path


class JobManager:
    def __init__(self, base_dir="."):
        self._base = Path(base_dir)
        self._job_dir = None

    def init(self, job_name: str) -> Path:
        self._job_dir = self._base / job_name
        (self._job_dir / "output" / "history").mkdir(parents=True, exist_ok=True)
        (self._job_dir / "output" / "field").mkdir(parents=True, exist_ok=True)
        (self._job_dir / "tmp").mkdir(parents=True, exist_ok=True)
        return self._job_dir

    @property
    def output_dir(self) -> Path:
        if self._job_dir is None:
            raise RuntimeError("JobManager not initialized. Call init() first.")
        return self._job_dir / "output"

    def save_input_copy(self, content: str) -> None:
        (self._job_dir / "input_copy.inp").write_text(content, encoding="utf-8")

    def cleanup(self) -> None:
        tmp = self._job_dir / "tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
