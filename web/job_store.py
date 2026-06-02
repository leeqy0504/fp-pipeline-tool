# web/job_store.py
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class JobStore:
    def __init__(self, project_root: str):
        self._logs_dir = Path(project_root) / "logs"

    async def write_metadata(self, job_id: str, data: dict) -> None:
        path = self._logs_dir / job_id / "metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.rename(path)

    async def read_metadata(self, job_id: str) -> dict | None:
        path = self._logs_dir / job_id / "metadata.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    async def mark_failed_on_startup(self) -> None:
        if not self._logs_dir.exists():
            return
        for child in self._logs_dir.iterdir():
            if not child.is_dir():
                continue
            meta = await self.read_metadata(child.name)
            if meta and meta.get("status") == "running":
                meta["status"] = "failed"
                meta["error_message"] = "Server restarted"
                path = self._logs_dir / child.name / "metadata.json"
                path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                logger.warning("Recovery: marked job %s as failed", child.name)

    async def list_jobs(self) -> list[dict]:
        if not self._logs_dir.exists():
            return []
        jobs = []
        for child in sorted(self._logs_dir.iterdir(), reverse=True):
            if child.is_dir():
                meta = await self.read_metadata(child.name)
                if meta:
                    jobs.append(meta)
        return jobs

    async def get_log(self, job_id: str) -> str | None:
        path = self._logs_dir / job_id / "run.log"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    async def delete_job(self, job_id: str) -> bool:
        path = self._logs_dir / job_id
        if not path.exists():
            return False
        import shutil
        shutil.rmtree(path)
        return True
