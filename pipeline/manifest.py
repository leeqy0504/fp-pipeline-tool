"""Manifest: track pipeline stage status and outputs."""

import json
from datetime import datetime, timezone
from pathlib import Path


class Manifest:
    """Tracks per-stage status, output paths, and timing."""

    def __init__(
        self,
        task: str,
        config_path: str,
        created_at: str | None = None,
        metadata: dict | None = None,
    ):
        self.task = task
        self.config_path = config_path
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.stages: dict[str, dict] = {}
        self.metadata = metadata or {}

    def mark_stage_done(self, name: str, output_dir: str, duration_s: float):
        self.stages[name] = {
            "status": "done",
            "output_dir": output_dir,
            "duration_s": duration_s,
        }

    def mark_stage_failed(self, name: str):
        self.stages[name] = {
            "status": "failed",
            "output_dir": None,
            "duration_s": None,
        }

    def mark_stage_skipped(self, name: str):
        self.stages[name] = {
            "status": "skipped",
            "output_dir": None,
            "duration_s": 0,
        }

    def is_stage_done(self, name: str) -> bool:
        s = self.stages.get(name)
        return s is not None and s["status"] == "done"

    def get_output_dir(self, name: str) -> str | None:
        s = self.stages.get(name)
        if s:
            return s.get("output_dir")
        return None

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "config_path": self.config_path,
            "created_at": self.created_at,
            "stages": self.stages,
            "metadata": self.metadata,
        }

    @classmethod
    def load(cls, path: str) -> "Manifest":
        with open(path) as f:
            data = json.load(f)
        m = cls(
            task=data["task"],
            config_path=data["config_path"],
            created_at=data.get("created_at"),
            metadata=data.get("metadata", {}),
        )
        m.stages = data.get("stages", {})
        return m

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
