import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class RunContext:
    """Run-level environment shared across stages.

    This object is intentionally about execution environment, not stage data
    dependencies. Manifest writes stay in the scheduler/orchestrator.
    """

    run_id: str | None
    task_name: str
    logger: logging.Logger | None = None
    resolved_config_path: Path | None = None
    job_id: str | None = None
    stop_event: threading.Event | None = None
    progress_callback: Callable[[int, int, str], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataContext:
    """Data-flow view for a stage: task data, run data, named inputs, output."""

    task_dir: Path
    run_dir: Path
    output_dir: Path
    inputs: dict[str, Path] = field(default_factory=dict)

    @property
    def output(self) -> Path:
        return self.output_dir

    def input(self, name: str) -> Path:
        return self.inputs[name]

    def get_input(self, name: str) -> Path | None:
        return self.inputs.get(name)


@dataclass
class StageContext:
    run: RunContext | None = None
    data: DataContext | None = None
    stage_name: str | None = None
    logger: logging.Logger | None = None
    job_id: str | None = None
    stop_event: threading.Event | None = None
    progress_callback: Callable[[int, int, str], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.run:
            self.job_id = self.job_id or self.run.job_id or self.run.run_id
            self.stop_event = self.stop_event or self.run.stop_event
            self.progress_callback = self.progress_callback or self.run.progress_callback
            if not self.metadata:
                self.metadata = self.run.metadata
            if self.logger is None and self.run.logger:
                if self.stage_name:
                    self.logger = self.run.logger.getChild(self.stage_name)
                else:
                    self.logger = self.run.logger

    @property
    def output_dir(self) -> Path | None:
        return self.data.output if self.data else None

    @property
    def inputs(self) -> dict[str, Path]:
        return self.data.inputs if self.data else {}

    def input(self, name: str) -> Path:
        if not self.data:
            raise KeyError(name)
        return self.data.input(name)

    def log(self, level: int, msg: str, *args, **kwargs) -> None:
        if self.logger:
            self.logger.log(level, msg, *args, **kwargs)

    def report_progress(self, current: int, total: int, message: str = "") -> None:
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def is_stopped(self) -> bool:
        if self.stop_event:
            return self.stop_event.is_set()
        return False
