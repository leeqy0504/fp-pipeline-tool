import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class StageContext:
    logger: logging.Logger | None = None
    job_id: str | None = None
    stop_event: threading.Event | None = None
    progress_callback: Callable[[int, int, str], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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
