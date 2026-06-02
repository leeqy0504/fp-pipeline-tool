# web/log_setup.py
import asyncio
import logging
from pathlib import Path


class WebSocketLogHandler(logging.Handler):
    def __init__(self, ws_manager, job_id: str):
        super().__init__()
        self._ws = ws_manager
        self._job_id = job_id
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(
                lambda m=msg: asyncio.create_task(
                    self._ws.broadcast(self._job_id, m)))
        except RuntimeError:
            pass
        except Exception:
            pass


def create_job_logger(job_id: str, ws_manager, project_root: str) -> logging.Logger:
    log_dir = Path(project_root) / "logs" / job_id
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"pipeline.job.{job_id}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"))
    logger.addHandler(fh)

    wh = WebSocketLogHandler(ws_manager, job_id)
    wh.setLevel(logging.INFO)
    logger.addHandler(wh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("[%(levelname)-5s] %(message)s"))
    logger.addHandler(ch)

    return logger
