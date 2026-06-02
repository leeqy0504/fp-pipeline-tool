# web/scheduler.py
import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class JobInfo:
    job_id: str
    task_name: str
    preset: str
    start_time: float
    status: JobStatus = JobStatus.PENDING
    current_stage: str | None = None
    stages_completed: list[str] = field(default_factory=list)
    asyncio_task: asyncio.Task | None = None
    error_message: str | None = None
    end_time: float | None = None


class ConflictError(Exception):
    pass


class Scheduler:
    def __init__(self, ws_manager, job_store, project_root: str):
        self._jobs: dict[str, JobInfo] = {}
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self.ws_manager = ws_manager
        self.job_store = job_store
        self.project_root = project_root

    async def submit(self, task_name: str, preset: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = JobInfo(job_id=job_id, task_name=task_name, preset=preset,
                       start_time=time.time(), status=JobStatus.PENDING)
        async with self._lock:
            for existing in self._jobs.values():
                if existing.task_name == task_name and existing.status == JobStatus.RUNNING:
                    raise ConflictError(
                        f"Task '{task_name}' is already running (job {existing.job_id})")
            self._jobs[job_id] = job
        job.asyncio_task = asyncio.create_task(self._demo_job(job))
        return job_id

    async def _demo_job(self, job: JobInfo):
        from web.log_setup import create_job_logger
        job_log = create_job_logger(job.job_id, self.ws_manager, self.project_root)
        try:
            job.status = JobStatus.RUNNING
            await self.job_store.write_metadata(job.job_id, {
                "job_id": job.job_id, "task_name": job.task_name,
                "preset": job.preset, "start_time": job.start_time,
                "status": job.status.value,
            })
            job_log.info("Demo job started — if you see this in browser, the chain works")
            await asyncio.sleep(0.5)
            job_log.info("Step 2: async sleep done, executor thread next")
            loop = asyncio.get_running_loop()

            def _blocking_work():
                job_log.info("Running in executor thread — thread-safe logging verified")
                return "ok"

            await loop.run_in_executor(self._executor, _blocking_work)
            job_log.info("Demo job complete — all 3 handlers worked")
            job.status = JobStatus.COMPLETED
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job_log.error("Demo job failed: %s", e)
        finally:
            job.end_time = time.time()
            await self.job_store.write_metadata(job.job_id, {
                "job_id": job.job_id, "task_name": job.task_name,
                "preset": job.preset, "start_time": job.start_time,
                "status": job.status.value, "end_time": job.end_time,
                "error_message": job.error_message,
            })
            for h in list(job_log.handlers):
                h.close()
                job_log.removeHandler(h)

    def get(self, job_id: str) -> JobInfo | None:
        return self._jobs.get(job_id)

    async def stop(self, job_id: str) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.RUNNING:
            return False
        job.status = JobStatus.STOPPED
        if job.asyncio_task and not job.asyncio_task.done():
            job.asyncio_task.cancel()
        return True

    def shutdown(self):
        self._executor.shutdown(wait=False)
