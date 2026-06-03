import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from subprocess import Popen

from pipeline.stages.context import StageContext

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
    process: Popen | None = None
    stop_event: threading.Event | None = None
    error_message: str | None = None
    end_time: float | None = None


class ConflictError(Exception):
    pass


MAX_INMEMORY_JOBS = 100


class Scheduler:
    def __init__(self, ws_manager, job_store, project_root: str):
        self._jobs: dict[str, JobInfo] = {}
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self.ws_manager = ws_manager
        self.job_store = job_store
        self.project_root = project_root

    # ── Public API ────────────────────────────────────────────

    async def submit(self, task_name: str, preset: str,
                     config_path: str | None = None) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = JobInfo(job_id=job_id, task_name=task_name, preset=preset,
                       start_time=time.time())
        async with self._lock:
            for existing in self._jobs.values():
                if existing.task_name == task_name and existing.status == JobStatus.RUNNING:
                    raise ConflictError(
                        f"Task '{task_name}' is already running (job {existing.job_id})")
            self._jobs[job_id] = job
            self._evict_if_needed()
        job.asyncio_task = asyncio.create_task(self._run_job(job, config_path))
        return job_id

    def get(self, job_id: str) -> JobInfo | None:
        return self._jobs.get(job_id)

    async def list_running(self) -> list[JobInfo]:
        async with self._lock:
            return [j for j in self._jobs.values() if j.status == JobStatus.RUNNING]

    async def list_all(self) -> list[JobInfo]:
        async with self._lock:
            return list(self._jobs.values())

    async def stop(self, job_id: str) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.RUNNING:
            return False
        # Signal the stop_event so stages checking context.is_stopped() can bail early.
        # Note: executor thread work is cooperative cancellation only —
        # stages must periodically check context.is_stopped() to be responsive.
        if job.stop_event:
            job.stop_event.set()
        if job.process and job.process.poll() is None:
            job.process.terminate()
            try:
                job.process.wait(timeout=5)
            except Exception:
                job.process.kill()
        if job.asyncio_task and not job.asyncio_task.done():
            job.asyncio_task.cancel()
        return True

    def shutdown(self):
        self._executor.shutdown(wait=False)

    @staticmethod
    def _serialize_job(job: JobInfo) -> dict:
        return {
            "job_id": job.job_id, "task_name": job.task_name,
            "preset": job.preset, "start_time": job.start_time,
            "status": job.status.value, "current_stage": job.current_stage,
            "stages_completed": job.stages_completed,
            "error_message": job.error_message, "end_time": job.end_time,
        }

    # ── Job lifecycle state machine ───────────────────────────

    async def _transition_job(self, job: JobInfo, new_status: JobStatus,
                               job_log: logging.Logger | None = None,
                               error_message: str | None = None):
        """Single entry point for job state changes.

        Updates in-memory fields, persists metadata, and broadcasts
        [JOB STATUS] via the job logger so WebSocket clients see the
        terminal state before handlers are closed.
        """
        job.status = new_status
        if new_status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED):
            job.end_time = job.end_time or time.time()
            job.current_stage = None
        if error_message is not None:
            job.error_message = error_message
        await self.job_store.write_metadata(job.job_id, self._serialize_job(job))
        if job_log:
            job_log.info("[JOB %s]", new_status.value.upper())

    # ── Private: job execution ────────────────────────────────

    def _evict_if_needed(self):
        if len(self._jobs) <= MAX_INMEMORY_JOBS:
            return
        terminal = [
            (jid, j) for jid, j in self._jobs.items()
            if j.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED)
        ]
        excess = len(self._jobs) - MAX_INMEMORY_JOBS
        terminal.sort(key=lambda item: item[1].end_time or item[1].start_time)
        for jid, _ in terminal[:excess]:
            del self._jobs[jid]

    async def _run_job(self, job: JobInfo, config_path: str | None):
        from web.log_setup import create_job_logger
        from pipeline.config import load_config
        from pipeline.pipeline import PipelineOrchestrator
        from pipeline.manifest import Manifest

        job_log = create_job_logger(job.job_id, self.ws_manager, self.project_root)
        stop_evt = threading.Event()
        job.stop_event = stop_evt

        try:
            await self._transition_job(job, JobStatus.RUNNING, job_log)

            cfg_path = config_path or "configs/foundationpose.yaml"
            config = load_config(cfg_path)
            orch = PipelineOrchestrator()
            stages = orch.resolve_preset(job.preset)

            manifest_dir = Path(config.output_dir) / config.task
            manifest_path = manifest_dir / "manifest.json"
            manifest = (Manifest.load(str(manifest_path)) if manifest_path.exists()
                        else Manifest(task=config.task, config_path=str(manifest_dir)))

            loop = asyncio.get_running_loop()

            for stage_name in stages:
                if job.status == JobStatus.STOPPED:
                    break
                if manifest.is_stage_done(stage_name):
                    job_log.info("[%s] skip (already done)", stage_name)
                    job.stages_completed.append(stage_name)
                    continue

                job.current_stage = stage_name
                await self.job_store.write_metadata(job.job_id, self._serialize_job(job))
                job_log.info("[%s] starting...", stage_name)

                from pipeline.stages import get_stage
                stage = get_stage(stage_name)
                output_dir = Path(config.output_dir) / config.task / stage_name

                context = StageContext(
                    logger=job_log, job_id=job.job_id, stop_event=stop_evt)

                start = time.time()
                try:
                    result_path = await loop.run_in_executor(
                        self._executor,
                        lambda: stage.run(config, output_dir, context=context))
                    elapsed = time.time() - start
                    manifest.mark_stage_done(stage_name, str(result_path), elapsed)
                    job.stages_completed.append(stage_name)
                    job_log.info("[%s] done (%.1fs)", stage_name, elapsed)
                except Exception:
                    manifest.mark_stage_failed(stage_name)
                    manifest.save(str(manifest_path))
                    raise
                manifest.save(str(manifest_path))

            if job.status != JobStatus.STOPPED:
                await self._transition_job(job, JobStatus.COMPLETED, job_log)

        except asyncio.CancelledError:
            await self._transition_job(job, JobStatus.STOPPED, job_log)
            job_log.warning("Job cancelled by user")
        except Exception as e:
            await self._transition_job(job, JobStatus.FAILED, job_log, error_message=str(e))
            job_log.error("Job failed: %s", e, exc_info=True)
        finally:
            job.stop_event = None
            for h in list(job_log.handlers):
                h.close()
                job_log.removeHandler(h)
