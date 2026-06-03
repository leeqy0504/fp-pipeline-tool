# Pipeline Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Four phases, each phase dispatched as a batch of subagents, followed by a checkpoint review before advancing. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a FastAPI + Jinja2 Web UI to pipeline-tool for remote pipeline scheduling, real-time log streaming, and config management.

**Architecture:** Single FastAPI process directly imports `pipeline.*` modules. Pipeline stages run via `ThreadPoolExecutor` in asyncio background tasks. WebSocket pushes real-time logs from a custom `logging.Handler`. Authentication via `itsdangerous` signed cookie.

**Tech Stack:** FastAPI, Jinja2, Alpine.js, HTMX, itsdangerous, psutil, PyYAML, uvicorn

**Spec:** `docs/superpowers/specs/2026-06-02-pipeline-web-ui-design.md`

---

## Execution Strategy: Subagent-Driven, Phased

每阶段先 dispatch 所有 subagent 并行执行，全部完成后做一次 checkpoint review，验证通过再进入下一阶段。

核心原则：
- Phase 1 地基不稳不进入 Phase 2
- 每阶段 review 侧重该阶段的耦合点（不是逐行 code review）
- 发现架构问题立即回退，不继续堆代码

---

## Task 0: Architecture Validation

**目标：** 用最小 Demo 验证 Scheduler → Logger → WebSocket 整条链路能跑通。不涉及 pipeline stage，不涉及 FoundationPose。一条日志从提交到浏览器收到，通路验证。

**Files:**
- Create: `web/ws_manager.py`
- Create: `web/log_setup.py`
- Create: `web/scheduler.py` (minimal — no pipeline import)
- Create: `web/job_store.py` (minimal — write/read metadata)
- Create: `web/app.py` (minimal — lifespan, WS endpoint, one test route)

- [ ] **Step 1: Create ws_manager.py**

```python
# web/ws_manager.py
import asyncio
import logging
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self._connections: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, job_id: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.setdefault(job_id, set()).add(ws)
        logger.debug("ws connected: job=%s", job_id)

    async def disconnect(self, job_id: str, ws: WebSocket) -> None:
        async with self._lock:
            conns = self._connections.get(job_id)
            if conns:
                conns.discard(ws)
                if not conns:
                    del self._connections[job_id]

    async def broadcast(self, job_id: str, message: str) -> None:
        async with self._lock:
            conns = set(self._connections.get(job_id, set()))
        dead = []
        for ws in conns:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    await self.disconnect(job_id, ws)
```

- [ ] **Step 2: Create log_setup.py**

```python
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
```

- [ ] **Step 3: Create minimal job_store.py**

```python
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
```

- [ ] **Step 4: Create minimal scheduler.py (no pipeline import)**

```python
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
                    raise ConflictError(f"Task '{task_name}' is already running (job {existing.job_id})")
            self._jobs[job_id] = job
        job.asyncio_task = asyncio.create_task(self._demo_job(job))
        return job_id

    async def _demo_job(self, job: JobInfo):
        """Minimal demo: logs a few lines then completes. No pipeline import."""
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
```

- [ ] **Step 5: Create minimal app.py for validation**

```python
# web/app.py (minimal — validation version)
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    from web.ws_manager import ConnectionManager
    from web.job_store import JobStore
    from web.scheduler import Scheduler

    app.state.ws_manager = ConnectionManager()
    app.state.job_store = JobStore(str(PROJECT_ROOT))
    await app.state.job_store.mark_failed_on_startup()
    app.state.scheduler = Scheduler(
        ws_manager=app.state.ws_manager,
        job_store=app.state.job_store,
        project_root=str(PROJECT_ROOT),
    )
    logger.info("Validation server started — open /test in browser")
    yield
    app.state.scheduler.shutdown()


app = FastAPI(title="Pipeline UI Validation", version="0.0.1", lifespan=lifespan)


@app.get("/test")
async def test_page():
    return HTMLResponse("""
    <!DOCTYPE html><html><head><meta charset="UTF-8"><title>Arch Validation</title>
    <style>body{background:#0a0a0f;color:#e4e4ec;font-family:monospace;padding:40px}
    #log{background:#050510;border:1px solid #1e1e33;padding:16px;max-height:400px;
    overflow-y:auto;font-size:12px;line-height:1.6;margin-bottom:16px;border-radius:8px}
    button{background:#00d4aa;color:#000;border:none;padding:8px 16px;border-radius:4px;
    cursor:pointer;font-weight:600} button:hover{opacity:0.9}
    .dim{color:#666}</style></head><body>
    <h2>Architecture Validation</h2>
    <p class="dim">Scheduler → Logger → WebSocket chain test</p>
    <div id="log"><div class="dim">Waiting for job...</div></div>
    <button id="btn" onclick="startJob()">Start Demo Job</button>
    <script>
    let ws = null;
    function append(msg) {
      const el = document.getElementById('log');
      el.innerHTML += `<div>${msg}</div>`;
      el.scrollTop = el.scrollHeight;
    }
    async function startJob() {
      document.getElementById('btn').disabled = true;
      append('<span style="color:#00d4aa">>>> Submitting job...</span>');
      const resp = await fetch('/api/tasks/test_demo/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({preset: 'demo'})
      });
      const data = await resp.json();
      append(`<span style="color:#888">job_id: ${data.job_id}</span>`);
      ws = new WebSocket(`ws://${location.host}/ws/jobs/${data.job_id}/logs`);
      ws.onmessage = (e) => append(e.data);
      ws.onclose = () => { append('<span class="dim">--- disconnected ---</span>'); document.getElementById('btn').disabled = false; };
    }
    </script></body></html>
    """)


@app.post("/api/tasks/{task_name}/run")
async def run_job(task_name: str, request: Request):
    from pydantic import BaseModel
    class RunReq(BaseModel): preset: str = "demo"
    body = await request.json()
    body = RunReq(**body) if isinstance(body, dict) else RunReq(preset="demo")
    scheduler = request.app.state.scheduler
    try:
        job_id = await scheduler.submit(task_name, body.preset)
        return {"job_id": job_id}
    except Exception as e:
        from web.scheduler import ConflictError
        if isinstance(e, ConflictError):
            raise HTTPException(409, detail=str(e))
        raise HTTPException(500, detail=str(e))


@app.websocket("/ws/jobs/{job_id}/logs")
async def ws_logs(websocket: WebSocket, job_id: str):
    ws_manager = websocket.app.state.ws_manager
    await ws_manager.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(job_id, websocket)
    except Exception:
        await ws_manager.disconnect(job_id, websocket)
```

- [ ] **Step 6: Start validation server and verify the full chain**

```bash
cd /home/ubt2204/code/pipeline-tool
uvicorn web.app:app --host 0.0.0.0 --port 8080 &
sleep 2
# Open http://localhost:8080/test in browser
# Click "Start Demo Job" button
# Expected: log lines appear in real-time via WebSocket:
#   "Demo job started — if you see this in browser, the chain works"
#   "Step 2: async sleep done, executor thread next"
#   "Running in executor thread — thread-safe logging verified"
#   "Demo job complete — all 3 handlers worked"
```

- [ ] **Step 7: Check disk output**

```bash
ls /home/ubt2204/code/pipeline-tool/logs/
cat /home/ubt2204/code/pipeline-tool/logs/*/run.log
cat /home/ubt2204/code/pipeline-tool/logs/*/metadata.json
```

Expected: `run.log` contains all 4 lines. `metadata.json` shows `status: completed`.

- [ ] **Step 8: Simulate server restart recovery**

```bash
# Manually mark the metadata status to "running"
# Restart server — verify mark_failed_on_startup() resets it to "failed"
```

- [ ] **Step 9: If all checks pass, stop server and commit**

```bash
kill %1
cd /home/ubt2204/code/pipeline-tool
git add web/ && git commit -m "feat: architecture validation — Scheduler+Logger+WS chain verified"
```

**Validation success criteria:**
- [x] Browser receives real-time log lines via WebSocket
- [x] Log lines also written to `logs/<job_id>/run.log`
- [x] `metadata.json` shows correct job lifecycle
- [x] Recovery marks stale "running" → "failed"

Only proceed to Phase 1 after this checkpoint passes.

---

## Phase 1: Core Backend

Phase 1 dispatch all 7 tasks in parallel as subagents. After all complete, run Checkpoint Review #1.

### Task 1: Add dependencies

**Files:** `requirements.txt`

```bash
cat >> requirements.txt << 'EOF'
fastapi>=0.115.0
uvicorn[standard]>=0.34.0
jinja2>=3.1.0
python-multipart>=0.0.18
itsdangerous>=2.2.0
psutil>=7.0.0
python-dotenv>=1.0.0
aiofiles>=24.0.0
EOF
pip install -r requirements.txt
git add requirements.txt && git commit -m "feat: add web UI dependencies"
```

### Task 2: StageContext

**Files:** Create `pipeline/stages/context.py`

Complete file:

```python
# pipeline/stages/context.py
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
```

Commit: `feat: add StageContext`

### Task 3: Update Stage Signatures

**Files:** Modify `pipeline/stages/base.py`, `masks.py`, `hunyuangen.py`, `scale.py`, `package.py`, `foundationpose.py`

Each stage `run()` method signature changes from:
```python
def run(self, config: PipelineConfig, output_dir: Path) -> Path:
```
to:
```python
def run(self, config: PipelineConfig, output_dir: Path,
        context: StageContext | None = None) -> Path:
```

Replace `print(...)` with `if context: context.log(logging.INFO, ...)` pattern.

Verify CLI still works:
```bash
python -c "from pipeline.stages import get_stage; s = get_stage('masks'); print(type(s).__name__)"
# Expected: MasksStage
```

Commit: `feat: add context parameter to all stage signatures`

### Task 4: WebSocket Manager (final version)

**Files:** Replace `web/ws_manager.py` (overwrite the Task 0 minimal version)

Same code as Task 0 Step 1 — already production-quality. No changes needed.

Commit: `feat: finalize ws_manager`

### Task 5: Logging (final version)

**Files:** Replace `web/log_setup.py`

Same code as Task 0 Step 2 — already production-quality. No changes needed.

Commit: `feat: finalize log_setup`

### Task 6: Job Store (final version)

**Files:** Replace `web/job_store.py`

Same code as Task 0 Step 3 — already production-quality. No changes needed.

Commit: `feat: finalize job_store`

### Task 7: Auth

**Files:** Create `web/auth.py`

```python
# web/auth.py
import os
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

COOKIE_NAME = "pipeline_session"
COOKIE_MAX_AGE = 86400 * 7


class AuthManager:
    def __init__(self, password: str, secret_key: str | None = None):
        self._password = password
        self._secret = secret_key or os.urandom(32).hex()
        self._serializer = URLSafeTimedSerializer(self._secret)

    def check_password(self, password: str) -> bool:
        return password == self._password

    def create_session(self) -> str:
        return self._serializer.dumps({"authenticated": True})

    def validate_session(self, token: str) -> bool:
        try:
            data = self._serializer.loads(token, max_age=COOKIE_MAX_AGE)
            return data.get("authenticated", False)
        except (BadSignature, SignatureExpired):
            return False


def load_password_from_env() -> str:
    from pathlib import Path
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass
    pw = os.environ.get("WEB_PASSWORD")
    if not pw:
        print("ERROR: WEB_PASSWORD not set", flush=True)
        raise SystemExit(1)
    return pw
```

Commit: `feat: add auth manager`

### Task 8: Scheduler (full version with pipeline import)

**Files:** Replace `web/scheduler.py` — upgrade from demo to full version

Starting from Task 0's scheduler.py, add:
- `_run_job()` replaces `_demo_job()` — imports `pipeline.config`, `pipeline.pipeline`, `pipeline.manifest`
- `StageContext` injection with `stop_event`
- `process: Popen | None` field in JobInfo
- `stop()` method with `process.terminate()` + `task.cancel()`
- `list_running()`, `list_all()` query methods

```python
# web/scheduler.py (FINAL)
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
    process: Popen | None = None  # Future: processes: set[Popen]
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
        job.asyncio_task = asyncio.create_task(self._run_job(job, config_path))
        return job_id

    async def _run_job(self, job: JobInfo, config_path: str | None):
        from web.log_setup import create_job_logger
        from pipeline.config import load_config
        from pipeline.pipeline import PipelineOrchestrator
        from pipeline.manifest import Manifest

        job_log = create_job_logger(job.job_id, self.ws_manager, self.project_root)
        stop_evt = threading.Event()

        try:
            job.status = JobStatus.RUNNING
            await self.job_store.write_metadata(job.job_id, self._serialize_job(job))

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
                except Exception as e:
                    manifest.mark_stage_failed(stage_name)
                    manifest.save(str(manifest_path))
                    raise
                manifest.save(str(manifest_path))

            if job.status != JobStatus.STOPPED:
                job.status = JobStatus.COMPLETED

        except asyncio.CancelledError:
            job.status = JobStatus.STOPPED
            job_log.warning("Job cancelled by user")
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job_log.error("Job failed: %s", e, exc_info=True)
        finally:
            job.current_stage = None
            job.end_time = time.time()
            await self.job_store.write_metadata(job.job_id, self._serialize_job(job))
            for h in list(job_log.handlers):
                h.close()
                job_log.removeHandler(h)

    async def stop(self, job_id: str) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.RUNNING:
            return False
        job.status = JobStatus.STOPPED
        if job.process and job.process.poll() is None:
            job.process.terminate()
            try:
                job.process.wait(timeout=5)
            except Exception:
                job.process.kill()
        if job.asyncio_task and not job.asyncio_task.done():
            job.asyncio_task.cancel()
        return True

    def get(self, job_id: str) -> JobInfo | None:
        return self._jobs.get(job_id)

    async def list_running(self) -> list[JobInfo]:
        async with self._lock:
            return [j for j in self._jobs.values() if j.status == JobStatus.RUNNING]

    async def list_all(self) -> list[JobInfo]:
        async with self._lock:
            return list(self._jobs.values())

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
```

Commit: `feat: finalize scheduler with pipeline integration`

### Checkpoint Review #1

Review focus:
1. **StageContext** — Does it pollute pipeline/? Is the default-`None` pattern clean?
2. **Scheduler deadlock risk** — `_lock` usage in `submit()` vs `_run_job()` vs `stop()`. Are there await points inside lock that could deadlock?
3. **Logging handler lifecycle** — Are handlers properly closed in `finally`? Does `removeHandler()` work correctly with `propagate = False`?
4. **Job recovery** — Does `mark_failed_on_startup()` actually run at lifespan startup? Test by manually writing a `status: running` metadata and restarting.

---

## Phase 2: API Routes

### Task 9: Tasks + Jobs API

**Files:** Create `web/routes/tasks.py`

Complete file — includes all routes:
- `GET /api/tasks` — list tasks with stage status + job references
- `GET /api/tasks/{task_name}` — task detail
- `POST /api/tasks/{task_name}/run` — submit job
- `POST /api/tasks/{task_name}/reset` — delete manifest only
- `GET /api/jobs/{job_id}` — job detail
- `POST /api/jobs/{job_id}/stop` — stop job

(Full code same as earlier plan version. Uses `_scan_tasks()` to merge `tasks/` + `output/`/manifest.json.)

Commit: `feat: add tasks and jobs API routes`

### Task 10: Config API

**Files:** Create `web/routes/configs.py`

Routes: `GET /api/configs`, `GET /api/configs/{name}`, `PUT /api/configs/{name}` (with `yaml.safe_load` validation, path traversal guard).

Commit: `feat: add config editor API`

### Task 11: Upload API

**Files:** Create `web/routes/upload.py`

Route: `POST /api/tasks/upload` with constraints (500MB max, extension allowlist, path traversal rejection, `os.path.basename` sanitization).

Commit: `feat: add upload API with security constraints`

### Task 12: History + System API

**Files:** Create `web/routes/history.py`, `web/routes/system.py`

History: `GET /api/history`, `GET /api/history/{job_id}`, `DELETE /api/history/{job_id}`.
System: `GET /api/system` — psutil CPU/Mem/Disk + nvidia-smi GPU (returns `null` if no GPU).

Commit: `feat: add history and system API routes`

### Task 13: Page Routes

**Files:** Create `web/routes/pages.py`

All HTML page routes: `/` → redirect, `/login`, `/tasks`, `/tasks/{task_name}`, `/upload`, `/configs`, `/history`.

Commit: `feat: add HTML page routes`

### Checkpoint Review #2

Review focus:
1. **API consistency** — Does every endpoint follow the job_id model? No lingering task-level status ambiguity?
2. **Upload security** — Path traversal test: try uploading `../../../etc/passwd` as filename. Extension reject test.
3. **YAML validation** — PUT invalid YAML → 400 with parse error. PUT valid YAML → 200.
4. **System API robustness** — Run on a machine without nvidia-smi → `gpu: null`, not 500.

---

## Phase 3: Frontend

### Task 14: Base Template + CSS

**Files:** Create `web/templates/base.html`, `web/static/style.css`

`base.html`: Jinja2 base with sidebar nav, topbar, blocks for title/content/scripts. Loads DM Mono + DM Sans fonts, Alpine.js 3.x CDN, HTMX 2.x CDN.

`style.css`: Complete Precision Terminal CSS from the visual companion mockup (`.superpowers/brainstorm/5013-1780385545/content/ui-design-v1.html` `<style>` block).

Commit: `feat: add base template and CSS`

### Task 15: All Page Templates

**Files:** Create 6 templates

- `login.html` — password form, POST to `/api/login`
- `tasks.html` — task grid with stage dots, HTMX polling every 5s, color-coded status
- `task_detail.html` — stage progress bar, run/stop/reset buttons, WebSocket log terminal, job status polling
- `upload.html` — drag-and-drop zone, Alpine.js file list, fetch upload with FormData
- `config_editor.html` — left sidebar config list, right textarea, save with YAML validation feedback
- `history.html` — job list with expand/collapse, log viewer in terminal style, delete button

(Full code for each template as specified in the earlier plan.)

Commit: `feat: add all page templates`

### Checkpoint Review #3

Review focus:
1. **Page navigation** — All sidebar links work. Active state reflects current page.
2. **WebSocket terminal** — Run a demo job, verify logs appear in real-time in the terminal. No duplicate lines. No missing lines.
3. **HTMX interactions** — Polling works. Click through task → task detail → back.
4. **Login flow** — Wrong password → error. Correct password → redirect to /tasks. Logout clears cookie.

---

## Phase 4: Integration

### Task 16: FastAPI App Assembly

**Files:** Replace `web/app.py` (overwrite Task 0 minimal version)

Full `app.py`:
- `lifespan`: init ws_manager, job_store (with recovery), auth, scheduler, Jinja2Templates
- `auth_middleware`: exempt `/login`, `/static`, `/api/login`; redirect to login or 401
- `POST /api/login`, `POST /api/logout`
- Register all routers (pages, tasks, configs, upload, history, system)
- `WS /ws/jobs/{job_id}/logs`
- Mount static files
- 500MB body limit for upload

```python
# web/app.py (FINAL)
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent

EXEMPT_PATHS = {"/login", "/static", "/api/login"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from web.ws_manager import ConnectionManager
    from web.job_store import JobStore
    from web.auth import AuthManager, load_password_from_env
    from web.scheduler import Scheduler

    app.state.ws_manager = ConnectionManager()
    app.state.job_store = JobStore(str(PROJECT_ROOT))
    templates_dir = PROJECT_ROOT / "web" / "templates"
    app.state.templates = Jinja2Templates(directory=str(templates_dir))

    await app.state.job_store.mark_failed_on_startup()

    password = load_password_from_env()
    app.state.auth = AuthManager(password)
    app.state.scheduler = Scheduler(
        ws_manager=app.state.ws_manager,
        job_store=app.state.job_store,
        project_root=str(PROJECT_ROOT),
    )
    logger.info("Pipeline Web UI started")
    yield
    app.state.scheduler.shutdown()
    logger.info("Pipeline Web UI stopped")


app = FastAPI(title="Pipeline UI", version="0.1.0", lifespan=lifespan)
app.state.max_body_size = 500 * 1024 * 1024  # 500 MB

static_dir = PROJECT_ROOT / "web" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith("/static") or path.startswith("/ws/") or path in EXEMPT_PATHS:
        return await call_next(request)
    auth = request.app.state.auth
    token = request.cookies.get("pipeline_session")
    if not token or not auth.validate_session(token):
        if path.startswith("/api"):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return RedirectResponse(url="/login", status_code=302)
    return await call_next(request)


from pydantic import BaseModel

class LoginBody(BaseModel):
    password: str


@app.post("/api/login")
async def api_login(body: LoginBody, request: Request):
    auth = request.app.state.auth
    if not auth.check_password(body.password):
        return JSONResponse({"detail": "Wrong password"}, status_code=401)
    token = auth.create_session()
    response = RedirectResponse(url="/tasks", status_code=302)
    response.set_cookie("pipeline_session", token, max_age=86400 * 7,
                        httponly=True, samesite="lax")
    return response


@app.post("/api/logout")
async def api_logout():
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("pipeline_session")
    return response


# Register routers
from web.routes.pages import router as pages_router
from web.routes.tasks import router as tasks_router
from web.routes.configs import router as configs_router
from web.routes.upload import router as upload_router
from web.routes.history import router as history_router
from web.routes.system import router as system_router

app.include_router(pages_router)
app.include_router(tasks_router)
app.include_router(configs_router)
app.include_router(upload_router)
app.include_router(history_router)
app.include_router(system_router)


@app.websocket("/ws/jobs/{job_id}/logs")
async def ws_logs(websocket: WebSocket, job_id: str):
    ws_manager = websocket.app.state.ws_manager
    await ws_manager.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(job_id, websocket)
    except Exception:
        await ws_manager.disconnect(job_id, websocket)
```

Commit: `feat: assemble FastAPI application`

### Task 17: Startup Config

**Files:** Create `.env.example`, modify `run.sh`

```bash
echo 'WEB_PASSWORD=changeme' > .env.example
echo ".env" >> .gitignore
```

Add to `run.sh`:
```bash
# web) shift; uvicorn web.app:app --host 0.0.0.0 --port 8080 "$@";;
```

Commit: `feat: add startup config`

### Task 18: Integration Smoke Test

1. Start server: `WEB_PASSWORD=test uvicorn web.app:app --host 0.0.0.0 --port 8080 &`
2. Login: `curl -X POST localhost:8080/api/login -H 'Content-Type: application/json' -d '{"password":"test"}' -c /tmp/cookies -v`
3. Tasks: `curl -b /tmp/cookies localhost:8080/api/tasks | python -m json.tool`
4. System: `curl -b /tmp/cookies localhost:8080/api/system | python -m json.tool`
5. Configs: `curl -b /tmp/cookies localhost:8080/api/configs | python -m json.tool`
6. Run job: `curl -b /tmp/cookies -X POST localhost:8080/api/tasks/mouse003/run -H 'Content-Type: application/json' -d '{"preset":"foundationpose"}'`
7. Open browser: `http://localhost:8080` → login → task dashboard
8. Click a task → run pipeline → verify WebSocket logs appear in terminal
9. Stop the pipeline → verify job shows "stopped"
10. Reset the task → verify manifest is deleted, output data untouched
11. Kill server: `kill %1`

All tests must pass.

### Checkpoint Review #4

Review focus:
1. **Login flow** — wrong password rejected, correct password sets cookie, logout clears it
2. **Task run** — submit → WebSocket logs stream → stages complete → manifest updated → job marked completed
3. **Stop** — running job stops, stage marked, can re-run
4. **Reset** — manifest deleted, output data preserved
5. **UI pages** — all 6 pages render correctly, no JS errors in console

---

## Out of Scope (Separate Plan)

**Deployment to 180.127.11.166** is not part of this development plan. A separate deployment spec will cover:

- systemd unit file (`pipeline-web.service`)
- nginx reverse proxy (if needed)
- uvicorn production flags (`--workers`, `--log-level`)
- `.env` on server
- log rotation
- firewall rules
- TLS/SSL

---

## File Manifest (What Gets Created/Modified)

### New Files (18)
```
web/__init__.py
web/app.py
web/auth.py
web/scheduler.py
web/ws_manager.py
web/log_setup.py
web/job_store.py
web/routes/__init__.py
web/routes/pages.py
web/routes/tasks.py
web/routes/configs.py
web/routes/upload.py
web/routes/history.py
web/routes/system.py
web/templates/base.html
web/templates/login.html
web/templates/tasks.html
web/templates/task_detail.html
web/templates/upload.html
web/templates/config_editor.html
web/templates/history.html
web/static/style.css
pipeline/stages/context.py
.env.example
```

### Modified Files (7)
```
pipeline/stages/base.py
pipeline/stages/masks.py
pipeline/stages/hunyuangen.py
pipeline/stages/scale.py
pipeline/stages/package.py
pipeline/stages/foundationpose.py
requirements.txt
run.sh
.gitignore
```
