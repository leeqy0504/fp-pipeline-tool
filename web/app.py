import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

EXEMPT_PATHS = {"/login", "/static", "/api/login"}
TEMPLATES_DIR = PROJECT_ROOT / "web" / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from web.ws_manager import ConnectionManager
    from web.job_store import JobStore
    from web.auth import AuthManager, load_password_from_env
    from web.scheduler import Scheduler

    app.state.ws_manager = ConnectionManager()
    app.state.job_store = JobStore(str(PROJECT_ROOT))
    app.state.project_root = PROJECT_ROOT
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

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
app.state.max_body_size = 2 * 1024 * 1024 * 1024  # 2 GB

static_dir = PROJECT_ROOT / "web" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Auth middleware ────────────────────────────────────────────


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


# ── Auth routes ────────────────────────────────────────────────


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


# ── Register API routers ───────────────────────────────────────


from web.routes.tasks import router as tasks_router
from web.routes.jobs import router as jobs_router
from web.routes.configs import router as configs_router
from web.routes.upload import router as upload_router
from web.routes.history import router as history_router
from web.routes.system import router as system_router

app.include_router(tasks_router)
app.include_router(jobs_router)
app.include_router(configs_router)
app.include_router(upload_router)
app.include_router(history_router)
app.include_router(system_router)


# ── WebSocket ──────────────────────────────────────────────────


@app.websocket("/ws/jobs/{job_id}/logs")
async def ws_logs(websocket: WebSocket, job_id: str):
    token = websocket.cookies.get("pipeline_session")
    auth = websocket.app.state.auth
    if not token or not auth.validate_session(token):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    ws_manager = websocket.app.state.ws_manager
    await ws_manager.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(job_id, websocket)
    except Exception:
        await ws_manager.disconnect(job_id, websocket)


# ── Page routes ────────────────────────────────────────────────


def _render(request: Request, template: str, context: dict = None):
    """Starlette >=1.2: TemplateResponse(request, name, context)."""
    return request.app.state.templates.TemplateResponse(request, template, context or {})


@app.get("/login")
async def login_page(request: Request):
    return _render(request, "login.html")


@app.get("/")
async def root():
    return RedirectResponse(url="/tasks", status_code=302)


@app.get("/tasks")
async def tasks_page(request: Request):
    scheduler = request.app.state.scheduler
    job_store = request.app.state.job_store
    output_dir = PROJECT_ROOT / "output"

    all_jobs = await scheduler.list_all()
    running_map = {}
    latest_map = {}
    for j in all_jobs:
        if j.status.value == "running":
            running_map[j.task_name] = j.job_id
        if j.task_name not in latest_map or j.start_time > latest_map[j.task_name][0]:
            latest_map[j.task_name] = (j.start_time, j.job_id)

    disk_jobs = await job_store.list_jobs()
    disk_latest = {}
    for dj in disk_jobs:
        key = dj.get("task_name")
        if not key:
            continue
        st = dj.get("start_time", 0)
        if key not in disk_latest or st > disk_latest[key][0]:
            disk_latest[key] = (st, dj["job_id"])

    tasks = []
    tasks_dir = PROJECT_ROOT / "tasks"
    if tasks_dir.exists():
        for task_dir in sorted(tasks_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            manifest = None
            manifest_path = output_dir / task_name / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            running_job_id = running_map.get(task_name)
            if task_name in latest_map:
                latest_job_id = latest_map[task_name][1]
            elif task_name in disk_latest:
                latest_job_id = disk_latest[task_name][1]
            else:
                latest_job_id = None

            tasks.append({
                "task_name": task_name,
                "manifest": manifest,
                "latest_job_id": latest_job_id,
                "running_job_id": running_job_id,
            })

    return _render(request, "tasks.html", {"tasks": tasks, "page": "tasks"})


@app.get("/tasks/{task_name}")
async def task_detail_page(task_name: str, request: Request):
    tasks_dir = PROJECT_ROOT / "tasks"
    if not (tasks_dir / task_name).is_dir():
        return HTMLResponse("<h1>Task not found</h1>", status_code=404)

    scheduler = request.app.state.scheduler
    job_store = request.app.state.job_store
    output_dir = PROJECT_ROOT / "output"

    manifest = None
    manifest_path = output_dir / task_name / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    running = await scheduler.list_running()
    running_job_id = None
    for j in running:
        if j.task_name == task_name:
            running_job_id = j.job_id
            break

    # Fetch recent jobs for this task from job_store (disk)
    all_disk_jobs = await job_store.list_jobs()
    jobs = [j for j in all_disk_jobs if j.get("task_name") == task_name]

    return _render(request, "task_detail.html", {
        "task_name": task_name,
        "manifest": manifest,
        "running_job_id": running_job_id,
        "jobs": jobs,
        "page": "tasks",
    })


@app.get("/upload")
async def upload_page(request: Request):
    return _render(request, "upload.html", {"page": "upload"})


@app.get("/tasks/{task_name}/pick")
async def pick_points_page(task_name: str, request: Request):
    tasks_dir = PROJECT_ROOT / "tasks" / task_name
    if not tasks_dir.is_dir():
        return HTMLResponse("<h1>Task not found</h1>", status_code=404)
    return _render(request, "pick_points.html", {"task_name": task_name, "page": "tasks"})


@app.get("/configs")
async def configs_page(request: Request):
    return _render(request, "config_editor.html", {"page": "configs"})


@app.get("/history")
async def history_page(request: Request):
    return _render(request, "history.html", {"page": "history"})


# ── HTMX partial endpoints ────────────────────────────────────
# These render HTML fragments for HTMX polling on pages.


@app.get("/_/tasks-list")
async def htmx_tasks_list(request: Request):
    """Return rendered tasks_list.html partial for HTMX polling."""
    scheduler = request.app.state.scheduler
    job_store = request.app.state.job_store
    output_dir = PROJECT_ROOT / "output"

    all_jobs = await scheduler.list_all()
    running_map = {j.task_name: j.job_id for j in all_jobs if j.status.value == "running"}
    latest_map = {}
    for j in all_jobs:
        if j.task_name not in latest_map or j.start_time > latest_map[j.task_name][0]:
            latest_map[j.task_name] = (j.start_time, j.job_id)

    disk_jobs = await job_store.list_jobs()
    disk_latest = {}
    for dj in disk_jobs:
        key = dj.get("task_name")
        if key and (key not in disk_latest or dj.get("start_time", 0) > disk_latest[key][0]):
            disk_latest[key] = (dj.get("start_time", 0), dj["job_id"])

    tasks = []
    tasks_dir = PROJECT_ROOT / "tasks"
    if tasks_dir.exists():
        for task_dir in sorted(tasks_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            manifest = None
            mp = output_dir / task_name / "manifest.json"
            if mp.exists():
                try:
                    manifest = json.loads(mp.read_text(encoding="utf-8"))
                except Exception:
                    pass

            running_job_id = running_map.get(task_name)
            if task_name in latest_map:
                latest_job_id = latest_map[task_name][1]
            elif task_name in disk_latest:
                latest_job_id = disk_latest[task_name][1]
            else:
                latest_job_id = None

            tasks.append({
                "task_name": task_name, "manifest": manifest,
                "latest_job_id": latest_job_id, "running_job_id": running_job_id,
            })

    return _render(request, "tasks_list.html", {"tasks": tasks})
