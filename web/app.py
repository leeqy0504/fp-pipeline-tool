# web/app.py
"""Architecture validation — minimal demo server."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

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
    logger.info("Validation server started — open http://localhost:8080/test")
    yield
    app.state.scheduler.shutdown()
    logger.info("Validation server stopped")


app = FastAPI(title="Pipeline UI Validation", version="0.0.1", lifespan=lifespan)


@app.get("/test")
async def test_page():
    return HTMLResponse("""
    <!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><title>Arch Validation</title>
    <style>
      *{margin:0;padding:0;box-sizing:border-box}
      body{background:#07070c;color:#e4e4ec;font-family:'DM Mono',monospace;padding:40px}
      h2{font-size:22px;font-weight:600;margin-bottom:4px}
      .dim{color:#666;font-size:12px;margin-bottom:20px}
      #log{background:#050510;border:1px solid #1e1e33;padding:16px;max-height:400px;
           overflow-y:auto;font-size:12px;line-height:1.8;margin-bottom:16px;border-radius:8px}
      .log-line{padding:1px 0}
      .green{color:#00d4aa}.amber{color:#f0a030}.red{color:#ff4d5a}.dim{color:#5a5a72}
      button{background:#00d4aa;color:#000;border:none;padding:10px 20px;border-radius:6px;
             cursor:pointer;font-weight:600;font-family:inherit;font-size:13px}
      button:hover{box-shadow:0 0 20px rgba(0,212,170,0.3)}
      button:disabled{opacity:0.4;cursor:not-allowed}
      .status{font-family:inherit;font-size:11px;color:#5a5a72;margin-top:8px}
    </style></head><body>
    <h2>Architecture Validation</h2>
    <p class="dim">Scheduler &rarr; Logger &rarr; WebSocket chain test</p>
    <div id="log"><div class="log-line dim">Waiting for job...</div></div>
    <button id="btn" onclick="startJob()">Start Demo Job</button>
    <div id="status" class="status"></div>
    <script>
    let ws = null;
    function append(msg, cls) {
      const el = document.getElementById('log');
      const c = cls || '';
      el.innerHTML += `<div class="log-line ${c}">${msg}</div>`;
      el.scrollTop = el.scrollHeight;
    }
    async function startJob() {
      document.getElementById('btn').disabled = true;
      document.getElementById('status').textContent = 'Submitting...';
      append('>>> Submitting job...', 'green');
      try {
        const resp = await fetch('/api/tasks/val_test/run', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({preset: 'demo'})
        });
        const data = await resp.json();
        document.getElementById('status').textContent = 'job: ' + data.job_id;
        append('job_id: ' + data.job_id, 'dim');
        ws = new WebSocket(`ws://${location.host}/ws/jobs/${data.job_id}/logs`);
        ws.onmessage = (e) => append(e.data);
        ws.onclose = () => {
          append('--- disconnected ---', 'dim');
          document.getElementById('btn').disabled = false;
          document.getElementById('status').textContent = 'Done. Check logs/ for persisted output.';
        };
        ws.onerror = () => append('WebSocket error!', 'red');
      } catch(e) {
        append('Error: ' + e, 'red');
        document.getElementById('btn').disabled = false;
      }
    }
    </script></body></html>
    """)


from pydantic import BaseModel

class RunReq(BaseModel):
    preset: str = "demo"


@app.post("/api/tasks/{task_name}/run")
async def run_job(task_name: str, request: Request):
    body_data = await request.json()
    body = RunReq(**body_data) if isinstance(body_data, dict) else RunReq(preset="demo")
    scheduler = request.app.state.scheduler
    try:
        job_id = await scheduler.submit(task_name, body.preset)
        return {"job_id": job_id}
    except Exception as e:
        from web.scheduler import ConflictError
        from fastapi.responses import JSONResponse
        if isinstance(e, ConflictError):
            return JSONResponse({"detail": str(e)}, status_code=409)
        return JSONResponse({"detail": str(e)}, status_code=500)


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
