"""
Phase 4 Integration Smoke Test

Checklist (from plan):
  1. Start server with WEB_PASSWORD
  2. Login — wrong password rejected, correct password sets cookie
  3. GET /api/tasks, /api/system, /api/configs
  4. POST /api/tasks/{name}/run → job created
  5. GET /api/jobs/{id} → status query
  6. Verify metadata.json + run.log on disk
  7. Verify [JOB RUNNING] + [JOB FAILED] in log
  8. POST /api/tasks/{name}/reset → manifest deleted
  9. GET /api/history → list + delete

Run:  python web/smoke_test.py
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
from httpx import AsyncHTTPTransport

BASE = "http://127.0.0.1:8000"
PROJECT = Path("/home/ubt2204/code/pipeline-tool")

pass_count = 0
fail_count = 0


def check(desc, ok, detail=""):
    global pass_count, fail_count
    if ok:
        print(f"  PASS: {desc}")
        pass_count += 1
    else:
        print(f"  FAIL: {desc}  {detail}")
        fail_count += 1


async def main():
    global pass_count, fail_count
    transport = httpx.AsyncHTTPTransport(proxy=None, retries=0)
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as c:

        # ── 1. Login: wrong password ──
        print("── 1. Login flow ──")
        r = await c.post(f"{BASE}/api/login", json={"password": "wrong"})
        check("wrong password → 401", r.status_code == 401, r.text)

        # ── 2. Login: correct password ──
        r = await c.post(f"{BASE}/api/login", json={"password": "test"})
        check("correct password → 302", r.status_code == 302)
        cookie = r.cookies.get("pipeline_session")
        check("cookie set", cookie is not None)
        if not cookie:
            print("FATAL: no cookie, stopping")
            return
        jar = {"pipeline_session": cookie}

        # ── 3. Logout clears cookie ──
        r = await c.post(f"{BASE}/api/logout")
        check("logout → 302", r.status_code == 302)
        # Login again
        r = await c.post(f"{BASE}/api/login", json={"password": "test"})
        jar = {"pipeline_session": r.cookies.get("pipeline_session")}

        # ── 4. API endpoints (authenticated) ──
        print("\n── 2. Authenticated API ──")
        r = await c.get(f"{BASE}/api/tasks", cookies=jar)
        tasks = r.json()
        check("GET /api/tasks → 200", r.status_code == 200 and isinstance(tasks, list),
              f"status={r.status_code} count={len(tasks) if isinstance(tasks, list) else '?'}")
        task_names = [t["task_name"] for t in tasks] if isinstance(tasks, list) else []

        r = await c.get(f"{BASE}/api/system", cookies=jar)
        sys_data = r.json() if r.status_code == 200 else {}
        check("GET /api/system → 200", r.status_code == 200)
        check("  cpu_percent present", "cpu_percent" in sys_data)
        check("  gpu field present", "gpu" in sys_data)

        r = await c.get(f"{BASE}/api/configs", cookies=jar)
        check("GET /api/configs → 200", r.status_code == 200)

        r = await c.get(f"{BASE}/api/history", cookies=jar)
        check("GET /api/history → 200", r.status_code == 200)

        # ── 5. Page routes ──
        print("\n── 3. Page routes ──")
        pages = [
            ("/login", None, ["PIPELINE UI"]),
            ("/tasks", jar, ["sidebar", "Tasks"]),
            ("/upload", jar, ["Upload"]),
            ("/configs", jar, ["Configs"]),
            ("/history", jar, ["History"]),
        ]
        for path, cookies, keywords in pages:
            r = await c.get(f"{BASE}{path}", cookies=cookies)
            ok = r.status_code == 200
            for kw in keywords:
                if kw.lower() not in r.text.lower():
                    ok = False
            check(f"GET {path} → 200 + content", ok, f"status={r.status_code}")

        if task_names:
            task = task_names[0]
            r = await c.get(f"{BASE}/tasks/{task}", cookies=jar)
            check(f"GET /tasks/{task} → 200", r.status_code == 200)

        # ── 6. Submit a job ──
        print("\n── 4. Job lifecycle ──")
        test_task = "mouse002"
        r = await c.post(
            f"{BASE}/api/tasks/{test_task}/run",
            json={"preset": "foundationpose"},
            cookies=jar,
        )
        data = r.json()
        check("POST run → 200", r.status_code == 200, str(data))
        job_id = data.get("job_id", "")
        check("  job_id returned", len(job_id) == 12)

        # Wait for job to finish
        await asyncio.sleep(1)

        # ── 7. Job status ──
        r = await c.get(f"{BASE}/api/jobs/{job_id}", cookies=jar)
        job = r.json() if r.status_code == 200 else {}
        check("GET /api/jobs/{id} → 200", r.status_code == 200)
        check("  status is failed", job.get("status") == "failed",
              f"actual={job.get('status')}")
        check("  task_name correct", job.get("task_name") == test_task)

        # ── 8. Disk verification ──
        log_dir = PROJECT / "logs" / job_id
        meta_path = log_dir / "metadata.json"
        log_path = log_dir / "run.log"
        check("  metadata.json exists", meta_path.exists())
        check("  run.log exists", log_path.exists())

        if log_path.exists():
            log_text = log_path.read_text()
            check("  [JOB RUNNING] in log", "[JOB RUNNING]" in log_text)
            check("  [JOB FAILED] in log", "[JOB FAILED]" in log_text)
            check("  error traceback in log", "StageError" in log_text or "Error" in log_text)

        # ── 9. Duplicate run protection ──
        print("\n── 5. Conflict protection ──")
        r = await c.post(
            f"{BASE}/api/tasks/{test_task}/run",
            json={"preset": "foundationpose"},
            cookies=jar,
        )
        check("duplicate submit → 200 (allowed after fail)", r.status_code == 200,
              f"status={r.status_code}")

        # ── 10. History detail + delete ──
        print("\n── 6. History ──")
        r = await c.get(f"{BASE}/api/history/{job_id}", cookies=jar)
        hist = r.json() if r.status_code == 200 else {}
        check("GET /api/history/{id} → 200", r.status_code == 200)
        check("  metadata present", "metadata" in hist)
        check("  log present", "log" in hist and len(hist.get("log", "")) > 100)

        r = await c.delete(f"{BASE}/api/history/{job_id}", cookies=jar)
        check("DELETE /api/history/{id} → 200", r.status_code == 200)

        r = await c.get(f"{BASE}/api/history/{job_id}", cookies=jar)
        check("  GET after delete → 404", r.status_code == 404)

        # ── 11. Static files ──
        print("\n── 7. Static files ──")
        r = await c.get(f"{BASE}/static/style.css")
        check("GET /static/style.css → 200", r.status_code == 200)
        check("  CSS contains tokens", "--bg-primary" in r.text)

        print(f"\n{'='*50}")
        print(f"Results: {pass_count} passed, {fail_count} failed "
              f"({pass_count + fail_count} total)")
        print(f"{'ALL PASSED' if fail_count == 0 else 'SOME FAILED'}")


if __name__ == "__main__":
    asyncio.run(main())
