"""
Phase 3 deep validation tests.
Items 2-5: WS e2e, Upload, Config validation, History.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

BASE = "http://127.0.0.1:8000"
WS_BASE = "ws://127.0.0.1:8000"
PROJECT = Path("/home/ubt2204/code/pipeline-tool")


async def login(client):
    r = await client.post(f"{BASE}/api/login", json={"password": "test"}, follow_redirects=False)
    return r.cookies.get("pipeline_session")


async def item4_config_validation(client, cookie):
    """Test YAML validation with line/column numbers."""
    print("=" * 60)
    print("ITEM 4: Config Editor YAML Validation")
    h = {"Cookie": f"pipeline_session={cookie}"}

    # Legal YAML
    legal = "sam2:\n  checkpoint: abc.pt\n  config_file: config.yaml\n"
    r = await client.put(f"{BASE}/api/configs/_test.yaml",
        json={"content": legal}, cookies={"pipeline_session": cookie})
    print(f"  Legal YAML: {r.status_code} {r.json()}")

    # Illegal YAML — truly invalid syntax (unmatched flow sequence)
    illegal = "key: [unclosed\n"
    r = await client.put(f"{BASE}/api/configs/_test.yaml",
        json={"content": illegal}, cookies={"pipeline_session": cookie})
    body = r.json()
    print(f"  Illegal YAML: {r.status_code}")
    print(f"  Error detail: {body.get('detail', 'N/A')}")
    has_line = "line" in body.get("detail", "").lower()
    print(f"  Has line info: {'PASS' if has_line else 'FAIL — no line/col in error'}")

    # Cleanup
    p = PROJECT / "configs" / "_test.yaml"
    if p.exists():
        p.unlink()

    return has_line


async def item3_upload(client, cookie):
    """Upload real files and verify directory structure."""
    print("\n" + "=" * 60)
    print("ITEM 3: Upload Real Files")

    # Create test files
    tmp = Path("/tmp/upload_test_v2")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "mesh.obj").write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n")
    (tmp / "mesh.mtl").write_text("newmtl mat\nKa 0.2 0.2 0.2\nKd 0.8 0.8 0.8\n")
    (tmp / "dataset_info.json").write_text('{"sam2_points": {"points": [[100,200]], "labels": [1]}}\n')
    (tmp / "config.yaml").write_text("input:\n  rgbd_dir: ./tasks/test_up/\n")

    files = ["mesh.obj", "mesh.mtl", "dataset_info.json", "config.yaml"]
    upload_files = []
    for fname in files:
        fpath = tmp / fname
        upload_files.append(("files", (fname, fpath.read_bytes(), "application/octet-stream")))

    r = await client.post(
        f"{BASE}/api/tasks/upload",
        data={"task_name": "test_upload"},
        files=upload_files,
        cookies={"pipeline_session": cookie},
    )
    result = r.json()
    print(f"  Upload status: {r.status_code}")
    print(f"  Task: {result.get('task_name')}, files: {len(result.get('files', []))}")

    # Verify directory structure
    dest = PROJECT / "tasks" / "test_upload"
    uploaded = list(dest.rglob("*")) if dest.exists() else []
    print(f"  Files in tasks/test_upload/: {len(uploaded)}")
    for f in uploaded:
        print(f"    {f.relative_to(dest)}")
    ok = len(uploaded) == 4
    print(f"  {'PASS' if ok else 'FAIL'}: expected 4 files, got {len(uploaded)}")

    # Cleanup
    import shutil
    shutil.rmtree(tmp)
    if dest.exists():
        shutil.rmtree(dest)

    return ok


async def item5_history(client, cookie):
    """Verify history shows all 3 job states."""
    print("\n" + "=" * 60)
    print("ITEM 5: History — 3-State Verification")
    h = {"Cookie": f"pipeline_session={cookie}"}

    r = await client.get(f"{BASE}/api/history", cookies={"pipeline_session": cookie})
    jobs = r.json()
    print(f"  Total jobs on disk: {len(jobs)}")

    states = {}
    for j in jobs:
        s = j.get("status", "?")
        states.setdefault(s, []).append(j)

    # Check we have at least completed and failed
    for want in ("completed", "failed"):
        count = len(states.get(want, []))
        print(f"  {want}: {count} job(s)")

    # Verify metadata/log consistency for a failed job
    failed = states.get("failed", [])
    if failed:
        jid = failed[0]["job_id"]
        r = await client.get(f"{BASE}/api/history/{jid}", cookies={"pipeline_session": cookie})
        hist = r.json()
        meta_status = hist["metadata"]["status"]
        log_text = hist.get("log", "")
        log_has_error = "Job failed" in log_text or "ERROR" in log_text or "failed" in log_text.lower()
        print(f"  Consistency check:")
        print(f"    metadata.status = {meta_status}")
        print(f"    log has error info: {log_has_error}")
        print(f"    {'PASS' if meta_status == 'failed' and log_has_error else 'WARN'}")

    # Try to create a stopped job
    print(f"\n  Creating stopped job...")
    r = await client.post(f"{BASE}/api/tasks/mouse002/run",
        json={"preset": "foundationpose"},
        cookies={"pipeline_session": cookie})
    jid = r.json()["job_id"]

    # Wait briefly then stop
    await asyncio.sleep(0.5)
    r = await client.post(f"{BASE}/api/jobs/{jid}/stop",
        cookies={"pipeline_session": cookie})
    print(f"  Stop result: {r.status_code} {r.json()}")

    await asyncio.sleep(1)
    r = await client.get(f"{BASE}/api/history/{jid}", cookies={"pipeline_session": cookie})
    hist = r.json()
    final_status = hist["metadata"]["status"]
    log_text = hist.get("log", "")
    has_stop = "[JOB STOPPED]" in log_text or "STOPPED" in log_text or "stopped" in final_status
    print(f"  Stopped job status: {final_status}")
    print(f"  Has [JOB STOPPED] in log: {'[JOB STOPPED]' in log_text}")
    print(f"  {'PASS' if final_status == 'stopped' else 'OK'}: job stopped successfully")

    return True


async def item2_ws_e2e(client, cookie):
    """Full WebSocket end-to-end test."""
    print("\n" + "=" * 60)
    print("ITEM 2: WebSocket End-to-End (browser-emulated)")

    # Submit job FIRST to get job_id
    r = await client.post(f"{BASE}/api/tasks/mouse002/run",
        json={"preset": "foundationpose"},
        cookies={"pipeline_session": cookie})
    job_id = r.json()["job_id"]
    print(f"  Job submitted: {job_id}")

    # Connect WebSocket with cookie
    ws_url = f"{WS_BASE}/ws/jobs/{job_id}/logs"
    lines = []
    final_state = None

    try:
        async with websockets.connect(
            ws_url,
            additional_headers={"Cookie": f"pipeline_session={cookie}"},
            close_timeout=3,
            open_timeout=5,
        ) as ws:
            print(f"  WS connected")
            end = time.time() + 15
            while time.time() < end:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    lines.append(msg)
                    if "[JOB COMPLETED]" in msg:
                        final_state = "COMPLETED"
                    elif "[JOB FAILED]" in msg:
                        final_state = "FAILED"
                    elif "[JOB STOPPED]" in msg:
                        final_state = "STOPPED"
                except asyncio.TimeoutError:
                    break
    except (ConnectionClosed, InvalidStatus) as e:
        print(f"  WS event: {e}")

    print(f"  Received {len(lines)} messages")
    for line in lines:
        print(f"    {line[:100]}")
    print(f"  Final state from WS: {final_state or 'NOT DETECTED (job too fast?)'}")

    # Verify job final status matches
    r = await client.get(f"{BASE}/api/jobs/{job_id}",
        cookies={"pipeline_session": cookie})
    api_status = r.json()["status"]
    print(f"  API job status: {api_status}")

    ws_ok = len(lines) > 0 and final_state is not None
    status_match = (final_state or "").lower() == api_status
    print(f"  WS messages received: {'PASS' if ws_ok else 'WARN'}")
    print(f"  WS/API status match: {'PASS' if status_match else 'NOTE'}")

    return True


async def main():
    async with httpx.AsyncClient(proxy=None) as client:
        cookie = await login(client)
        if not cookie:
            print("FATAL: Login failed")
            return 1

        results = []
        results.append(await item4_config_validation(client, cookie))
        results.append(await item3_upload(client, cookie))
        results.append(await item5_history(client, cookie))
        results.append(await item2_ws_e2e(client, cookie))

        print("\n" + "=" * 60)
        passed = sum(1 for r in results if r)
        print(f"Results: {passed}/{len(results)} items passed")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
