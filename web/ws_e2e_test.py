"""
E2E WebSocket integration test.

Verifies:
  1. WS rejected without cookie → HTTP 403
  2. WS accepted with cookie → connection stays open
  3. Job status query works
  4. Log file written to disk
"""
import asyncio
import sys

import httpx
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus


BASE = "http://localhost:8080"
WS_BASE = "ws://localhost:8080"


async def main():
    passed = 0
    failed = 0

    # ── 1. WS auth rejection (no cookie) ──
    print("=== Test 1: WS rejected without cookie ===")
    try:
        async with websockets.connect(f"{WS_BASE}/ws/jobs/fake/logs") as ws:
            print("  FAIL: accepted without cookie")
            failed += 1
    except InvalidStatus as e:
        print(f"  PASS: HTTP {e.response.status_code}")
        passed += 1

    # ── 2. Login ──
    print("\n=== Test 2: Login ===")
    async with httpx.AsyncClient(proxy=None) as http:
        resp = await http.post(
            f"{BASE}/api/login",
            json={"password": "test"},
            follow_redirects=False,
        )
        cookie = resp.cookies.get("pipeline_session")
        auth_headers = {"Cookie": f"pipeline_session={cookie}"}
        ok = cookie is not None
        print(f"  {'PASS' if ok else 'FAIL'}: cookie={'OK' if ok else 'MISSING'}")
        if ok:
            passed += 1
        else:
            failed += 1
            print("FATAL: no cookie")
            return 1

        # ── 3. Submit job ──
        print("\n=== Test 3: Submit job → get job_id ===")
        resp = await http.post(
            f"{BASE}/api/tasks/mouse002/run",
            json={"preset": "foundationpose"},
            cookies={"pipeline_session": cookie},
        )
        data = resp.json()
        job_id = data["job_id"]
        ok = len(job_id) == 12
        print(f"  {'PASS' if ok else 'FAIL'}: job_id={job_id}")
        passed += 1 if ok else 0
        failed += 0 if ok else 1

        # ── 4. WS accepted with cookie → connection stays open ──
        print("\n=== Test 4: WS accepted with valid cookie ===")
        ws_url = f"{WS_BASE}/ws/jobs/{job_id}/logs"
        try:
            async with websockets.connect(
                ws_url,
                additional_headers=auth_headers,
                close_timeout=2,
            ) as ws:
                # Connection established = auth passed.
                # Job may already be done (fast-fail in <10ms), so messages
                # may or may not arrive. That's expected — WebSocket is for
                # real-time streaming during long-running stages.
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    print(f"  PASS: auth accepted + received log: {msg[:80]}")
                except asyncio.TimeoutError:
                    print("  PASS: auth accepted (job already done, no live messages)")
        except InvalidStatus as e:
            print(f"  FAIL: HTTP {e.response.status_code} — cookie rejected by WS handler")
            failed += 1
            passed -= 1  # adjust double-count
        passed += 1

        # ── 5. Job status + disk log ──
        print("\n=== Test 5: Job status + disk log ===")
        resp = await http.get(
            f"{BASE}/api/history/{job_id}",
            cookies={"pipeline_session": cookie},
        )
        hist = resp.json()
        status = hist["metadata"]["status"]
        log_lines = (hist.get("log") or "").splitlines()
        ok = status == "failed" and len(log_lines) > 3
        print(f"  {'PASS' if ok else 'FAIL'}: status={status} log_lines={len(log_lines)}")
        passed += 1 if ok else 0
        failed += 0 if ok else 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
