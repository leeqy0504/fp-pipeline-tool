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
