from pathlib import Path

from fastapi import Request


def _project_root(request: Request) -> Path:
    return request.app.state.project_root
