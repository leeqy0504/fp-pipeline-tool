import logging
import shutil

import psutil
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/system")
async def system_info():
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    gpu = _get_gpu_info()

    return {
        "cpu_percent": cpu,
        "memory": {
            "total_bytes": mem.total,
            "used_bytes": mem.used,
            "percent": mem.percent,
        },
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "percent": disk.percent,
        },
        "gpu": gpu,
    }


def _get_gpu_info():
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    import subprocess
    try:
        result = subprocess.run(
            [nvidia_smi,
             "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "name": parts[0],
                    "memory_total_mb": float(parts[1]),
                    "memory_used_mb": float(parts[2]),
                    "memory_free_mb": float(parts[3]),
                    "gpu_utilization_pct": float(parts[4]),
                    "temperature_c": float(parts[5]),
                })

        return gpus if gpus else None
    except Exception:
        logger.warning("nvidia-smi query failed", exc_info=True)
        return None
