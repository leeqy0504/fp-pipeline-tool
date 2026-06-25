#!/usr/bin/env bash
# Pipeline tool entrypoint — manages Python environment and dispatches commands.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Use conda obe environment if available, else system python
if command -v conda &> /dev/null; then
    source "$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")/etc/profile.d/conda.sh"
    if conda env list | grep -q "^obe "; then
        conda activate obe
    fi
fi

CMD="${1:-}"
shift || true

case "$CMD" in
    web)
        # Load .env if present
        if [ -f "${SCRIPT_DIR}/.env" ]; then
            export $(grep -v '^#' "${SCRIPT_DIR}/.env" | xargs)
        fi
        if [ -z "${WEB_PASSWORD:-}" ]; then
            echo "ERROR: WEB_PASSWORD not set. Create .env from .env.example or export it."
            exit 1
        fi
        echo "Starting Pipeline Web UI on http://0.0.0.0:8001"
        uvicorn web.app:app --host 0.0.0.0 --port "${2:-8001}"
        ;;
    *)
        PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" python -m pipeline.cli "$CMD" "$@"
        ;;
esac
