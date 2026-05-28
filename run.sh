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

PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" python -m pipeline.cli "$@"
