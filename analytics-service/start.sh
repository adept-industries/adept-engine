#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Resolve uvicorn if virtualenv exists
if ! command -v uvicorn &> /dev/null; then
    if [ -f "$DIR/../.venv/bin/uvicorn" ]; then
        export PATH="$DIR/../.venv/bin:$PATH"
    elif [ -f "$DIR/.venv/bin/uvicorn" ]; then
        export PATH="$DIR/.venv/bin:$PATH"
    fi
fi

echo "Starting PR Risk Analytics Service on http://0.0.0.0:8000..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
