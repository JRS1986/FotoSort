#!/usr/bin/env bash
# Convenience wrapper: ./fotosort.sh /path/to/photos [options]
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Setting up virtualenv (first run only) ..."
  python3.11 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
fi
exec .venv/bin/python -m fotosort "$@"
