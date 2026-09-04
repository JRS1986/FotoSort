#!/usr/bin/env bash
# Convenience wrapper: ./fotosort.sh /path/to/photos [options]
# Creates a virtualenv and installs the package (with judge support) on first use.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -x "$DIR/.venv/bin/fotosort" ]; then
  echo "Setting up virtualenv (first run only) ..."
  PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
  "$PY" -m venv "$DIR/.venv" && "$DIR/.venv/bin/pip" install -q -e "$DIR[judge]"
fi
exec "$DIR/.venv/bin/fotosort" "$@"
