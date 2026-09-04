#!/usr/bin/env bash
# Convenience wrapper: ./fotosort.sh /path/to/photos [options]
# Creates a virtualenv and installs the package (with judge support) on first use.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/fotosort ]; then
  echo "Setting up virtualenv (first run only) ..."
  python3.11 -m venv .venv && .venv/bin/pip install -q -e ".[judge]"
fi
exec .venv/bin/fotosort "$@"
