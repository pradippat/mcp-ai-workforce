#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
PY=python3; command -v python3 >/dev/null || PY=python
[ -x venv/bin/python ] || "$PY" -m venv venv || { echo "Could not create venv (install python3-venv)"; exit 1; }
venv/bin/python -m pip install --quiet --upgrade pip
venv/bin/python -m pip install --quiet -r requirements.txt || exit 1
venv/bin/python server.py --setup
