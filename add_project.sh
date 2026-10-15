#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
venv/bin/python server.py --add-project "$@"
