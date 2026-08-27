#!/usr/bin/env bash
#
# BL View -- one command, end to end.
#
#   ./run.sh                 generate synthetic data, run the pipeline, validate
#                            it against the injected truth, then serve the quicklook
#   ./run.sh --test          run the test suite and the validation harness, then stop
#   ./run.sh --no-serve      build and validate, but do not start the server
#   ./run.sh --port 9000     serve on a different port
#   ./run.sh --hours 48      generate a longer synthetic series
#   ./run.sh --regenerate    rebuild the synthetic file even if one exists
#
# Anything not recognised here is passed straight through to `blview demo`.

set -euo pipefail

cd "$(dirname "$0")"

VENV="${BLVIEW_VENV:-.venv}"
PYTHON_BIN="${PYTHON:-python3}"
MODE="serve"
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test)  MODE="test"; shift ;;
    -h|--help)
      sed -n '3,14p' "$0" | sed 's/^#\{1,\} \{0,1\}//'
      exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

# ---------------------------------------------------------------- environment
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "==> creating virtual environment in $VENV"
  "$PYTHON_BIN" -m venv "$VENV"
fi
PY="$VENV/bin/python"

# Install only when the dependency set has changed since the last run.
STAMP="$VENV/.blview-requirements"
if [[ ! -f "$STAMP" ]] || ! cmp -s requirements.txt "$STAMP"; then
  echo "==> installing dependencies"
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r requirements.txt
  cp requirements.txt "$STAMP"
fi

if [[ "$MODE" == "test" ]]; then
  echo "==> unit tests"
  "$PY" -m pytest -q
  echo
  echo "==> validation harness (full pipeline against injected synthetic truth)"
  exec "$PY" scripts/validate.py
fi

echo "==> BL View demo: generate -> ingest -> preprocess -> detect -> track -> serve"
exec "$PY" -m blview.cli demo "${ARGS[@]}"
