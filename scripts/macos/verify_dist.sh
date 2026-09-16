#!/usr/bin/env bash
#
# Smoke-test a built Mycol.app.
#
# Checks the bundle contains what it needs, that its config parses, that the
# heavy imports work in both embedded interpreters, that the interpreter is
# genuinely relocatable, and that the app serves a page. Run in CI right after
# make_dist.sh so a broken bundle never reaches a release.
#
# Usage: ./scripts/macos/verify_dist.sh [path/to/Mycol.app]

set -uo pipefail

APP="${1:-dist/Mycol.app}"
[[ -d "$APP" ]] || { echo "No such app bundle: $APP" >&2; exit 1; }
APP="$(cd "$APP" && pwd)"

CONTENTS="$APP/Contents"
RES="$CONTENTS/Resources"
MAIN_PY="$RES/bin/python_main/bin/python3"
WORKER_PY="$RES/bin/python_worker/bin/python3"

failures=0
ok()   { echo "  OK   $1"; }
bad()  { echo "  FAIL $1"; failures=$((failures+1)); }
check_file() { if [[ -e "$CONTENTS/$1" ]]; then ok "$2"; else bad "$2 (missing: Contents/$1)"; fi; }

echo ""
echo "Verifying $APP"

echo ""
echo "[1/6] Bundle structure..."
check_file "MacOS/Mycol"                      "launcher executable"
check_file "Info.plist"                       "Info.plist"
check_file "Resources/bootstrap.py"           "bootstrap"
check_file "Resources/app.py"                 "app entry point"
# app.py calls st.logo("logo.png") during startup.
check_file "Resources/logo.png"               "logo asset"
check_file "Resources/.streamlit/config.toml" "streamlit config"
check_file "Resources/src/helpers"            "src tree"
check_file "Resources/bin/python_main/bin/python3"   "main interpreter"
check_file "Resources/bin/python_worker/bin/python3" "worker interpreter"

if [[ -x "$CONTENTS/MacOS/Mycol" ]]; then ok "launcher is executable"; else bad "launcher is not executable"; fi

echo ""
echo "[2/6] Pre-baked models..."
check_file "Resources/models/mobile_sam.pt"          "MobileSAM weights"
check_file "Resources/models/cellpose/cyto3"         "Cellpose cyto3"
check_file "Resources/models/cellpose/cyto2torch_0"  "Cellpose cyto2"
check_file "Resources/models/cellpose/nucleitorch_0" "Cellpose nuclei"

echo ""
echo "[3/6] Interpreter is relocatable..."
# A `uv venv` would symlink back to the build machine's Python and break on any
# other Mac. A python-build-standalone copy must have no such link.
if [[ -L "$RES/bin/python_main/bin/python3" ]] && \
   [[ "$(readlink "$RES/bin/python_main/bin/python3")" == /* ]]; then
    bad "main interpreter points outside the bundle ($(readlink "$RES/bin/python_main/bin/python3"))"
else
    ok "main interpreter is self-contained"
fi
if [[ -f "$RES/bin/python_main/pyvenv.cfg" ]]; then
    bad "main interpreter is a virtualenv (not relocatable)"
else
    ok "not a virtualenv"
fi
prefix="$("$MAIN_PY" -c 'import sys; print(sys.prefix)' 2>/dev/null)"
if [[ "$prefix" == "$RES"/* ]]; then ok "sys.prefix resolves inside the bundle"; else bad "sys.prefix is $prefix"; fi

echo ""
echo "[4/6] Config parses..."
if "$MAIN_PY" -c "import tomllib,sys; tomllib.load(open(sys.argv[1],'rb'))" "$RES/.streamlit/config.toml" 2>/dev/null; then
    ok "config.toml is valid TOML"
else
    bad "config.toml is not valid TOML"
fi

echo ""
echo "[5/6] Imports..."
if "$MAIN_PY" -c "import streamlit, torch, cellpose, mobile_sam, huggingface_hub, plotly, cv2, skimage" 2>/dev/null; then
    ok "main imports ($("$MAIN_PY" -c 'import torch,sys; print("py"+sys.version.split()[0], "torch"+torch.__version__)'))"
else
    bad "main interpreter imports"
fi
if "$WORKER_PY" -c "import torch, cellpose, optuna, numpy" 2>/dev/null; then
    ok "worker imports ($("$WORKER_PY" -c 'import numpy,sys; print("py"+sys.version.split()[0], "numpy"+numpy.__version__)'))"
else
    bad "worker interpreter imports"
fi

echo ""
echo "[6/6] App serves a page..."
# A fixed port silently invalidates this check: anything else already listening
# there answers the request and the bundle appears to work when it never started.
PORT="$("$MAIN_PY" -c 'import socket
s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
echo "  using port $PORT"
# exec, so $! is the Streamlit process itself rather than the subshell wrapping
# it - otherwise the kill below leaves Streamlit running and holding the port.
( cd "$RES" && exec "$MAIN_PY" -m streamlit run app.py --server.headless true --server.port $PORT \
    --server.address 127.0.0.1 --server.fileWatcherType none --browser.gatherUsageStats false \
    >/tmp/mycol_verify_streamlit.log 2>&1 ) &
SERVER_PID=$!
# Belt and braces: never leave a server behind, however this script exits.
trap 'kill $SERVER_PID 2>/dev/null' EXIT

served=0
for _ in $(seq 1 180); do
    kill -0 $SERVER_PID 2>/dev/null || break
    if curl -fsS -o /dev/null "http://127.0.0.1:$PORT" 2>/dev/null; then served=1; break; fi
    sleep 1
done
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null

if [[ "$served" == "1" ]]; then
    ok "app returned HTTP 200"
else
    bad "app never served a page"
    echo "--- streamlit log ---"; tail -20 /tmp/mycol_verify_streamlit.log
fi

echo ""
if (( failures > 0 )); then
    echo "FAILED: $failures check(s)"
    exit 1
fi
echo "All checks passed."
