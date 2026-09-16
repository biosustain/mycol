#!/usr/bin/env bash
#
# Build a development Mycol.app that runs straight from the working tree.
#
# The release bundle copies src/ into itself, so testing a code change means a
# full rebuild. This one symlinks src/, app.py and bootstrap.py back to the repo
# and keeps the heavy parts - the two interpreters and the model weights - in a
# cache under build/dev/. The result:
#
#   - first run:      a few minutes (populates the cache, or adopts a release build)
#   - later runs:     ~2 seconds
#   - code edits:     no rebuild at all; the app reloads them as you save
#
# Usage:
#   ./scripts/dev_app.sh            build/refresh, then launch
#   ./scripts/dev_app.sh --no-open  build/refresh only
#   ./scripts/dev_app.sh --reset    discard the cached environment and rebuild it

set -euo pipefail

OPEN_APP=1
RESET=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-open) OPEN_APP=0; shift ;;
        --reset)   RESET=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

PY_MAIN=3.12
PY_WORKER=3.10

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "$(uname -m)" in
    arm64)  PY_ARCH="aarch64" ;;
    x86_64) PY_ARCH="x86_64" ;;
    *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

CACHE="$ROOT/build/dev"
APP="$ROOT/dist/Mycol-dev.app"
RES="$APP/Contents/Resources"
RELEASE_APP="$ROOT/dist/Mycol.app"

step()   { echo ""; echo "==> $*"; }
detail() { echo "    - $*"; }
die()    { echo "ERROR: $*" >&2; exit 1; }

[[ "$RESET" == "1" ]] && { detail "removing cached environment"; rm -rf "$CACHE"; }

# --------------------------------------------------------------- environment
# Populated once, then reused. A release build already contains exactly what we
# need, so adopt it rather than downloading and installing everything again.
if [[ ! -d "$CACHE/python_main" ]]; then
    step "Populating dev environment cache (one time)..."
    mkdir -p "$CACHE"

    # -c clones on APFS: copy-on-write, so adopting ~2 GB is near-instant and
    # costs almost no extra disk. Falls back to a real copy on other filesystems.
    copy_tree() { cp -Rc "$1" "$2" 2>/dev/null || cp -R "$1" "$2"; }

    if [[ -d "$RELEASE_APP/Contents/Resources/bin/python_main" ]]; then
        detail "adopting interpreters from dist/Mycol.app"
        copy_tree "$RELEASE_APP/Contents/Resources/bin/python_main" "$CACHE/python_main"
        copy_tree "$RELEASE_APP/Contents/Resources/bin/python_worker" "$CACHE/python_worker"
        [[ -d "$RELEASE_APP/Contents/Resources/models" ]] && \
            copy_tree "$RELEASE_APP/Contents/Resources/models" "$CACHE/models"
    else
        command -v uv >/dev/null || die "uv not found"
        install_python() {
            local version="$1" dest="$2"
            uv python install "$version" >/dev/null 2>&1 || true
            local src
            src="$(ls -d "$(uv python dir)"/cpython-"$version".*-macos-"$PY_ARCH"-none 2>/dev/null | sort -V | tail -1)"
            [[ -n "$src" && -d "$src" ]] || die "no uv-managed CPython $version for $PY_ARCH"
            cp -R "$src" "$dest"
            find "$dest" -name EXTERNALLY-MANAGED -delete
        }
        detail "installing main interpreter (py$PY_MAIN)"
        install_python "$PY_MAIN" "$CACHE/python_main"
        detail "installing worker interpreter (py$PY_WORKER)"
        install_python "$PY_WORKER" "$CACHE/python_worker"

        mkdir -p "$ROOT/build"
        detail "resolving dependencies"
        uv export --no-dev --python "$PY_MAIN" -o "$ROOT/build/req_main.txt" >/dev/null
        ( cd src/training && uv export --no-dev --python "$PY_WORKER" -o "$ROOT/build/req_worker.txt" >/dev/null )
        detail "installing dependencies (this is the slow part)"
        uv pip install --python "$CACHE/python_main/bin/python3"   -r "$ROOT/build/req_main.txt" --quiet
        uv pip install --python "$CACHE/python_worker/bin/python3" -r "$ROOT/build/req_worker.txt" --quiet
    fi

    if [[ ! -d "$CACHE/models" ]]; then
        detail "pre-baking model weights"
        "$CACHE/python_main/bin/python3" "$ROOT/scripts/fetch_models.py" "$CACHE"
    fi
else
    detail "reusing cached environment at build/dev"
fi

# ------------------------------------------------------------------ launcher
step "Building launcher..."
command -v cargo >/dev/null || die "cargo not found. Install Rust from https://rustup.rs, then: . \"\$HOME/.cargo/env\""
( cd tools/launcher && cargo build --release --quiet )

# ------------------------------------------------------------------- bundle
step "Assembling Mycol-dev.app..."
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$RES/bin"

cp tools/launcher/target/release/launcher "$APP/Contents/MacOS/Mycol"
chmod +x "$APP/Contents/MacOS/Mycol"

# Symlinks, not copies: this is the entire point. Editing src/ in the repo
# changes what the running app executes.
ln -s "$ROOT/src"          "$RES/src"
ln -s "$ROOT/app.py"       "$RES/app.py"
ln -s "$ROOT/src/bootstrap.py" "$RES/bootstrap.py"
ln -s "$ROOT/logo.png"     "$RES/logo.png"
ln -s "$ROOT/.streamlit"   "$RES/.streamlit"
ln -s "$CACHE/python_main"   "$RES/bin/python_main"
ln -s "$CACHE/python_worker" "$RES/bin/python_worker"
ln -s "$CACHE/models"        "$RES/models"

# Read by bootstrap.py to turn on the file watcher and auto-rerun.
touch "$RES/.devmode"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Mycol (dev)</string>
    <key>CFBundleDisplayName</key><string>Mycol (dev)</string>
    <!-- A distinct identifier, so macOS never confuses this with the real app. -->
    <key>CFBundleIdentifier</key><string>io.github.biosustain.mycol.dev</string>
    <key>CFBundleVersion</key><string>0.0.0-dev</string>
    <key>CFBundleShortVersionString</key><string>0.0.0-dev</string>
    <key>CFBundleExecutable</key><string>Mycol</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

echo ""
echo "========================================"
echo "Dev app ready: $APP"
echo ""
echo "  Edit anything under src/ and save - the open app reloads it."
echo "  Only rerun this script after changing the launcher (Rust) or"
echo "  the dependency list; use --reset to rebuild the environment."
echo ""
echo "  Logs: ~/Library/Logs/Mycol/mycol.log"
echo "========================================"

if [[ "$OPEN_APP" == "1" ]]; then
    step "Launching..."
    open "$APP"
fi
