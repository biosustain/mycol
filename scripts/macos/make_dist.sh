#!/usr/bin/env bash
#
# Build the portable macOS app bundle for Mycol.
#
# Produces Mycol.app, containing its own Python interpreters, all dependencies
# and the model weights, so the target Mac needs no Python, uv or git. The app
# is built for the architecture of the machine running this script; cross-arch
# builds need a runner of that architecture.
#
# Usage:
#   ./scripts/macos/make_dist.sh [--version 0.2.0] [--skip-models] [--no-dmg]
#
# Signing (optional, both must be set):
#   MYCOL_SIGN_IDENTITY   "Developer ID Application: Name (TEAMID)"
#   MYCOL_NOTARY_PROFILE  notarytool keychain profile name
# Without them the app is ad-hoc signed, which runs locally but triggers
# Gatekeeper on any machine that downloaded it.

set -euo pipefail

VERSION="0.1.0"
SKIP_MODELS=0
MAKE_DMG=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --version=*) VERSION="${1#*=}"; shift ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        --no-dmg) MAKE_DMG=0; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

PY_MAIN=3.12      # cellpose pins numpy<2.1, which has no cp313 wheels
PY_WORKER=3.10

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

ARCH="$(uname -m)"                                   # arm64 or x86_64
case "$ARCH" in
    arm64)  PY_ARCH="aarch64" ;;
    x86_64) PY_ARCH="x86_64" ;;
    *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

DIST_DIR="$PROJECT_ROOT/dist"
APP="$DIST_DIR/Mycol.app"
CONTENTS="$APP/Contents"
RESOURCES="$CONTENTS/Resources"
MACOS_DIR="$CONTENTS/MacOS"
BIN_DIR="$RESOURCES/bin"
BUILD_DIR="$PROJECT_ROOT/build"

step()   { echo ""; echo "==> $*"; }
detail() { echo "    - $*"; }
die()    { echo "ERROR: $*" >&2; exit 1; }

command -v uv >/dev/null || die "uv not found. See https://docs.astral.sh/uv/getting-started/installation/"

echo "========================================"
echo "Building Mycol $VERSION for macOS ($ARCH)"
echo "========================================"

# ---------------------------------------------------------------- 1. skeleton
step "[1/9] Creating app bundle skeleton..."
rm -rf "$APP"
mkdir -p "$MACOS_DIR" "$RESOURCES" "$BIN_DIR" "$BUILD_DIR"

# ------------------------------------------------------------- 2. interpreters
# A uv-managed (python-build-standalone) interpreter is relocatable: its dylib
# is referenced via @rpath and sys.prefix follows the directory. A `uv venv`
# would NOT work here - it only symlinks back to the Python that built it, so
# the bundle would break on any other machine.
install_python() {
    local version="$1" dest="$2"
    uv python install "$version" >/dev/null 2>&1 || true
    local src
    src="$(ls -d "$(uv python dir)"/cpython-"$version".*-macos-"$PY_ARCH"-none 2>/dev/null | sort -V | tail -1)"
    [[ -n "$src" && -d "$src" ]] || die "no uv-managed CPython $version for $PY_ARCH; run: uv python install $version"
    detail "$(basename "$src")"
    cp -R "$src" "$dest"
    # Installing into the copy is the whole point of making one.
    find "$dest" -name EXTERNALLY-MANAGED -delete
}

step "[2/9] Main interpreter (Python $PY_MAIN)..."
install_python "$PY_MAIN" "$BIN_DIR/python_main"
MAIN_PY="$BIN_DIR/python_main/bin/python3"

step "[3/9] Worker interpreter (Python $PY_WORKER)..."
install_python "$PY_WORKER" "$BIN_DIR/python_worker"
WORKER_PY="$BIN_DIR/python_worker/bin/python3"

# ------------------------------------------------------------- 4. dependencies
step "[4/9] Resolving and installing dependencies..."
detail "exporting main requirements"
uv export --no-dev --python "$PY_MAIN" -o "$BUILD_DIR/req_main.txt" >/dev/null
detail "exporting worker requirements"
( cd src/training && uv export --no-dev --python "$PY_WORKER" -o "$BUILD_DIR/req_worker.txt" >/dev/null )

detail "installing main (py$PY_MAIN)"
uv pip install --python "$MAIN_PY" -r "$BUILD_DIR/req_main.txt" --quiet
detail "installing worker (py$PY_WORKER)"
uv pip install --python "$WORKER_PY" -r "$BUILD_DIR/req_worker.txt" --quiet

# ------------------------------------------------------------ 5. app source
step "[5/9] Copying application files..."
if command -v rsync >/dev/null; then
    rsync -a --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
        --exclude='*.egg-info' --exclude='.mypy_cache' src "$RESOURCES/"
else
    cp -R src "$RESOURCES/src"
    find "$RESOURCES/src" \( -name '__pycache__' -o -name '.venv' \) -type d -prune -exec rm -rf {} +
fi

cp src/bootstrap.py "$RESOURCES/bootstrap.py"
cp app.py "$RESOURCES/app.py"
# app.py calls st.logo("logo.png") during startup.
cp logo.png "$RESOURCES/logo.png"
[[ -f logo_icon.png ]] && cp logo_icon.png "$RESOURCES/logo_icon.png"
# load_demo_data() expects this beside app.py.
[[ -f example_session.zip ]] && cp example_session.zip "$RESOURCES/example_session.zip"
cp LICENSE "$RESOURCES/LICENSE"

if [[ -d .streamlit ]]; then
    cp -R .streamlit "$RESOURCES/.streamlit"
    # printf, not echo: bash's echo leaves a literal \n and the result is
    # invalid TOML, which stops Streamlit from starting at all.
    printf '\n[client]\ntoolbarMode = "viewer"\n' >> "$RESOURCES/.streamlit/config.toml"
fi
[[ -d demo_data ]] && cp -R demo_data "$RESOURCES/demo_data"

# --------------------------------------------------------------- 6. weights
step "[6/9] Pre-baking model weights..."
if [[ "$SKIP_MODELS" == "1" ]]; then
    detail "skipped (--skip-models)"
else
    "$MAIN_PY" "$PROJECT_ROOT/scripts/fetch_models.py" "$RESOURCES"
fi

# -------------------------------------------------------------- 7. launcher
step "[7/9] Building native launcher..."
command -v cargo >/dev/null || die "cargo not found. Install Rust from https://rustup.rs"
( cd tools/launcher && cargo build --release --quiet )
cp tools/launcher/target/release/launcher "$MACOS_DIR/Mycol"
chmod +x "$MACOS_DIR/Mycol"

# ------------------------------------------------------- 8. bundle metadata
step "[8/9] Writing bundle metadata..."

# .icns from the square mark: sips -z ignores aspect ratio, so the wide
# sidebar lockup would come out stretched.
ICON_SRC="logo_icon.png"; [[ -f "$ICON_SRC" ]] || ICON_SRC="logo.png"
ICONSET="$BUILD_DIR/Mycol.iconset"
rm -rf "$ICONSET"; mkdir -p "$ICONSET"
for size in 16 32 64 128 256 512; do
    sips -z $size $size "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null 2>&1
    sips -z $((size*2)) $((size*2)) "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null 2>&1
done
iconutil -c icns "$ICONSET" -o "$RESOURCES/Mycol.icns" 2>/dev/null || detail "icon generation failed (continuing)"
rm -rf "$ICONSET"

cat > "$CONTENTS/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Mycol</string>
    <key>CFBundleDisplayName</key><string>Mycol</string>
    <key>CFBundleIdentifier</key><string>io.github.biosustain.mycol</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleExecutable</key><string>Mycol</string>
    <key>CFBundleIconFile</key><string>Mycol</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <!-- The UI is a WebView window, not a background service. -->
    <key>LSBackgroundOnly</key><false/>
</dict>
</plist>
PLIST

echo "APPL????" > "$CONTENTS/PkgInfo"

# --------------------------------------------------------- 9. sign + package
step "[9/9] Signing and packaging..."
detail "free disk: $(df -h "$DIST_DIR" | awk 'NR==2 {print $4}')"
if [[ -n "${MYCOL_SIGN_IDENTITY:-}" ]]; then
    detail "signing with Developer ID"
    # --options runtime is required for notarization.
    codesign --force --deep --options runtime --timestamp \
        --sign "$MYCOL_SIGN_IDENTITY" "$APP"
else
    # arm64 binaries need at least an ad-hoc signature to execute at all.
    detail "ad-hoc signing (set MYCOL_SIGN_IDENTITY for a distributable build)"
    codesign --force --deep --sign - "$APP"
fi
codesign --verify --deep "$APP" && detail "signature verifies"

if [[ "$MAKE_DMG" == "1" ]]; then
    # No version in the name: the README links to
    # releases/latest/download/<name>, which needs a stable filename.
    DMG="$DIST_DIR/mycol-macos-$ARCH.dmg"
    rm -f "$DMG"
    detail "building $(basename "$DMG")"
    STAGE="$BUILD_DIR/dmg"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    # -c clones on APFS, so staging a 2 GB app costs no extra disk. CI runners
    # have little headroom and the plain copy pushed them over.
    cp -Rc "$APP" "$STAGE/" 2>/dev/null || cp -R "$APP" "$STAGE/"
    ln -s /Applications "$STAGE/Applications"      # drag-to-install target
    # Not -quiet: it hides the reason when this fails.
    hdiutil create -volname "Mycol" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
    rm -rf "$STAGE"

    if [[ -n "${MYCOL_NOTARY_PROFILE:-}" ]]; then
        detail "submitting to Apple for notarization (this takes a few minutes)"
        xcrun notarytool submit "$DMG" --keychain-profile "$MYCOL_NOTARY_PROFILE" --wait
        xcrun stapler staple "$DMG"
        detail "notarized and stapled"
    else
        detail "not notarized - Gatekeeper will block this on other Macs"
    fi
fi

echo ""
echo "========================================"
echo "Build complete"
echo "  App: $APP"
[[ "$MAKE_DMG" == "1" ]] && echo "  DMG: ${DMG:-}"
du -sh "$APP" | awk '{print "  Size: " $1}'
echo "========================================"
