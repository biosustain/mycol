#!/bin/bash
set -e

VERSION="${1:-0.1.0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$PROJECT_ROOT/dist/MyCol"
BIN_DIR="$DIST_DIR/bin"
BUILD_DIR="$PROJECT_ROOT/build"

echo "========================================"
echo "Building Portable MyCol v$VERSION (macOS/Linux)"
echo "========================================"

# 1. Clean
echo "[1/6] Cleaning dist directory..."
rm -rf "$DIST_DIR"
mkdir -p "$BIN_DIR"
mkdir -p "$BUILD_DIR"

echo "[2/6] Creating Main environment (3.13)..."
MAIN_ENV="$BIN_DIR/python_main"

get_python_url() {
    local version=$1
    local os=$(uname -s)
    local arch=$(uname -m)
    
    #TODO: this might NOT work on Linux/MacOS
    echo "Creating venv in $2..."
    uv venv --python $version "$2"
}

get_python_url 3.13 "$MAIN_ENV"
MAIM_PY="$MAIN_ENV/bin/python"

# setup Worker Python (3.10)
echo "[3/6] Creating Worker environment (3.10)..."
WORKER_ENV="$BIN_DIR/python_worker"
get_python_url 3.10 "$WORKER_ENV"
WORKER_PY="$WORKER_ENV/bin/python"

# install dependencies
echo "[4/6] Installing dependencies with uv..."

echo "  - resolving main requirements..."
uv export --no-dev --python 3.13 -o "$BUILD_DIR/req_main.txt"

echo "  - resolving worker requirements..."
cd src/training
uv export --no-dev --python 3.10 -o "../../$BUILD_DIR/req_worker.txt"
cd ../..

echo "  - Installing Main deps..."
uv pip install --python "$MAIM_PY" -r "$BUILD_DIR/req_main.txt"

echo "  - Installing Worker deps..."
uv pip install --python "$WORKER_PY" -r "$BUILD_DIR/req_worker.txt"

# copy source
echo "[5/6] Copying source code..."
# Use rsync for exclusion if available, else plain cp (and manual remove)
if command -v rsync &> /dev/null; then
    rsync -av \
        --exclude='.venv' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='.pytest_cache' \
        --exclude='*.egg-info' \
        src "$DIST_DIR/"
else
    cp -r src "$DIST_DIR/src"
    # Basic cleanup if rsync missing
    find "$DIST_DIR/src" -name ".venv" -type d -exec rm -rf {} +
    find "$DIST_DIR/src" -name "__pycache__" -type d -exec rm -rf {} +
fi

cp src/bootstrap.py "$DIST_DIR/bootstrap.py"
cp app.py "$DIST_DIR/app.py"

# Copy .streamlit and demo_data if they exist
if [ -d ".streamlit" ]; then
    cp -r ".streamlit" "$DIST_DIR/.streamlit"
    echo '[client]\ntoolbarMode = "viewer"' >> "$DIST_DIR/.streamlit/config.toml"
fi
[ -d "demo_data" ] && cp -r "demo_data" "$DIST_DIR/demo_data"

# build launcher
echo "[6/6] Building Native Launcher (Rust)..."

if command -v cargo &> /dev/null; then
    cd tools/launcher
    echo "  - Compiling launcher..."
    cargo build --release
    cd ../..
    
    LAUNCHER_SRC="tools/launcher/target/release/launcher"
    if [ -f "$LAUNCHER_SRC" ]; then
        cp "$LAUNCHER_SRC" "$DIST_DIR/MyCol"
        echo "  - Launcher copied to MyCol"
    else
        echo "Error: Launcher compilation failed."
    fi
else
    echo "Cargo not found. Falling back to shell script."
    LAUNCHER="$DIST_DIR/MyCol.sh"
    cat > "$LAUNCHER" << EOF
#!/bin/bash
HERE="\$(dirname "\$0")"
"\$HERE/bin/python_main/bin/python" "\$HERE/bootstrap.py" "\$@"
EOF
    chmod +x "$LAUNCHER"
fi

# cleanup
rm -f "$DIST_DIR/pwa.py"

echo ""
echo "========================================"
echo "Build Complete!"
echo "Portable App: dist/MyCol"
echo "Run: ./dist/MyCol/MyCol"
echo "========================================"
