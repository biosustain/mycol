#Requires -Version 7.0
param(
    [string]$Version = "0.1.0"
)

$ErrorActionPreference = "Stop"

# URLs for Python Embeddable Distributions
$PYTHON_MAIN_URL = "https://www.python.org/ftp/python/3.13.1/python-3.13.1-embed-amd64.zip"
$PYTHON_WORKER_URL = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"

# Project Root
$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $ProjectRoot

# Destination Directory: dist/MyCol/
$DistDir = "$ProjectRoot\dist\MyCol"
$BinDir = "$DistDir\bin"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building Portable MyCol v$Version" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Clean Dist
Write-Host "`n[1/6] Cleaning dist directory..." -ForegroundColor Yellow
if (Test-Path $DistDir) { Remove-Item $DistDir -Recurse -Force }
New-Item -ItemType Directory -Path $BinDir -Force | Out-Null

# 2. Download and Extract Python Main (3.13)
Write-Host "[2/6] Setting up Main environment (Python 3.13)..." -ForegroundColor Yellow
$MainDir = "$BinDir\python_main"
New-Item -ItemType Directory -Path $MainDir | Out-Null

if (-not (Test-Path "build\python-3.13.zip")) {
    New-Item -ItemType Directory -Path "build" -Force | Out-Null
    Write-Host "  - Downloading Python 3.13..." -ForegroundColor Gray
    Invoke-WebRequest -Uri $PYTHON_MAIN_URL -OutFile "build\python-3.13.zip"
}
Write-Host "  - Extracting Python 3.13..." -ForegroundColor Gray
Expand-Archive "build\python-3.13.zip" -DestinationPath $MainDir

# Enable site-packages for Main
Set-Content "$MainDir\python313._pth" "python313.zip`n.`n`nimport site`n"

# 3. Download and Extract Python Worker (3.10)
Write-Host "[3/6] Setting up Worker environment (Python 3.10)..." -ForegroundColor Yellow
$WorkerDir = "$BinDir\python_worker"
New-Item -ItemType Directory -Path $WorkerDir | Out-Null

if (-not (Test-Path "build\python-3.10.zip")) {
    Write-Host "  - Downloading Python 3.10..." -ForegroundColor Gray
    Invoke-WebRequest -Uri $PYTHON_WORKER_URL -OutFile "build\python-3.10.zip"
}
Write-Host "  - Extracting Python 3.10..." -ForegroundColor Gray
Expand-Archive "build\python-3.10.zip" -DestinationPath $WorkerDir

# Enable site-packages for Worker
Set-Content "$WorkerDir\python310._pth" "python310.zip`n.`n`nimport site`n"

# 4. Install Dependencies with UV
Write-Host "[4/6] Installing dependencies with uv..." -ForegroundColor Yellow

# Generate requirements if they dont exist
Write-Host "  - resolving main requirements..." -ForegroundColor Gray
uv export --no-dev --python 3.13 -o build/req_main.txt

Write-Host "  - resolving worker requirements..." -ForegroundColor Gray
Push-Location src/training
uv export --no-dev --python 3.10 -o ../../build/req_worker.txt
Pop-Location

# Install Main
Write-Host "  - Installing Py3.13 deps..." -ForegroundColor Gray
uv pip install --python "$MainDir\python.exe" -r build/req_main.txt --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match

# Install Worker
Write-Host "  - Installing Py3.10 deps..." -ForegroundColor Gray
uv pip install --python "$WorkerDir\python.exe" -r build/req_worker.txt --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match

# 5. Copy Source Code
Write-Host "[5/6] Copying source code..." -ForegroundColor Yellow
$SrcDest = "$DistDir\src"

$RoboArgs = @(
    "src", 
    "$SrcDest", 
    "/E", 
    "/XD", ".venv", "__pycache__", ".git", ".pytest_cache", "dist", "build", "*.egg-info", ".mypy_cache", 
    "/nfl", "/ndl", "/njh", "/njs", "/nc", "/ns", "/np" # Silent mode
)

Invoke-Expression "robocopy $RoboArgs" | Out-Null
if ($LASTEXITCODE -gt 7) { 
    Write-Error "Robocopy failed with exit code $LASTEXITCODE" 
} else {
    $global:LASTEXITCODE = 0
}

# Copy bootstrap.py
Copy-Item -Path "src\bootstrap.py" -Destination "$DistDir\bootstrap.py"

# Copy app.py (Main Streamlit App)
Copy-Item -Path "app.py" -Destination "$DistDir\app.py"

# Copy .streamlit folder
if (Test-Path ".streamlit") {
    Copy-Item -Recurse -Path ".streamlit" -Destination "$DistDir\.streamlit"
    # Ensure toolbarMode is viewer
    Add-Content -Path "$DistDir\.streamlit\config.toml" -Value "`n[client]`ntoolbarMode = `"viewer`""
}

# Copy demo_data folder
if (Test-Path "demo_data") {
    Copy-Item -Recurse -Path "demo_data" -Destination "$DistDir\demo_data"
}

# 6. Build Native Launcher
Write-Host "[6/6] Building Native Launcher (Rust)..." -ForegroundColor Yellow

# Ensure cargo is available
if (Get-Command cargo -ErrorAction SilentlyContinue) {
    Push-Location "tools\launcher"
    Write-Host "  - Compiling launcher..." -ForegroundColor Gray
    cargo build
    cargo build --release
    Pop-Location
    
    $LauncherSrcRelease = "tools\launcher\target\release\launcher.exe"
    if (Test-Path $LauncherSrcRelease) {
        Copy-Item -Path $LauncherSrcRelease -Destination "$DistDir\mycol.exe"
        Write-Host "  - Launcher copied to mycol.exe" -ForegroundColor Gray
    } else {
        Write-Error "Launcher compilation failed or output not found."
    }

    $LauncherSrcDebug = "tools\launcher\target\debug\launcher.exe"
    if (Test-Path $LauncherSrcDebug) {
        Copy-Item -Path $LauncherSrcDebug -Destination "$DistDir\mycol_debug.exe"
        Write-Host "  - Launcher copied to mycol_debug.exe" -ForegroundColor Gray
    } else {
        Write-Error "Launcher compilation failed or output not found."
    }
} else {
    Write-Error "Cargo (Rust) not found. Cannot build native launcher."
    Write-Host "Falling back to batch file..." -ForegroundColor Yellow
    $LauncherContent = @"
@echo off
set "HERE=%~dp0"
"%HERE%bin\python_main\python.exe" "%HERE%bootstrap.py" %*
"@
    Set-Content "$DistDir\MyCol_fallback.bat" $LauncherContent
}

# Cleanup unnecessary files from dist if exist
if (Test-Path "$DistDir\pwa.py") { Remove-Item "$DistDir\pwa.py" }

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Build Complete!" -ForegroundColor Green
Write-Host "Portable App: dist\MyCol" -ForegroundColor Cyan
Write-Host "Run: dist\MyCol\MyCol.exe" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Green
