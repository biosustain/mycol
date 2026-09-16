#Requires -Version 7.0
<#
.SYNOPSIS
    Build the portable Windows bundle(s) for Mycol.

.DESCRIPTION
    Produces a self-contained folder under dist\ that needs no Python, pip, uv or
    git on the target machine, and zips it for upload to GitHub Releases.

.PARAMETER Variant
    cpu   - CPU-only torch. The default, and what most users want.
    cuda  - NVIDIA CUDA 12.6 torch, roughly 2.5 GB larger.
    both  - Build both, one after the other.

.PARAMETER SkipModels
    Skip pre-baking the model weights. The app then downloads them on first use,
    which is the behaviour we ship bundles to avoid — for local test builds only.
#>
param(
    [string]$Version = "0.1.0",
    [ValidateSet("cpu", "cuda", "both")]
    [string]$Variant = "cpu",
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"

# Embeddable distributions: the main app runs on 3.12, the training worker on
# 3.10 (cellpose 3.x pins numpy<2, which constrains what the worker can use).
#
# 3.12 rather than 3.13 is deliberate and load-bearing: cellpose requires
# numpy<2.1, and numpy only gained cp313 wheels in 2.1.0. On 3.13 there is no
# numpy satisfying both, so uv falls back to building it from an sdist, which
# fails in the embeddable environment (no Python headers for meson to find).
# 3.12 is also what the project's own .venv uses.
$PYTHON_MAIN_URL = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"
$PYTHON_WORKER_URL = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"

$CPU_INDEX = "https://download.pytorch.org/whl/cpu"
$CUDA_INDEX = "https://download.pytorch.org/whl/cu126"

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

function Write-Step($msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Detail($msg) { Write-Host "  - $msg" -ForegroundColor Gray }

function Get-EmbeddedPython {
    param($Url, $Zip, $Target, $PthName, $ZipName)

    if (-not (Test-Path $Zip)) {
        Write-Detail "Downloading $(Split-Path $Zip -Leaf)..."
        Invoke-WebRequest -Uri $Url -OutFile $Zip
    }
    Write-Detail "Extracting to $(Split-Path $Target -Leaf)..."
    Expand-Archive $Zip -DestinationPath $Target -Force

    # The embeddable build ships with site-packages disabled; re-enable it so uv
    # can install into Lib\site-packages.
    Set-Content "$Target\$PthName" "$ZipName`n.`nLib\site-packages`n`nimport site`n"
}

function Build-Bundle {
    param([string]$BuildVariant)

    $DistDir = "$ProjectRoot\dist\mycol-windows-$BuildVariant"
    $BinDir = "$DistDir\bin"
    $BuildDir = "$ProjectRoot\build"

    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Building Mycol v$Version ($BuildVariant)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    Write-Step "`n[1/8] Cleaning..."
    if (Test-Path $DistDir) { Remove-Item $DistDir -Recurse -Force }
    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
    New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null

    Write-Step "[2/8] Main environment (Python 3.12)..."
    $MainDir = "$BinDir\python_main"
    Get-EmbeddedPython -Url $PYTHON_MAIN_URL -Zip "$BuildDir\python-3.12.zip" `
        -Target $MainDir -PthName "python312._pth" -ZipName "python312.zip"

    Write-Step "[3/8] Worker environment (Python 3.10)..."
    $WorkerDir = "$BinDir\python_worker"
    Get-EmbeddedPython -Url $PYTHON_WORKER_URL -Zip "$BuildDir\python-3.10.zip" `
        -Target $WorkerDir -PthName "python310._pth" -ZipName "python310.zip"

    Write-Step "[4/8] Resolving dependencies..."
    Write-Detail "exporting main requirements..."
    uv export --no-dev --python 3.12 -o "$BuildDir\req_main.txt"
    if ($LASTEXITCODE -ne 0) { throw "uv export (main) failed" }

    Write-Detail "exporting worker requirements..."
    Push-Location src\training
    uv export --no-dev --python 3.10 -o "$BuildDir\req_worker.txt"
    $exportCode = $LASTEXITCODE
    Pop-Location
    if ($exportCode -ne 0) { throw "uv export (worker) failed" }

    Write-Step "[5/8] Installing dependencies..."
    # The lock pins +cpu local versions, which only exist on the PyTorch index,
    # so it has to be reachable even though the requirements carry no index
    # directive of their own.
    Write-Detail "installing main (py3.12)..."
    uv pip install --python "$MainDir\python.exe" -r "$BuildDir\req_main.txt" `
        --extra-index-url $CPU_INDEX --index-strategy unsafe-best-match
    if ($LASTEXITCODE -ne 0) { throw "main dependency install failed" }

    Write-Detail "installing worker (py3.10)..."
    uv pip install --python "$WorkerDir\python.exe" -r "$BuildDir\req_worker.txt" `
        --extra-index-url $CPU_INDEX --index-strategy unsafe-best-match
    if ($LASTEXITCODE -ne 0) { throw "worker dependency install failed" }

    if ($BuildVariant -eq "cuda") {
        # Overlay the CUDA build over the CPU one. Keeping this out of the lock
        # means the default `uv sync` stays small for everyone else.
        # --reinstall is load-bearing: the CPU wheels already satisfy a bare
        # "torch" requirement, so without it uv reports "Checked 2 packages",
        # installs nothing, and the cuda bundle ships as CPU-only.
        Write-Detail "overlaying CUDA torch (cu126)..."
        foreach ($py in @("$MainDir\python.exe", "$WorkerDir\python.exe")) {
            uv pip install --python $py --reinstall torch torchvision --index-url $CUDA_INDEX
            if ($LASTEXITCODE -ne 0) { throw "CUDA overlay failed for $py" }
        }
        # Fail the build rather than ship a mislabelled bundle.
        $torchBuild = & "$MainDir\python.exe" -c "import torch; print(torch.version.cuda)"
        if (-not $torchBuild -or $torchBuild -eq "None") {
            throw "cuda variant still has a CPU torch (torch.version.cuda = '$torchBuild')"
        }
        Write-Detail "torch CUDA runtime: $torchBuild"
    }

    Write-Step "[6/8] Copying application files..."
    $RoboArgs = @(
        "src", "$DistDir\src", "/E",
        "/XD", ".venv", "__pycache__", ".git", ".pytest_cache", "dist", "build", "*.egg-info", ".mypy_cache",
        "/nfl", "/ndl", "/njh", "/njs", "/nc", "/ns", "/np"
    )
    robocopy @RoboArgs | Out-Null
    # Robocopy uses exit codes 0-7 for success; 8+ is a real failure.
    if ($LASTEXITCODE -gt 7) { throw "robocopy failed with exit code $LASTEXITCODE" }
    $global:LASTEXITCODE = 0

    Copy-Item -Path "src\bootstrap.py" -Destination "$DistDir\bootstrap.py"
    Copy-Item -Path "app.py" -Destination "$DistDir\app.py"
    # app.py calls st.logo("logo.png") at import time, so the bundle is broken
    # without it.
    Copy-Item -Path "logo.png" -Destination "$DistDir\logo.png"
    if (Test-Path "logo_icon.png") {
        Copy-Item -Path "logo_icon.png" -Destination "$DistDir\logo_icon.png"
    }
    # load_demo_data() expects this beside app.py.
    if (Test-Path "example_session.zip") {
        Copy-Item -Path "example_session.zip" -Destination "$DistDir\example_session.zip"
    }
    Copy-Item -Path "LICENSE" -Destination "$DistDir\LICENSE"

    if (Test-Path ".streamlit") {
        Copy-Item -Recurse -Path ".streamlit" -Destination "$DistDir\.streamlit"
        Add-Content -Path "$DistDir\.streamlit\config.toml" -Value "`n[client]`ntoolbarMode = `"viewer`""
    }
    if (Test-Path "demo_data") {
        Copy-Item -Recurse -Path "demo_data" -Destination "$DistDir\demo_data"
    }

    $readme = @"
Mycol $Version ($BuildVariant build)

To start: double-click mycol.exe

Windows may warn that it "protected your PC" because this download is not
code-signed. Choose "More info", then "Run anyway".

If nothing appears, run mycol_debug.exe to see the startup output, or read the
log at %LOCALAPPDATA%\Mycol\mycol.log

Keep this folder together - mycol.exe expects bin\ and src\ beside it.

Docs: https://biosustain.github.io/mycol/
"@
    Set-Content -Path "$DistDir\README.txt" -Value $readme

    Write-Step "[7/8] Pre-baking model weights..."
    if ($SkipModels) {
        Write-Detail "skipped (-SkipModels)"
    } else {
        & "$MainDir\python.exe" "$ProjectRoot\scripts\fetch_models.py" $DistDir
        if ($LASTEXITCODE -ne 0) { throw "model pre-bake failed" }
    }

    Write-Step "[8/8] Building native launcher..."
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw "cargo (Rust) not found - install from https://rustup.rs to build the launcher."
    }
    Push-Location "tools\launcher"
    cargo build --release
    $relCode = $LASTEXITCODE
    cargo build
    $dbgCode = $LASTEXITCODE
    Pop-Location
    if ($relCode -ne 0 -or $dbgCode -ne 0) { throw "launcher build failed" }

    Copy-Item -Path "tools\launcher\target\release\launcher.exe" -Destination "$DistDir\mycol.exe"
    # Same binary without windows_subsystem="windows", so users can see errors.
    Copy-Item -Path "tools\launcher\target\debug\launcher.exe" -Destination "$DistDir\mycol_debug.exe"
    Write-Detail "mycol.exe + mycol_debug.exe"

    if (Test-Path "$DistDir\pwa.py") { Remove-Item "$DistDir\pwa.py" }

    # No version in the name: the README links to
    # releases/latest/download/<name>, which needs a stable filename.
    $Zip = "$ProjectRoot\dist\mycol-windows-$BuildVariant.zip"
    if (Test-Path $Zip) { Remove-Item $Zip -Force }
    Write-Detail "compressing to $(Split-Path $Zip -Leaf)..."
    Compress-Archive -Path "$DistDir\*" -DestinationPath $Zip -CompressionLevel Optimal

    $bytes = (Get-Item $Zip).Length
    $sizeMb = [math]::Round($bytes / 1MB, 0)
    Write-Host "`nBuilt: $Zip ($sizeMb MB)" -ForegroundColor Green

    # GitHub caps a single release asset at 2 GiB. The CUDA bundle can approach
    # that, and the upload fails late and unhelpfully when it does.
    if ($bytes -gt 2GB) {
        Write-Warning ("$(Split-Path $Zip -Leaf) is $sizeMb MB, over GitHub's 2 GiB " +
            "release asset limit. Host it elsewhere or split it before publishing.")
    }
}

$targets = if ($Variant -eq "both") { @("cpu", "cuda") } else { @($Variant) }
foreach ($t in $targets) { Build-Bundle -BuildVariant $t }

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Build complete. Artifacts in dist\" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
