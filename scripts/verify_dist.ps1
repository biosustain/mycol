#Requires -Version 7.0
<#
.SYNOPSIS
    Smoke-test a built Mycol bundle.

.DESCRIPTION
    Checks that the bundle contains everything it needs, that its config parses,
    that the heavy imports work in the embedded interpreters, and that the app
    actually serves a page. Intended to run in CI straight after make_dist.ps1
    so a broken bundle never reaches a release.
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$DistDir
)

$ErrorActionPreference = "Stop"
$failures = @()

function Assert-BundleFile($relPath, $label) {
    $full = Join-Path $DistDir $relPath
    if (Test-Path $full) {
        Write-Host "  OK   $label" -ForegroundColor Green
    } else {
        Write-Host "  FAIL $label (missing: $relPath)" -ForegroundColor Red
        $script:failures += $label
    }
}

Write-Host "`nVerifying bundle: $DistDir" -ForegroundColor Cyan

Write-Host "`n[1/5] Required files..." -ForegroundColor Yellow
Assert-BundleFile "mycol.exe"                      "launcher"
Assert-BundleFile "mycol_debug.exe"                "debug launcher"
Assert-BundleFile "bootstrap.py"                   "bootstrap"
Assert-BundleFile "app.py"                         "app entry point"
# app.py calls st.logo("logo.png") during startup; omitting it broke the bundle
# before, and nothing else would have caught it.
Assert-BundleFile "logo.png"                       "logo asset"
Assert-BundleFile "README.txt"                     "user readme"
Assert-BundleFile ".streamlit/config.toml"         "streamlit config"
Assert-BundleFile "src/helpers"                    "src tree"
Assert-BundleFile "bin/python_main/python.exe"     "main interpreter"
Assert-BundleFile "bin/python_worker/python.exe"   "worker interpreter"

Write-Host "`n[2/5] Pre-baked models..." -ForegroundColor Yellow
Assert-BundleFile "models/mobile_sam.pt"           "MobileSAM weights"
Assert-BundleFile "models/cellpose/cyto3"          "Cellpose cyto3"
Assert-BundleFile "models/cellpose/cyto2torch_0"   "Cellpose cyto2"
Assert-BundleFile "models/cellpose/nucleitorch_0"  "Cellpose nuclei"

$mainPy = Join-Path $DistDir "bin/python_main/python.exe"
$workerPy = Join-Path $DistDir "bin/python_worker/python.exe"

Write-Host "`n[3/5] Config parses..." -ForegroundColor Yellow
# A previous bug appended a literal "\n" here and produced invalid TOML, which
# only showed up as a failure to start.
& $mainPy -c "import tomllib,sys; tomllib.load(open(sys.argv[1],'rb')); print('config.toml parses')" (Join-Path $DistDir ".streamlit/config.toml")
if ($LASTEXITCODE -ne 0) { $failures += "config.toml is not valid TOML" }

Write-Host "`n[4/5] Imports in embedded interpreters..." -ForegroundColor Yellow
& $mainPy -c @"
import streamlit, torch, cellpose, mobile_sam, huggingface_hub, plotly, cv2, skimage
print('main imports OK  torch', torch.__version__, '| cuda build:', torch.version.cuda)
"@
if ($LASTEXITCODE -ne 0) { $failures += "main interpreter imports" }

& $workerPy -c @"
import torch, cellpose, optuna, numpy
print('worker imports OK  torch', torch.__version__, '| numpy', numpy.__version__)
"@
if ($LASTEXITCODE -ne 0) { $failures += "worker interpreter imports" }

Write-Host "`n[5/5] App serves a page..." -ForegroundColor Yellow
# A fixed port silently invalidates this check: anything else already listening
# there answers the request and the bundle appears to work when it never started.
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
$listener.Start()
$port = $listener.LocalEndpoint.Port
$listener.Stop()
Write-Host "  using port $port" -ForegroundColor Gray
$proc = Start-Process -FilePath $mainPy -PassThru -WindowStyle Hidden `
    -WorkingDirectory $DistDir `
    -ArgumentList @(
        "-m", "streamlit", "run", "app.py",
        "--server.headless", "true",
        "--server.port", "$port",
        "--server.address", "127.0.0.1",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false"
    )

$served = $false
$deadline = (Get-Date).AddSeconds(180)
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) { break }
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$port" -TimeoutSec 5 -UseBasicParsing
        if ($resp.StatusCode -eq 200) { $served = $true; break }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

if (-not $proc.HasExited) { $proc.Kill(); $proc.WaitForExit() }

if ($served) {
    Write-Host "  OK   app returned HTTP 200" -ForegroundColor Green
} else {
    Write-Host "  FAIL app never served a page" -ForegroundColor Red
    $failures += "app startup"
}

Write-Host ""
if ($failures.Count -gt 0) {
    Write-Host "FAILED: $($failures.Count) check(s)" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
Write-Host "All checks passed." -ForegroundColor Green
