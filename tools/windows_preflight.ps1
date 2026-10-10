[CmdletBinding()]
param(
    [switch]$InstallRuntime,
    [switch]$ValidateLocalData
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($null -ne $PyLauncher) {
    & py -3.11 -m venv .venv
} else {
    & python -m venv .venv
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Expected virtual-environment Python at $Python"
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements-dev.txt
& $Python -m compileall -q src tests
& $Python -m pytest
& $Python -m ruff check src tests

if ($InstallRuntime) {
    Write-Host "Installing Ultralytics runtime packages..."
    Write-Host "Install the appropriate PyTorch/torchvision build from pytorch.org first if it is not already present."
    & $Python -m pip install -r requirements.txt
}

if ($ValidateLocalData) {
    & $Python -m src.validate_data --config config.yaml
}

Write-Host "Windows preflight completed successfully."
