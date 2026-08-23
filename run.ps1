# Helpa - ONE command (Windows PowerShell)
#   irm https://raw.githubusercontent.com/Yusasif-A/Helpa/main/run.ps1 | iex
$ErrorActionPreference = "Stop"
function Say($m){ Write-Host "`n> $m" -ForegroundColor Green }

if (-not (Test-Path ".\app.py")) {
  Say "getting the app..."
  if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "git is required. Install from https://git-scm.com then re-run."
    exit 1
  }
  git clone --depth 1 https://github.com/Yusasif-A/Helpa.git "$env:TEMP\helpa-src"
  Set-Location "$env:TEMP\helpa-src"
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Host "Python required. Install from https://python.org (tick 'Add to PATH'), then re-run."
  exit 1
}
Say "using $(python --version)"

$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
python -c "import gradio, requests, PIL, cv2" *> $null
$depsOk = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $prevEAP

if (-not $depsOk) {
  Say "installing dependencies..."
  python -m pip install --quiet -r requirements.txt
}

Say "starting Helpa..."
Write-Host "  First run downloads the model (~3.2 GB) and vision projector (~940 MB) once."
Write-Host "  Then opens at http://localhost:7861 - runs on THIS computer from then on."
Write-Host "  Ctrl+C to stop.`n"
python app.py
