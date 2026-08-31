# Builds the frontend, then launches the Sarathi/Arjun Tauri app.
#
# The app is a Tauri 2 desktop app. The release binary at
# `target/release/sarathi.exe` embeds the frontend from `dist/`
# at build time. If `dist/` is stale, the webview inside the
# app shows "localhost refused to connect" because the binary
# still has the old bundle embedded.
#
# This script:
# 1. Stops any running Sarathi app.
# 2. Builds the frontend (`npm run build`) so dist/ is current.
# 3. Builds the Tauri release binary via `npx tauri build --no-bundle`
#    so the new dist/ is embedded in the binary.
# 4. Launches the release binary.

$ErrorActionPreference = "Stop"
Set-Location "c:\Users\lenovo\Desktop\Arjun-1"

Write-Host "Stopping any running Sarathi app..." -ForegroundColor Yellow
Get-Process -Name "sarathi" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  Killing PID $($_.Id)" -ForegroundColor Gray
    $_ | Stop-Process -Force
}
Start-Sleep -Seconds 1

Write-Host ""
Write-Host "Building frontend dist..." -ForegroundColor Yellow
npm run build 2>&1 | Select-Object -Last 5

Write-Host ""
Write-Host "Building Tauri release binary (embeds the new dist/)..." -ForegroundColor Yellow
npx tauri build --no-bundle 2>&1 | Select-Object -Last 5

Write-Host ""
Write-Host "Launching Sarathi app..." -ForegroundColor Green
Write-Host "  Binary: C:\Users\lenovo\Desktop\Arjun-1\src-tauri\target\release\sarathi.exe" -ForegroundColor Gray
Write-Host ""
Start-Process -FilePath "C:\Users\lenovo\Desktop\Arjun-1\src-tauri\target\release\sarathi.exe"

Write-Host "App launched. The window should open shortly." -ForegroundColor Green