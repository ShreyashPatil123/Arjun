# Stop the running Sarathi app, rebuild the release binary with the latest
# router changes, and inform the user how to relaunch.

$ErrorActionPreference = "Stop"

Write-Host "Stopping running Sarathi app..." -ForegroundColor Yellow
Get-Process -Name "sarathi" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  Killing PID $($_.Id) (path: $($_.Path))" -ForegroundColor Gray
    $_ | Stop-Process -Force
}
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "Building release binary with latest router changes..." -ForegroundColor Yellow
Set-Location "c:\Users\lenovo\Desktop\Arjun-1\src-tauri"
cargo build --release 2>&1 | Select-Object -Last 5

Write-Host ""
Write-Host "Build complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To use the updated app, run one of:" -ForegroundColor Cyan
Write-Host "  1. Run the release binary directly:" -ForegroundColor White
Write-Host "     C:\Users\lenovo\Desktop\Arjun-1\src-tauri\target\release\sarathi.exe" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Or use npm run tauri:dev for dev mode (uses debug build):" -ForegroundColor White
Write-Host "     cd C:\Users\lenovo\Desktop\Arjun-1" -ForegroundColor Gray
Write-Host "     npm run tauri:dev" -ForegroundColor Gray
Write-Host ""
Write-Host "The router now prefers the orchestrator (gemma-4-12b-it)." -ForegroundColor Green