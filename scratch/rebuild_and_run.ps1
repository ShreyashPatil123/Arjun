# Stops the running Sarathi app, rebuilds the Tauri release binary
# (which embeds the latest dist/ and Rust code), and informs the
# user how to relaunch.

$ErrorActionPreference = "Stop"
Set-Location "c:\Users\lenovo\Desktop\Arjun-1"

Write-Host "Stopping running Sarathi app..." -ForegroundColor Yellow
Get-Process -Name "sarathi" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  Killing PID $($_.Id)" -ForegroundColor Gray
    $_ | Stop-Process -Force
}
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "Rebuilding frontend dist..." -ForegroundColor Yellow
npm run build 2>&1 | Select-Object -Last 5

Write-Host ""
Write-Host "Rebuilding Tauri release binary (embeds the new dist/)..." -ForegroundColor Yellow
npx tauri build --no-bundle 2>&1 | Select-Object -Last 5

Write-Host ""
Write-Host "Build complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To use the updated app, run:" -ForegroundColor Cyan
Write-Host "  C:\Users\lenovo\Desktop\Arjun-1\src-tauri\target\release\sarathi.exe" -ForegroundColor Gray
Write-Host ""
Write-Host "Or use scratch/run_app.ps1 which does this for you." -ForegroundColor White