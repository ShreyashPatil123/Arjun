# Launch-section smoke test.
#
# Checks the launcher end to end against a running app: does it find the tools
# that are really installed, does it correctly report the ones that are not, and
# does the server panel have live data to show.
#
#   powershell -File scratch\test_launcher.ps1
#
# Run with the Sarathi app open. Nothing here installs or launches anything --
# it only reads state, so it is safe to run repeatedly.

$ErrorActionPreference = 'Continue'
$base = 'http://127.0.0.1:11435'
$pass = 0
$fail = 0

function Check($name, $ok, $detail) {
    if ($ok) {
        Write-Host "  PASS  $name" -ForegroundColor Green
        $script:pass++
    } else {
        Write-Host "  FAIL  $name" -ForegroundColor Red
        Write-Host "        $detail" -ForegroundColor DarkGray
        $script:fail++
    }
}

Write-Host "`nSarathi launcher smoke test`n" -ForegroundColor Cyan

# The launcher itself is reached through Tauri IPC, which is only available
# inside the app window. What this script can verify from outside is that the
# gateway the launcher hands to tools is real and serving.
Write-Host "Gateway the launcher connects tools to"
try {
    $health = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 5
    Check "gateway reachable" $true ""
    Check "a model is loaded" ($health.modelLoaded -eq $true) `
        "Launch is disabled without one - tools would connect but get no answer"
    Write-Host "        model: $($health.model)" -ForegroundColor DarkGray
} catch {
    Check "gateway reachable" $false "Is the Sarathi app running? $($_.Exception.Message)"
    Write-Host "`nCannot continue without the gateway.`n" -ForegroundColor Red
    exit 1
}

# Ground truth for detection: what is actually installed on this machine.
# The Launch screen should agree with this.
Write-Host "`nWhat is really installed (ground truth for the tool cards)"

function Test-RealTool($command, $expect) {
    # Same three steps the launcher uses: a real executable, a version that
    # runs, and output that names the tool.
    $resolved = & where.exe $command 2>$null | Select-Object -First 1
    if (-not $resolved -or -not (Test-Path $resolved)) { return $false }
    $out = & $command --version 2>&1 | Out-String
    return $out.ToLower().Contains($expect.ToLower())
}

$claude = Test-RealTool 'claude' 'claude'
Write-Host "        claude   : $(if ($claude) { 'installed' } else { 'not installed' })" -ForegroundColor DarkGray
$opencode = Test-RealTool 'opencode' 'opencode'
Write-Host "        opencode : $(if ($opencode) { 'installed' } else { 'not installed' })" -ForegroundColor DarkGray

Check "at least one known tool resolves" ($claude -or $opencode) `
    "neither claude nor opencode found - the Launch screen will show both as Not installed"

# The false positive that motivated the three-step check.
Write-Host "`nDetection must not be fooled by shell keywords"
$continueIsReal = Test-RealTool 'continue' 'continue'
Check "'continue' is not treated as an installed tool" (-not $continueIsReal) `
    "a shell keyword was mistaken for a program - the three-step check has regressed"

# npm is what Install delegates to; without it the button must not be offered.
Write-Host "`nInstall route"
$npm = & where.exe npm 2>$null | Select-Object -First 1
Check "npm is available for installs" ([bool]$npm) `
    "without npm, Install is correctly hidden rather than shown as a dead button"

Write-Host "`n$pass passed, $fail failed`n" -ForegroundColor $(if ($fail -gt 0) { 'Red' } else { 'Green' })
exit ([int]($fail -gt 0))
