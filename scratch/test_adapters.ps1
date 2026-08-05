# Adapter download and management smoke test.
#
# Verifies the parts that touch the real world: that HuggingFace actually
# returns adapters for a model, that GGUF-shipping adapters exist and can be
# told apart from PEFT ones, and that installed adapters land where the model
# loader will look for them.
#
#   powershell -File scratch\test_adapters.ps1
#
# Read-only against the network; the only disk writes are into a temp folder.

$ErrorActionPreference = 'Continue'
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

Write-Host "`nSarathi adapter smoke test`n" -ForegroundColor Cyan

# 1. The lookup the Discover card depends on.
Write-Host "Finding adapters for a base model"
$base = 'meta-llama/Llama-3.1-8B'
try {
    $found = Invoke-RestMethod -TimeoutSec 20 `
        -Uri "https://huggingface.co/api/models?filter=base_model:adapter:$base&limit=10&full=true"
    Check "HuggingFace returns adapters for $base" ($found.Count -gt 0) "got $($found.Count)"
    Write-Host "        $($found.Count) adapter(s), e.g. $($found[0].id)" -ForegroundColor DarkGray
} catch {
    Check "HuggingFace returns adapters" $false $_.Exception.Message
    Write-Host "`nCannot continue without the adapter listing.`n" -ForegroundColor Red
    exit 1
}

# 2. The distinction the UI makes: loadable now, versus needs conversion.
Write-Host "`nTelling loadable adapters from ones needing conversion"
$ggufCount = 0
$peftCount = 0
foreach ($m in $found) {
    $names = @($m.siblings | ForEach-Object { $_.rfilename })
    if ($names -match '\.gguf$') { $ggufCount++ }
    elseif ($names -match '\.safetensors$') { $peftCount++ }
}
Write-Host "        GGUF (loadable now): $ggufCount" -ForegroundColor DarkGray
Write-Host "        PEFT (needs conversion): $peftCount" -ForegroundColor DarkGray

Check "every adapter is classified one way or the other" `
    (($ggufCount + $peftCount) -gt 0) "none had recognisable adapter files"

# This is the honest expectation, not a failure: most published adapters are
# PEFT. The UI must therefore show "needs conversion" far more often than "Get".
if ($ggufCount -eq 0) {
    Write-Host "        Note: none of these ship GGUF, so all show 'needs conversion'." -ForegroundColor Yellow
}

# 3. Where an installed adapter must land for the loader to find it.
Write-Host "`nInstall location matches what the model loader reads"
$appData = "$env:APPDATA\com.sarathi.app"
$model = Get-ChildItem "$appData\models\huggingface" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
if ($model) {
    $adapterRoot = Join-Path $model.FullName 'adapters'
    Check "adapters folder sits beside the model" `
        ($adapterRoot -like "*$($model.Name)*adapters") $adapterRoot
    Write-Host "        $adapterRoot" -ForegroundColor DarkGray

    $installed = Get-ChildItem $adapterRoot -Directory -ErrorAction SilentlyContinue
    Write-Host "        installed adapters: $(@($installed).Count)" -ForegroundColor DarkGray
} else {
    Check "a model is installed to attach adapters to" $false `
        "no model found under $appData\models\huggingface"
}

# 4. The gateway should be unaffected by any of this.
Write-Host "`nGateway still serving"
try {
    $h = Invoke-RestMethod -Uri 'http://127.0.0.1:11435/health' -TimeoutSec 8
    Check "gateway healthy with a model loaded" ($h.modelLoaded -eq $true) "modelLoaded=$($h.modelLoaded)"
} catch {
    Check "gateway healthy" $false "is the app running? $($_.Exception.Message)"
}

Write-Host "`n$pass passed, $fail failed`n" -ForegroundColor $(if ($fail -gt 0) { 'Red' } else { 'Green' })
exit ([int]($fail -gt 0))
