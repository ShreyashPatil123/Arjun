# Gateway smoke test.
#
# Checks the local gateway from the outside, exactly as Claude Code and opencode
# would reach it. Run this with the Sarathi app open.
#
#   powershell -File scratch\test_gateway.ps1
#
# Every check reports PASS or FAIL and says what the result means, so a failure
# points at the cause instead of just going red.

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

Write-Host "`nSarathi gateway smoke test -> $base`n" -ForegroundColor Cyan

# 1. Is it up at all?
Write-Host "Server reachable"
try {
    $health = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 5
    Check "GET /health responds" $true ""
    Write-Host "        model loaded: $($health.modelLoaded)  ($($health.model))" -ForegroundColor DarkGray
    $modelLoaded = $health.modelLoaded
} catch {
    Check "GET /health responds" $false "Is the Sarathi app running? $($_.Exception.Message)"
    Write-Host "`nCannot continue without the server.`n" -ForegroundColor Red
    exit 1
}

# 2. Model listing (OpenAI discovery endpoint)
Write-Host "`nModel listing"
try {
    $models = Invoke-RestMethod -Uri "$base/v1/models" -TimeoutSec 5
    Check "GET /v1/models returns a list" ($models.object -eq 'list') "got object=$($models.object)"
} catch {
    Check "GET /v1/models returns a list" $false $_.Exception.Message
}

# 3. Browser requests must be blocked; CLI requests must pass.
Write-Host "`nOrigin guard (blocks web pages, allows tools)"
try {
    Invoke-WebRequest -Uri "$base/health" -Headers @{ Origin = 'https://evil.com' } -TimeoutSec 5 | Out-Null
    Check "rejects a web page Origin" $false "request was allowed - the guard is not working"
} catch {
    $code = $_.Exception.Response.StatusCode.value__
    Check "rejects a web page Origin" ($code -eq 403) "expected 403, got $code"
}

try {
    Invoke-WebRequest -Uri "$base/health" -Headers @{ Host = 'evil.com' } -TimeoutSec 5 | Out-Null
    Check "rejects a foreign Host header" $false "request was allowed - DNS rebinding is possible"
} catch {
    $code = $_.Exception.Response.StatusCode.value__
    Check "rejects a foreign Host header" ($code -eq 403) "expected 403, got $code"
}

if (-not $modelLoaded) {
    Write-Host "`nNo model is loaded, so generation checks are skipped." -ForegroundColor Yellow
    Write-Host "Open Sarathi, load a model, and run this again.`n" -ForegroundColor Yellow
    Write-Host "$pass passed, $fail failed`n"
    exit ([int]($fail -gt 0))
}

# 4. OpenAI shape - what opencode / openclaw / Cursor send.
Write-Host "`nOpenAI endpoint (opencode, openclaw, Cursor)"
$body = @{
    model    = 'gpt-4o'          # deliberately unknown; Sarathi serves its loaded model
    messages = @(@{ role = 'user'; content = 'Reply with exactly: OK' })
    stream   = $false
    max_tokens = 32
} | ConvertTo-Json -Depth 5

try {
    $r = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 180
    Check "non-streaming reply" ($null -ne $r.choices[0].message.content) "no content in choices[0]"
    Check "reports a finish reason" ($null -ne $r.choices[0].finish_reason) "finish_reason missing"
    Write-Host "        model said: $($r.choices[0].message.content.Trim())" -ForegroundColor DarkGray
} catch {
    Check "non-streaming reply" $false $_.Exception.Message
}

# 5. Anthropic shape - what Claude Code sends.
Write-Host "`nAnthropic endpoint (Claude Code)"
$body = @{
    model      = 'claude-sonnet-4-5'
    max_tokens = 32
    system     = 'You are terse.'
    messages   = @(@{ role = 'user'; content = 'Reply with exactly: OK' })
    stream     = $false
} | ConvertTo-Json -Depth 5

try {
    $r = Invoke-RestMethod -Uri "$base/v1/messages" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 180
    Check "non-streaming reply" ($r.type -eq 'message') "expected type=message, got $($r.type)"
    Check "content is a text block" ($r.content[0].type -eq 'text') "expected a text block"
    Check "reports a stop reason" ($null -ne $r.stop_reason) "stop_reason missing"
    Write-Host "        model said: $($r.content[0].text.Trim())" -ForegroundColor DarkGray
} catch {
    Check "non-streaming reply" $false $_.Exception.Message
}

# 6. Streaming - the path both tools actually use day to day.
#
# Uses curl.exe rather than Invoke-WebRequest. PowerShell's client buffers and
# post-processes the response, and throws a NullReferenceException on a
# server-sent-event stream -- which looks exactly like a gateway failure but is
# not one. curl.exe ships with Windows 10+ and reads SSE as plain bytes.
Write-Host "`nStreaming"

function Invoke-Stream($path, $json) {
    $tmp = [System.IO.Path]::GetTempFileName()
    try {
        # UTF8Encoding($false) = no byte-order mark. PowerShell 5.1's
        # `Set-Content -Encoding utf8` prepends a BOM, curl sends it as the first
        # bytes of the body, and the server rejects the request as invalid JSON --
        # which shows up here as "no chunks in the stream" rather than a parse error.
        [System.IO.File]::WriteAllText($tmp, $json, (New-Object System.Text.UTF8Encoding $false))
        & curl.exe -s -m 180 -X POST "$base$path" -H "Content-Type: application/json" --data-binary "@$tmp"
    } finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
}

$openaiStream = Invoke-Stream '/v1/chat/completions' '{"messages":[{"role":"user","content":"Count: 1 2 3"}],"stream":true,"max_tokens":32}'
$openaiText = $openaiStream -join "`n"
Check "OpenAI stream sends chunks"   ($openaiText -match 'chat\.completion\.chunk') "no chunk objects in the stream"
Check "OpenAI stream ends with [DONE]" ($openaiText -match '\[DONE\]') "missing the [DONE] sentinel clients wait for"

$anthropicStream = Invoke-Stream '/v1/messages' '{"model":"claude-sonnet-4-5","max_tokens":32,"messages":[{"role":"user","content":"Count: 1 2 3"}],"stream":true}'
$anthropicText = $anthropicStream -join "`n"
# Claude Code dispatches on these event names and needs them in this order.
Check "Anthropic stream opens with message_start" ($anthropicText -match 'message_start') "missing message_start"
Check "Anthropic stream sends text deltas"        ($anthropicText -match 'content_block_delta') "missing content_block_delta"
Check "Anthropic stream closes properly"          ($anthropicText -match 'message_stop') "missing message_stop"

Write-Host "`n$pass passed, $fail failed`n" -ForegroundColor $(if ($fail -gt 0) { 'Red' } else { 'Green' })
exit ([int]($fail -gt 0))
