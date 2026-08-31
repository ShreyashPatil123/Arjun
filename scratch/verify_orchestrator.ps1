$registryPath = "$env:APPDATA\com.sarathi.app\models\registry.json"
$content = Get-Content -Raw $registryPath | ConvertFrom-Json
$orchEntry = $content.models | Where-Object { $_.id -eq "orchestrator" -or ($_.id -like "orchestrator.*") }
if ($orchEntry) {
    Write-Host "Found orchestrator entry:"
    Write-Host "  id: $($orchEntry.id)"
    Write-Host "  name: $($orchEntry.name)"
    Write-Host "  path: $($orchEntry.path)"
} else {
    Write-Host "No orchestrator entry found"
}