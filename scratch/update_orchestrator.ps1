$registryPath = "$env:APPDATA\com.sarathi.app\models\registry.json"
$content = Get-Content -Raw $registryPath | ConvertFrom-Json

# Find gemma-4-12b-it entry to use as template
$gemmaEntry = $content.models | Where-Object { $_.id -eq "google_gemma-4-12b-it_UD-Q4_K_XL" } | Select-Object -First 1
if (-not $gemmaEntry) {
    Write-Error "Could not find google_gemma-4-12b-it_UD-Q4_K_XL in registry"
    exit 1
}

# Create a new orchestrator entry based on gemma-4-12b-it
$newEntry = $gemmaEntry.PSObject.Copy()
$newEntry.id = "orchestrator.gemma-4-12b-it"

# Remove any existing orchestrator entries
$content.models = @($content.models | Where-Object { $_.id -ne "orchestrator" -and -not ($_.id -like "orchestrator.*") })

# Add the new orchestrator entry at the beginning
$content.models = @($newEntry) + @($content.models)

# Save back to file
$json = $content | ConvertTo-Json -Depth 10
Set-Content -Path $registryPath -Value $json

Write-Host "Added orchestrator entry: $newEntry.id"
Write-Host "Total models: $($content.models.Count)"