# Sets the GitHub "About" description on both Aureon remotes.
# Requires: gh auth login (once) OR GITHUB_TOKEN / GH_TOKEN with repo scope.

$Description = Get-Content -Path "$PSScriptRoot\..\.github\description.txt" -Raw
$Description = $Description.Trim()

$Topics = @(
    "machine-learning",
    "supervised-learning",
    "neural-network",
    "fastapi",
    "postgresql",
    "railway",
    "backpropagation",
    "llm"
)

$gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $gh) {
    $candidate = "C:\Program Files\GitHub CLI\gh.exe"
    if (Test-Path $candidate) { $gh = $candidate }
}
if (-not $gh) {
    Write-Error "GitHub CLI (gh) not found. Install from https://cli.github.com/ then run: gh auth login"
    exit 1
}

foreach ($repo in @("houseofasher/SOLIA", "ZorakCorp/Aureon-LLM", "shep95/Aureon_Elion-LLM")) {
    Write-Host "Updating $repo ..."
    & $gh repo edit $repo --description $Description
    & $gh repo edit $repo --add-topic ($Topics -join ",")
    if ($LASTEXITCODE -eq 0) { Write-Host "OK: $repo" } else { Write-Warning "Failed: $repo (check gh auth)" }
}
