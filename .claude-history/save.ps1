# Save Claude Code history of this project INTO the repo (.claude-history/sessions).
# Run before commit:   powershell -ExecutionPolicy Bypass -File .claude-history\save.ps1

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

# Claude Code names the project folder by replacing every non-alphanumeric char with '-'
# e.g. D:\Huan\2.Python\learning\neu_det -> D--Huan-2-Python-learning-neu-det
$encoded = $repoRoot -replace '[^A-Za-z0-9]', '-'
$src = Join-Path $HOME ".claude\projects\$encoded"
$dst = Join-Path $PSScriptRoot "sessions"

if (-not (Test-Path $src)) {
    Write-Host "No Claude Code history found at: $src"
    exit 1
}

New-Item -ItemType Directory -Force $dst | Out-Null
# memory\ is excluded on purpose: it belongs to the user profile, not to the project
Get-ChildItem $src -Exclude "memory" | Copy-Item -Destination $dst -Recurse -Force

$n = (Get-ChildItem $dst -Filter *.jsonl).Count
Write-Host "Saved $n session(s): $src -> $dst"
