# Restore Claude Code history from the repo into this machine (~\.claude\projects\...).
# Run once after clone:   powershell -ExecutionPolicy Bypass -File .claude-history\restore.ps1
# Then open the folder in VS Code -> Claude Code -> past conversations (or: claude --resume)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

# Folder name is computed from the CURRENT clone path, so any location works
$encoded = $repoRoot -replace '[^A-Za-z0-9]', '-'
$src = Join-Path $PSScriptRoot "sessions"
$dst = Join-Path $HOME ".claude\projects\$encoded"

if (-not (Test-Path $src)) {
    Write-Host "No saved sessions in: $src"
    exit 1
}

New-Item -ItemType Directory -Force $dst | Out-Null

# Only copy a session if it is missing here or the saved copy is newer (never lose local progress)
foreach ($item in Get-ChildItem $src) {
    $target = Join-Path $dst $item.Name
    if ((Test-Path $target) -and ((Get-Item $target).LastWriteTime -ge $item.LastWriteTime)) {
        Write-Host "Skip (local is newer or same): $($item.Name)"
        continue
    }
    Copy-Item $item.FullName -Destination $dst -Recurse -Force
    Write-Host "Restored: $($item.Name)"
}

Write-Host "Done -> $dst"
