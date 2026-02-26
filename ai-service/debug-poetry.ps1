# debug-poetry.ps1 - Find all references to 'ai_service' in project
param(
    [string]$SearchPath = ".",
    [string]$PackageName = "ai_service"
)

Write-Host "🔍 Searching for '$PackageName' references in $SearchPath" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Gray

# 1. Check pyproject.toml explicitly
Write-Host "`n📄 Checking pyproject.toml:" -ForegroundColor Yellow
if (Test-Path "pyproject.toml") {
    $content = Get-Content "pyproject.toml" -Raw
    Write-Host $content | Select-String -Pattern $PackageName -Context 2,2
} else {
    Write-Host "❌ pyproject.toml not found!" -ForegroundColor Red
}

# 2. Search all TOML/Python/Config files for the package name
Write-Host "`n🔎 Searching all files for '$PackageName':" -ForegroundColor Yellow
Get-ChildItem -Recurse -Include *.toml,*.py,*.yml,*.yaml,*.json,*.env -Exclude .venv,__pycache__,node_modules | 
    Select-String -Pattern $PackageName -List | 
    ForEach-Object {
        Write-Host "  $($_.Path):$($_.LineNumber): $($_.Line.Trim())" -ForegroundColor Gray
    }

# 3. Check what Poetry thinks the package path is
Write-Host "`n📦 Poetry configuration:" -ForegroundColor Yellow
poetry config --list | Select-String "package"

# 4. Show actual directory structure
Write-Host "`n📁 Actual directories in project root:" -ForegroundColor Yellow
Get-ChildItem -Directory | Select-Object Name | Format-Table -AutoSize

# 5. Check if __init__.py exists in expected locations
Write-Host "`n🔍 Checking for __init__.py files:" -ForegroundColor Yellow
Get-ChildItem -Recurse -Filter __init__.py | Select-Object FullName

# 6. Show what Poetry will try to install
Write-Host "`n🚀 Poetry build preview:" -ForegroundColor Yellow
poetry build --dry-run 2>&1 | Select-String -Pattern "ai_service|app|package"

Write-Host "`n✅ Diagnostic complete!" -ForegroundColor Green