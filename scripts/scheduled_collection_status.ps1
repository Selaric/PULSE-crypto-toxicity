$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$statusScript = Join-Path $PSScriptRoot "check_collection_status.py"
$log = Join-Path $projectRoot "data\collection_status.log"

"--- $(Get-Date -Format o) ---" | Add-Content $log
& $python $statusScript --raw-dir (Join-Path $projectRoot "data\raw") *>> $log