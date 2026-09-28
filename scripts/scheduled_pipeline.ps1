$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pipeline = Join-Path $PSScriptRoot "run_pipeline.py"
$log = Join-Path $projectRoot "data\pipeline_runs.log"

"--- $(Get-Date -Format o) ---" | Add-Content $log
& $python $pipeline *>> $log