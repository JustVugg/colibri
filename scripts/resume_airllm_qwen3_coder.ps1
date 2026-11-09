$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:HF_HUB_CACHE = 'D:/colibri-models/.airllm-hub'
$env:TRANSFORMERS_CACHE = 'D:/colibri-models/.airllm-hub'
$env:AIRLLM_CACHE = 'D:/colibri-models/.airllm-split'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PATH = 'D:\Colibri\.venv\Scripts' + [System.IO.Path]::PathSeparator + $env:PATH
$script = 'D:\Colibri\scripts\resume_airllm_qwen3_coder.py'
if (-not (Test-Path $script)) { Write-Host 'missing script: ' $script; exit 1 }
$start = Get-Date
Write-Host "start: " $start
& python $script
$rc = $LASTEXITCODE
Write-Host "end: " (Get-Date)
Write-Host "rc=$rc"
exit $rc
