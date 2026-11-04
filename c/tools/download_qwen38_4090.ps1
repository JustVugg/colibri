# Download Qwen3.8-Flash-Next-FP8 (Qwen/Qwen3.8-Flash-Next-FP8, 185.6 GB) for the RTX 4090 / 192 GiB /
# two-NVMe machine. Every layer's routed experts and the PLE table's parts are dealt between C: and D:
# in proportion to their measured read speeds (17.5 MB unbuffered reads, 3 readers: C: 10.4 GB/s,
# D: 6.55 GB/s, 16.9 GB/s together -- the cut Kimi K3 uses), so every layer reads from both drives at
# once: experts 0-313 of every layer and PLE parts 0-78 on C:, experts 314-511 and parts 79-127 on D:.
# tools/q38_download.py does the work; rerun the same command to continue after an interruption.
param(
    # The engine's model folder: config, tokenizer, the resident weights, 61% of every
    # layer's experts and of the PLE table
    [string]$ModelDir = 'C:\AI\Qwen3.8-Flash-Next-FP8',
    [double]$ModelWeight = 10.4,
    # The second drive's folder (searched after the model folder): the other 39%
    [string]$ShardDir = 'D:\AI\Models\Qwen3.8-Flash-Next-FP8',
    [double]$ShardWeight = 6.55,
    # Files fetched at once
    [ValidateRange(1,16)][int]$Jobs = 3,
    [ValidateSet('auto', 'hf', 'http')][string]$Backend = 'auto',
    # Fetch the small files and the shards' headers, print the plan and the space it needs, stop
    [switch]$DryRun,
    # Re-hash every file against the published bytes
    [switch]$Verify,
    # Put every shard back together as the published file in the model folder (for other tools)
    [switch]$Join,
    [switch]$Status,
    # Free space to leave on each drive, GB
    [double]$ReserveGB = 20
)
$ErrorActionPreference = 'Stop'
$python = Get-Command python -ErrorAction SilentlyContinue
if (!$python) { $python = Get-Command py -ErrorAction SilentlyContinue }
if (!$python) { throw 'Python 3 is needed (python.org); huggingface_hub with hf_xet makes it faster' }
$tool = Join-Path $PSScriptRoot 'q38_download.py'
$argv = @('-I', $tool, '--drive', "$ModelDir=$ModelWeight", '--drive', "$ShardDir=$ShardWeight",
          '--jobs', $Jobs, '--backend', $Backend, '--reserve-gb', $ReserveGB)
if ($DryRun) { $argv += '--dry-run' }
if ($Verify) { $argv += '--verify' }
if ($Join) { $argv += '--join' }
if ($Status) { $argv += '--status' }
# Python's warnings go to stderr; Windows PowerShell must not make them errors
$ErrorActionPreference = 'Continue'
& $python.Source @argv
exit $LASTEXITCODE
