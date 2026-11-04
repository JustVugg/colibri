# Download Kimi K3 (moonshotai/Kimi-K3, 1.56 TB) for the RTX 4090 / 192 GiB / two-NVMe machine.
# Every MoE layer's shard is cut between C: and D: in proportion to their measured read speeds
# (17.5 MB unbuffered reads, 3 readers: C: 10.4 GB/s, D: 6.55 GB/s, 16.9 GB/s together), so
# every layer's experts come from both drives at once. tools/k3_download.py does the work;
# rerun the same command to continue after an interruption. See docs/kimi_k3-4090.md.
param(
    # The engine's model folder: config, tokenizer, the files read at load, 61% of every
    # layer's experts
    [string]$ModelDir = 'C:\AI\Kimi-K3',
    [double]$ModelWeight = 10.4,
    # The second drive's folder (K3_DIRS): the other 39%
    [string]$ShardDir = 'D:\AI\Models\Kimi-K3',
    [double]$ShardWeight = 6.55,
    # Files fetched at once
    [ValidateRange(1,16)][int]$Jobs = 3,
    [ValidateSet('auto', 'hf', 'http')][string]$Backend = 'auto',
    # Fetch the small files, print the plan and the space it needs, stop
    [switch]$DryRun,
    # Re-hash every file against the published bytes
    [switch]$Verify,
    # Put the cut shards back together as the published files (for another engine)
    [switch]$Join,
    [switch]$Status,
    # Free space to leave on each drive, GB
    [double]$ReserveGB = 20
)
$ErrorActionPreference = 'Stop'
$python = Get-Command python -ErrorAction SilentlyContinue
if (!$python) { $python = Get-Command py -ErrorAction SilentlyContinue }
if (!$python) { throw 'Python 3 is needed (python.org); huggingface_hub with hf_xet makes it faster' }
$tool = Join-Path $PSScriptRoot 'k3_download.py'
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
