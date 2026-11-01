# Kimi K3 on the RTX 4090 / 192 GiB RAM / two-NVMe machine (docs/kimi_k3-4090.md).
# The checkpoint as tools/download_kimi_k3_4090.ps1 lays it out: the model folder on C:
# and every MoE layer's other part on D:, read by two readers per drive. The largest
# dense matrices go to the GPU (K3_CUDA_DENSE), their host copies released for the
# expert cache. A decode step's routed experts run on the CPU, which reads them in place;
# a prompt step's go to the GPU too as they are read, with its other dense matrices
# (coli_cuda_k3.dll, the CPU's bytes), unless -CpuExperts.
param(
    # The engine's model folder (config, tokenizer, 61% of every layer's experts)
    [string]$ModelDir = 'C:\AI\Kimi-K3',
    # The other drive's folders, ';'-separated (K3_DIRS); "" for a model in one folder
    [string]$ShardDir = 'D:\AI\Models\Kimi-K3',
    # The model name the API answers to
    [string]$ModelId = 'kimi-k3',
    # The whole process's RAM ceiling, GB: the expert cache gets what the rest leaves
    [ValidateRange(16,190)][double]$RamGB = 174,
    [ValidateRange(1,32)][int]$Threads = 16,
    [ValidateRange(1,8)][int]$IoPerDrive = 2,
    # Read the next MoE layer's predicted experts ahead (decode)
    [ValidateRange(0,1)][int]$Prefetch = 1,
    # Dense matrices on the GPU, GiB, and the VRAM left free beside them; -DenseGB 21.5
    # -ReserveGB 1 puts 35 more on the GPU (0.78 -> 0.81 tok/s measured) and leaves about
    # 0.6 GiB free: past the free memory Windows pages the GPU's to RAM and every call slows
    [ValidateRange(0,23)][double]$DenseGB = 20,
    [ValidateRange(0.5,8)][double]$ReserveGB = 2,
    # Context, tokens: the MLA caches take 55 KB a token (7 GB at 128k), counted against RamGB
    [ValidateRange(512,1048576)][int]$Context = 131072,
    # The most tokens a reply may have when a request does not ask for fewer (reasoning included)
    [ValidateRange(256,1048576)][int]$MaxOutput = 32768,
    # Prompt tokens a prefill step takes at once: each expert a step needs is read once for
    # all of them (a step of 4,096 or more reads nearly every one, 1.4 TB), so larger steps
    # read the disks less per prompt token (a 2,637-token prompt 352 s in steps of 2,048, 268 s
    # in one of 4,096; a 5,862-token one 254 s in steps of 4,096, 200 s in one of 8,192); past
    # 512 a step's token holds 1.4 MB of RAM, taken from the expert cache (with the read-ahead
    # pool 84 slots a layer at 512, 81 at 4,096, 78 at 8,192: decode 0.81 -> 0.80 tok/s; about
    # 71 at 16,384)
    [ValidateRange(1,16384)][int]$Chunk = 8192,
    # A long prompt's read-ahead pool, GB: the next MoE layer's experts read while this one
    # computes (2,637 tokens 170 -> 137 s at 20); its RAM comes out of the expert cache
    # (decode 0.83 -> 0.81 tok/s), 0 gives decode all of it
    [ValidateRange(0,64)][double]$ReadAheadGB = 20,
    # Photos of the recurrent state at every prompt's and reply's end (COLI_K3_CKPT, about
    # 434 MB each): a turn whose client sends the history back without its reasoning, or
    # edits a tool result, resumes from the deepest photo its prompt still matches and
    # prefills only the rest, instead of the whole conversation again; 0 = off
    [ValidateRange(0,8)][int]$Photos = 4,
    [ValidateRange(1,65535)][int]$Port = 18080,
    # No GPU: the dense matrices stay on the CPU (a CPU build, or to compare)
    [switch]$Cpu,
    # A prompt step's routed experts and other dense matrices on the CPU alone (K3_GPU_EXPERTS=0)
    [switch]$CpuExperts
)
$ErrorActionPreference = 'Stop'
$EngineDir = Split-Path -Parent $PSScriptRoot
$need = @('kimi_k3.exe', 'coli')
if (!$Cpu) { $need += 'coli_cuda.dll' }
foreach ($file in $need) {
    if (!(Test-Path -LiteralPath (Join-Path $EngineDir $file) -PathType Leaf)) {
        throw "Missing $file in $EngineDir. See docs/kimi_k3-4090.md for the build commands (or pass -Cpu)."
    }
}
if (!$Cpu -and !$CpuExperts -and !(Test-Path -LiteralPath (Join-Path $EngineDir 'coli_cuda_k3.dll') -PathType Leaf)) {
    Write-Warning "No coli_cuda_k3.dll in $EngineDir (c\build_k3_cuda.bat): a prompt's experts stay on the CPU"
}
foreach ($file in @('config.json', 'tokenizer.json')) {
    if (!(Test-Path -LiteralPath (Join-Path $ModelDir $file) -PathType Leaf)) {
        throw "Missing $file in $ModelDir (tools/download_kimi_k3_4090.ps1 fetches the model and writes tokenizer.json)"
    }
}
$shardDirs = @($ShardDir -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($dir in $shardDirs) {
    if (!(Test-Path -LiteralPath $dir -PathType Container)) { throw "Missing shard directory: $dir" }
}
$layout = Join-Path $ModelDir 'k3_layout.json'
if (Test-Path -LiteralPath $layout -PathType Leaf) {
    $state = Get-Content -LiteralPath $layout -Raw | ConvertFrom-Json
    if (!$state.complete) {
        throw "$ModelDir is not completely downloaded yet (k3_layout.json); rerun tools/download_kimi_k3_4090.ps1"
    }
}
$env:K3_DIRS = $shardDirs -join ';'
$env:K3_CUDA = '0'                     # no VRAM tier: a decode step's experts on the CPU, read in place
$env:K3_GPU_EXPERTS = $(if ($Cpu -or $CpuExperts) { '0' } else { '1' })
$env:K3_READAHEAD_GB = $(if ($Cpu -or $CpuExperts) { '0' } else { [string]$ReadAheadGB })
$env:K3_CUDA_DENSE = $(if ($Cpu -or $DenseGB -le 0) { '0' } else { '1' })
$env:K3_CUDA_DENSE_GB = [string]$DenseGB
$env:K3_CUDA_RESERVE_GB = [string]$ReserveGB
$env:K3_CUDA_RELEASE_HOST = '1'
$env:K3_EXPERT_GB = '1000'             # the cache takes what RamGB leaves (the engine clamps it)
$env:K3_MAXT = [string]$Context
$env:K3_CHUNK = [string]$Chunk
$env:K3_DIRECT = '1'
$env:K3_PIPE = '1'
$env:K3_IO_PER_DRIVE = [string]$IoPerDrive
$env:K3_PREFETCH = [string]$Prefetch
$env:COLI_K3_CKPT = [string]$Photos
$env:OMP_NUM_THREADS = [string]$Threads
$env:COLI_NO_OMP_TUNE = '1'
$env:COLI_VULKAN = '0'
$env:COLI_CUDA = '0'                   # the launcher's own GPU tier is for other engines
$env:COLI_GPU = '0'
foreach ($key in @('K3_LAYERS', 'K3_TRACE', 'K3_LOGITS', 'K3_X0', 'K3_MMAP', 'K3_IO_THREADS', 'K3_LOAD_THREADS',
                   'K3_IO_DELAY_US', 'K3_VK', 'COLI_GPU_FAIL_AFTER', 'K3_VAL_LOGITS', 'K3_VALIDATE_LAYER',
                   'COLI_K3_CKPT_DIR')) {
    Remove-Item -LiteralPath "Env:$key" -ErrorAction SilentlyContinue
}
Write-Host "Kimi K3: $ModelDir$(if ($shardDirs.Count) { '; experts also from ' + ($shardDirs -join '; ') })"
Write-Host "RAM $RamGB GB, CPU $Threads, readers $IoPerDrive per drive, prefetch $Prefetch, $(if ($env:K3_CUDA_DENSE -eq '1') { "dense matrices on the GPU up to $DenseGB GiB" } else { 'dense matrices on the CPU' })$(if ($env:K3_GPU_EXPERTS -eq '1') { ', a prompt step''s experts on the GPU too' })"
Write-Host "Context $Context tokens, replies up to $MaxOutput tokens, prompt steps of $Chunk (read-ahead $($env:K3_READAHEAD_GB) GB), $Photos turn photos"
Write-Host "API: http://127.0.0.1:$Port/v1, model $ModelId"
& python (Join-Path $EngineDir 'coli') serve --model $ModelDir --ram $RamGB --ctx $Context `
    --ngen $MaxOutput --host 127.0.0.1 --port $Port --model-id $ModelId --kv-slots 1 `
    --queue-timeout 7200 --max-queue 8
exit $LASTEXITCODE
