# Native MiMo CUDA profile for RTX 4090 / 192 GiB RAM / split model shards.
param(
    # Which checkpoint: base (MiMo-V2.6-Pro-MOPD) or heretic (MiMo-V2.6-Pro-MOPD-heretic, the
    # base with its 70 o_proj edited). Each picks the folders and the API model id below;
    # -ModelDir, -ShardDir and -ModelId given on the command line win over it.
    [ValidateSet('base', 'heretic')][string]$Model = 'base',
    [string]$ModelDir,
    # More folders the shards are read from, ';'-separated: a shard is taken from the first
    # of ModelDir, then these, that holds it, so a checkpoint that differs from another in
    # a few shards keeps only those and reads the rest where the other keeps them. "": none.
    [string]$ShardDir,
    # The model name the API answers to (requests naming another get 404)
    [string]$ModelId,
    [ValidateRange(8,104)][int]$Cap = 104,
    [ValidateRange(1,32)][int]$Threads = 16,
    [ValidateRange(1,8)][int]$IoPerDrive = 2,
    [ValidateRange(0,2)][int]$Prefetch = 1,
    [ValidateRange(1,21)][double]$DenseGB = 20,
    # VRAM left free for scratch and other programs; -DenseGB 21 -ReserveGB 1 puts
    # five more o_proj on the GPU (+2% measured) and leaves about 1 GiB free
    [ValidateRange(0.5,8)][double]$ReserveGB = 2,
    # The model's full 1M positions. The attention caches take memory as a conversation
    # grows (102 KB a token); past what RAM holds beside the expert cache (about 300k
    # tokens here) the expert pool gives memory back and decoding slows (docs/mimo.md).
    [ValidateRange(512,1048576)][int]$Context = 1048576,
    # The most tokens a reply may have when a request does not ask for fewer; reasoning
    # counts toward it.
    [ValidateRange(256,1048576)][int]$MaxOutput = 65536,
    [ValidateRange(1,65535)][int]$Port = 18080,
    # Faster and NOT exact: every o_proj on the GPU as int8 (docs/mimo-cuda.md)
    [switch]$OprojInt8
)
$ErrorActionPreference = 'Stop'
$EngineDir = Split-Path -Parent $PSScriptRoot
foreach ($file in @('mimo.exe', 'coli_cuda_mimo.dll', 'coli')) {
    if (!(Test-Path -LiteralPath (Join-Path $EngineDir $file) -PathType Leaf)) {
        throw "Missing $file in $EngineDir. See docs/mimo-cuda.md for the build commands."
    }
}
# heretic differs from base in one shard (model_pp0_ep0_shard0: the 70 o_proj), which it
# must hold itself; its other 129 are the base's bytes and are read from the base's folders,
# split over the two drives as the base reads them.
$presets = @{
    base    = @{ ModelDir = 'D:\AI\Models\MiMo-V2.6-Pro-MOPD'; ShardDir = 'C:\AI\MiMoPro-Shards';
                 ModelId = 'mimo-v2.6-pro-mopd'; Own = @() }
    heretic = @{ ModelDir = 'D:\AI\Models\MiMo-V2.6-Pro-MOPD-heretic';
                 ShardDir = 'C:\AI\MiMoPro-Shards;D:\AI\Models\MiMo-V2.6-Pro-MOPD';
                 ModelId = 'mimo-v2.6-pro-mopd-heretic'; Own = @('model_pp0_ep0_shard0.safetensors') }
}
$preset = $presets[$Model]
if (!$PSBoundParameters.ContainsKey('ModelDir')) { $ModelDir = $preset.ModelDir }
if (!$PSBoundParameters.ContainsKey('ShardDir')) { $ShardDir = $preset.ShardDir }
if (!$PSBoundParameters.ContainsKey('ModelId')) { $ModelId = $preset.ModelId }
if (!(Test-Path -LiteralPath (Join-Path $ModelDir 'config.json') -PathType Leaf)) {
    throw "Missing model config in $ModelDir"
}
$shardDirs = @($ShardDir -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($dir in $shardDirs) {
    if (!(Test-Path -LiteralPath $dir -PathType Container)) { throw "Missing shard directory: $dir" }
}
foreach ($file in $preset.Own) {
    if (!$PSBoundParameters.ContainsKey('ModelDir') -and
        !(Test-Path -LiteralPath (Join-Path $ModelDir $file) -PathType Leaf)) {
        throw "$ModelDir must hold its own $file; without it the shard of the same name in $ShardDir would be read"
    }
}
# Every shard the index names, somewhere on the search path
$index = Join-Path $ModelDir 'model.safetensors.index.json'
if (Test-Path -LiteralPath $index -PathType Leaf) {
    $names = New-Object 'System.Collections.Generic.HashSet[string]'
    foreach ($m in [regex]::Matches([IO.File]::ReadAllText($index), '"([^"\\/]+\.safetensors)"')) {
        [void]$names.Add($m.Groups[1].Value)
    }
    $missing = @($names | Where-Object {
        $name = $_
        !(@(@($ModelDir) + $shardDirs) | Where-Object { Test-Path -LiteralPath (Join-Path $_ $name) -PathType Leaf })
    })
    if ($missing.Count) {
        throw "$($missing.Count) shard(s) of the index are in none of $ModelDir; $ShardDir (first: $($missing[0]))"
    }
}
$env:MIMO_DIRS = $shardDirs -join ';'
$env:MIMO_CUDA = '1'
if ($OprojInt8 -and -not $PSBoundParameters.ContainsKey('DenseGB')) { $DenseGB = 20.2 }
$env:MIMO_CUDA_DENSE_GB = [string]$DenseGB
$env:MIMO_CUDA_OPROJ_INT8 = $(if ($OprojInt8) { '1' } else { '0' })
$env:MIMO_CUDA_RESERVE_GB = [string]$ReserveGB
$env:MIMO_CUDA_RELEASE_HOST = '1'
$env:MIMO_CTX = [string]$Context
$env:MIMO_DENSE_BITS = '0'
$env:MIMO_IDOT = '0'
$env:MIMO_DIRECT = '1'
$env:MIMO_IO_PER_DRIVE = [string]$IoPerDrive
$env:MIMO_PREFETCH = [string]$Prefetch
$env:OMP_NUM_THREADS = [string]$Threads
$env:COLI_NO_OMP_TUNE = '1'
$env:COLI_VULKAN = '0'
$env:COLI_GPU = '0'
# MIMO_CHUNK unset: prompts go to the GPU in blocks as large as its memory allows (docs/mimo-cuda.md)
foreach ($key in @('MIMO_CHUNK', 'MIMO_CUDA_FAULT', 'MIMO_LAYERS', 'MIMO_TRACE', 'MIMO_LOGITS', 'MIMO_ROUTE_TRACE',
                   'MIMO_IO_THREADS', 'MIMO_READ_THREADS')) {
    Remove-Item -LiteralPath "Env:$key" -ErrorAction SilentlyContinue
}
Write-Host "Model ${Model}: $ModelDir$(if ($shardDirs.Count) { '; shards also from ' + ($shardDirs -join '; ') })"
Write-Host "MiMo native CUDA: dense budget $DenseGB GiB, cache $Cap experts/layer, CPU $Threads, readers $IoPerDrive per drive, prefetch $Prefetch$(if ($OprojInt8) { ', o_proj int8 (not exact)' })"
Write-Host "Context $Context tokens, replies up to $MaxOutput tokens"
Write-Host "API: http://127.0.0.1:$Port/v1, model $ModelId"
& python (Join-Path $EngineDir 'coli') serve --model $ModelDir --cap $Cap --ram 174 --ctx $Context `
    --ngen $MaxOutput --host 127.0.0.1 --port $Port --model-id $ModelId --kv-slots 1 `
    --queue-timeout 7200 --max-queue 8
exit $LASTEXITCODE
