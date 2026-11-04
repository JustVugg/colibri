# Qwen3.8-Flash-Next-FP8 on the RTX 4090 / 192 GiB RAM / two-NVMe machine.
# The checkpoint as tools/download_qwen38_4090.ps1 lays it out: the model folder on C: and every
# layer's other experts and PLE parts on D: (Q38_DIRS). Every routed expert is read into RAM at the
# start (Q38_PRELOAD: 123 GB, about 20 s from both drives), so no step waits on a disk read; the
# layers run on the GPU as one chain (COLI_CUDA_CHAIN), a decode step's routed experts hottest in
# this conversation in VRAM (the tier, about 2,000-2,800 of 25,088) and the rest on the CPU, a
# prompt's (64 tokens or more) all on the GPU, copied up from RAM layer by layer with the same bits
# (Q38_PROMPT_GPU: 3,763 tokens in 10 s instead of 74); the MTP head drafts tokens that one forward
# checks (the reply is the one plain decoding gives).
param(
    # The engine's model folder (config, tokenizer, the resident weights, 61% of the experts)
    [string]$ModelDir = 'C:\AI\Qwen3.8-Flash-Next-FP8',
    # The other drive's folders, ';'-separated (Q38_DIRS); "" for a model in one folder
    [string]$ShardDir = 'D:\AI\Models\Qwen3.8-Flash-Next-FP8',
    # The model name the API answers to
    [string]$ModelId = 'qwen3.8-flash-next',
    # The whole process's RAM ceiling, GB: every expert in RAM takes 120 GB of it
    [ValidateRange(32,190)][double]$RamGB = 174,
    [ValidateRange(1,32)][int]$Threads = 16,
    # Context, tokens (K and V as float16 when serving: about 27 KB a token)
    [ValidateRange(4096,262144)][int]$Context = 65536,
    # The most tokens a reply may have when a request does not ask for fewer (reasoning included)
    [ValidateRange(256,262144)][int]$MaxOutput = 32768,
    # Tokens the MTP head drafts for each verify; 0 = no drafts
    [ValidateRange(0,3)][int]$Drafts = 2,
    [ValidateRange(1,65535)][int]$Port = 18080,
    # Read the experts from the disks as they are first routed instead of all at the start
    [switch]$NoPreload,
    # No GPU: every layer and every expert on the CPU
    [switch]$Cpu
)
$ErrorActionPreference = 'Stop'
$EngineDir = Split-Path -Parent $PSScriptRoot
$need = @('qwen38.exe', 'coli')
if (!$Cpu) { $need += 'coli_cuda.dll' }
foreach ($file in $need) {
    if (!(Test-Path -LiteralPath (Join-Path $EngineDir $file) -PathType Leaf)) {
        throw "Missing $file in $EngineDir (mingw32-make -C c qwen38 CUDA_DLL=1 ARCH=native VK=0; coli_cuda.dll with the chain: the Makefile's cuda-dll target)"
    }
}
foreach ($file in @('config.json', 'tokenizer.json')) {
    if (!(Test-Path -LiteralPath (Join-Path $ModelDir $file) -PathType Leaf)) {
        throw "Missing $file in $ModelDir (tools/download_qwen38_4090.ps1 fetches the model)"
    }
}
$shardDirs = @($ShardDir -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($dir in $shardDirs) {
    if (!(Test-Path -LiteralPath $dir -PathType Container)) { throw "Missing shard directory: $dir" }
}
$layout = Join-Path $ModelDir 'q38_layout.json'
if (Test-Path -LiteralPath $layout -PathType Leaf) {
    $state = Get-Content -LiteralPath $layout -Raw | ConvertFrom-Json
    if (!$state.complete) {
        throw "$ModelDir is not completely downloaded yet (q38_layout.json); rerun tools/download_qwen38_4090.ps1"
    }
}
$env:Q38_DIRS = $shardDirs -join ';'
$env:Q38_PRELOAD = $(if ($NoPreload) { '0' } else { '1' })
$env:COLI_CUDA = $(if ($Cpu) { '0' } else { '1' })
$env:COLI_CUDA_CHAIN = $(if ($Cpu) { '0' } else { '1' })
$env:COLI_GPU = '0'
$env:Q38_MTP = $(if ($Drafts -gt 0) { '1' } else { '0' })
if ($Drafts -gt 0) { $env:Q38_MTP_DRAFTS = [string]$Drafts } else { Remove-Item Env:Q38_MTP_DRAFTS -ErrorAction SilentlyContinue }
$env:OMP_NUM_THREADS = [string]$Threads
$env:COLI_NO_OMP_TUNE = '1'
$env:COLI_VULKAN = '0'
foreach ($key in @('CUDA_EXPERT_GB', 'Q38_TRUNK_GPU', 'Q38_MOE_GROUPED', 'Q38_NATIVE_FP8', 'Q38_FP8_KERNEL',
                   'Q38_EXPERT_INT4', 'COLI_CUDA_CHAIN_MOE', 'COLI_MAP_EXPERTS', 'KV_SLOTS', 'Q38_PROMPT_GPU')) {
    Remove-Item -LiteralPath "Env:$key" -ErrorAction SilentlyContinue
}
Write-Host "Qwen3.8-Flash-Next: $ModelDir$(if ($shardDirs.Count) { '; shards also from ' + ($shardDirs -join '; ') })"
Write-Host "RAM $RamGB GB, CPU $Threads, $(if ($NoPreload) { 'experts read as first routed' } else { 'every expert read into RAM at the start' }), $(if ($Cpu) { 'no GPU' } else { 'the layers and a prompt''s experts on the GPU, the hottest experts in VRAM' }), MTP drafts $Drafts"
Write-Host "Context $Context tokens, replies up to $MaxOutput tokens"
Write-Host "API: http://127.0.0.1:$Port/v1, model $ModelId"
& python (Join-Path $EngineDir 'coli') serve --model $ModelDir --ram $RamGB --ctx $Context `
    --ngen $MaxOutput --host 127.0.0.1 --port $Port --model-id $ModelId --kv-slots 1 `
    --queue-timeout 7200 --max-queue 8
exit $LASTEXITCODE
