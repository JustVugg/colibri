<#
.SYNOPSIS
  Launch colibri on this machine. Thin wrapper over the `coli` launcher, which on
  Windows already auto-detects the NVIDIA GPU and sizes the VRAM expert tier from
  real free VRAM. We only resolve the model path and pass through options.

.DESCRIPTION
  Earlier this script hand-set COLI_CUDA / CUDA_EXPERT_GB env vars. That was wrong:
  the launcher wants the `--vram` flag (GB, 0=auto), not CUDA_EXPERT_GB, and it
  plans RAM+VRAM itself. A bare `coli chat` auto-enables CUDA when colibri.exe is a
  CUDA build and coli_cuda.dll sits next to it (both true after Build-Colibri.ps1).

.PARAMETER Model
  Path to the quantized model directory (e.g. C:\Users\...\GLM5.2).

.PARAMETER Action
  chat (default) | serve | plan | doctor  -> `coli <action>`.

.PARAMETER VramGB
  VRAM expert-tier budget in GB. 0 = auto (default; launcher sizes from free VRAM).
  A positive value forces the size via `--auto-tier --vram`.

.PARAMETER RamGB
  RAM budget in GB. 0 = auto (engine uses ~88% of available RAM).

.PARAMETER Mirror
  Optional second copy of the model on another drive (COLI_MODEL_MIRROR) to
  aggregate read bandwidth.

.EXAMPLE
  .\Run-Colibri.ps1 -Model C:\Models\GLM5.2 -Action plan
.EXAMPLE
  .\Run-Colibri.ps1 -Model C:\Models\GLM5.2               # chat, GPU auto
.EXAMPLE
  .\Run-Colibri.ps1 -Model ...\GLM5.2 -VramGB 12   # cap the expert tier at 12 GB
#>
[CmdletBinding()]
param(
    # Omit -Model to get the interactive wizard: it finds your models, asks what you
    # want in 3 questions, picks every parameter for you and explains each choice.
    [string]$Model,
    [ValidateSet('chat','serve','plan','doctor','web','run')][string]$Action = 'chat',
    # -Mtp: GLM native speculation under CUDA (COLI_CUDA_MTP=1). Experimental: cold
    # experts verify on CPU kernels whose FP order differs, so acceptance drops to
    # ~30-50% (upstream #163) — still a net decode win when it holds.
    [switch]$Mtp,
    [double]$VramGB = 0,
    [int]$RamGB = 0,
    # -CudaDense: put the ~11 GB dense weights ON the GPU (experimental resident-dense
    # path). Frees that much RAM for more warm-expert cache -> higher residency, and
    # fills VRAM. Best fit for a 16 GB-VRAM / 32 GB-RAM box.
    [switch]$CudaDense,
    # -VramExperts: keep the routed experts resident in VRAM (GPU expert tier). Faster for
    # LONG-prefill / batch runs, but ~3x SLOWER single-token decode (GPU is latency-bound at
    # S=1). Default OFF: experts stay in RAM and decode on the CPU (best for interactive chat).
    [switch]$VramExperts,
    [string]$Mirror,
    [string]$RepoRoot = (Join-Path $PSScriptRoot '..'),
    [Parameter(ValueFromRemainingArguments=$true)]$Passthrough
)

$ErrorActionPreference = 'Stop'
# Pick up CUDA/gcc installed after this shell started.
$env:PATH = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' +
            [System.Environment]::GetEnvironmentVariable('Path','User') + ';' + $env:PATH

# ========================= INSTANCE PREFLIGHT ===============================
# ONE engine per box. A second colibri fights the first for RAM, VRAM and the
# page cache — measured: a stray `coli web` during a GLM run inflated prefill
# 27 s -> 336 s. Two cases:
#  1) A `coli serve`/`coli web` is answering on :8000 -> don't spawn anything:
#     chat ATTACHES to the warm engine (coli chat does this natively — the
#     34-136 s load and the expert-cache warmth are paid once and survive
#     sessions), run sends the one-shot through its API, web opens the
#     dashboard that is already being served.
#  2) A raw engine (benchmark, another chat's private engine) has no API to
#     attach to -> wait for it, or quit; never silently double up.
$AttachBase = $null
try {
    $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2 -ErrorAction Stop
    if ("$($h.status)" -eq 'ok') { $AttachBase = 'http://127.0.0.1:8000' }
} catch {}
$CDirPre = Join-Path (Resolve-Path $RepoRoot) 'c'
if ($AttachBase) {
    $mid = $null
    try { $mods = Invoke-RestMethod -Uri "$AttachBase/v1/models" -TimeoutSec 2 -ErrorAction Stop
          if ($mods.data -and $mods.data.Count -ge 1) { $mid = $mods.data[0].id } } catch {}
    Write-Host "Running colibri server found at $AttachBase$(if($mid){ " (model: $mid)" }) - reusing it, no second engine." -ForegroundColor Green
    switch ($Action) {
        'web'   { Start-Process "$AttachBase/"; Write-Host "Dashboard opened (already served by the running instance)." -ForegroundColor Cyan; return }
        'serve' { Write-Host "A server is already up. To replace it: python coli stop  (then re-run)." -ForegroundColor Yellow; return }
        'run'   {
            $oneshot = if ($Passthrough) { "$Passthrough" } else { Read-Host "  prompt" }
            $body = @{ model=$(if($mid){$mid}else{'colibri'}); stream=$false; max_tokens=1024;
                       messages=@(@{ role='user'; content=$oneshot }) } | ConvertTo-Json -Depth 4
            $resp = Invoke-RestMethod -Uri "$AttachBase/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600
            Write-Host ""; Write-Host $resp.choices[0].message.content; return
        }
        default {   # chat (and anything chat-like): coli chat auto-attaches on :8000
            Set-Location $CDirPre
            & python coli chat
            return
        }
    }
}
$runningEngine = @(Get-Process colibri -ErrorAction SilentlyContinue)
if ($runningEngine.Count -gt 0) {
    $p0 = $runningEngine[0]
    Write-Host ""
    Write-Host "A colibri engine is ALREADY RUNNING (PID $($p0.Id), started $($p0.StartTime.ToString('HH:mm:ss')), $([math]::Round($p0.WorkingSet64/1GB,1)) GB RAM) with no API to attach to (benchmark or private chat engine)." -ForegroundColor Yellow
    Write-Host "Starting a second engine would fight it for RAM, VRAM and disk cache and slow BOTH down." -ForegroundColor Yellow
    $ans = Read-Host "  [w] wait for it to finish, then continue   [c] continue anyway (not recommended)   [q] quit (default)"
    if ($ans -match '^[wW]') {
        Write-Host "  waiting for PID $($p0.Id) to exit (Ctrl-C aborts)..." -ForegroundColor DarkGray
        while (Get-Process -Id $p0.Id -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 5 }
        Write-Host "  engine exited - continuing." -ForegroundColor Green
    } elseif ($ans -notmatch '^[cC]') { return }
}
# ======================= END INSTANCE PREFLIGHT =============================

# ============================== WIZARD ======================================
# No -Model given -> ask three questions, pick every parameter, explain each.
$WizardPrompt = $null
if(-not $Model){
    Write-Host ""
    Write-Host "  (\   colibri launcher" -ForegroundColor Cyan
    Write-Host "   )>  three questions; every technical knob is picked for you." -ForegroundColor DarkGray
    Write-Host ""

    # ---- Q1: which model? (any dir with a config.json under the workspace) ----
    $roots = @()
    if($env:COLI_MODELS_DIR){ $roots += $env:COLI_MODELS_DIR }
    $roots += Join-Path $env:USERPROFILE 'Documents\Workspace'
    $roots += 'C:\Models'
    $models = @()
    foreach($r in ($roots | Select-Object -Unique)){
        if(Test-Path $r){
            foreach($d in (Get-ChildItem $r -Directory -ErrorAction SilentlyContinue)){
                if(Test-Path (Join-Path $d.FullName 'config.json')){ $models += $d.FullName }
            }
        }
    }
    Write-Host "  Which model?" -ForegroundColor White
    for($i=0; $i -lt $models.Count; $i++){
        # top-level shard files only: fast even on a 358 GB dir
        $gb = [math]::Round((Get-ChildItem $models[$i] -File -ErrorAction SilentlyContinue |
                             Measure-Object Length -Sum).Sum / 1GB, 1)
        $arch = 'model'
        try { $cfgj = Get-Content (Join-Path $models[$i] 'config.json') -Raw | ConvertFrom-Json
              if($cfgj.model_type){ $arch = $cfgj.model_type } } catch {}
        Write-Host ("   [{0}] {1}  ({2} GB, {3})" -f ($i+1), (Split-Path $models[$i] -Leaf), $gb, $arch)
    }
    Write-Host ("   [{0}] other (type a path)" -f ($models.Count+1))
    $pick = Read-Host "  choice"
    $pickN = 0; [void][int]::TryParse($pick, [ref]$pickN)
    if($pickN -lt 1 -or $pickN -gt $models.Count){ $Model = Read-Host "  model directory" }
    else { $Model = $models[$pickN-1] }

    # ---- Q2: what do you want to do? (skipped if -Action was passed explicitly) ----
    if(-not $PSBoundParameters.ContainsKey('Action')){
        Write-Host ""
        Write-Host "  What do you want to do?" -ForegroundColor White
        Write-Host "   [1] Chat        - interactive conversation in this terminal (recommended)"
        Write-Host "   [2] Serve       - keep the model warm in the background; chats attach instantly"
        Write-Host "                     (pays the load ONCE; later prompts skip the multi-minute warmup)"
        Write-Host "   [3] Web         - browser dashboard: chat + live tier/disk metrics"
        Write-Host "   [4] One answer  - ask a single question, print the answer, exit"
        switch(Read-Host "  choice"){
            '2' { $Action='serve' }
            '3' { $Action='web' }
            '4' { $Action='run'; $WizardPrompt = Read-Host "  your question" }
            default { $Action='chat' }
        }
    }

    # ---- Q3: speed profile ----
    Write-Host ""
    Write-Host "  Speed profile?" -ForegroundColor White
    Write-Host "   [1] Interactive (recommended) - fastest replies while you chat."
    Write-Host "       experts compute on the CPU; the GPU runs attention; its idle VRAM"
    Write-Host "       becomes a PCIe cache that serves misses ~5x faster than the disk;"
    Write-Host "       the hot-expert set re-tunes itself between replies at zero disk cost."
    Write-Host "   [2] Long documents - fastest LARGE-prompt ingestion (summaries, code review)."
    Write-Host "       experts live in VRAM and compute on the GPU: big prefills fly,"
    Write-Host "       but short back-and-forth replies are ~3x slower than [1]."
    Write-Host "   [3] Plain - engine defaults only, no tuning (for debugging)."
    $prof = Read-Host "  choice"
    switch($prof){
        '2' { $VramExperts=$true; $CudaDense=$true }
        '3' { }
        default { $CudaDense=$true; $env:REPIN='16' }   # [1]: live re-pin every 16 tokens
    }

    # ---- Q4 (GLM only): native speculation ----
    $isGlm = $false
    try { $cfgj = Get-Content (Join-Path $Model 'config.json') -Raw | ConvertFrom-Json
          if("$($cfgj.model_type)" -match 'glm'){ $isGlm = $true } } catch {}
    # GLM: bound AUTOPIN. As .coli_usage history grows, confidence-scaled pinning
    # gets greedier and can trip the engine's projected-peak OOM guard, which
    # refuses to start. Applied after the env scrub, like WizTier.
    if($isGlm -and $prof -ne '3'){ $script:WizGlmPin = $true }
    if($isGlm -and $prof -ne '3'){
        Write-Host ""
        Write-Host "  GLM speculation (MTP)? The model drafts several tokens per step and" -ForegroundColor White
        Write-Host "  verifies them in one pass. On RAM-tight machines decode gets SLOWER:"
        Write-Host "  drafts are accepted often, but the batched verify reads more unique"
        Write-Host "  experts from disk than it saves in passes. Say no"
        Write-Host "  unless this machine has 64 GB+ RAM (then it becomes a real speedup)."
        if((Read-Host "  enable? [y/N]") -match '^[yY]'){ $Mtp = $true }
    }
    # Interactive profile, small-dense models (Qwen-class): a mid-size VRAM expert
    # tier + the engine's async CPU||GPU overlap (COLI_GROUP_ASYNC=1) lifts decode
    # measurably; larger tiers plateau. GLM keeps tier 0: its ~10 GB dense owns the
    # VRAM there.
    # NOTE: only the flag is set here — the stale-env scrub below (Remove-Item line)
    # clears CUDA_EXPERT_GB/COLI_GROUP_ASYNC, so the actual envs are applied in the
    # experts block AFTER the scrub, keyed on $script:WizTier.
    if($prof -ne '2' -and $prof -ne '3' -and -not $isGlm){ $script:WizTier=$true }
    $profName = 'interactive'
    if($prof -eq '2'){ $profName = 'long-documents' } elseif($prof -eq '3'){ $profName = 'plain' }
    Write-Host ""
    Write-Host "  -> $(Split-Path $Model -Leaf) - $Action - profile: $profName$(if($Mtp){' + MTP'})" -ForegroundColor Cyan
    Write-Host ""
}
# ============================ END WIZARD ====================================

# Self-heal: an earlier version of this script set these process env vars, and in
# PowerShell they PERSIST in the calling shell after the script exits. A stale
# CUDA_EXPERT_GB=auto crashes the coli launcher. Clear them so the launcher plans
# fresh; we drive the GPU via the --vram flag, not env vars.
Remove-Item Env:CUDA_EXPERT_GB, Env:COLI_CUDA, Env:COLI_GPU, Env:PIN_FILL, Env:CUDA_RELEASE_HOST, Env:COLI_GROUP_ASYNC, Env:PIN_GB -ErrorAction SilentlyContinue

# GLM wizard profile: bound AUTOPIN (see the wizard Q4 block for the why/measurement)
# and tune the disk/GPU knobs that only pay off on the huge disk-bound models.
if($script:WizGlmPin){
    # Size the pin from RAM that is actually free NOW, not from a constant tuned on a
    # quiet box. Empirical peak model from the engine's own OOM-guard reports:
    # peak ~= 14.2 GB (dense + reserve) + pin, so pin = free - 15 keeps ~0.8 GB
    # slack. A fixed PIN_GB refused to start whenever other apps held RAM.
    $freeGB = [math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)
    $pin = [math]::Max(2, [math]::Min(12, $freeGB - 15))
    $env:PIN_GB = "$pin"
    Write-Host "GLM AUTOPIN bounded: PIN_GB=$pin (sized from $freeGB GB free RAM; 12 is the ceiling)" -ForegroundColor DarkGray
    # NVMe saturates near queue depth 16 on ~19 MB expert reads; match the reader
    # pool to the logical CPU count instead of the fixed default 8.
    $env:PIPE_WORKERS = "$([Environment]::ProcessorCount)"
    Write-Host "Disk readers: PIPE_WORKERS=$($env:PIPE_WORKERS) (reader pool sized to CPU count)" -ForegroundColor DarkGray
}
else { Remove-Item Env:PIPE_WORKERS -ErrorAction SilentlyContinue }

$CDir = Join-Path (Resolve-Path $RepoRoot) 'c'
if(-not (Test-Path $Model)){ throw "Model dir not found: $Model" }
$exe = Join-Path $CDir 'colibri.exe'
if(-not (Test-Path $exe)){ throw "colibri.exe not built at $exe. Run .\Build-Colibri.ps1 first." }

$hasDll = Test-Path (Join-Path $CDir 'coli_cuda.dll')
if($hasDll){ Write-Host "GPU: coli_cuda.dll present -> launcher will auto-enable CUDA" -ForegroundColor Green }
else       { Write-Host "GPU: coli_cuda.dll NOT found -> CPU-only. Re-run Build-Colibri.ps1 for GPU." -ForegroundColor Yellow }

if($Mirror){ $env:COLI_MODEL_MIRROR = (Resolve-Path $Mirror).Path }

# CUDA_DENSE: attention/dense weights on the GPU. Helps decode a little (attention off the CPU).
# FREE_HOST releases the ~10 GB RAM twin of those weights after the GPU upload (GLM: the RAM
# LRU goes from cap 1 to hundreds). Rare CPU fallbacks reload the affected tensor from disk once.
if($CudaDense){ $env:CUDA_DENSE = '1'; $env:CUDA_DENSE_FREE_HOST = '1'; Write-Host "CUDA_DENSE=1 + FREE_HOST: attention on GPU, dense RAM copies released" -ForegroundColor Green }
else { Remove-Item Env:CUDA_DENSE, Env:CUDA_DENSE_FREE_HOST -ErrorAction SilentlyContinue }

# -Mtp / wizard Q4: GLM native speculation under CUDA (see param help).
if($Mtp){ $env:COLI_CUDA_MTP = '1'; Write-Host "COLI_CUDA_MTP=1: GLM drafts+verifies several tokens per forward (experimental)" -ForegroundColor Green }
else { Remove-Item Env:COLI_CUDA_MTP -ErrorAction SilentlyContinue }

# EXPERTS STAY IN RAM (computed on CPU), by design: a VRAM-resident expert tier
# makes single-token DECODE measurably SLOWER — for S=1 the GPU
# expert path is latency-bound (48 tiny H2D->kernel->D2H round-trips per token) while on CPU
# it is one memory-bound matmul. So suppress the launcher's auto VRAM expert tier by pinning
# CUDA_EXPERT_GB=0; experts live in RAM (RAM_FILL prewarms them) and decode on the CPU. The
# GPU still does attention (CUDA_DENSE) and would still win for LONG-prefill batches — set
# -VramExperts to opt back into the VRAM tier for document/prefill-heavy runs.
if($VramExperts){ Remove-Item Env:CUDA_EXPERT_GB -ErrorAction SilentlyContinue; Write-Host "VRAM expert tier ON (better for long prefills, slower single-token decode)" -ForegroundColor Yellow }
elseif($script:WizTier){ $env:CUDA_EXPERT_GB='8'; $env:COLI_GROUP_ASYNC='1'; Write-Host "Hybrid experts: 8 GB VRAM tier + async CPU||GPU overlap" -ForegroundColor Green }
else { $env:CUDA_EXPERT_GB = '0'; Write-Host "Experts in RAM / CPU decode (fast interactive chat). -VramExperts to change." -ForegroundColor DarkGray }

# RAM budget: when the user does not pass -RamGB, compute one that is BOTH greedy and safe.
# Two failure modes to avoid:
#  1) The engine's own auto (0.88 x MemAvailable) over-commits on a fresh boot (MemAvailable
#     ~= total) -> cap 8->107 -> projected peak ~= physical -> alloc fails -> SIGSEGV.
#  2) A fixed fraction of TOTAL ignores what other apps already use: the budget can
#     exceed what is actually free -> RSS crosses physical -> Windows pages -> the
#     resident experts thrash and every response crawls.
# So: budget = min(0.72 x TOTAL, AVAILABLE - 4 GB). The fraction caps greed on an empty box;
# the (available - reserve) term prevents oversubscription when apps are open. Override with
# -RamGB <n>. Close other apps to raise it; a persistent `coli serve` pays the load only once.
$effRamGB = $RamGB
if($effRamGB -le 0){
    $totalGB = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    $availGB = 0
    try { $availGB = [math]::Floor((Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes / 1024) } catch {}
    $ceil = [math]::Floor($totalGB * 0.80)   # greedier ceiling; the engine's RSS guard
    if($availGB -gt 0){ $effRamGB = [math]::Min($ceil, $availGB - 2) }   # ...evicts if RSS crosses the budget, so avail-2 is safe
    else             { $effRamGB = [math]::Floor($totalGB * 0.65) }   # perf class missing: conservative fixed fraction
    if($effRamGB -lt 6){ $effRamGB = 0 }   # too little free: defer to the engine's own auto (it will size down)
    if($effRamGB -gt 0){ Write-Host "RAM budget: --ram $effRamGB GB (of $totalGB GB total, $availGB GB free now; capped to avoid paging. Close apps or pass -RamGB to raise it)" -ForegroundColor DarkGray }
}

# Build `coli` argv. Let the launcher plan/enable the GPU; only pass explicit knobs.
$modelPath = (Resolve-Path $Model).Path
$cliArgs = @('coli', $Action, '--model', $modelPath)
if($VramGB -gt 0){ $cliArgs += @('--auto-tier','--vram', "$VramGB") }  # --vram needs --auto-tier to take effect
if($effRamGB -gt 0){ $cliArgs += @('--ram', "$effRamGB") }
if($WizardPrompt){ $cliArgs += $WizardPrompt }   # one-shot question from the wizard
if($Passthrough){ $cliArgs += $Passthrough }

$py = Get-Command python -ErrorAction SilentlyContinue
if(-not $py){ $py = Get-Command python3 -ErrorAction SilentlyContinue }
if(-not $py){ throw "python not found on PATH (the coli launcher needs it)." }

Write-Host "-> $($py.Source) $($cliArgs -join ' ')`n" -ForegroundColor Cyan
Push-Location $CDir
try { & $py.Source @cliArgs }
finally { Pop-Location }
