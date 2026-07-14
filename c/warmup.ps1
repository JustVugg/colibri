# warmup.ps1 - overnight expert-cache warmup for colibri
#
# Runs `coli run` in a loop with diverse prompts so the engine records which
# routed experts your workload actually uses into .coli_usage. At startup the
# engine pins the hottest experts into RAM; the more history it has, the bigger
# and more accurate that pin gets. This does NOT load random experts - it loads
# whatever the model actually routes to for these prompts, then promotes the
# frequent ones.
#
# Usage (from the c\ directory):
#   .\warmup.ps1                          # defaults: model next to repo, 3 rounds
#   .\warmup.ps1 -Model D:\glm52_i4 -Rounds 10 -Ngen 400
#
# Let it run while you sleep. Each iteration logs selections count + hit rate.
# Ctrl-C is safe: each run saves usage atomically only on clean completion, so
# the file is never corrupted (but a killed mid-generation run saves nothing).
#
# Why diverse prompts? Expert routing is content-dependent. Coding prompts
# activate different experts than poetry or math. A spread of topics builds a
# general-purpose pin that helps whatever YOU ask later. If you only ever warm
# on one topic, the pin overfits to that topic.

param(
    [string]$Model = (Resolve-Path (Join-Path $PSScriptRoot "..\glm52_i4")).Path,
    [int]$Rounds = 3,
    # Default 32 (not 500): on a cold QLC cache a 500-token run takes hours and
    # a killed mid-generation run saves nothing (usage_save runs only on clean
    # completion). 32 tokens finishes in ~5-10 min even cold, so usage saves
    # frequently and the loop accumulates selections steadily overnight. Each
    # 32-token prompt still records ~90k expert selections.
    [int]$Ngen = 32,
    [string]$Log = (Join-Path $PSScriptRoot "warmup.log")
)

# "Continue" (not "Stop"): the engine writes status to stderr, which "Stop"
# treats as a fatal error and aborts the whole warmup loop on every prompt.
$ErrorActionPreference = "Continue"
$Coli = Join-Path $PSScriptRoot "coli"

if (-not (Test-Path $Coli)) { Write-Error "coli not found at $Coli - run from the c\ directory"; exit 1 }
if (-not (Test-Path $Model)) { Write-Error "model not found at $Model"; exit 1 }

# Diverse prompts across domains - each touches a different expert distribution.
# Kept open-ended ("explain", "write", "list") so generation runs to NGEN tokens
# and routes through many experts rather than stopping early on a short answer.
$Prompts = @(
    "Explain how a transformer neural network works, covering attention, feed-forward layers, and backpropagation in detail.",
    "Write a Python function that implements quicksort with in-place partitioning, including comments explaining each step.",
    "Describe the causes and major events of the French Revolution in chronological order.",
    "What is the difference between TCP and UDP? Explain handshakes, reliability, and use cases.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Explain the theory of general relativity, including the equivalence principle and gravitational time dilation.",
    "List and describe the major organ systems of the human body and their primary functions.",
    "How does photosynthesis work? Explain the light-dependent reactions and the Calvin cycle.",
    "Write a C program that reads a file line by line and counts word frequency using a hash table.",
    "Summarize the plot of Shakespeare's Hamlet, act by act.",
    "Explain the difference between supervised, unsupervised, and reinforcement learning with examples of each.",
    "What causes climate change? Describe the greenhouse effect, carbon cycle, and major greenhouse gases.",
    "Write a recipe for a classic French onion soup, with step-by-step instructions.",
    "Describe how the internet works, from typing a URL to rendering a webpage, including DNS, TCP, HTTP, and browsers.",
    "Explain database normalization, including first, second, and third normal forms with examples.",
    "What is quantum entanglement? Explain it as if to a curious high school student.",
    "Write a poem about the ocean and the passage of time.",
    "Describe the water cycle, including evaporation, condensation, precipitation, and transpiration.",
    "How do vaccines work? Explain the immune response, antibodies, and mRNA vaccine technology.",
    "Explain the Big Bang theory and the evidence supporting it, including cosmic microwave background and redshift.",
    "Write a Python class for a binary search tree with insert, search, and inorder traversal methods.",
    "What are the major branches of philosophy? Describe epistemology, ethics, metaphysics, and logic.",
    "Explain how a CPU executes an instruction, covering fetch, decode, execute, and writeback.",
    "Describe the life cycle of a star, from protostar to main sequence to red giant and beyond.",
    "How does public key cryptography work? Explain RSA, including key generation, encryption, and signing.",
    "Write a dialogue between two characters debating whether artificial intelligence can be conscious.",
    "Explain the economic concepts of supply and demand, elasticity, and market equilibrium.",
    "What is CRISPR gene editing and how does it work? Explain Cas9, guide RNA, and applications.",
    "Describe the major causes and consequences of World War I.",
    "How does a compiler work? Explain lexing, parsing, semantic analysis, optimization, and code generation."
)

function Get-Selections {
    $u = Join-Path $Model ".coli_usage"
    if (-not (Test-Path $u)) { return 0 }
    $tot = 0
    Get-Content $u | ForEach-Object {
        $p = $_ -split '\s+'
        if ($p.Count -eq 3) { $tot += [int]$p[2] }
    }
    return $tot
}

$start = Get-Date
$baseline = Get-Selections
$line = "=" * 72
"$line"                                  | Tee-Object -FilePath $Log -Append
"colibri warmup - started $start"        | Tee-Object -FilePath $Log -Append
"  model:    $Model"                     | Tee-Object -FilePath $Log -Append
"  rounds:   $Rounds x $($Prompts.Count) prompts" | Tee-Object -FilePath $Log -Append
"  ngen:     $Ngen tokens/prompt"        | Tee-Object -FilePath $Log -Append
"  baseline: $baseline selections"       | Tee-Object -FilePath $Log -Append
"$line"                                  | Tee-Object -FilePath $Log -Append

$iter = 0
$total = $Rounds * $Prompts.Count
for ($r = 1; $r -le $Rounds; $r++) {
    for ($i = 0; $i -lt $Prompts.Count; $i++) {
        $iter++
        $prompt = $Prompts[$i]
        $now = Get-Date -Format "HH:mm:ss"
        $sel = Get-Selections
        $header = "[$now] round $r/$Rounds prompt {0,2}/$($Prompts.Count)  (iter $iter/$total)  selections: $sel" -f ($i+1)
        $header | Tee-Object -FilePath $Log -Append
        "  prompt: $($prompt.Substring(0, [Math]::Min(70, $prompt.Length)))..." | Tee-Object -FilePath $Log -Append

        $t0 = Get-Date
        # coli run writes status to stderr (normal) and may exit non-zero on
        # EOS-early; neither is a real failure for our purpose. Relax the
        # error preference and collect ALL output streams so stderr text
        # doesn't abort the loop.
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $output = & python $Coli run --model $Model --ngen $Ngen $prompt 2>&1 |
                      Select-Object -Last 4
        } catch {
            $output = @("  (engine run threw: $($_.Exception.Message))")
        }
        $ErrorActionPreference = $prev
        $elapsed = ((Get-Date) - $t0).TotalSeconds
        $after = Get-Selections
        $delta = $after - $sel

        $output | ForEach-Object { "  $_" | Tee-Object -FilePath $Log -Append }
        "  -> {0:N0}s, +{1} selections (now {2})" -f $elapsed, $delta, $after | Tee-Object -FilePath $Log -Append
        "" | Tee-Object -FilePath $Log -Append
    }
}

$end = Get-Date
$final = Get-Selections
$gain = $final - $baseline
$duration = ($end - $start).ToString("hh\:mm\:ss")
"$line"                                           | Tee-Object -FilePath $Log -Append
"colibri warmup - finished $end"                  | Tee-Object -FilePath $Log -Append
"  duration:    $duration"                        | Tee-Object -FilePath $Log -Append
"  selections:  $baseline -> $final (+$gain)"     | Tee-Object -FilePath $Log -Append
"  next: python coli chat --model $Model"         | Tee-Object -FilePath $Log -Append
"$line"                                           | Tee-Object -FilePath $Log -Append
