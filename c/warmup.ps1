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
    [string]$Model = $env:COLI_MODEL,
    [int]$Rounds = 3,
    # Default 32 (not 500): on a cold QLC cache a 500-token run takes hours and
    # a killed mid-generation run saves nothing (usage_save runs only on clean
    # completion). 32 tokens finishes in ~5-10 min even cold, so usage saves
    # frequently and the loop accumulates selections steadily overnight. Each
    # 32-token prompt still records ~90k expert selections.
    [int]$Ngen = 32,
    [string]$Log = (Join-Path $PSScriptRoot "warmup.log"),
    # Backend: 'auto' lets the launcher auto-enable CUDA (default, matches how you
    # infer). 'gpu' forces device 0; 'cpu' forces the pure-CPU path (--gpu none).
    # NOTE: routing differs slightly between CPU (int8-dot) and GPU (float) matmuls,
    # so the .coli_usage pin is backend-flavoured. Warm on the SAME backend you run.
    [ValidateSet('auto','gpu','cpu')][string]$Backend = 'auto',
    # Optional file with one extra prompt per line (blank lines and # comments skipped) -
    # appended to the built-in set for domain-specific warmups.
    [string]$PromptFile
)

# "Continue" (not "Stop"): the engine writes status to stderr, which "Stop"
# treats as a fatal error and aborts the whole warmup loop on every prompt.
$ErrorActionPreference = "Continue"
$Coli = Join-Path $PSScriptRoot "coli"

# Make CUDA/gcc discoverable when launched from a shell opened before they were installed.
$env:PATH = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' +
            [System.Environment]::GetEnvironmentVariable('Path','User') + ';' + $env:PATH

$BackendArgs = switch ($Backend) {
    'cpu' { @('--gpu','none') }
    'gpu' { @('--gpu','0') }
    default { @() }
}

if (-not (Test-Path $Coli)) { Write-Error "coli not found at $Coli - run from the c\ directory"; exit 1 }
if ([string]::IsNullOrWhiteSpace($Model)) {
    Write-Error "no model specified. Pass -Model <dir> or set `$env:COLI_MODEL (e.g. C:\Users\...\GLM5.2)."; exit 1
}
if (-not (Test-Path $Model)) { Write-Error "model not found at: $Model"; exit 1 }

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
    "How does a compiler work? Explain lexing, parsing, semantic analysis, optimization, and code generation.",
    # --- diversity extension: the .coli_usage pin is only as good as the token
    # coverage of the warmup - multilingual text, code in several languages, and
    # structured formats route through experts the original 30 English prompts
    # never touch, so mixed workloads start with a warmer cache. ---
    "Explique en francais ce qu'est la photosynthese et pourquoi les feuilles sont vertes.",
    "Raconte en francais l'histoire de la Revolution francaise en un paragraphe detaille.",
    "Explica en espanol como funciona el sistema solar y por que los planetas orbitan el sol.",
    "Beschreibe auf Deutsch, wie ein Verbrennungsmotor funktioniert, Schritt fur Schritt.",
    "Spiega in italiano come si prepara una vera pizza napoletana, passo dopo passo.",
    "用中文解释什么是机器学习，以及它与传统编程的区别。",
    "日本語で、寿司の作り方と歴史について説明してください。",
    "Объясните по-русски, как работает интернет и что такое протокол TCP/IP.",
    "اشرح باللغة العربية كيف تعمل الطاقة الشمسية ولماذا هي مهمة للمستقبل.",
    "Translate this paragraph into French, then explain each grammar choice: The scientists discovered that the ancient river had changed course twice.",
    "Write a JavaScript async function that fetches JSON from three URLs in parallel and merges the results, with error handling.",
    "Write a Rust function that parses a CSV line respecting quoted fields, and explain ownership choices.",
    "Write a SQL schema for a library: books, members, loans, with foreign keys, then three useful queries.",
    "Write a regular expression that validates an email address and explain each component.",
    "Write a Bash script that finds the ten largest files under a directory and prints them human-readable.",
    "Produce a JSON object describing a fictional company: name, founded, employees array with roles and salaries, offices by city.",
    "Create a markdown table comparing four programming languages by typing, speed, ecosystem, and learning curve.",
    "Solve step by step: a rectangle's length is twice its width and its perimeter is 36 cm. Find its area.",
    "Prove that the square root of 2 is irrational, step by step.",
    "Compute the derivative of f(x) = x^3 * ln(x) and explain each rule you used.",
    "A bag has 5 red, 3 blue, 2 green marbles. What is the probability of drawing two red without replacement? Show the work.",
    "Draft a formal business email requesting a deadline extension on a client project, with a proposed new timeline.",
    "Write the terms-of-service summary for a mobile app in plain language: data collected, user rights, cancellation.",
    "Explain the difference between a stock and a bond, and how interest rates affect each.",
    "Describe the symptoms, causes, and standard treatments of type 2 diabetes.",
    "Explain how a court trial proceeds in a common-law system, from filing to verdict.",
    "Write a dialogue between a customer and a support agent resolving a billing error, then summarize it in two sentences.",
    "Write a sonnet about a city waking up in winter, then explain its rhyme scheme.",
    "Write the opening paragraph of a mystery novel set in a lighthouse during a storm.",
    "List the steps to change a car tire safely, numbered, with a tools checklist first.",
    "Explain chess strategy for beginners: openings, center control, piece development, and common mistakes.",
    "Describe the rules of association football (soccer) including offside, in detail.",
    "Explain how vaccines achieve herd immunity, with the math of R0 thresholds.",
    "Describe the nitrogen cycle and why fertilizer runoff causes algal blooms.",
    "Explain what happens inside a black hole's event horizon according to general relativity.",
    "Write a recipe for vegetarian chili with exact quantities and timing, then a shopping list.",
    "Explain the causes of the 2008 financial crisis: subprime mortgages, securitization, and leverage.",
    "Compare Buddhism and Stoicism: their views on suffering, desire, and the good life.",
    "Explain how GPS determines your position, including why relativity corrections are needed.",
    "Describe the water treatment process from reservoir to tap, stage by stage."
)
if ($PromptFile -and (Test-Path $PromptFile)) {
    $extra = Get-Content $PromptFile | Where-Object { $_.Trim() -and -not $_.Trim().StartsWith('#') }
    if ($extra) { $Prompts += $extra; Write-Host "Loaded $($extra.Count) extra prompts from $PromptFile" }
}

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
"  backend:  $Backend"                   | Tee-Object -FilePath $Log -Append
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
            $output = & python $Coli run --model $Model --ngen $Ngen @BackendArgs $prompt 2>&1 |
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
