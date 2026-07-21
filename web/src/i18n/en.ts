const en: Record<string, string> = {
  // nav
  "nav.chat": "Chat",
  "nav.brain": "Brain",
  "nav.profiling": "Profiling",

  // brand
  "brand.tagline": "local giant, tiny footprint",

  // sidebar — connection
  "sidebar.connection": "Connection",
  "sidebar.endpoint": "API endpoint",
  "sidebar.apiKey": "API key",
  "sidebar.apiKeyPlaceholder": "optional",
  "sidebar.apiKeyHelp": "Kept in memory only · sent to this endpoint",
  "sidebar.probe": "Probe server",
  "status.connected": "Engine reachable",
  "status.notConnected": "Not connected",
  "status.runtimeUnavailable": "Runtime metrics unavailable",
  "status.serverError": "Could not reach the server.",
  "status.generationFailed": "Generation failed.",

  // sidebar — runtime
  "sidebar.runtime": "Runtime",
  "sidebar.runtimeProbe": "Probe the server to inspect runtime state.",
  "sidebar.schedulerOnline": "Scheduler online",
  "dashboard.active": "Active",
  "dashboard.queued": "Queued",
  "dashboard.completed": "Completed",
  "dashboard.failures": "Failures",
  "dashboard.session": "Session:",
  "dashboard.prompt": "prompt",
  "dashboard.completion": "completion",

  // sidebar — tiers
  "tier.vram": "VRAM",
  "tier.ram": "RAM",
  "tier.disk": "Disk",
  "tier.ariaLabel": "Experts: {{vram}} VRAM, {{ram}} RAM, {{disk}} disk",

  // sidebar — inference
  "sidebar.inference": "Inference",
  "sidebar.model": "Model",
  "sidebar.kvSession": "KV session",
  "sidebar.kvSessionHelp": "Isolated context · conversation follows the selected slot",
  "sidebar.sessionLabel": "Session {{slot}}",
  "sidebar.temperature": "Temperature",
  "sidebar.maxTokens": "Max output tokens",
  "sidebar.reasoning": "Reasoning",
  "sidebar.transport": "OpenAI-compatible transport",

  // top bar
  "topbar.activeModel": "ACTIVE MODEL",
  "topbar.tokens": "{{n}} tokens",
  "topbar.tokPerSec": "{{n}} tok/s",
  "topbar.slot": "slot {{n}}",
  "topbar.clear": "Clear",

  // hero / empty state
  "hero.title": "COLIBRÌ ENGINE",
  "hero.subtitle": "Ask the giant.",
  "hero.tagline": "Keep the machine yours.",
  "hero.description": "Connect to a local colibrì server and stream responses directly from your hardware. Nothing leaves the endpoint you choose.",
  "prompts.routing": "Explain how expert routing works",
  "prompts.benchmark": "Write a small C benchmark",
  "prompts.caching": "Compare RAM and VRAM caching",

  // chat
  "chat.you": "You",
  "chat.colibri": "colibrì",
  "chat.placeholder": "Message colibrì…",
  "chat.inputHint": "Enter to send · Shift+Enter for newline",
  "chat.stop": "Stop generation",
  "chat.send": "Send message",

  // brain
  "brain.title": "Expert Cortex",
  "brain.waiting": "waiting for engine",
  "brain.layers": "{{rows}} layers × {{cols}} experts",
  "brain.brightnessHint": "brightness = routing heat",
  "brain.flashHint": "⚡ white flash = routed this turn",
  "brain.connectHint": "Connect to the engine to see the cortex.",
  "brain.neverRouted": "never routed",
  "brain.selections": "~2^{{heat}} selections",
  "brain.specialist": "⭐ Specialist: {{top}}",
  "brain.generalist": "Generalist",
  "brain.mtp": "MTP head — drafts the next token for speculative decoding",
  "brain.early": "early layers — surface features: tokens, spelling, local syntax",
  "brain.lowerMiddle": "lower-middle — phrase structure, word relations, simple facts",
  "brain.upperMiddle": "upper-middle — semantics, long-range context, reasoning steps",
  "brain.late": "late layers — planning the answer, style, coherence",
  "brain.final": "final layers — output shaping: picks the actual next-token distribution",

  // profiling
  "profile.title": "Profiling — where the engine spends each turn",
  "profile.ioWait": "I/O wait",
  "profile.expertMatmul": "Expert matmul",
  "profile.attention": "Attention",
  "profile.lmHead": "LM head",
  "profile.other": "Other",
  "profile.empty": "No profiled turns yet — send a chat message and the breakdown appears here.",
  "profile.connectHint": "Connect to the engine to collect per-turn timings.",
  "profile.lastTurn": "Last turn",
  "profile.wallTime": "Wall time",
  "profile.batching": "Batching",
  "profile.tokensPerForward": "tokens / forward",
  "profile.diskService": "Disk service",
  "profile.overlapped": "overlapped with compute",
  "profile.window": "Window · last {{n}} turns",
  "profile.throughputTitle": "Throughput per turn (tok/s)",
  "profile.phaseTitle": "Turn wall time by phase (s)",
  "profile.turnCol": "Turn",
  "profile.tokensCol": "Tokens",
  "profile.wallCol": "Wall",
  "profile.turnsLabel": "{{n}} turns · oldest → newest",
  "profile.oneTurn": "1 turn",
  "profile.diskNote": "Disk service is time spent reading experts on I/O threads; it overlaps with compute, so only the I/O wait the compute thread felt counts inside the wall-time stack. With multiple KV sessions the shares describe the whole engine over the turn's window.",

  // error boundary
  "error.title": "colibrì UI hit an error",
  "error.hint": "The engine is unaffected. Try refreshing.",
  "error.retry": "Retry",
}

export default en
