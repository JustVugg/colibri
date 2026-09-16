const id: Record<string, string> = {
  // nav
  "nav.chat": "Chat",
  "nav.brain": "Peta Pakar",
  "nav.profiling": "Profiling",

  // brand
  "brand.tagline": "raksasa lokal, jejak kecil",

  // sidebar — connection
  "sidebar.connection": "Koneksi",
  "sidebar.endpoint": "Endpoint API",
  "sidebar.apiKey": "Kunci API",
  "sidebar.apiKeyPlaceholder": "opsional",
  "sidebar.apiKeyHelp": "Hanya disimpan di memori · dikirim ke endpoint ini",
  "sidebar.probe": "Periksa server",
  "status.connected": "Engine dapat diakses",
  "status.notConnected": "Tidak terhubung",
  "status.runtimeUnavailable": "Metrik runtime tidak tersedia",
  "status.serverError": "Tidak dapat menghubungi server.",
  "status.generationFailed": "Generasi gagal.",

  // sidebar — runtime
  "sidebar.runtime": "Runtime",
  "sidebar.runtimeProbe": "Periksa server untuk melihat status runtime.",
  "sidebar.schedulerOnline": "Scheduler online",
  "dashboard.active": "Aktif",
  "dashboard.queued": "Dalam antrean",
  "dashboard.completed": "Selesai",
  "dashboard.failures": "Kegagalan",
  "dashboard.session": "Sesi:",
  "dashboard.prompt": "prompt",
  "dashboard.completion": "completion",

  // sidebar — tiers
  "tier.vram": "VRAM",
  "tier.ram": "RAM",
  "tier.disk": "Disk",
  "tier.ariaLabel": "Pakar: {{vram}} VRAM, {{ram}} RAM, {{disk}} disk",

  // sidebar — inference
  "sidebar.inference": "Inferensi",
  "sidebar.model": "Model",
  "sidebar.kvSession": "Sesi KV",
  "sidebar.kvSessionHelp": "Konteks terisolasi · percakapan mengikuti slot yang dipilih",
  "sidebar.sessionLabel": "Sesi {{slot}}",
  "sidebar.temperature": "Temperatur",
  "sidebar.maxTokens": "Maksimum token output",
  "sidebar.reasoning": "Penalaran",
  "sidebar.reasoning.off": "Nonaktif",
  "sidebar.reasoning.low": "Rendah",
  "sidebar.reasoning.medium": "Sedang",
  "sidebar.reasoning.high": "Tinggi",
  "sidebar.reasoning.max": "Maks",
  "sidebar.transport": "Transport kompatibel OpenAI",

  // top bar
  "topbar.activeModel": "MODEL AKTIF",
  "topbar.tokens": "{{n}} token",
  "topbar.tokPerSec": "{{n}} tok/s",
  "topbar.slot": "slot {{n}}",
  "topbar.truncated": "Terpotong",
  "topbar.truncatedHelp": "Respons mencapai batas maksimum token output dan terpotong. Naikkan \"Maksimum token output\" untuk melihat jawaban lengkap.",
  "topbar.clear": "Bersihkan",

  // hero / empty state
  "hero.title": "COLIBRÌ ENGINE",
  "hero.subtitle": "Tanya sang raksasa.",
  "hero.tagline": "Mesin tetap milikmu.",
  "hero.description": "Hubungkan ke server colibrì lokal dan alirkan respons langsung dari perangkat keras. Tidak ada data yang keluar dari endpoint yang dipilih.",
  "prompts.routing": "Jelaskan cara kerja routing pakar",
  "prompts.benchmark": "Tulis benchmark C kecil",
  "prompts.caching": "Bandingkan caching RAM dan VRAM",

  // chat
  "chat.you": "Anda",
  "chat.colibri": "colibrì",
  "chat.placeholder": "Pesan untuk colibrì…",
  "chat.inputHint": "Enter untuk mengirim · Shift+Enter untuk baris baru",
  "chat.attachImage": "Lampirkan gambar",
  "chat.removeImage": "Hapus gambar",
  "chat.stop": "Hentikan generasi",
  "chat.send": "Kirim pesan",

  // brain
  "brain.title": "Korteks Pakar",
  "brain.waiting": "menunggu engine",
  "brain.layers": "{{rows}} lapisan × {{cols}} pakar",
  "brain.brightnessHint": "kecerahan = aktivitas routing",
  "brain.flashHint": "⚡ kilatan putih = dirutekan pada giliran ini",
  "brain.connectHint": "Hubungkan ke engine untuk melihat korteks.",
  "brain.neverRouted": "belum pernah dirutekan",
  "brain.selections": "~2^{{heat}} kali dipilih",
  "brain.specialist": "⭐ Spesialis: {{top}}",
  "brain.generalist": "Generalis",
  "brain.mtp": "Head MTP — membuat draf token berikutnya untuk decoding spekulatif",
  "brain.early": "lapisan awal — fitur permukaan: token, ejaan, sintaks lokal",
  "brain.lowerMiddle": "lapisan menengah bawah — struktur frasa, relasi kata, fakta sederhana",
  "brain.upperMiddle": "lapisan menengah atas — semantik, konteks jarak jauh, langkah penalaran",
  "brain.late": "lapisan akhir — perencanaan jawaban, gaya, koherensi",
  "brain.final": "lapisan terakhir — pembentukan output: memilih distribusi token berikutnya yang sebenarnya",

  // profiling
  "profile.title": "Profiling — waktu yang dihabiskan engine pada setiap giliran",
  "profile.ioWait": "Waktu tunggu I/O",
  "profile.expertMatmul": "Matmul pakar",
  "profile.attention": "Attention",
  "profile.lmHead": "Head LM",
  "profile.other": "Lainnya",
  "profile.empty": "Belum ada giliran yang diprofilkan — kirim pesan chat, lalu rinciannya akan muncul di sini.",
  "profile.connectHint": "Hubungkan ke engine untuk merekam waktu tiap giliran.",
  "profile.lastTurn": "Giliran terakhir",
  "profile.wallTime": "Waktu total",
  "profile.batching": "Batching",
  "profile.tokensPerForward": "token / forward",
  "profile.diskService": "Layanan disk",
  "profile.overlapped": "tumpang tindih dengan komputasi",
  "profile.window": "Rentang · {{n}} giliran terakhir",
  "profile.throughputTitle": "Throughput per giliran (tok/s)",
  "profile.phaseTitle": "Waktu total per giliran berdasarkan fase (s)",
  "profile.turnCol": "Giliran",
  "profile.tokensCol": "Token",
  "profile.wallCol": "Total",
  "profile.turnsLabel": "{{n}} giliran · terlama → terbaru",
  "profile.oneTurn": "1 giliran",
  "profile.diskNote": "Layanan disk adalah waktu yang digunakan untuk membaca pakar di thread I/O; proses ini tumpang tindih dengan komputasi, sehingga hanya waktu tunggu I/O yang dialami thread komputasi yang dihitung dalam waktu total. Dengan beberapa sesi KV, proporsi menggambarkan seluruh engine selama rentang waktu giliran.",

  // error boundary
  "error.title": "UI colibrì mengalami kesalahan",
  "error.hint": "Engine tidak terpengaruh. Coba muat ulang.",
  "error.retry": "Coba lagi",
}

export default id
