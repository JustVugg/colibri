const de: Record<string, string> = {
  // navigation
  "nav.chat": "Chat",
  "nav.brain": "Expertenkarte",
  "nav.profiling": "Profiling",

  // Marke
  "brand.tagline": "lokaler Riese, kleiner Fußabdruck",

  // Seitenleiste – Verbindung
  "sidebar.connection": "Verbindung",
  "sidebar.endpoint": "API-Endpunkt",
  "sidebar.apiKey": "API-Schlüssel",
  "sidebar.apiKeyPlaceholder": "optional",
  "sidebar.apiKeyHelp": "Nur im Speicher · wird an diesen Endpunkt gesendet",
  "sidebar.probe": "Server prüfen",
  "status.connected": "Engine erreichbar",
  "status.notConnected": "Nicht verbunden",
  "status.runtimeUnavailable": "Laufzeitmetriken nicht verfügbar",
  "status.serverError": "Server konnte nicht erreicht werden.",
  "status.generationFailed": "Generierung fehlgeschlagen.",

  // Seitenleiste – Laufzeit
  "sidebar.runtime": "Laufzeit",
  "sidebar.runtimeProbe": "Server prüfen, um den Laufzeitstatus zu sehen.",
  "sidebar.schedulerOnline": "Scheduler online",
  "dashboard.active": "Aktiv",
  "dashboard.queued": "Wartend",
  "dashboard.completed": "Abgeschlossen",
  "dashboard.failures": "Fehler",
  "dashboard.session": "Sitzung:",
  "dashboard.prompt": "Prompt",
  "dashboard.completion": "Antwort",

  // Seitenleiste – Tiers
  "tier.vram": "VRAM",
  "tier.ram": "RAM",
  "tier.disk": "Datenträger",
  "tier.ariaLabel": "Experten: {{vram}} VRAM, {{ram}} RAM, {{disk}} Datenträger",

  // Seitenleiste – Inferenz
  "sidebar.inference": "Inferenz",
  "sidebar.model": "Modell",
  "sidebar.kvSession": "KV-Sitzung",
  "sidebar.kvSessionHelp": "Isolierter Kontext · Unterhaltung folgt dem gewählten Slot",
  "sidebar.sessionLabel": "Sitzung {{slot}}",
  "sidebar.temperature": "Temperatur",
  "sidebar.maxTokens": "Maximale Ausgabetokens",
  "sidebar.reasoning": "Reasoning",
  "sidebar.transport": "OpenAI-kompatibler Transport",

  // Kopfzeile
  "topbar.activeModel": "AKTIVES MODELL",
  "topbar.tokens": "{{n}} Tokens",
  "topbar.tokPerSec": "{{n}} Tok/s",
  "topbar.slot": "Slot {{n}}",
  "topbar.truncated": "Abgeschnitten",
  "topbar.truncatedHelp": "Die Antwort erreichte die maximale Tokenanzahl und wurde abgeschnitten. Erhöhe \"Maximale Ausgabetokens\", um die vollständige Antwort zu sehen.",
  "topbar.clear": "Leeren",

  // Startansicht
  "hero.title": "COLIBRÌ ENGINE",
  "hero.subtitle": "Frag den Riesen.",
  "hero.tagline": "Deine Hardware bleibt deine.",
  "hero.description": "Mit einem lokalen Colibrì-Server verbinden und Antworten direkt von deiner Hardware streamen. Nichts verlässt den gewählten Endpunkt.",
  "prompts.routing": "Erkläre, wie Experten-Routing funktioniert",
  "prompts.benchmark": "Schreibe einen kleinen C-Benchmark",
  "prompts.caching": "Vergleiche RAM- und VRAM-Caching",

  // Chat
  "chat.you": "Du",
  "chat.colibri": "colibrì",
  "chat.placeholder": "Nachricht an colibrì…",
  "chat.inputHint": "Eingabetaste zum Senden · Umschalt+Eingabetaste für Zeilenumbruch",
  "chat.attachImage": "Bild anhängen",
  "chat.removeImage": "Bild entfernen",
  "chat.stop": "Generierung stoppen",
  "chat.send": "Nachricht senden",

  // Expertenkarte
  "brain.title": "Expertenkortex",
  "brain.waiting": "warte auf Engine",
  "brain.layers": "{{rows}} Schichten × {{cols}} Experten",
  "brain.brightnessHint": "Helligkeit = Routing-Aktivität",
  "brain.flashHint": "⚡ weißes Blinken = in diesem Zug geroutet",
  "brain.connectHint": "Verbinde dich mit der Engine, um den Kortex zu sehen.",
  "brain.neverRouted": "nie geroutet",
  "brain.selections": "~2^{{heat}} Auswahlen",
  "brain.specialist": "⭐ Spezialist: {{top}}",
  "brain.generalist": "Generalist",
  "brain.mtp": "MTP-Head – entwirft das nächste Token für spekulative Dekodierung",
  "brain.early": "frühe Schichten – Oberflächenmerkmale: Tokens, Schreibweise, lokale Syntax",
  "brain.lowerMiddle": "untere mittlere Schichten – Phrasenstruktur, Wortbeziehungen, einfache Fakten",
  "brain.upperMiddle": "obere mittlere Schichten – Semantik, langer Kontext, Reasoning-Schritte",
  "brain.late": "späte Schichten – Antwortplanung, Stil, Kohärenz",
  "brain.final": "letzte Schichten – Ausgabeformung: wählt die tatsächliche nächste-Token-Verteilung",

  // Profiling
  "profile.title": "Profiling – wo die Engine Zeit pro Zug verbringt",
  "profile.ioWait": "I/O-Wartezeit",
  "profile.expertMatmul": "Experten-Matmul",
  "profile.attention": "Attention",
  "profile.lmHead": "LM-Head",
  "profile.other": "Sonstiges",
  "profile.empty": "Noch keine profilierten Züge – sende eine Chat-Nachricht, dann erscheint die Aufschlüsselung.",
  "profile.connectHint": "Verbinde dich mit der Engine, um Zeiten pro Zug zu erfassen.",
  "profile.lastTurn": "Letzter Zug",
  "profile.wallTime": "Gesamtzeit",
  "profile.batching": "Batching",
  "profile.tokensPerForward": "Tokens / Forward",
  "profile.diskService": "Datenträgerdienst",
  "profile.overlapped": "mit Berechnung überlappt",
  "profile.window": "Fenster · letzte {{n}} Züge",
  "profile.throughputTitle": "Durchsatz pro Zug (Tok/s)",
  "profile.phaseTitle": "Gesamtzeit pro Zug nach Phase (s)",
  "profile.turnCol": "Zug",
  "profile.tokensCol": "Tokens",
  "profile.wallCol": "Gesamt",
  "profile.turnsLabel": "{{n}} Züge · ältester → neuester",
  "profile.oneTurn": "1 Zug",
  "profile.diskNote": "Datenträgerdienst ist die Zeit zum Lesen von Experten in I/O-Threads; sie überlappt mit der Berechnung. Nur die I/O-Wartezeit des Rechenthreads fließt in die Gesamtzeit ein. Bei mehreren KV-Sitzungen beschreiben die Anteile die ganze Engine im Zeitfenster des Zuges.",

  // Fehlergrenze
  "error.title": "Die colibrì-Oberfläche hat einen Fehler festgestellt",
  "error.hint": "Die Engine ist nicht betroffen. Versuche, die Seite neu zu laden.",
  "error.retry": "Erneut versuchen",
}

export default de
