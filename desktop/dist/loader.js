// What the window shows while it starts colibri: it asks the Rust side for the
// step it is at (`status`) and offers to start again (`retry`) when one failed.
(function () {
  var it = (navigator.language || "").toLowerCase().indexOf("it") === 0;
  var coli = navigator.userAgent.indexOf("Windows") >= 0 ? "coli.cmd" : "coli";
  var T = it ? {
    python: "Cerco Python…",
    checking: "Cerco il server di colibri…",
    starting: "Avvio {d}…",
    loading: "Il modello si sta caricando… {d} s",
    waiting: "Aspetto il server su {d}…",
    page: "Apro la chat…",
    no_python: "Python 3 non c'è su questo computer.",
    no_python_hint: "Il launcher e il server di colibri sono in Python. Installalo da python.org spuntando \"Add python.exe to PATH\", poi Riprova.",
    no_coli: "Non trovo coli.",
    no_coli_hint: "Questo programma va nella cartella di colibri, accanto a <code>coli</code>.",
    no_ui: "La pagina della chat non c'è (ui/server.py).",
    no_port: "Nessuna porta libera fra 18790 e 18799.",
    not_set_up: "Non c'è ancora un modello.",
    not_set_up_hint: "<code>{c} setup</code> sceglie e scarica un modello adatto a questo computer; oppure <code>{c} serve --model CARTELLA</code>. Poi Riprova.",
    start_failed: "Il server non è partito.",
    ui_failed: "Il server della pagina si è fermato.",
    waiting_hint: "Controlla l'indirizzo dato con --backend.",
    retry: "Riprova"
  } : {
    python: "Looking for Python…",
    checking: "Looking for the colibri server…",
    starting: "Starting {d}…",
    loading: "The model is loading… {d} s",
    waiting: "Waiting for the server at {d}…",
    page: "Opening the chat…",
    no_python: "Python 3 is not on this computer.",
    no_python_hint: "colibri's launcher and server are Python. Install it from python.org with \"Add python.exe to PATH\" ticked, then Retry.",
    no_coli: "coli is not here.",
    no_coli_hint: "This program goes in the colibri folder, next to <code>coli</code>.",
    no_ui: "The chat page is missing (ui/server.py).",
    no_port: "No free port between 18790 and 18799.",
    not_set_up: "There is no model yet.",
    not_set_up_hint: "<code>{c} setup</code> picks a model that fits this computer and downloads it; or <code>{c} serve --model DIR</code>. Then Retry.",
    start_failed: "The server did not start.",
    ui_failed: "The chat page's server stopped.",
    waiting_hint: "Check the address given with --backend.",
    retry: "Retry"
  };
  document.documentElement.lang = it ? "it" : "en";
  var $ = function (id) { return document.getElementById(id); };
  var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return "&#" + c.charCodeAt(0) + ";"; }); };
  var fill = function (s, d) { return s.replace("{d}", d).split("{c}").join(coli); };
  $("retry").textContent = T.retry;
  var shown = "";

  function show(st) {
    var key = st.key || "python", d = st.detail || "";
    var line = st.bad && key === "waiting" ? fill(T.waiting, d) : fill(T[key] || key, d);
    var now = key + "|" + d + "|" + st.bad;
    if (now === shown) return;
    shown = now;
    $("status").textContent = line;
    document.body.className = st.bad ? "bad" : "";
    var hint = T[key + "_hint"];
    $("hint").hidden = !(st.bad && hint);
    if (hint) $("hint").innerHTML = fill(hint, esc(d));
    var log = st.bad && d && key !== "waiting" && key !== "no_ui";
    $("detail").hidden = !log;
    $("detail").textContent = log ? d : "";
    var ending = st.bad && key !== "waiting";
    $("retry").hidden = !ending;
  }

  function invoke(cmd) {
    var ipc = window.__TAURI_INTERNALS__;
    return ipc ? ipc.invoke(cmd) : Promise.reject(new Error("no ipc"));
  }

  function poll() {
    invoke("status").then(show, function () {}).then(function () { setTimeout(poll, 300); });
  }

  $("retry").onclick = function () {
    shown = "";
    $("retry").hidden = true;
    invoke("retry");
  };
  show({ key: "python" });
  poll();
})();
