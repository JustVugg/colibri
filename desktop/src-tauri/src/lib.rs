//! colibrì desktop: the chat page (`ui/`) in a window of its own.
//!
//! It finds `coli` next to itself (a release archive) or above it (a source
//! tree), makes sure a colibri server answers (the one `coli setup` configured,
//! started with `coli start` when it is not running), serves the chat page with
//! `ui/server.py` on a fixed local port and shows it. Closing the window stops
//! the page's server, and the colibri server when it was this window that
//! started it.

use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::Value;
use tauri::webview::NewWindowResponse;
use tauri::{AppHandle, Manager, RunEvent, Url, WebviewUrl, WebviewWindowBuilder};

/// The page keeps its conversations in the browser storage of its origin: a port
/// that changed at every start would show an empty page every time.
const UI_PORT: u16 = 18790;
const DEFAULT_BACKEND: &str = "http://127.0.0.1:8000";

#[derive(Clone, Default, Serialize)]
struct Status {
    key: String,
    detail: String,
    bad: bool,
}

#[derive(Default)]
struct Shell {
    status: Status,
    /// the Python and the `coli` found at the start, for the stop at the end
    tools: Option<(PathBuf, PathBuf)>,
    /// `ui/server.py`, ours to stop
    ui: Option<Child>,
    /// a `coli` command in flight, killed if the window closes meanwhile
    pending: Option<Child>,
    /// the port of the colibri server this window started, stopped on exit
    started: Option<u16>,
    running: bool,
    quitting: bool,
}

type State = Arc<Mutex<Shell>>;
type Failure = (&'static str, String);

fn set(state: &State, key: &str, detail: impl Into<String>, bad: bool) {
    state.lock().unwrap().status = Status {
        key: key.into(),
        detail: detail.into(),
        bad,
    };
}

fn quitting(state: &State) -> bool {
    state.lock().unwrap().quitting
}

/// No console window for the children of a windowed program.
fn hidden(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd
}

/// The path of a Python 3 that runs, from the interpreter itself: the py
/// launcher is not the process to stop later, and the Microsoft Store's
/// `python` alias is on PATH without being a Python.
fn probe_python(prog: &Path, pre: &[&str]) -> Option<PathBuf> {
    let out = hidden(Command::new(prog).args(pre).args([
        "-c",
        "import sys; print(sys.executable if sys.version_info[0] == 3 else '')",
    ]))
    .stdin(Stdio::null())
    .stderr(Stdio::null())
    .output()
    .ok()?;
    let path = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    (out.status.success() && path.is_file()).then_some(path)
}

fn find_python() -> Option<PathBuf> {
    if let Some(p) = env::var_os("COLI_PYTHON") {
        return probe_python(Path::new(&p), &[]);
    }
    let tries: &[(&str, &[&str])] = if cfg!(windows) {
        &[("py", &["-3"]), ("python", &[]), ("python3", &[])]
    } else {
        &[("python3", &[]), ("python", &[])]
    };
    for (prog, pre) in tries {
        if let Some(p) = probe_python(Path::new(prog), pre) {
            return Some(p);
        }
    }
    known_pythons()
        .into_iter()
        .find_map(|p| probe_python(&p, &[]))
}

/// Where the Windows installers put Python, for a machine with none on PATH.
fn known_pythons() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if !cfg!(windows) {
        return out;
    }
    if let Some(home) = env::var_os("USERPROFILE").map(PathBuf::from) {
        for d in ["anaconda3", "miniconda3", "miniforge3"] {
            out.push(home.join(d).join("python.exe"));
        }
    }
    for d in [r"C:\ProgramData\anaconda3", r"C:\ProgramData\miniconda3"] {
        out.push(Path::new(d).join("python.exe"));
    }
    let mut roots = Vec::new();
    if let Some(local) = env::var_os("LOCALAPPDATA").map(PathBuf::from) {
        roots.push(local.join("Programs").join("Python"));
    }
    if let Some(pf) = env::var_os("ProgramFiles").map(PathBuf::from) {
        roots.push(pf);
    }
    roots.push(PathBuf::from(r"C:\"));
    for root in roots {
        // Python313 before Python39: by the minor version, not by the name
        let mut found: Vec<(u32, PathBuf)> = fs::read_dir(&root)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_ascii_lowercase();
                let minor: String = name
                    .strip_prefix("python3")?
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect();
                Some((minor.parse().unwrap_or(0), e.path().join("python.exe")))
            })
            .collect();
        found.sort_by_key(|a| std::cmp::Reverse(a.0));
        out.extend(found.into_iter().map(|(_, p)| p));
    }
    out
}

/// `coli` next to this program (a release archive) or above it (a source tree:
/// desktop/src-tauri/target/release), or where COLIBRI_COLI says.
fn find_coli() -> Option<PathBuf> {
    if let Some(p) = env::var_os("COLIBRI_COLI").map(PathBuf::from) {
        return p.is_file().then_some(p);
    }
    let mut dir = env::current_exe().ok()?.parent()?.to_path_buf();
    for _ in 0..7 {
        for c in [dir.join("coli"), dir.join("c").join("coli")] {
            if c.is_file() {
                return Some(c);
            }
        }
        dir = dir.parent()?.to_path_buf();
    }
    None
}

/// The page's server: next to coli in an archive, one level up in a checkout
/// (the two places `coli ui` looks).
fn find_ui_server(coli: &Path) -> Option<PathBuf> {
    let here = coli.parent()?;
    [
        Some(here.join("ui").join("server.py")),
        here.parent().map(|p| p.join("ui").join("server.py")),
    ]
    .into_iter()
    .flatten()
    .find(|p| p.is_file())
}

/// A plain HTTP/1.0 GET: (status code, headers, body). No TLS and no redirects:
/// it only talks to colibri's own servers.
fn http_get(url: &str, timeout: Duration) -> Option<(u16, String, String)> {
    let u = Url::parse(url).ok()?;
    let host = u.host_str()?.to_string();
    let port = u.port_or_known_default()?;
    let addr = (host.as_str(), port).to_socket_addrs().ok()?.next()?;
    let mut s = TcpStream::connect_timeout(&addr, timeout).ok()?;
    s.set_read_timeout(Some(timeout)).ok()?;
    s.set_write_timeout(Some(timeout)).ok()?;
    let path = match u.query() {
        Some(q) => format!("{}?{}", u.path(), q),
        None => u.path().to_string(),
    };
    write!(
        s,
        "GET {path} HTTP/1.0\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
    )
    .ok()?;
    let mut buf = Vec::new();
    let _ = s.take(1 << 20).read_to_end(&mut buf);
    let text = String::from_utf8_lossy(&buf).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let code = head.split_whitespace().nth(1)?.parse().ok()?;
    Some((code, head.to_string(), body.to_string()))
}

/// None when no colibri server answers at `base`, Some(true) once its model
/// answers, Some(false) while it loads or restarts. A backend given by hand
/// may sit behind a proxy that rewrites the Server header: any server counts.
fn health(base: &str, any_server: bool) -> Option<bool> {
    let (code, head, _) = http_get(
        &format!("{}/health", base.trim_end_matches('/')),
        Duration::from_millis(1500),
    )?;
    let colibri = head
        .lines()
        .any(|l| l.to_ascii_lowercase().starts_with("server: colibri"));
    (any_server || colibri).then_some(code == 200)
}

/// Runs `coli ARGS` to the end, its output in a file and the process in the
/// shared state meanwhile, so that closing the window can kill it.
fn coli_run(state: &State, py: &Path, coli: &Path, args: &[&str]) -> Option<(bool, String)> {
    let out_path = env::temp_dir().join(format!("colibri-desktop-coli-{}.txt", std::process::id()));
    let out = File::create(&out_path).ok()?;
    let err = out.try_clone().ok()?;
    let mut cmd = Command::new(py);
    cmd.arg(coli)
        .args(args)
        .env("PYTHONUNBUFFERED", "1")
        .env("NO_COLOR", "1")
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err);
    let child = hidden(&mut cmd).spawn().ok()?;
    state.lock().unwrap().pending = Some(child);
    let ok = loop {
        let mut s = state.lock().unwrap();
        if s.quitting {
            return None;
        }
        match s.pending.as_mut().map(Child::try_wait) {
            Some(Ok(Some(code))) => {
                s.pending = None;
                break code.success();
            }
            Some(Ok(None)) => {}
            _ => {
                s.pending = None;
                return None;
            }
        }
        drop(s);
        thread::sleep(Duration::from_millis(100));
    };
    let text = fs::read_to_string(&out_path).unwrap_or_default();
    let _ = fs::remove_file(&out_path);
    Some((ok, text))
}

/// The JSON that `--json` prints, after anything printed before it.
fn json_in(text: &str) -> Option<Value> {
    serde_json::from_str(&text[text.find('{')?..]).ok()
}

fn last_lines(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// The base URL of a server in `coli status --json` and `coli start --json`.
fn base_of(server: &Value) -> Option<String> {
    Some(
        server["urls"]["anthropic_base_url"]
            .as_str()?
            .trim_end_matches('/')
            .to_string(),
    )
}

fn port_of(base: &str) -> Option<u16> {
    Url::parse(base).ok()?.port_or_known_default()
}

fn explicit_backend() -> Option<String> {
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--backend" {
            return args.next();
        }
        if let Some(v) = a.strip_prefix("--backend=") {
            return Some(v.to_string());
        }
    }
    env::var("COLIBRI_BACKEND")
        .ok()
        .filter(|v| !v.trim().is_empty())
}

/// Which colibri server the page talks to, and whether it was given by hand.
/// A server that answers already is used as it is: a second engine next to it
/// would only run the machine out of memory.
fn backend(state: &State, py: &Path, coli: &Path) -> Result<(String, bool), Failure> {
    if let Some(url) = explicit_backend() {
        return Ok((url.trim_end_matches('/').to_string(), true));
    }
    set(state, "checking", "", false);
    let status = coli_run(state, py, coli, &["status", "--json"])
        .and_then(|(_, t)| json_in(&t))
        .unwrap_or(Value::Null);
    let configured = if status["server"].is_object() {
        base_of(&status["server"])
    } else {
        None
    };
    if let Some(base) = &configured {
        if matches!(
            status["server"]["state"].as_str(),
            Some("ready" | "loading")
        ) {
            return Ok((base.clone(), false));
        }
    }
    if health(DEFAULT_BACKEND, false).is_some() {
        return Ok((DEFAULT_BACKEND.to_string(), false));
    }
    let (Some(base), Some(true)) = (configured, status["setup"]["ready"].as_bool()) else {
        return Err(("not_set_up", String::new()));
    };
    set(
        state,
        "starting",
        status["setup"]["model"]["name"].as_str().unwrap_or(""),
        false,
    );
    // recorded before the start: a window closed meanwhile still stops it
    state.lock().unwrap().started = port_of(&base);
    let (ok, text) = coli_run(state, py, coli, &["start", "--json", "--no-browser"])
        .ok_or(("start_failed", String::new()))?;
    match json_in(&text).as_ref().and_then(base_of) {
        Some(base) if ok => {
            // `coli start` moves to the next port when another program holds this one
            state.lock().unwrap().started = port_of(&base);
            Ok((base, false))
        }
        _ => Err(("start_failed", last_lines(&text, 12))),
    }
}

/// The fixed port, or the next free one when another program holds it.
fn ui_port() -> Option<u16> {
    (UI_PORT..UI_PORT + 10).find(|&p| {
        TcpStream::connect_timeout(&([127, 0, 0, 1], p).into(), Duration::from_millis(200)).is_err()
            && TcpListener::bind(("127.0.0.1", p)).is_ok()
    })
}

fn ui_log() -> PathBuf {
    env::temp_dir().join("colibri-desktop-ui.log")
}

fn spawn_ui(py: &Path, server: &Path, port: u16, backend: &str) -> std::io::Result<Child> {
    let out = File::create(ui_log())?;
    let err = out.try_clone()?;
    let mut cmd = Command::new(py);
    cmd.arg(server)
        .args([
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--backend",
            backend,
        ])
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    hidden(&mut cmd).spawn()
}

/// Stops a child and whatever it started.
fn kill_tree(child: &mut Child) {
    let pid = child.id().to_string();
    if cfg!(windows) {
        let _ = hidden(Command::new("taskkill").args(["/T", "/F", "/PID", &pid]))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    } else {
        let _ = Command::new("kill")
            .args(["-TERM", &format!("-{pid}")])
            .status();
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        if let Ok(Some(_)) = child.try_wait() {
            return;
        }
        thread::sleep(Duration::from_millis(50));
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Everything between the window opening and the page showing, off the UI thread.
fn start(app: AppHandle, state: State) {
    {
        let mut s = state.lock().unwrap();
        if s.running {
            return;
        }
        s.running = true;
    }
    let result = bring_up(&app, &state);
    let mut s = state.lock().unwrap();
    s.running = false;
    if let Err((key, detail)) = result {
        s.status = Status {
            key: key.into(),
            detail,
            bad: true,
        };
    }
}

fn bring_up(app: &AppHandle, state: &State) -> Result<(), Failure> {
    set(state, "python", "", false);
    let py = find_python().ok_or(("no_python", String::new()))?;
    let coli = find_coli().ok_or(("no_coli", String::new()))?;
    let server = find_ui_server(&coli).ok_or(("no_ui", coli.display().to_string()))?;
    state.lock().unwrap().tools = Some((py.clone(), coli.clone()));
    let (base, explicit) = backend(state, &py, &coli)?;

    // the page's server; a retry replaces the one before
    let old = state.lock().unwrap().ui.take();
    if let Some(mut old) = old {
        kill_tree(&mut old);
    }
    let port = ui_port().ok_or(("no_port", String::new()))?;
    let child = spawn_ui(&py, &server, port, &base).map_err(|e| ("ui_failed", e.to_string()))?;
    state.lock().unwrap().ui = Some(child);

    // the model: the page works with a server that loads too, but it says little
    let t0 = Instant::now();
    let mut checked = Instant::now();
    loop {
        if quitting(state) {
            return Ok(());
        }
        let secs = t0.elapsed().as_secs().to_string();
        match health(&base, explicit) {
            Some(true) => break,
            Some(false) => set(state, "loading", secs, false),
            None if explicit => set(
                state,
                "waiting",
                base.clone(),
                t0.elapsed() > Duration::from_secs(20),
            ),
            None => {
                // the server of `coli setup` binds its port once the engine runs:
                // until then only `coli status` tells a load from a crash
                if checked.elapsed() > Duration::from_secs(10) {
                    checked = Instant::now();
                    let status = coli_run(state, &py, &coli, &["status", "--json"])
                        .and_then(|(_, t)| json_in(&t));
                    if status.is_some_and(|s| s["server"]["state"].as_str() == Some("stopped")) {
                        let log = coli_run(state, &py, &coli, &["logs", "-n", "15"])
                            .map(|(_, t)| t)
                            .unwrap_or_default();
                        return Err(("start_failed", last_lines(&log, 12)));
                    }
                }
                set(state, "loading", secs, false)
            }
        }
        thread::sleep(Duration::from_millis(500));
    }

    set(state, "page", "", false);
    let url = format!("http://127.0.0.1:{port}/");
    let t0 = Instant::now();
    while http_get(&url, Duration::from_millis(800)).is_none_or(|(code, _, _)| code != 200) {
        let died = state
            .lock()
            .unwrap()
            .ui
            .as_mut()
            .is_none_or(|c| matches!(c.try_wait(), Ok(Some(_))));
        if died || t0.elapsed() > Duration::from_secs(30) {
            let log = fs::read_to_string(ui_log()).unwrap_or_default();
            return Err(("ui_failed", last_lines(&log, 12)));
        }
        if quitting(state) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(200));
    }
    if let Some(win) = app.get_webview_window("main") {
        let _ = win.navigate(Url::parse(&url).unwrap());
    }
    Ok(())
}

/// The window closed: what this program started stops with it.
fn shut_down(state: &State) {
    let (pending, ui, started, tools) = {
        let mut s = state.lock().unwrap();
        s.quitting = true;
        (
            s.pending.take(),
            s.ui.take(),
            s.started.take(),
            s.tools.clone(),
        )
    };
    for mut child in [pending, ui].into_iter().flatten() {
        kill_tree(&mut child);
    }
    if let (Some(port), Some((py, coli))) = (started, tools) {
        let _ = hidden(
            Command::new(py)
                .arg(coli)
                .args(["stop", "--port", &port.to_string()]),
        )
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    }
}

#[tauri::command]
fn status(state: tauri::State<'_, State>) -> Status {
    state.lock().unwrap().status.clone()
}

#[tauri::command]
fn retry(app: AppHandle, state: tauri::State<'_, State>) {
    let state = state.inner().clone();
    thread::spawn(move || start(app, state));
}

/// A link that leaves colibri opens in the system's browser, not in this window.
fn open_outside(url: &Url) {
    let u = url.as_str();
    let _ = if cfg!(windows) {
        hidden(Command::new("rundll32").args(["url.dll,FileProtocolHandler", u])).spawn()
    } else if cfg!(target_os = "macos") {
        Command::new("open").arg(u).spawn()
    } else {
        Command::new("xdg-open").arg(u).spawn()
    };
}

fn is_ours(url: &Url) -> bool {
    matches!(url.scheme(), "tauri" | "about" | "data" | "blob")
        || matches!(
            url.host_str(),
            Some("127.0.0.1" | "localhost" | "tauri.localhost" | "ipc.localhost")
        )
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let state: State = Arc::new(Mutex::new(Shell::default()));
    let app = tauri::Builder::default()
        .manage(state.clone())
        .invoke_handler(tauri::generate_handler![status, retry])
        .setup(|app| {
            WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("colibrì")
                .inner_size(1280.0, 820.0)
                .min_inner_size(860.0, 600.0)
                .center()
                .on_navigation(|url| {
                    if is_ours(url) {
                        return true;
                    }
                    open_outside(url);
                    false
                })
                .on_new_window(|url, _| {
                    if is_ours(&url) {
                        return NewWindowResponse::Allow;
                    }
                    open_outside(&url);
                    NewWindowResponse::Deny
                })
                .build()?;
            let handle = app.handle().clone();
            let state = app.state::<State>().inner().clone();
            thread::spawn(move || start(handle, state));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build the colibri desktop application");
    app.run(move |_, event| {
        if let RunEvent::Exit = event {
            shut_down(&state);
        }
    });
}
