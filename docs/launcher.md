# Desktop launcher

The optional desktop launcher starts an **existing** local Colibri installation
and model on Windows or Linux. It provides folder pickers, saved models, Web Chat
and API Server modes, CPU/CUDA selection, readiness checks, and Start/Stop controls.

The launcher does not install Colibri, download or convert models, or install
drivers. Start with the [quick-start guide](quickstart.md) if those are missing.
Colibri's normal command-line installation remains dependency-free; Qt is needed
only for this interface.

## Open the launcher

From a checkout containing this feature, use Python 3.10 or newer:

```sh
python -m pip install -e ".[launcher]"
coli-launcher
```

`python -m colibri.launcher` is an equivalent entrypoint. On Linux the Python
command may be named `python3`. On Windows the GUI entrypoint opens without a
terminal window.

Extract the complete Windows ZIP or Linux tar.gz archive into your Colibri folder
before opening the app (`tar -xzf colibri-launcher-Linux.tar.gz` on Linux).
A native packaged build is a folder containing `ColibriLauncher.exe` on Windows
or `ColibriLauncher` on Linux. Keep its accompanying `_internal` directory beside
the executable. Place this `ColibriLauncher` folder directly inside your Colibri
folder, or place the executable and `_internal` directly beside `coli`. The app
automatically uses that Colibri copy, including when started through a shortcut
with a different working folder. There is no installation-location setting.
Source invocations use the current Colibri folder. Old saved installation paths,
`COLIBRI_HOME`, and other Colibri copies on `PATH` do not override this location.
The package contains the interface and its Qt runtime; a compatible external
Python interpreter and local model files are still required. Builds must be made
on the target OS.
Linux packages inherit the build machine's minimum glibc version; the CI build
uses Ubuntu 24.04 (glibc 2.39). Build locally when targeting an older distribution.

## Launch a model

1. Open the launcher inside your Colibri folder. Extracted releases, source
   checkouts, and installed prefixes containing `bin/coli` and `libexec/colibri`
   are supported. Python is found automatically; use **Python interpreter**
   only if you need to select a particular Python installation or environment.
2. Add the folder containing the model's `config.json`, tokenizer, and weights.
   The installed Colibri identifies the model family and its engine.
3. Choose **Web Chat** or **API Server**, then **Automatic**, **CPU only**, or a
   compatible **NVIDIA CUDA** configuration.
4. Review the readiness messages and press **Start**. Large models can take many
   minutes to load. The loading indicator is deliberately not a percentage.
5. Web Chat opens in your browser when ready. API Server exposes a local API
   address and model identifier to copy into a client.
6. Press **Stop** to release the launched process tree. Closing an active
   launcher offers to stop the model and quit or cancel closing.

Advanced controls set memory budgets, context, output length, and port. Automatic
values use Colibri's defaults and resource planning. The application remembers
the model library and per-model settings, but always opens with inference stopped.
Renaming or removing a library entry never changes model files.
Choose **Light** or **Dark** from **Theme** at the bottom of the sidebar. The
launcher remembers your choice; changing it does not interrupt a running model.
The GPU device selector lists detected devices by name and remembers the choice
for each model. Changing the Python interpreter clears previous readiness
results before checking Colibri again.

Settings are stored in `%LOCALAPPDATA%\Colibri\launcher.json` on Windows and
`$XDG_CONFIG_HOME/colibri/launcher.json` on Linux (default
`~/.config/colibri/launcher.json`). Invalid settings are preserved beside that
file with a `.corrupt-...` suffix before defaults are restored.

## GPU support

GPU readiness depends on the **model family, installed engine, driver/runtime,
and device**. A GPU in the computer is not proof the engine can use it. Automatic
uses CUDA only when the selected configuration can be verified, otherwise CPU
with an explanation. Explicit GPU requests must not silently become CPU runs.
The interface reports the requested backend; it does not claim to have measured
actual GPU utilization. Engine messages that report a CPU fallback are surfaced
in the status and log.

Detected NVIDIA cards are listed even when CPU mode is selected or the model's
engine cannot use CUDA. The CUDA choice is checked again when selected; choosing
CPU does not prevent switching back on a compatible installation.

The first launcher version targets CPU and NVIDIA CUDA. Colibri also has HIP and
Vulkan backends, but their integration is outside this version's simple selector.
CPU mode clears conflicting inherited backend controls, including family-specific
switches. It changes only the child process environment.

The launcher uses the installed Colibri's registry and diagnostic reports. It
does not promise support for arbitrary old or future CLI schemas. Source `dev`
and release v1.11.0 are the initial compatibility targets. Real model/GPU tests
are separate from the launcher's simulated integration tests.
Older installations that only expose a default-engine CUDA probe must also pass
their legacy launch check and show CUDA support in the selected engine itself.
Unsupported combinations remain available in CPU mode;
the selected installation's dedicated DeepSeek V4 probe is used when available.

## Troubleshooting

- **Colibri was not found:** place the complete launcher folder directly inside
  your Colibri folder. For a source invocation, first change to that folder.
- **Python was not found:** choose a working Python 3.10+ interpreter under
  **Python interpreter**.
- **The engine or model is incomplete:** follow the reported missing-file or
  runtime message. The launcher does not build or repair the installation.
- **Web Chat files are missing:** use API Server, or build the existing web UI
  using its documented setup. The launcher checks for `web/dist/index.html`
  rather than opening a broken dashboard.
- **The port is in use:** choose another port. The launcher never stops an
  unrelated service occupying the requested port.
- **Loading takes a long time:** inspect Details for continuing output; Stop
  remains available. A listening socket alone is not treated as model readiness.
- **A failure occurs:** fix the reported selection or dependency and retry.
  Copy the redacted command or export the log from Details when seeking help.

Command previews quote arguments for PowerShell on Windows and a POSIX shell on
Linux. They omit environment values; use Start to apply the launcher's complete
backend and resource configuration.

Ordinary Colibri diagnostics can create/update `.coli_analysis.json` in the model
directory. They do not load model tensor payloads or initialize inference.
The launcher runs diagnostics asynchronously and does not repeat them on every
model-library refresh.

This version binds only to `127.0.0.1`. Network exposure, remote machines, WSL
installation management, multiple simultaneous model servers, and automatic
installation/downloads are outside its scope.

## Develop, test, and package

```sh
python -m pip install -e ".[launcher]" "PySide6==6.11.2" "pyinstaller>=6,<7"
python -m unittest discover -s tests/launcher -v
python -m PyInstaller --noconfirm --distpath desktop/dist --workpath desktop/build desktop/launcher.spec
```

On Ubuntu, install the Qt display libraries before developing or packaging:

```sh
sudo apt-get install libegl1 libopengl0 libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 libxcb-xkb1
```

Packaged builds pin Qt to 6.11.2 so the checked-in third-party notices match the
runtime. The build collects Python and available distribution runtime notices
without network access, and omits unused Qt Virtual Keyboard and PDF plugins.
Review `desktop/launcher-licenses/README.md` before changing the packaged Qt
version or redistributing a build made with a different runtime distribution.

After building, verify the packaged interface using
`python tests/launcher/packaged_smoke.py desktop/dist/ColibriLauncher/ColibriLauncher .`
(use `ColibriLauncher.exe` on Windows). This starts the interface with temporary
settings, verifies installation discovery and the bundled model-inspection helper,
and closes it without loading a model.

Use `QT_QPA_PLATFORM=offscreen` for headless widget tests. The optional
`desktop launcher` workflow runs the suite and builds separate Windows and Linux
artifacts. CI configuration is not a claim that a workflow has already run.
Process tests use a small local HTTP fixture; no model downloads are required.
The normal repository `make check` remains the engine's contribution gate.

The source lives in `colibri/launcher/`. The existing [Tauri desktop chat
client](../desktop/README.md) remains available independently; it connects to a
server, while this application manages the local process.
