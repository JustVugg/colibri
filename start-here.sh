#!/bin/sh
# colibri: one-step setup and start for Linux and macOS.
#
#   ./start-here.sh
#
# First run: finds your hardware, recommends a model that fits, builds the
# engine (with Vulkan or CUDA when a GPU is usable) or fetches a prebuilt one,
# downloads the model (safe to interrupt: run this again to continue), then
# opens the dashboard in your browser. Later runs start colibri straight away.
#
# Options go to `coli setup` (c/coli setup --help lists them), for example:
#   ./start-here.sh --yes                     no questions, take the recommendation
#   ./start-here.sh --model qwen3-coder-30b   a specific model
#   ./start-here.sh --reconfigure             choose another model
#   ./start-here.sh --no-gpu                  CPU only
#   ./start-here.sh --backend vulkan          the GPU through Vulkan rather than CUDA
# Stop: Ctrl+C in this terminal, or `c/coli stop` from another one.
# The engine build itself is c/setup.sh, unchanged.
here=$(cd "$(dirname "$0")" && pwd)
launcher="$here/c/coli"
[ -f "$launcher" ] || launcher="$here/coli"

for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        exec "$candidate" "$launcher" setup "$@"
    fi
done

echo "colibri needs Python 3.10 or newer, and it was not found."
case "$(uname -s)" in
Darwin) echo "Install it with:  brew install python   (or from https://www.python.org/downloads/)" ;;
*)
    if command -v apt >/dev/null 2>&1; then echo "Install it with:  sudo apt install python3"
    elif command -v dnf >/dev/null 2>&1; then echo "Install it with:  sudo dnf install python3"
    elif command -v pacman >/dev/null 2>&1; then echo "Install it with:  sudo pacman -S python"
    elif command -v zypper >/dev/null 2>&1; then echo "Install it with:  sudo zypper install python3"
    else echo "Install Python 3.10 or newer from your distribution."
    fi ;;
esac
echo "Then run ./start-here.sh again."
exit 1
