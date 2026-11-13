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
#
# Python lookup: COLIBRI_PYTHON_DIRS adds colon-separated directories to
# search for a Python, ahead of the built-in locations below. It exists
# for installs in unusual places and for tests (see
# tests/test_start_here_python_detection.sh).
here=$(cd "$(dirname "$0")" && pwd)
launcher="$here/c/coli"
[ -f "$launcher" ] || launcher="$here/coli"

# True when the given interpreter is Python 3.10 or newer.
python_usable() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null
}

# Prints the first usable python3* found in the given directory, or nothing.
try_python_dir() {
    for name in python3 python3.15 python3.14 python3.13 python3.12 python3.11 python3.10; do
        candidate="$1/$name"
        if [ -x "$candidate" ] && python_usable "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

# Prints the first usable Python 3.10+, or nothing.
find_python() {
    # Anything on PATH first, including versioned names, so a plain
    # `python3` that is too old does not hide a newer python3.12.
    for candidate in python3 python3.15 python3.14 python3.13 python3.12 python3.11 python3.10 python; do
        if command -v "$candidate" >/dev/null 2>&1 && python_usable "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    # Extra directories from COLIBRI_PYTHON_DIRS, then the well-known
    # install locations a Homebrew or pyenv Python lives in when it is
    # not on PATH (for example when the shell only reaches it through an
    # alias, as in #2812).
    if [ -n "${COLIBRI_PYTHON_DIRS:-}" ]; then
        _saved_ifs=$IFS
        IFS=:
        for dir in $COLIBRI_PYTHON_DIRS; do
            IFS=$_saved_ifs
            if found=$(try_python_dir "$dir"); then
                printf '%s\n' "$found"
                IFS=$_saved_ifs
                return 0
            fi
            IFS=:
        done
        IFS=$_saved_ifs
    fi
    for dir in /opt/homebrew/bin /usr/local/bin; do
        if found=$(try_python_dir "$dir"); then
            printf '%s\n' "$found"
            return 0
        fi
    done
    if [ -n "${HOME:-}" ]; then
        if found=$(try_python_dir "$HOME/.pyenv/shims"); then
            printf '%s\n' "$found"
            return 0
        fi
        for candidate in "$HOME"/.pyenv/versions/*/bin/python3; do
            if [ -x "$candidate" ] && python_usable "$candidate"; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
    fi
    # Keg-only Homebrew python@3.x cellars, which are never linked into bin.
    for ver in 3.15 3.14 3.13 3.12 3.11 3.10; do
        for prefix in /opt/homebrew/opt/python@"$ver" /usr/local/opt/python@"$ver"; do
            candidate="$prefix/bin/python3"
            if [ -x "$candidate" ] && python_usable "$candidate"; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
    done
    return 1
}

_python=$(find_python)
if [ -n "$_python" ]; then
    exec "$_python" "$launcher" setup "$@"
fi

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
