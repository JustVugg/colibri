#!/bin/sh
# Regression test for colibri #2812.
#
# start-here.sh must find a usable Python 3.10+ that is not on PATH
# (a Homebrew install reached only through a shell alias, a pyenv shim,
# a versioned name) instead of failing first run with "Python 3.10 or
# newer ... was not found".
#
# Run from the repo root:
#   sh tests/test_start_here_python_detection.sh
# Exit 0 when every case passes, 1 otherwise. Needs only sh, coreutils
# and one Python 3.10+ on PATH to stand in for the installed Python.

failures=0
cases=0

here=$(cd "$(dirname "$0")/.." && pwd)
script="$here/start-here.sh"
shell_path=$(command -v sh || true)
if [ -z "$shell_path" ]; then
    echo "SKIP: no sh on PATH"
    exit 0
fi
good_python=$(command -v python3 || true)
if [ -z "$good_python" ]; then
    echo "SKIP: no python3 on PATH to stand in for the installed Python"
    exit 0
fi
if ! "$good_python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    echo "SKIP: python3 on PATH is older than 3.10"
    exit 0
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/colibri-start-here-test.XXXXXX") || exit 1
trap 'rm -rf "$work"' EXIT INT TERM

# Minimal PATH pieces: the utilities start-here.sh needs, but no python.
fakebin="$work/fakebin"
mkdir -p "$fakebin"
for tool in uname dirname; do
    path=$(command -v "$tool") || continue
    ln -s "$path" "$fakebin/$tool"
done

# A python too old to use: answers --version, fails the 3.10 probe.
olddir="$work/oldbin"
mkdir -p "$olddir"
cat > "$olddir/python3" <<'EOF'
#!/bin/sh
if [ "$1" = "--version" ]; then echo "Python 3.9.6"; exit 0; fi
exit 1
EOF
chmod +x "$olddir/python3"

# The "installed" Python, reachable only outside PATH.
gooddir="$work/homebrew/bin"
mkdir -p "$gooddir"
ln -s "$good_python" "$gooddir/python3"
verdir="$work/verbin"
mkdir -p "$verdir"
ln -s "$good_python" "$verdir/python3.12"

# A pyenv-style home with only a shim.
fakehome="$work/fakehome"
mkdir -p "$fakehome/.pyenv/shims"
ln -s "$good_python" "$fakehome/.pyenv/shims/python3"
emptyhome="$work/emptyhome"
mkdir -p "$emptyhome"

# A fake repo checkout: the script under test plus a stub launcher that
# reports it ran instead of doing a real setup.
repo="$work/repo"
mkdir -p "$repo/c"
cp "$script" "$repo/start-here.sh"
cat > "$repo/c/coli" <<'EOF'
import sys
print("LAUNCHER OK python=" + sys.executable)
EOF

# The "missing" cases below assume no usable Python hides in the
# built-in locations start-here.sh probes. If this machine has one
# (for example a Homebrew Python on macOS CI), those cases cannot fail
# and are skipped instead of reported as failures.
builtin_python=0
for _dir in /opt/homebrew/bin /usr/local/bin; do
    for _name in python3 python3.15 python3.14 python3.13 python3.12 python3.11 python3.10; do
        if [ -x "$_dir/$_name" ] && "$_dir/$_name" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            builtin_python=1
        fi
    done
done
for _ver in 3.15 3.14 3.13 3.12 3.11 3.10; do
    for _prefix in /opt/homebrew/opt/python@"$_ver" /usr/local/opt/python@"$_ver"; do
        if [ -x "$_prefix/bin/python3" ] && "$_prefix/bin/python3" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            builtin_python=1
        fi
    done
done

run_case() {
    # run_case <name> <PATH> <HOME> <COLIBRI_PYTHON_DIRS> <expect ok|missing>
    name=$1; path=$2; home=$3; extra_dirs=$4; expect=$5
    cases=$((cases + 1))
    out=$(env -i PATH="$path" HOME="$home" COLIBRI_PYTHON_DIRS="$extra_dirs" "$shell_path" "$repo/start-here.sh" 2>&1)
    status=$?
    if [ "$expect" = ok ]; then
        case "$out" in
            *"LAUNCHER OK"*) echo "ok - $name" ;;
            *) echo "FAIL - $name: expected the launcher to run, got status $status: $out"; failures=$((failures + 1)) ;;
        esac
    else
        if [ "$status" -eq 1 ]; then
            case "$out" in
                *"needs Python 3.10"*) echo "ok - $name" ;;
                *) echo "FAIL - $name: expected the not-found message, got status $status: $out"; failures=$((failures + 1)) ;;
            esac
        else
            echo "FAIL - $name: expected exit 1, got $status: $out"; failures=$((failures + 1))
        fi
    fi
}

# 1. Baseline: a good python3 on PATH still wins.
run_case "PATH python3 works" "$fakebin:$gooddir" "$emptyhome" "" ok
# 2. The reported bug: only a too-old python3 on PATH, a good one in a
# Homebrew-style directory that is not on PATH.
run_case "Homebrew python off PATH is found" "$fakebin:$olddir" "$emptyhome" "$gooddir" ok
# 3. Only a versioned name on PATH, no plain python3.
run_case "versioned python3.12 on PATH is found" "$fakebin:$verdir" "$emptyhome" "" ok
# 4. Only a pyenv shim, not on PATH.
run_case "pyenv shim off PATH is found" "$fakebin:$olddir" "$fakehome" "" ok
# 5. Nothing usable anywhere: still the not-found message, exit 1.
if [ "$builtin_python" -eq 1 ]; then
    echo "skip - no usable python still fails (a built-in location on this machine has a usable Python)"
    echo "skip - too-old python in extra dir still fails (a built-in location on this machine has a usable Python)"
else
    run_case "no usable python still fails" "$fakebin:$olddir" "$emptyhome" "" missing
    # 6. An extra directory holding only a too-old python does not count.
    run_case "too-old python in extra dir still fails" "$fakebin:$olddir" "$emptyhome" "$olddir" missing
fi

echo "$((cases - failures))/$cases cases passed"
[ "$failures" -eq 0 ]
