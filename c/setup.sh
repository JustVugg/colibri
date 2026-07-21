#!/usr/bin/env bash
# colibrì — installazione su una macchina nuova (Linux x86-64, macOS, Windows/MinGW).
# Compila il motore e fa un self-test. Il MODELLO (~372 GB int4) va copiato a parte
# o rigenerato con: coli convert --model <dir-su-ext4/NVMe>
set -e
cd "$(dirname "$0")"
echo "🐦 colibrì — setup"

UNAME_S=$(uname -s)

# 1) dipendenze
command -v make >/dev/null || { echo "make is missing"; exit 1; }
case "$UNAME_S" in
Darwin)
    command -v clang >/dev/null || { echo "clang is missing (run: xcode-select --install)"; exit 1; }
    echo "  clang: $(clang --version | head -1) · $(sysctl -n hw.ncpu) core"
    echo -n "  OpenMP: "
    if [ -f "$(brew --prefix libomp 2>/dev/null)/lib/libomp.dylib" ]; then echo "ok (libomp)"
    else echo "libomp is missing -> single-threaded build (recommended: brew install libomp)"; fi
    ;;
MINGW*|MSYS*)
    command -v gcc  >/dev/null || { echo "gcc is missing (MinGW-w64). Install: pacman -S mingw-w64-x86_64-gcc make"; exit 1; }
    echo "  gcc: $(gcc -dumpversion) · MinGW-w64"
    echo -n "  OpenMP: "; echo 'int main(){return 0;}' | gcc -fopenmp -xc - -o /tmp/_omp 2>/dev/null && echo ok || { echo "libgomp is missing (pacman -S mingw-w64-x86_64-gcc)"; exit 1; }
    ;;
*)
    command -v gcc  >/dev/null || { echo "gcc is missing (for example: sudo apt install build-essential)"; exit 1; }
    echo "  gcc: $(gcc -dumpversion) · $(nproc) core"
    echo -n "  OpenMP: "; echo 'int main(){return 0;}' | gcc -fopenmp -xc - -o /tmp/_omp 2>/dev/null && echo ok || { echo "libgomp is missing"; exit 1; }
    ;;
esac

# 2) build: nativa (veloce, per QUESTA macchina). Per un binario da distribuire: make portable
echo "  building (ARCH=${ARCH:-native})…"
make -s colibri ARCH="${ARCH:-native}"

# 3) self-test sull'oracolo tiny, se presente
if [ -d glm_tiny ] && [ -f ref_glm.json ]; then
    r=$(SNAP=./glm_tiny TF=1 ./colibri 64 16 16 2>/dev/null | grep -oE "[0-9]+/[0-9]+ positions" || true)
    echo "  engine self-test: ${r:-?}  (expected 32/32)"
fi

# 4) info macchina (la velocità dipende da QUESTI due numeri, non dalla GPU)
case "$UNAME_S" in
Darwin)
    ram=$(( $(sysctl -n hw.memsize) / 1000000000 ))
    ;;
MINGW*|MSYS*)
    # MSYS2 fornisce /proc/meminfo come symlink (più affidabile di wmic, deprecato)
    ram=$(awk '/MemTotal/{printf "%.0f", $2/1e6}' /proc/meminfo 2>/dev/null || echo "?")
    ;;
*)
    ram=$(awk '/MemTotal/{printf "%.0f", $2/1e6}' /proc/meminfo 2>/dev/null || echo "?")
    ;;
esac
echo "  RAM: ${ram} GB   (more RAM = more cached experts = faster inference)"
echo
echo "ready. Next steps:"
echo "  ./coli build           # already done"
echo "  ./coli convert --model /path/on/NVMe/glm52_i4     # generate the int4 model (hours)"
echo "  ./coli info  --model /path/on/NVMe/glm52_i4"
echo "  ./coli chat  --model /path/on/NVMe/glm52_i4 --ram <GB>"
echo
echo "IMPORTANT: keep the model on fast storage (NVMe/ext4), never on /mnt/c or a network mount."
