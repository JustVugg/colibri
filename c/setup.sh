#!/usr/bin/env bash
# colibrì — installazione su una macchina nuova (Linux x86-64, macOS, Windows/MinGW).
# Compila il motore e fa un self-test. Il MODELLO (~372 GB int4) va copiato a parte
# o rigenerato con: coli convert --model <dir-su-ext4/NVMe>
set -e
cd "$(dirname "$0")"
echo "🐦 colibrì — setup"

UNAME_S=$(uname -s)

# 1) dipendenze
command -v make >/dev/null || { echo "manca make"; exit 1; }
case "$UNAME_S" in
Darwin)
    command -v clang >/dev/null || { echo "manca clang (xcode-select --install)"; exit 1; }
    echo "  clang: $(clang --version | head -1) · $(sysctl -n hw.ncpu) core"
    echo -n "  OpenMP: "
    if brew --prefix libomp >/dev/null 2>&1; then echo "ok (libomp)"
    else echo "libomp assente -> build single-thread (consigliato: brew install libomp)"; fi
    ;;
MINGW*|MSYS*)
    command -v gcc  >/dev/null || { echo "manca gcc (MinGW-w64). Installa: pacman -S mingw-w64-x86_64-gcc make"; exit 1; }
    echo "  gcc: $(gcc -dumpversion) · MinGW-w64"
    echo -n "  OpenMP: "; echo 'int main(){return 0;}' | gcc -fopenmp -xc - -o /tmp/_omp 2>/dev/null && echo ok || { echo "manca libgomp (pacman -S mingw-w64-x86_64-gcc)"; exit 1; }
    ;;
*)
    command -v gcc  >/dev/null || { echo "manca gcc (es: sudo apt install build-essential)"; exit 1; }
    echo "  gcc: $(gcc -dumpversion) · $(nproc) core"
    echo -n "  OpenMP: "; echo 'int main(){return 0;}' | gcc -fopenmp -xc - -o /tmp/_omp 2>/dev/null && echo ok || { echo "manca (libgomp)"; exit 1; }
    ;;
esac

# 2) build: nativa (veloce, per QUESTA macchina). Per un binario da distribuire: make portable
echo "  compilo (ARCH=${ARCH:-native})…"
make -s glm ARCH="${ARCH:-native}"

# 3) self-test sull'oracolo tiny, se presente
if [ -d glm_tiny ] && [ -f ref_glm.json ]; then
    r=$(SNAP=./glm_tiny TF=1 ./glm 64 16 16 2>/dev/null | grep -oE "[0-9]+/[0-9]+ posizioni" || true)
    echo "  self-test motore: ${r:-?}  (atteso 32/32)"
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
echo "  RAM: ${ram} GB   (più RAM = più expert in cache = più veloce)"
echo
echo "pronto. Prossimi passi:"
echo "  ./coli build           # (gia' fatto)"
echo "  ./coli convert --model /percorso/NVMe/glm52_i4     # genera il modello int4 (ore)"
echo "  ./coli info  --model /percorso/NVMe/glm52_i4"
echo "  ./coli chat  --model /percorso/NVMe/glm52_i4 --ram <GB>"
echo
echo "IMPORTANTE: tieni il modello su disco VELOCE (NVMe/ext4), MAI su /mnt/c o rete."
