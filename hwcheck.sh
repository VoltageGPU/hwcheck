#!/usr/bin/env bash
# VoltageGPU hardware check, version 1.1 (2026-10-24).
# Read-only: compares this machine with what its listing says. Nothing about the
# machine is sent anywhere; the speed test downloads 25 MB from
# speed.cloudflare.com (skip it with --no-net). Exit code 0 = matches, 2 = does not.
# --acceptance, as first option, runs acceptance.sh instead: a 10-minute test of a
# multi-GPU machine before a long job (README.md).
#
#   curl -fsSL https://voltagegpu.com/hwcheck.sh | bash -s -- --gpu "RTX 4090" --gpus 1 --vram 24 --ram 128 --disk 500
#   curl -fsSL https://voltagegpu.com/hwcheck.sh | bash -s -- --acceptance [--quick]
set -u

# ── --acceptance: only the acceptance.sh whose SHA-256 is pinned here runs ──
ACCEPTANCE_URL="https://voltagegpu.com/hwcheck-acceptance.sh"
ACCEPTANCE_SHA256="6bce8305ebf6903eb3c46b2a12423019817d81e8182542c8ef9fa1a3544f5e9d"
if [ "${1:-}" = "--acceptance" ]; then
  shift
  sha() { sha256sum "$1" 2>/dev/null | awk '{ print $1 }'; }
  # From a clone: the acceptance.sh next to this file. Piped: downloaded, then checked.
  SELF="${BASH_SOURCE[0]:-}"
  if [ -n "$SELF" ] && [ -f "$SELF" ] && [ "$(sha "$(dirname "$SELF")/acceptance.sh")" = "$ACCEPTANCE_SHA256" ]; then
    exec bash "$(dirname "$SELF")/acceptance.sh" "$@" < /dev/null
  fi
  ACC=$(mktemp) || exit 70
  trap 'rm -f "$ACC"' EXIT
  if ! curl -fsSL --max-time 60 "$ACCEPTANCE_URL" -o "$ACC"; then echo "could not download $ACCEPTANCE_URL" >&2; exit 70; fi
  if [ "$(sha "$ACC")" != "$ACCEPTANCE_SHA256" ]; then
    echo "$ACCEPTANCE_URL does not match the SHA-256 pinned in this hwcheck.sh: not running it" >&2; exit 70
  fi
  bash "$ACC" "$@" < /dev/null; exit $?
fi

GPU=""; GPUS=""; VRAM=""; RAM=""; DISK=""; NET=1
usage() {
  echo 'usage: hwcheck.sh --gpu "RTX 4090" --gpus 1 --vram 24 [--ram GB] [--disk GB] [--no-net]'
  echo '       hwcheck.sh --acceptance [--quick] [--gpus 8] [--bare-metal]   (10-minute test, README.md)'
}
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="${2:-}"; shift 2 ;;
    --gpus) GPUS="${2:-}"; shift 2 ;;
    --vram) VRAM="${2:-}"; shift 2 ;;
    --ram) RAM="${2:-}"; shift 2 ;;
    --disk) DISK="${2:-}"; shift 2 ;;
    --no-net) NET=0; shift ;;
    -h|--help) usage; exit 0 ;;
    --acceptance) echo "--acceptance must be the first option" >&2; usage >&2; exit 64 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 64 ;;
  esac
done

FAILS=""
WARNS=""
fail() { FAILS="${FAILS:+$FAILS, }$1"; }
warn() { WARNS="${WARNS:+$WARNS, }$1"; }
row() { printf '  %-14s %-24s %-34s %s\n' "$1" "$2" "$3" "$4"; }
num() { case "$1" in ''|*[!0-9.]*) return 1 ;; *) return 0 ;; esac; }
# measured >= listed * tolerance
at_least() { awk -v m="$1" -v l="$2" -v t="$3" 'BEGIN { exit !(m + 0 >= l * t) }'; }
norm() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/\(r\)|\(tm\)//g; s/[-_\/,]+/ /g; s/ +/ /g; s/^ //; s/ $//'; }
# Model tokens of the listing (the ones with a digit: 4090, h100, a6000, 80gb) must all be in the measured name.
model_matches() {
  local listed measured tok ok=1 any=0
  listed=$(norm "$1"); measured=" $(norm "$2") "
  for tok in $listed; do
    case "$tok" in *[0-9]*) any=1; case "$measured" in *" $tok "*) ;; *) ok=0 ;; esac ;; esac
  done
  [ "$any" = 1 ] && [ "$ok" = 1 ]
}

echo "VoltageGPU hardware check (read-only, nothing about this machine is sent anywhere)"
echo "  $(date -u '+%Y-%m-%d %H:%M UTC') · $(hostname 2>/dev/null || echo host)"
echo
row "" "LISTED" "MEASURED" ""

# ── GPU ────────────────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
  COUNT=$(printf '%s\n' "$NAMES" | grep -c .)
  FIRST=$(printf '%s\n' "$NAMES" | head -n 1)
  DISTINCT=$(printf '%s\n' "$NAMES" | sort -u | grep -c .)
  VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | head -n 1 | tr -dc '0-9')
  VRAM_GB=$(awk -v m="${VRAM_MIB:-0}" 'BEGIN { printf "%d", m / 1024 + 0.5 }')
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)
  CUDA=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')

  if [ -n "$GPU" ]; then
    if model_matches "$GPU" "$FIRST" && [ "$DISTINCT" = 1 ]; then row "GPU model" "$GPU" "$FIRST" "OK"
    else row "GPU model" "$GPU" "$FIRST$([ "$DISTINCT" != 1 ] && echo ' (mixed)')" "DOES NOT MATCH"; fail "GPU model"; fi
  else row "GPU model" "-" "$FIRST" "info"; fi

  if num "$GPUS"; then
    if [ "$COUNT" -ge "${GPUS%.*}" ]; then row "GPU count" "$GPUS" "$COUNT" "OK"
    else row "GPU count" "$GPUS" "$COUNT" "DOES NOT MATCH"; fail "GPU count"; fi
  else row "GPU count" "-" "$COUNT" "info"; fi

  if num "$VRAM"; then
    if at_least "$VRAM_GB" "$VRAM" 0.93; then row "VRAM per GPU" "$VRAM GB" "$VRAM_GB GB" "OK"
    else row "VRAM per GPU" "$VRAM GB" "$VRAM_GB GB" "DOES NOT MATCH"; fail "VRAM"; fi
  else row "VRAM per GPU" "-" "$VRAM_GB GB" "info"; fi

  row "Driver / CUDA" "" "${DRIVER:-?} / ${CUDA:-?}" "OK"
else
  row "GPU" "${GPU:--}" "no GPU visible (nvidia-smi failed)" "DOES NOT MATCH"; fail "no GPU visible"
  COUNT=0
fi

# ── CUDA compute (only when the image ships PyTorch) ──────────────────
if [ "${COUNT:-0}" -gt 0 ] && command -v python3 >/dev/null 2>&1 && python3 -c 'import torch' >/dev/null 2>&1; then
  OKGPUS=$(python3 - <<'PY' 2>/dev/null
import torch
ok = 0
for i in range(torch.cuda.device_count()):
    try:
        x = torch.ones((2048, 2048), device=f"cuda:{i}")
        if float((x @ x).sum()) > 0:
            ok += 1
    except Exception:
        pass
print(ok)
PY
)
  OKGPUS=${OKGPUS:-0}
  if [ "$OKGPUS" -ge "$COUNT" ]; then row "CUDA compute" "" "matmul on $OKGPUS/$COUNT GPU" "OK"
  else row "CUDA compute" "" "matmul on $OKGPUS/$COUNT GPU" "DOES NOT MATCH"; fail "CUDA compute"; fi
else
  row "CUDA compute" "" "not tested (no PyTorch)" "info"
fi

# ── CPU, RAM, disk ────────────────────────────────────────────────────
row "CPU" "" "$(nproc 2>/dev/null || echo ?) threads" "info"

MEM_KB=$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo 2>/dev/null)
MEM_GB=$(awk -v k="${MEM_KB:-0}" 'BEGIN { printf "%d", k / 1048576 + 0.5 }')
LIMIT=""
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  [ -r "$f" ] && LIMIT=$(tr -dc '0-9' < "$f") && break
done
if [ -n "$LIMIT" ] && awk -v l="$LIMIT" -v k="${MEM_KB:-0}" 'BEGIN { exit !(l / 1024 < k) }'; then
  MEM_GB=$(awk -v l="$LIMIT" 'BEGIN { printf "%d", l / 1073741824 + 0.5 }'); MEM_NOTE=" (container limit)"
else MEM_NOTE=""; fi
if num "$RAM"; then
  if at_least "$MEM_GB" "$RAM" 0.9; then row "RAM" "$RAM GB" "$MEM_GB GB$MEM_NOTE" "OK"
  else row "RAM" "$RAM GB" "$MEM_GB GB$MEM_NOTE" "below the listing"; warn "RAM"; fi
else row "RAM" "-" "$MEM_GB GB$MEM_NOTE" "info"; fi

DISK_GB=$(df -BG --output=size / 2>/dev/null | tail -n 1 | tr -dc '0-9')
FREE_GB=$(df -BG --output=avail / 2>/dev/null | tail -n 1 | tr -dc '0-9')
if num "$DISK"; then
  if at_least "${DISK_GB:-0}" "$DISK" 0.8; then row "Disk (/)" "$DISK GB" "${DISK_GB:-?} GB, ${FREE_GB:-?} GB free" "OK"
  else row "Disk (/)" "$DISK GB" "${DISK_GB:-?} GB, ${FREE_GB:-?} GB free" "below the listing"; warn "disk"; fi
else row "Disk (/)" "-" "${DISK_GB:-?} GB, ${FREE_GB:-?} GB free" "info"; fi
if [ -d /mnt ] && mountpoint -q /mnt 2>/dev/null; then
  row "Volume (/mnt)" "" "$(df -BG --output=size /mnt 2>/dev/null | tail -n 1 | tr -d ' ') mounted" "info"
fi

# ── Download speed (one stream, to Cloudflare) ────────────────────────
if [ "$NET" = 1 ] && command -v curl >/dev/null 2>&1; then
  read -r CODE BPS SIZE <<EOF
$(curl -s -o /dev/null -w '%{http_code} %{speed_download} %{size_download}' --max-time 20 'https://speed.cloudflare.com/__down?bytes=25000000' 2>/dev/null)
EOF
  if [ "${CODE:-0}" = 200 ] && [ "${SIZE:-0}" -gt 1000000 ] 2>/dev/null; then
    row "Download" "" "$(awk -v b="$BPS" 'BEGIN { printf "%d", b * 8 / 1000000 }') Mbit/s (one stream)" "info"
  else
    row "Download" "" "not measured (test server unreachable)" "info"
  fi
fi

echo
if [ -z "$FAILS" ]; then
  echo "RESULT: MATCHES THE LISTING${WARNS:+ (note: $WARNS below the listing)}"
  exit 0
fi
echo "RESULT: DOES NOT MATCH THE LISTING ($FAILS)"
echo "Release the pod from your dashboard and write to contact@voltagegpu.com with this output."
exit 2
