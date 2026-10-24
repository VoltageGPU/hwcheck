#!/usr/bin/env bash
# VoltageGPU hwcheck acceptance test, hwcheck version 1.1 (2026-10-24).
# Ten minutes on a rented multi-GPU machine (made for 8-GPU nodes, bare metal or VM) before a
# multi-day job: NVLink topology, P2P and NCCL bandwidth, a 5-minute load, ECC and Xid errors,
# leftovers of a previous tenant. Read-only: it writes its report files and a temporary
# directory removed at exit; nothing about the machine is sent anywhere. Only if PyTorch is
# missing, it is downloaded from download.pytorch.org into that temporary directory
# (--no-install forbids it). Exit code 0 = PASS, 1 = WARN, 2 = FAIL, 64 = bad option.
# Every threshold below is explained in README.md.
#
#   curl -fsSL https://voltagegpu.com/hwcheck.sh | bash -s -- --acceptance
#   bash acceptance.sh --quick        # from a clone, without the 5-minute load
set -u
VERSION="1.1"

# ── Thresholds (README.md, "How each verdict is decided") ──────────────────
# Measured on 8x H100 SXM, 2026-10-24 (examples/h100-8x-acceptance.txt): pairs 387 to 391 GB/s,
# median 389 (86% of the 450 GB/s line rate), NCCL bus bandwidth 478.5 GB/s, 669 to 689 TFLOPS,
# SM clock 1335 to 1380 MHz, hottest GPU 69 C for a median of 63.5, host-to-GPU 53.1 to 55.5 GB/s.
P2P_FAIL=0.25     # a GPU pair under 25% of the NVLink line rate is not really on NVLink
P2P_WARN=0.75     # median pair under 75% of the line rate (measured: 86% on 8x H100 SXM, 2026-10-24)
P2P_SPREAD=0.90   # one pair under 90% of the median pair (measured: slowest pair at 99% of it)
H2D_SPREAD=0.70   # one GPU's host-to-GPU copies under 70% of the median GPU
TFLOPS_FAIL=0.50  # one GPU under half the median matmul throughput
TFLOPS_WARN=0.85  # ... or under 85% of it
CLOCK_WARN=0.85   # one GPU's median SM clock under load under 85% of the median GPU
TEMP_SPREAD=10    # one GPU at least 10 C hotter than the median GPU
HWSLOW_FAIL=0.05  # hardware thermal slowdown or power brake in 5% of the samples or more
BUSY_MIN=90       # a GPU under 90% median utilisation did not really run the load
MEM_BUSY=2048     # MiB already used on a GPU before the test: someone else's process
XID_HW=" 48 62 64 74 79 92 94 95 119 120 140 "   # hardware Xids (NVIDIA Xid catalog)

# NVLink by model (NVIDIA datasheets): links per GPU, line rate per direction in GB/s (half
# the advertised two-way figure), NVSwitch chips on the 8-GPU board, and the NCCL all-reduce
# bus bandwidth on 8 GPUs: PASS from P_BUS, FAIL under P_BUS_FAIL (README.md: sources).
profile() {
  P_LABEL=""; P_SXM=0; P_LINKS=""; P_LINE=""; P_SWITCH=""; P_BUS=""; P_BUS_FAIL=""; P_BUS_REF=""
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    *pcie*|*nvl*|*gh200*|*gb200*|*gb300*) ;;
    *h100*hbm3*|*h100*sxm*) P_LABEL="H100 SXM"; P_SXM=1; P_LINKS=18; P_LINE=450; P_SWITCH=4
      P_BUS=340; P_BUS_FAIL=185; P_BUS_REF="478.5 GB/s measured on 8x H100 SXM (2026-10-24), about 370 in ring without NVLink SHARP" ;;
    *h200*) P_LABEL="H200 SXM"; P_SXM=1; P_LINKS=18; P_LINE=450; P_SWITCH=4
      P_BUS=340; P_BUS_FAIL=185; P_BUS_REF="about 370 GB/s in ring, about 480 with NVLink SHARP" ;;
    *a100*sxm*) P_LABEL="A100 SXM"; P_SXM=1; P_LINKS=12; P_LINE=300; P_SWITCH=6
      P_BUS=210; P_BUS_FAIL=115; P_BUS_REF="about 230 GB/s" ;;
    *b200*) P_LABEL="B200 SXM"; P_SXM=1; P_LINKS=18; P_LINE=900 ;;
    *b300*) P_LABEL="B300 SXM"; P_SXM=1; P_LINKS=18; P_LINE=900 ;;
  esac
}

QUICK=0; BURN=300; OUT="."; BARE=0; INSTALL=1; WANT=""
usage() {
  cat <<'EOF'
usage: acceptance.sh [--quick] [--burn-seconds N] [--gpus N] [--bare-metal] [--no-install] [--out DIR]
  --quick            skip the 5-minute load test (about 2 minutes instead of 10)
  --burn-seconds N   length of the load test in seconds, 10 to 3600 (default 300)
  --gpus N           GPU count of the listing: FAIL if fewer are visible
  --bare-metal       the listing says bare metal: FAIL if a hypervisor is detected
  --no-install       never download PyTorch; the tests that need it are skipped without it
  --out DIR          where to write the report files (default: the current directory)
EOF
}
int_in() { case "$1" in ''|*[!0-9]*) return 1 ;; esac; [ "$1" -ge "$2" ] && [ "$1" -le "$3" ]; }
value() { [ $# -ge 2 ] || { echo "$1 needs a value" >&2; usage >&2; exit 64; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --burn-seconds) value "$@"; BURN=$2; shift 2 ;;
    --gpus) value "$@"; WANT=$2; shift 2 ;;
    --bare-metal) BARE=1; shift ;;
    --no-install) INSTALL=0; shift ;;
    --out) value "$@"; OUT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 64 ;;
  esac
done
int_in "$BURN" 10 3600 || { echo "--burn-seconds: 10 to 3600" >&2; exit 64; }
[ -z "$WANT" ] || int_in "$WANT" 1 64 || { echo "--gpus: 1 to 64" >&2; exit 64; }
{ [ -d "$OUT" ] && [ -w "$OUT" ]; } || { echo "--out: $OUT is not a writable directory" >&2; exit 64; }

FS="${HWCHECK_FS_ROOT:-}"   # tests/run.sh only: look for leftovers under this directory, not /
STAMP=$(date -u +%Y%m%dT%H%M%SZ); STARTED=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
T0=$(awk '{ print $1 }' /proc/uptime 2>/dev/null); T0=${T0:-0}
BOOT=$(awk '/^btime/ { print $2 }' /proc/stat 2>/dev/null); BOOT=${BOOT:-0}
HOST=$(hostname 2>/dev/null || echo host)
TMPD=$(mktemp -d "${TMPDIR:-/tmp}/hwcheck-acceptance.XXXXXX") || { echo "cannot create a temporary directory" >&2; exit 70; }
BG=""
# shellcheck disable=SC2329  # called by the EXIT trap below
cleanup() { local p; for p in $BG; do kill "$p" 2>/dev/null; done; rm -rf "$TMPD"; }
trap cleanup EXIT
trap 'exit 130' INT TERM
NOTE=""
if [ -n "${CUDA_VISIBLE_DEVICES+x}" ]; then
  NOTE="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES was ignored: the test covers every GPU"; unset CUDA_VISIBLE_DEVICES
fi

# ── Helpers ────────────────────────────────────────────────────────────────
say() { printf '. %s\n' "$*" >&2; }
N=0
declare -A DET
# add SECTION NAME STATUS SUMMARY [JSON] [COUNTS]: one verdict line. STATUS is PASS, WARN or
# FAIL; N/A (does not apply here), INFO (measured, not judged) or SKIP (could not run; it makes
# the result WARN, except with COUNTS=0 when the user chose to skip it).
add() {
  C_SEC[N]=$1; C_NAME[N]=$2; C_ST[N]=$3; C_SUM[N]=$4; C_DATA[N]=${5:-null}; C_CNT[N]=${6:-1}
  N=$((N + 1)); say "$3  $2: $4"
}
detail() { DET[$1]="${DET[$1]:-}$2"$'\n'; }
# verdicts SECTION < lines "V|name|status|summary|json" and "D|detail" printed by an awk program
verdicts() {
  local tag a b c d
  while IFS='|' read -r tag a b c d; do
    case "$tag" in V) add "$1" "$a" "$b" "$c" "${d:-null}" ;; D) detail "$1" "$a" ;; esac
  done
}
worse() { case "$1 $2" in *FAIL*) echo FAIL ;; *WARN*) echo WARN ;; *) echo "$1" ;; esac; }
fge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 >= b + 0) }'; }
fmul() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.0f", a * b }'; }
isnum() { case "$1" in ''|*[!0-9.]*) return 1 ;; *) return 0 ;; esac; }
jnum() { if isnum "$1"; then printf '%s' "$1"; else printf 'null'; fi; }
kv() { sed -n "s/^$1=//p" "$2" 2>/dev/null | head -n 1; }
# hint FILE: last line of an error output; CUDA error 802 on an NVSwitch board means no fabric manager
hint() {
  grep -v '^[[:space:]]*$' "$1" 2>/dev/null | tail -n 1 | cut -c1-200 | tr -cd '[:print:]'
  if grep -q 'system not yet initialized' "$1" 2>/dev/null; then printf ' (is nvidia-fabricmanager running?)'; fi
}
# JSON string: ASCII only (other bytes become ?), quotes, backslashes and control characters escaped
jstr() {
  local s
  s=$(printf '%s' "$1" | LC_ALL=C tr -c '\011\012\015\040-\176' '?')
  s=${s//\\/\\\\}; s=${s//\"/\\\"}; s=${s//$'\t'/\\t}; s=${s//$'\r'/\\r}; s=${s//$'\n'/\\n}
  printf '"%s"' "$s"
}

# ── nvidia-smi: only the fields this driver knows (an unknown one fails the whole query) ──
SUPPORTED=""
# fields KEY=name[|alias]...: F_KEYS = our keys, F_NAMES = the driver's names, same order
fields() {
  local spec a; F_KEYS=""; F_NAMES=""
  for spec in "$@"; do
    for a in $(printf '%s' "${spec#*=}" | tr '|' ' '); do
      case "$SUPPORTED" in ""|*" $a "*) F_KEYS="${F_KEYS:+$F_KEYS,}${spec%%=*}"; F_NAMES="${F_NAMES:+$F_NAMES,}$a"; break ;; esac
    done
  done
}
smi() { nvidia-smi --query-gpu="$F_NAMES" --format=csv,noheader,nounits 2>/dev/null | sed 's/ *, */,/g'; }
snapshot() { local f=$1; shift; fields "$@"; { printf '%s\n' "$F_KEYS"; smi; } > "$f"; }
# col FILE KEY: that column, one value per GPU (nothing when the driver lacks the field)
col() { awk -F, -v k="$2" 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == k) c = i; next } c { print $c }' "$1"; }

INV_FIELDS=(idx=index name=name uuid=uuid serial=serial bus=pci.bus_id vbios=vbios_version
  mem=memory.total used=memory.used driver=driver_version ecc=ecc.mode.current
  gen=pcie.link.gen.current genmax=pcie.link.gen.max width=pcie.link.width.current
  widthmax=pcie.link.width.max plim=power.limit pdef=power.default_limit
  fstate=fabric.state fstatus=fabric.status mig=mig.mode.current)
ECC_FIELDS=(idx=index mode=ecc.mode.current cv=ecc.errors.corrected.volatile.total
  uv=ecc.errors.uncorrected.volatile.total ca=ecc.errors.corrected.aggregate.total
  ua=ecc.errors.uncorrected.aggregate.total 'sbe=retired_pages.single_bit_ecc.count|retired_pages.sbe'
  'dbe=retired_pages.double_bit.count|retired_pages.dbe' rpend=retired_pages.pending
  rrc=remapped_rows.correctable rru=remapped_rows.uncorrectable rrp=remapped_rows.pending
  rrf=remapped_rows.failure)
SAMPLE_FIELDS=(idx=index temp=temperature.gpu tmem=temperature.memory
  'sm=clocks.current.sm|clocks.sm' smmax=clocks.max.sm 'power=power.draw|power.draw.average'
  plim=power.limit util=utilization.gpu gen=pcie.link.gen.current width=pcie.link.width.current
  'pwrcap=clocks_event_reasons.sw_power_cap|clocks_throttle_reasons.sw_power_cap'
  'hwslow=clocks_event_reasons.hw_slowdown|clocks_throttle_reasons.hw_slowdown'
  'hwtherm=clocks_event_reasons.hw_thermal_slowdown|clocks_throttle_reasons.hw_thermal_slowdown'
  'brake=clocks_event_reasons.hw_power_brake_slowdown|clocks_throttle_reasons.hw_power_brake_slowdown'
  'swtherm=clocks_event_reasons.sw_thermal_slowdown|clocks_throttle_reasons.sw_thermal_slowdown')

HAVE_GPU=0; NGPU=0; FIRST=""; DRIVER=""; CUDA=""; MINMEM=0; NVLINK=na; VIRT=""; VIRT_SRC=""
RAW_TOPO=""; RAW_LSPCI=""; H2D=""; RUNPY=(); PY_WHY=""; TORCH=""; TORCH_CUDA=""; NCCL=""
DEVICES=0; TORCH_SRC=""; KLOG_SRC=""; HYG=""; SAMPLER=""
profile ""

inventory() {
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then return 0; fi
  SUPPORTED=" $(nvidia-smi --help-query-gpu 2>/dev/null | grep -oE '"[a-z0-9_.]+"' | tr -d '"' | tr '\n' ' ')"
  [ "$SUPPORTED" = " " ] && SUPPORTED=""
  snapshot "$TMPD/inv.csv" "${INV_FIELDS[@]}"
  NGPU=$(( $(wc -l < "$TMPD/inv.csv") - 1 ))
  [ "$NGPU" -ge 1 ] || { NGPU=0; return 0; }
  HAVE_GPU=1
  FIRST=$(col "$TMPD/inv.csv" name | head -n 1)
  DRIVER=$(col "$TMPD/inv.csv" driver | head -n 1)
  CUDA=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA (UMD )?Version: *[0-9.]+' | grep -oE '[0-9.]+$' | head -n 1)
  MINMEM=$(col "$TMPD/inv.csv" mem | sort -n | head -n 1); isnum "$MINMEM" || MINMEM=0
  profile "$FIRST"
}

detect_virt() {
  VIRT=$(systemd-detect-virt 2>/dev/null); VIRT_SRC="systemd-detect-virt"
  [ -n "$VIRT" ] && return
  VIRT_SRC="guessed from DMI"
  if [ -f /.dockerenv ] || [ -f /run/.containerenv ]; then VIRT=container; return; fi
  case "$(cat /sys/class/dmi/id/sys_vendor /sys/class/dmi/id/product_name 2>/dev/null)" in
    "") VIRT=unknown ;;
    *QEMU*|*KVM*|*VMware*|*VirtualBox*|*Xen*|*"Virtual Machine"*) VIRT=vm ;;
    *) VIRT=none ;;
  esac
}

# ── 1. Topology ────────────────────────────────────────────────────────────
gpus_line() {
  local onbus
  if [ "$HAVE_GPU" = 0 ]; then
    onbus=$(lspci -nn 2>/dev/null | grep -i '\[10de:' | grep -cE '\[030[02]\]')   # NVIDIA display or 3D controllers
    if [ "${onbus:-0}" -gt 0 ]; then
      add 1 "GPUs" FAIL "$onbus NVIDIA GPUs on the PCI bus but no working driver: install the NVIDIA driver (and nvidia-fabricmanager on NVSwitch boards), then run again"
    else add 1 "GPUs" FAIL "no NVIDIA GPU visible (nvidia-smi missing or failing)"; fi
    return
  fi
  local st=PASS s="${NGPU}x $FIRST"
  if [ "$(col "$TMPD/inv.csv" name | sort -u | grep -c .)" -gt 1 ]; then st=WARN; s="$s, mixed models"; fi
  if [ -n "$WANT" ] && [ "$NGPU" -lt "$WANT" ]; then st=FAIL; s="$s, the listing says $WANT"; fi
  if col "$TMPD/inv.csv" mig | grep -q '^Enabled'; then st=$(worse "$st" WARN); s="$s, MIG enabled (the GPUs are split)"; fi
  add 1 "GPUs" "$st" "$s"
}

topology() {
  if [ "$HAVE_GPU" = 0 ]; then
    add 1 "NVLink mesh" N/A "no GPU"; add 1 "NVSwitch" N/A "no GPU"; add 1 "Fabric state" N/A "no GPU"; return
  fi
  RAW_TOPO=$(nvidia-smi topo -m 2>&1 | tr -d '\033' | sed 's/\[[0-9;]*m//g')
  # "row column cell" for every GPU pair of the matrix (NV18, SYS, PIX...)
  printf '%s\n' "$RAW_TOPO" | awk -F'\t' '
    !h && /GPU0/ { for (i = 1; i <= NF; i++) { x = $i; gsub(/ /, "", x); if (x ~ /^GPU[0-9]+$/) c[i] = substr(x, 4) } h = 1; next }
    h && $1 ~ /^GPU[0-9]+$/ { r = substr($1, 4); for (i in c) if (c[i] != r) { x = $i; gsub(/ /, "", x); print r, c[i], x } }' > "$TMPD/topo.pairs"
  detail 1 "nvidia-smi topo -m:"
  detail 1 "$(printf '%s\n' "$RAW_TOPO" | awk 'NF == 0 { exit } { print "    " $0 }' | expand -t 8)"
  local pairs nv lo hi others s
  pairs=$(grep -c . "$TMPD/topo.pairs")
  read -r nv lo hi < <(awk '$3 ~ /^NV[0-9]+$/ { n++; v = substr($3, 3) + 0; if (lo == "" || v < lo) lo = v; if (v > hi) hi = v }
    END { print n + 0, lo + 0, hi + 0 }' "$TMPD/topo.pairs")
  others=$(awk '$3 !~ /^NV[0-9]+$/ { print $3 }' "$TMPD/topo.pairs" | sort | uniq -c | awk '{ printf "%s%s x%d", s, $2, $1; s = ", " }')
  if [ "$NGPU" -lt 2 ]; then NVLINK=none; add 1 "NVLink mesh" N/A "single GPU"
  elif [ "$pairs" = 0 ]; then add 1 "NVLink mesh" SKIP "could not read nvidia-smi topo -m"
  elif [ "$nv" = "$pairs" ]; then
    NVLINK=full
    if [ "$lo" = "$hi" ]; then s="$pairs/$pairs GPU pairs on NV$lo"; else s="$pairs/$pairs GPU pairs on NVLink, NV$lo to NV$hi"; fi
    if [ -n "$P_LINKS" ] && [ "$lo" -lt "$P_LINKS" ]; then add 1 "NVLink mesh" WARN "$s, NV$P_LINKS expected on $P_LABEL: links down"
    elif [ -z "$P_LINKS" ] && [ "$lo" != "$hi" ]; then add 1 "NVLink mesh" WARN "$s: uneven, links down?"
    else add 1 "NVLink mesh" PASS "$s${P_LINKS:+ (NV$P_LINKS expected on $P_LABEL)}"; fi
  elif [ "$nv" -gt 0 ]; then
    NVLINK=partial; s="$nv/$pairs GPU pairs on NVLink, the others: $others"
    if [ "$P_SXM" = 1 ]; then add 1 "NVLink mesh" FAIL "$s; every pair is linked on 8-GPU $P_LABEL boards"
    else add 1 "NVLink mesh" N/A "$s (bridged pairs: no full mesh on this model)"; fi
  else
    NVLINK=none
    if [ "$P_SXM" = 1 ]; then add 1 "NVLink mesh" FAIL "no NVLink between the GPUs ($others), NV$P_LINKS expected on $P_LABEL"
    else add 1 "NVLink mesh" N/A "no NVLink: the GPUs talk over PCIe ($others)"; fi
  fi
  nvswitch
  fabric
}

nvswitch() {
  local nsw
  if ! command -v lspci >/dev/null 2>&1; then add 1 "NVSwitch" SKIP "lspci not installed (package pciutils)" null 0; return; fi
  RAW_LSPCI=$(lspci -nn 2>/dev/null | grep -i '\[10de:')
  nsw=$(printf '%s\n' "$RAW_LSPCI" | grep -c '\[0680\]')   # NVSwitch: NVIDIA device of class 0680
  detail 1 "lspci -nn, NVIDIA devices:"
  detail 1 "$(printf '%s\n' "${RAW_LSPCI:-(none)}" | sed 's/^/    /')"
  if [ -n "$P_SWITCH" ] && [ "$nsw" = "$P_SWITCH" ]; then add 1 "NVSwitch" PASS "$nsw NVSwitch chips visible ($P_SWITCH on an 8-GPU $P_LABEL board)"
  elif [ -n "$P_SWITCH" ] && [ "$nsw" = 0 ] && [ "$VIRT" != none ]; then
    add 1 "NVSwitch" INFO "none visible from this VM (normal when the host keeps them; the NVLink lines judge the fabric)"
  elif [ -n "$P_SWITCH" ] && [ "$NGPU" -ge 8 ]; then add 1 "NVSwitch" WARN "$nsw NVSwitch chips visible, $P_SWITCH on an 8-GPU $P_LABEL board"
  elif [ "$nsw" -gt 0 ]; then add 1 "NVSwitch" INFO "$nsw NVSwitch chips visible"
  else add 1 "NVSwitch" N/A "none visible (none expected without an 8-GPU NVLink board)"; fi
}

fabric() {
  verdicts 1 < <(paste -d'|' <(col "$TMPD/inv.csv" idx) <(col "$TMPD/inv.csv" fstate) <(col "$TMPD/inv.csv" fstatus) | awk -F'|' -v n="$NGPU" '
    $2 == "" || $2 ~ /^\[?N\/A\]?$/ || tolower($2) == "not supported" { next }
    { k++; if ($2 != "Completed" || ($3 != "Success" && $3 !~ /^\[?N\/A\]?$/ && $3 != "")) bad = bad (bad ? "; " : "") "GPU " $1 ": " $2 " / " $3 }
    END {
      if (!k) print "V|Fabric state|N/A|not reported (no NVLink fabric on this GPU, or an older driver)"
      else if (bad) print "V|Fabric state|WARN|" bad
      else print "V|Fabric state|PASS|Completed / Success on " k "/" n " GPUs"
    }')
}

# PCIe link of each GPU: the best generation and width seen (GPUs drop to Gen1 when idle, so
# the samples taken under load count), and the host-to-GPU copies measured by the P2P test.
pcie() {
  if [ "$HAVE_GPU" = 0 ]; then add 1 "PCIe links" N/A "no GPU"; return; fi
  verdicts 1 < <(awk -F, -v h2d="$H2D" -v spread="$H2D_SPREAD" '
    function c(k) { return (k in col) ? $(col[k]) : "" }
    function median(a, n,   b, i, j, t) {
      for (i = 1; i <= n; i++) b[i] = a[i]
      for (i = 2; i <= n; i++) { t = b[i]; for (j = i - 1; j >= 1 && b[j] > t; j--) b[j + 1] = b[j]; b[j + 1] = t }
      return n == 0 ? 0 : (n % 2 ? b[(n + 1) / 2] : (b[n / 2] + b[n / 2 + 1]) / 2)
    }
    FNR == 1 { delete col; for (i = 1; i <= NF; i++) col[$i] = i; next }
    FILENAME == ARGV[1] { g = c("idx"); ids[++n] = g; gm[g] = c("genmax"); wm[g] = c("widthmax"); go[g] = c("gen") + 0; wo[g] = c("width") + 0; next }
    { g = c("idx"); if (c("util") + 0 >= 50) busy[g] = 1; if (c("gen") + 0 > go[g]) go[g] = c("gen") + 0; if (c("width") + 0 > wo[g]) wo[g] = c("width") + 0 }
    END {
      m = split(h2d, hv, ","); mh = median(hv, m)
      for (k = 1; k <= m; k++) { if (k == 1 || hv[k] + 0 < hlo) hlo = hv[k] + 0; if (hv[k] + 0 > hhi) hhi = hv[k] + 0 }
      print "D|PCIe links, best generation and width seen:"
      for (k = 1; k <= n; k++) {
        g = ids[k]
        if (gm[g] !~ /^[0-9]+$/ || wm[g] !~ /^[0-9]+$/) { unk++; continue }
        if (busy[g]) loaded++
        if (wo[g] < wm[g]) { if (busy[g]) f = f (f ? "; " : "") "GPU " g " at x" wo[g] " (x" wm[g] " possible)"; else w = w (w ? "; " : "") "GPU " g " at x" wo[g] " at idle (x" wm[g] " possible)" }
        if (go[g] < gm[g] && busy[g]) w = w (w ? "; " : "") "GPU " g " at Gen" go[g] " under load (Gen" gm[g] " possible)"
        if (go[g] > best) best = go[g]
        if (wo[g] > bw) bw = wo[g]
        print "D|    GPU " g ": Gen" go[g] " x" wo[g] " seen (Gen" gm[g] " x" wm[g] " possible)" (busy[g] ? ", under load" : ", idle only") (k <= m ? ", host-to-GPU copies " hv[k] " GB/s" : "")
      }
      for (k = 1; k <= m; k++) if (hv[k] + 0 < spread * mh) w = w (w ? "; " : "") "GPU " ids[k] " host-to-GPU copies at " hv[k] " GB/s, median GPU " mh
      if (unk == n) { print "V|PCIe links|SKIP|link generation and width not reported by the driver"; exit }
      s = "Gen" best " x" bw " on " n - unk "/" n " GPUs" (loaded ? " under load" : " (seen at idle only: GPUs lower the generation when idle)")
      if (m) s = s ", host-to-GPU copies " (hlo == hhi ? hlo : hlo " to " hhi) " GB/s"
      if (f) print "V|PCIe links|FAIL|" f (w ? "; also " w : ""); else if (w) print "V|PCIe links|WARN|" w; else print "V|PCIe links|PASS|" s
    }' "$TMPD/inv.csv" "$TMPD/samples.csv")
}

# ── 2-4. PyTorch tests: P2P copies, NCCL all-reduce, 5-minute load ────────────
write_python() {
  cat > "$TMPD/probe.py" <<'PY'
import sys
try:
    import torch
except Exception as e:
    print("ERROR=%s" % type(e).__name__)
    sys.exit(1)
print("TORCH=%s" % torch.__version__)
print("TORCH_CUDA=%s" % (torch.version.cuda or "none"))
try:
    print("DEVICES=%d" % (torch.cuda.device_count() if torch.cuda.is_available() else 0))
except Exception:
    print("DEVICES=0")
try:
    v = torch.cuda.nccl.version()
    print("NCCL=%s" % (".".join(str(x) for x in v) if isinstance(v, tuple) else v))
except Exception:
    print("NCCL=none")
PY
  cat > "$TMPD/p2p.py" <<'PY'
# Copies 256 MiB from every GPU to every other one (copy engines, as cudaMemcpyPeer), checks the
# bytes, and times host-to-GPU copies from pinned memory. GB/s = 10^9 bytes per second.
import json, statistics, time
import torch
n = torch.cuda.device_count()
SIZE, ITERS = 256 << 20, 20
src = [torch.randint(0, 256, (SIZE,), dtype=torch.uint8, device=i) for i in range(n)]
dst = [torch.empty(SIZE, dtype=torch.uint8, device=i) for i in range(n)]
def sync(*ds):
    for d in ds:
        torch.cuda.synchronize(d)
def timed(copy, ds, iters):
    for _ in range(3):
        copy()
    sync(*ds)
    t = time.perf_counter()
    for _ in range(iters):
        copy()
    sync(*ds)
    return SIZE * iters / (time.perf_counter() - t) / 1e9
h2d = []
try:
    host = torch.empty(SIZE, dtype=torch.uint8).pin_memory()
    for i in range(n):
        h2d.append(timed(lambda: dst[i].copy_(host, non_blocking=True), [i], 10))
except Exception:
    h2d = []
print("H2D=%s" % ",".join("%.1f" % x for x in h2d))
mat = [[None] * n for _ in range(n)]
peer = [[i == j or torch.cuda.can_device_access_peer(i, j) for j in range(n)] for i in range(n)]
bad = []
for i in range(n):
    for j in range(n):
        if i != j:
            mat[i][j] = timed(lambda: dst[j].copy_(src[i], non_blocking=True), [i, j], ITERS)
            if not torch.equal(dst[j], src[i].to(j)):
                bad.append("%d>%d" % (i, j))
pairs = sorted((mat[i][j], i, j) for i in range(n) for j in range(n) if i != j)
if pairs:
    print("P2P_MIN=%.0f" % pairs[0][0])
    print("P2P_MIN_PAIR=%d>%d" % (pairs[0][1], pairs[0][2]))
    print("P2P_MEDIAN=%.0f" % statistics.median(p[0] for p in pairs))
    print("P2P_NOPEER=%s" % ",".join("%d>%d" % (i, j) for i in range(n) for j in range(n) if not peer[i][j]))
    print("P2P_BAD=%s" % ",".join(bad))
    for i in range(n):
        print("ROW=%d %s" % (i, " ".join("-" if i == j else "%.0f" % mat[i][j] for j in range(n))))
print("JSON=" + json.dumps({"gb_per_s": [[None if v is None else round(v, 1) for v in r] for r in mat],
                            "peer_access": peer, "host_to_gpu_gb_per_s": [round(x, 1) for x in h2d],
                            "bytes": SIZE, "iterations": ITERS, "corrupted_pairs": bad}))
PY
  cat > "$TMPD/nccl.py" <<'PY'
# All-reduce of one float32 buffer on every GPU, one process per GPU; bus bandwidth as in
# nccl-tests: busbw = size / time * 2 * (n - 1) / n. The last all-reduce checks the values.
import sys, time
import torch, torch.distributed as dist, torch.multiprocessing as mp
def worker(rank, world, nbytes, iters, init, out):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=init, rank=rank, world_size=world)
    x = torch.ones(nbytes // 4, dtype=torch.float32, device="cuda")
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = torch.tensor([(time.perf_counter() - t) / iters], dtype=torch.float64, device="cuda")
    dist.all_reduce(dt, op=dist.ReduceOp.MAX)
    x.fill_(1.0)
    dist.all_reduce(x)
    wrong = torch.tensor([int((x != world).sum().item())], dtype=torch.int64, device="cuda")
    dist.all_reduce(wrong)
    if rank == 0:
        algbw = nbytes / dt.item() / 1e9
        with open(out, "w") as f:
            f.write("ALGBW=%.1f\nBUSBW=%.1f\nTIME_US=%.0f\nWRONG=%d\n" % (algbw, algbw * 2 * (world - 1) / world, dt.item() * 1e6, wrong.item()))
    dist.destroy_process_group()
if __name__ == "__main__":
    world = torch.cuda.device_count()
    nbytes, iters, init, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    mp.spawn(worker, args=(world, nbytes, iters, init, out), nprocs=world, join=True)
    print(open(out).read() + "RANKS=%d" % world)
PY
  cat > "$TMPD/burn.py" <<'PY'
# Matrix multiplications on one GPU for N seconds. The same product is computed again and
# again: a healthy GPU returns the same bits every time (MISMATCH counts the times it did not).
import sys, time
import torch
dur = float(sys.argv[1])
dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
N = 8192
g = torch.Generator().manual_seed(20261024)
a = torch.randn(N, N, generator=g).to("cuda", dt)
b = torch.randn(N, N, generator=g).to("cuda", dt)
c = torch.empty(N, N, device="cuda", dtype=dt)
torch.matmul(a, b, out=c)
torch.cuda.synchronize()
ref = c.clone()
iters = mismatch = 0
t0 = time.perf_counter()
while time.perf_counter() - t0 < dur:
    for _ in range(25):
        torch.matmul(a, b, out=c)
    iters += 25
    if not torch.equal(c, ref):
        mismatch += 1
elapsed = time.perf_counter() - t0
print("TFLOPS=%.1f" % (2 * N ** 3 * iters / elapsed / 1e12))
print("MISMATCH=%d" % mismatch)
print("FINITE=%d" % int(bool(torch.isfinite(ref).all())))
print("CHECKSUM=%d" % int(ref.view(torch.int16).to(torch.int64).sum().item()))
print("DTYPE=%s" % str(dt).replace("torch.", ""))
PY
}

# probe CMD...: true when PyTorch imports and sees at least one GPU
probe() {
  "$@" "$TMPD/probe.py" > "$TMPD/probe.out" 2>/dev/null || return 1
  TORCH=$(kv TORCH "$TMPD/probe.out"); TORCH_CUDA=$(kv TORCH_CUDA "$TMPD/probe.out")
  NCCL=$(kv NCCL "$TMPD/probe.out"); DEVICES=$(kv DEVICES "$TMPD/probe.out")
  int_in "${DEVICES:-0}" 1 1024
}

# PyTorch of the image if it sees the GPUs; else a pinned build in $TMPD, chosen by the CUDA
# version of the driver (pip ignores any pip.conf and keeps no cache: nothing stays behind).
setup_python() {
  local base idx tv free pip_env
  base=(env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_CACHE_PATH="$TMPD/cache/nv" XDG_CACHE_HOME="$TMPD/cache"
    TRITON_CACHE_DIR="$TMPD/cache/triton" PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8)
  write_python
  if command -v python3 >/dev/null 2>&1 && probe "${base[@]}" python3; then
    RUNPY=("${base[@]}" python3); TORCH_SRC="from the image"; return
  fi
  if [ "$INSTALL" = 0 ]; then PY_WHY="no PyTorch that sees the GPUs (--no-install)"; return; fi
  if ! command -v python3 >/dev/null 2>&1; then PY_WHY="python3 is not installed"; return; fi
  case "${CUDA%%.*}" in
    1[3-9]|[2-9][0-9]) idx=cu130; tv=2.14.1 ;;
    12) case "${CUDA#12.}" in 8*|9*) idx=cu128; tv=2.11.0 ;; *) idx=cu126; tv=2.14.1 ;; esac ;;
    11) idx=cu118; tv=2.7.1 ;;
    *) PY_WHY="no PyTorch build for this driver (CUDA ${CUDA:-unknown})"; return ;;
  esac
  free=$(df -Pk "$TMPD" | awk 'NR == 2 { print int($4 / 1048576) }')
  if [ "${free:-0}" -lt 12 ]; then PY_WHY="PyTorch missing and only ${free:-0} GB free for it in ${TMPD%/*} (12 GB needed)"; return; fi
  say "PyTorch not found: installing torch $tv ($idx) from download.pytorch.org into $TMPD (about 3 GB, deleted at the end)"
  pip_env=(env PIP_CONFIG_FILE=/dev/null PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 TMPDIR="$TMPD")
  if python3 -m venv "$TMPD/venv" >/dev/null 2>&1; then
    if "${pip_env[@]}" "$TMPD/venv/bin/python" -m pip install -q "torch==$tv" --index-url "https://download.pytorch.org/whl/$idx" > "$TMPD/pip.log" 2>&1 &&
      probe "${base[@]}" "$TMPD/venv/bin/python"; then
      RUNPY=("${base[@]}" "$TMPD/venv/bin/python"); TORCH_SRC="installed for this run, $idx"; return
    fi
  elif python3 -m pip --version >/dev/null 2>&1; then
    rm -rf "$TMPD/venv"
    if "${pip_env[@]}" python3 -m pip install -q --target "$TMPD/site" "torch==$tv" --index-url "https://download.pytorch.org/whl/$idx" > "$TMPD/pip.log" 2>&1 &&
      probe "${base[@]}" PYTHONPATH="$TMPD/site" python3; then
      RUNPY=("${base[@]}" PYTHONPATH="$TMPD/site" python3); TORCH_SRC="installed for this run, $idx"; return
    fi
  else
    PY_WHY="PyTorch missing and neither python3-venv nor pip to install it"; return
  fi
  PY_WHY="PyTorch install or GPU access failed ($(hint "$TMPD/pip.log"))"
}

# nvidia-smi every 5 s during the load (every second around the shorter tests), with the phase
sampler() {
  local ph
  while [ -f "$TMPD/sampling" ]; do
    ph=$(cat "$TMPD/phase" 2>/dev/null)
    smi | sed "s/^/$ph,$(date +%s),/" >> "$TMPD/samples.csv"
    if [ "$ph" = burn ]; then sleep 5; else sleep 1; fi
  done
}
start_sampler() {
  fields "${SAMPLE_FIELDS[@]}"
  printf 'phase,t,%s\n' "$F_KEYS" > "$TMPD/samples.csv"
  echo idle > "$TMPD/phase"; : > "$TMPD/sampling"
  sampler & SAMPLER=$!; BG="$BG $SAMPLER"
}
stop_sampler() { rm -f "$TMPD/sampling"; kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null; }

gpu_tests() {
  if [ "$HAVE_GPU" = 0 ]; then
    add 2 "P2P copies" N/A "no GPU"; add 3 "NCCL all-reduce" N/A "no GPU"; add 4 "Load test" N/A "no GPU"; return
  fi
  setup_python
  if [ ${#RUNPY[@]} -eq 0 ]; then
    add 2 "P2P copies" SKIP "$PY_WHY"; add 3 "NCCL all-reduce" SKIP "$PY_WHY"
    if [ "$QUICK" = 1 ]; then add 4 "Load test" SKIP "--quick: not run" null 0; else add 4 "Load test" SKIP "$PY_WHY"; fi
    return
  fi
  if [ "$DEVICES" -lt "$NGPU" ]; then add 2 "CUDA init" FAIL "PyTorch initialises $DEVICES of the $NGPU GPUs nvidia-smi lists"
  else add 2 "CUDA init" PASS "$DEVICES/$NGPU GPUs in PyTorch $TORCH ($TORCH_SRC)"; fi
  start_sampler
  p2p_test
  nccl_test
  burn_test
  stop_sampler
}

p2p_test() {
  echo p2p > "$TMPD/phase"
  if ! timeout 900 "${RUNPY[@]}" "$TMPD/p2p.py" > "$TMPD/p2p.out" 2> "$TMPD/p2p.err"; then
    add 2 "P2P copies" FAIL "the copy test failed: $(hint "$TMPD/p2p.err")"; return
  fi
  H2D=$(kv H2D "$TMPD/p2p.out")
  if [ "$NGPU" -lt 2 ]; then add 2 "P2P copies" N/A "single GPU"; return; fi
  local o="$TMPD/p2p.out" min pair med nopeer bad data st=PASS s floor="" low
  min=$(kv P2P_MIN "$o"); pair=$(kv P2P_MIN_PAIR "$o"); med=$(kv P2P_MEDIAN "$o")
  nopeer=$(kv P2P_NOPEER "$o"); bad=$(kv P2P_BAD "$o"); data=$(kv JSON "$o")
  case "$data" in "{"*"}") ;; *) data=null ;; esac
  detail 2 "GPU-to-GPU copies in GB/s (row = source GPU, column = destination GPU):"
  detail 2 "$(sed -n 's/^ROW=//p' "$o" | awk '{ printf "    GPU%-3s", $1; for (i = 2; i <= NF; i++) printf " %6s", $i; print "" }')"
  s="slowest pair $min GB/s (GPU $pair), median $med GB/s"
  if [ "$NVLINK" != full ] && [ "$P_SXM" = 0 ]; then add 2 "P2P copies" N/A "$s (no NVLink mesh: copies over PCIe, for information)" "$data"; return; fi
  [ -n "$bad" ] && { st=FAIL; s="$s; bytes differ after the copy on $bad"; }
  [ -n "$nopeer" ] && [ "$NVLINK" = full ] && { st=FAIL; s="$s; no peer access on $nopeer"; }
  if [ -n "$P_LINE" ]; then
    floor=$(fmul "$P_LINE" "$P2P_FAIL"); low=$(fmul "$P_LINE" "$P2P_WARN")
    if ! fge "$min" "$floor"; then st=FAIL; s="$s; under $floor GB/s (25% of the $P_LINE GB/s line rate of $P_LABEL): not on NVLink"
    elif ! fge "$med" "$low"; then st=$(worse "$st" WARN); s="$s; median under $low GB/s ($(fmul 100 "$P2P_WARN")% of the $P_LABEL line rate)"; fi
  elif ! fge "$min" "$(fmul "$med" "$P2P_FAIL")"; then st=FAIL; s="$s; a pair under 25% of the median"; fi
  if [ "$st" = PASS ] && ! fge "$min" "$(fmul "$med" "$P2P_SPREAD")"; then st=WARN; s="$s; a pair under $(fmul 100 "$P2P_SPREAD")% of the median"; fi
  if [ "$st" = PASS ] && [ -n "$floor" ]; then s="$s (FAIL under $floor, WARN under $low: $P_LABEL line rate $P_LINE GB/s)"; fi
  add 2 "P2P copies" "$st" "$s" "$data"
}

nccl_test() {
  echo nccl > "$TMPD/phase"
  if [ "$NGPU" -lt 2 ]; then add 3 "NCCL all-reduce" N/A "single GPU"; return; fi
  local bytes=1073741824 label="1 GiB" o="$TMPD/nccl.out" bus wrong data s
  if [ "$MINMEM" -ge 40000 ]; then bytes=4294967296; label="4 GiB"; fi
  env | grep -E '^NCCL_' | sort > "$TMPD/nccl.env"
  if [ -s "$TMPD/nccl.env" ]; then detail 3 "NCCL settings in the environment (they change the result): $(paste -sd ' ' "$TMPD/nccl.env")"; fi
  if ! timeout 600 "${RUNPY[@]}" "$TMPD/nccl.py" "$bytes" 20 "file://$TMPD/nccl.init" "$TMPD/nccl.res" > "$o" 2> "$TMPD/nccl.err"; then
    s=$(hint "$TMPD/nccl.err"); grep -qiE 'shm|shared memory' "$TMPD/nccl.err" && s="$s (container /dev/shm too small?)"
    add 3 "NCCL all-reduce" FAIL "the all-reduce on $NGPU GPUs did not complete: $s"; return
  fi
  bus=$(kv BUSBW "$o"); wrong=$(kv WRONG "$o")
  data="{\"bytes\":$bytes,\"ranks\":$(jnum "$(kv RANKS "$o")"),\"algbw_gb_per_s\":$(jnum "$(kv ALGBW "$o")"),\"busbw_gb_per_s\":$(jnum "$bus"),\"time_us\":$(jnum "$(kv TIME_US "$o")"),\"wrong_values\":$(jnum "$wrong")}"
  detail 3 "all-reduce of $label per GPU, 20 iterations: algorithm bandwidth $(kv ALGBW "$o") GB/s, $(kv TIME_US "$o") us per all-reduce"
  s="bus bandwidth $bus GB/s on $NGPU GPUs ($label, NCCL $NCCL)"
  if [ "${wrong:-1}" != 0 ]; then add 3 "NCCL all-reduce" FAIL "$s; ${wrong:-unknown} wrong values in the result" "$data"
  elif [ "$NVLINK" != full ] && [ "$P_SXM" = 0 ]; then add 3 "NCCL all-reduce" N/A "$s (no NVLink mesh: over PCIe, for information)" "$data"
  elif [ -n "$P_BUS" ]; then
    if fge "$bus" "$P_BUS"; then add 3 "NCCL all-reduce" PASS "$s; $P_LABEL: $P_BUS_REF, PASS from $P_BUS" "$data"
    elif fge "$bus" "$P_BUS_FAIL"; then add 3 "NCCL all-reduce" WARN "$s; under $P_BUS ($P_LABEL: $P_BUS_REF)" "$data"
    else add 3 "NCCL all-reduce" FAIL "$s; under $P_BUS_FAIL, half the effective NVLink rate of $P_LABEL" "$data"; fi
  else add 3 "NCCL all-reduce" INFO "$s; no reference value for $FIRST: not judged" "$data"; fi
}

burn_test() {
  if [ "$QUICK" = 1 ]; then add 4 "Load test" SKIP "--quick: not run" null 0; return; fi
  echo burn > "$TMPD/phase"
  say "load test: $BURN s of matrix multiplications on every GPU, nvidia-smi read every 5 s"
  local i p pids=() o
  for i in $(col "$TMPD/inv.csv" idx); do
    CUDA_VISIBLE_DEVICES=$i timeout $((BURN + 180)) "${RUNPY[@]}" "$TMPD/burn.py" "$BURN" > "$TMPD/burn-$i.out" 2> "$TMPD/burn-$i.err" &
    pids+=("$!"); BG="$BG $!"
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo idle > "$TMPD/phase"
  : > "$TMPD/burn.res"
  for i in $(col "$TMPD/inv.csv" idx); do
    o="$TMPD/burn-$i.out"
    printf '%s,%s,%s,%s,%s,%s\n' "$i" "$(kv TFLOPS "$o")" "$(kv MISMATCH "$o")" "$(kv FINITE "$o")" "$(kv CHECKSUM "$o")" "$(kv DTYPE "$o")" >> "$TMPD/burn.res"
  done
  verdicts 4 < <(awk -F, -v tfail="$TFLOPS_FAIL" -v twarn="$TFLOPS_WARN" -v cwarn="$CLOCK_WARN" -v tspread="$TEMP_SPREAD" \
      -v hwfail="$HWSLOW_FAIL" -v busy="$BUSY_MIN" '
    function c(k) { return (k in col) ? $(col[k]) : "" }
    function median(a, n,   b, i, j, t) {
      for (i = 1; i <= n; i++) b[i] = a[i]
      for (i = 2; i <= n; i++) { t = b[i]; for (j = i - 1; j >= 1 && b[j] > t; j--) b[j + 1] = b[j]; b[j + 1] = t }
      return n == 0 ? 0 : (n % 2 ? b[(n + 1) / 2] : (b[n / 2] + b[n / 2 + 1]) / 2)
    }
    function msg(line, kind, m) { if (kind == "F") F[line] = F[line] (F[line] == "" ? "" : "; ") m; else W[line] = W[line] (W[line] == "" ? "" : "; ") m }
    function out(line, name, pass, data) {
      print "V|" name "|" (F[line] != "" ? "FAIL|" F[line] (W[line] != "" ? "; also " W[line] : "") : W[line] != "" ? "WARN|" W[line] : "PASS|" pass) "|" data
    }
    FILENAME == ARGV[1] { ng++; g = $1; ids[ng] = g; tf[g] = $2; mm[g] = $3; fin[g] = $4; ck[g] = $5; if ($6 != "") dtype = $6; next }
    FNR == 1 { delete col; for (i = 1; i <= NF; i++) col[$i] = i; next }
    FILENAME == ARGV[2] { g = c("idx"); nm[g] = c("name"); pl[g] = c("plim"); pd[g] = c("pdef"); next }
    $1 != "burn" { next }
    {
      g = c("idx"); k = ++ns[g]; ut[g, k] = c("util") + 0; sm[g, k] = c("sm") + 0
      t = c("temp") + 0; if (t > tmax[g]) tmax[g] = t
      t = c("tmem"); if (t ~ /^[0-9]+$/ && t + 0 > tmem[g]) tmem[g] = t + 0
      t = c("power") + 0; if (t > pmax[g]) pmax[g] = t
      if (c("pwrcap") == "Active") cap[g]++
      if (c("hwslow") == "Active") hws[g]++
      if (c("hwtherm") == "Active") hwt[g]++
      if (c("brake") == "Active") brk[g]++
      if (c("swtherm") == "Active") swt[g]++
    }
    END {
      for (k = 1; k <= ng; k++) {
        g = ids[k]; n = ns[g] + 0; total += n
        for (j = 1; j <= n; j++) { a1[j] = ut[g, j]; a2[j] = sm[g, j] }
        um[g] = median(a1, n); smm[g] = median(a2, n); tma[k] = tmax[g] + 0; sma[k] = smm[g]
        if (tf[g] ~ /^[0-9.]+$/ && tf[g] + 0 > 0) { ntf++; tfa[ntf] = tf[g] + 0 }
      }
      mtf = median(tfa, ntf); mtm = median(tma, ng); msm = median(sma, ng)
      print "D|    GPU  samples  busy  max C  HBM C  SM MHz   max W  TFLOPS  slowdown samples: power cap / thermal / hw / brake"
      for (k = 1; k <= ng; k++) {
        g = ids[k]; n = ns[g] + 0
        if (tf[g] !~ /^[0-9.]+$/ || tf[g] + 0 <= 0) msg("T", "F", "GPU " g ": the load did not run")
        else {
          if (fin[g] != "1") msg("T", "F", "GPU " g ": non-finite matmul results")
          if (mm[g] + 0 > 0) msg("T", "F", "GPU " g ": the same product changed " mm[g] " times (silent data corruption?)")
          if (tf[g] < tfail * mtf) msg("T", "F", sprintf("GPU %s at %.0f TFLOPS, under half the median GPU (%.0f)", g, tf[g], mtf))
          else if (tf[g] < twarn * mtf) msg("T", "W", sprintf("GPU %s at %.0f TFLOPS, under 85%% of the median GPU (%.0f)", g, tf[g], mtf))
          if (nm[g] in ref) { if (ref[nm[g]] != ck[g]) msg("T", "W", "GPU " g " computed another result than GPU " refg[nm[g]] " (same model, same inputs)") }
          else { ref[nm[g]] = ck[g]; refg[nm[g]] = g }
          if (slow == "" || tf[g] + 0 < slow) { slow = tf[g] + 0; slowg = g }
        }
        lim = n * hwfail; if (lim < 1) lim = 1
        if (n > 0 && hwt[g] + brk[g] >= lim) msg("P", "F", "GPU " g ": hardware thermal slowdown or power brake in " hwt[g] + brk[g] " of " n " samples")
        else if (hwt[g] + brk[g] > 0) msg("P", "W", "GPU " g ": hardware thermal slowdown or power brake in " hwt[g] + brk[g] " sample(s)")
        if (swt[g] > 0) msg("P", "W", "GPU " g ": thermal slowdown in " swt[g] " of " n " samples")
        if (hws[g] > 0 && hwt[g] + brk[g] == 0) msg("P", "W", "GPU " g ": hardware slowdown in " hws[g] " of " n " samples")
        if (pl[g] ~ /^[0-9.]+$/ && pd[g] ~ /^[0-9.]+$/ && pl[g] + 1 < pd[g] + 0) msg("P", "W", sprintf("GPU %s power limit %.0f W, under its default %.0f W", g, pl[g], pd[g]))
        if (ng > 1 && tmax[g] >= mtm + tspread) msg("P", "W", sprintf("GPU %s reached %d C, the median GPU %d C", g, tmax[g], mtm))
        if (n > 0 && um[g] < busy) msg("C", "W", sprintf("GPU %s only %d%% busy (median)", g, um[g]))
        if (n > 0 && smm[g] < cwarn * msm) msg("C", "W", sprintf("GPU %s median SM clock %d MHz, under 85%% of the median GPU (%d MHz)", g, smm[g], msm))
        if (tmax[g] + 0 >= hot) { hot = tmax[g] + 0; hotg = g }
        if (lo == "" || smm[g] < lo) lo = smm[g]
        if (smm[g] > hi) hi = smm[g]
        caps += cap[g]
        print "D|" sprintf("    %-4s %7d %4d%% %6d %6s %7d %7.0f %7s  %d / %d / %d / %d", g, n, um[g], tmax[g], (g in tmem) ? tmem[g] : "-", smm[g], pmax[g], tf[g] == "" ? "-" : tf[g], cap[g], swt[g] + hwt[g], hws[g], brk[g])
        js = js (js == "" ? "" : ",") sprintf("{\"gpu\":%d,\"samples\":%d,\"busy_median\":%d,\"temp_max_c\":%d,\"sm_mhz_median\":%d,\"power_max_w\":%.0f,\"tflops\":%s,\"changed_results\":%d,\"power_cap_samples\":%d,\"thermal_samples\":%d,\"hw_slowdown_samples\":%d,\"power_brake_samples\":%d}", g, n, um[g], tmax[g], smm[g], pmax[g], tf[g] ~ /^[0-9.]+$/ ? tf[g] : "null", mm[g], cap[g], swt[g] + hwt[g], hws[g], brk[g])
      }
      out("T", "Throughput", sprintf("%d/%d GPUs ran it, median %.0f TFLOPS (%s), slowest GPU %s at %.0f", ntf, ng, mtf, dtype, slowg, slow), "[" js "]")
      if (total == 0) { print "V|Thermal / power|SKIP|no nvidia-smi sample during the load"; print "V|Clocks|SKIP|no nvidia-smi sample during the load"; exit }
      out("P", "Thermal / power", sprintf("max %d C (GPU %s), no thermal or hardware slowdown; at the power cap in %d%% of the samples (normal under full load)", hot, hotg, 100 * caps / total))
      out("C", "Clocks", sprintf("median SM clock %d MHz, every GPU between %d and %d MHz, at least %d%% busy", msm, lo, hi, busy))
    }' "$TMPD/burn.res" "$TMPD/inv.csv" "$TMPD/samples.csv")
}

# ── 5. Errors: ECC and retired or remapped memory, Xid in the kernel log ─────
ecc_verdict() {
  if [ "$HAVE_GPU" = 0 ]; then add 5 "ECC / memory" N/A "no GPU"; return; fi
  snapshot "$TMPD/ecc1.csv" "${ECC_FIELDS[@]}"
  verdicts 5 < <(awk -F, '
    function v(k) { return (k in col) ? $(col[k]) : "" }
    function n(x) { return x ~ /^[0-9]+$/ }
    function msg(kind, m) { if (kind == "F") f = f (f == "" ? "" : "; ") m; else w = w (w == "" ? "" : "; ") m }
    FNR == 1 { delete col; for (i = 1; i <= NF; i++) col[$i] = i; next }
    FILENAME == ARGV[1] { cv0[FNR] = v("cv"); next }
    {
      g = v("idx"); gpus++
      if (v("mode") == "Enabled") on++; else if (v("mode") == "Disabled") msg("W", "GPU " g " has ECC disabled")
      if (n(v("uv")) && v("uv") > 0) msg("F", "GPU " g ": " v("uv") " uncorrected ECC errors since the driver loaded")
      if (v("rrp") == "Yes") msg("F", "GPU " g ": row remapping pending (the GPU needs a reset)")
      if (v("rrf") == "Yes") msg("F", "GPU " g ": row remapping failed")
      if (v("rpend") == "Yes") msg("F", "GPU " g ": page retirement pending (the GPU needs a reset)")
      if (n(v("cv")) && n(cv0[FNR]) && v("cv") > cv0[FNR]) msg("W", "GPU " g ": " v("cv") - cv0[FNR] " corrected errors during the test")
      if (n(v("ua")) && v("ua") > 0) msg("W", "GPU " g ": " v("ua") " uncorrected errors in its lifetime")
      if (n(v("dbe")) && v("dbe") > 0) msg("W", "GPU " g ": " v("dbe") " pages retired after double-bit errors")
      if (n(v("rru")) && v("rru") > 0) msg("W", "GPU " g ": " v("rru") " rows remapped after uncorrectable errors")
      if (n(v("cv"))) cv += v("cv")
      js = js (js == "" ? "" : ",") sprintf("{\"gpu\":%d,\"ecc_mode\":\"%s\",\"corrected_volatile\":\"%s\",\"uncorrected_volatile\":\"%s\",\"corrected_aggregate\":\"%s\",\"uncorrected_aggregate\":\"%s\",\"retired_sbe\":\"%s\",\"retired_dbe\":\"%s\",\"retired_pending\":\"%s\",\"remapped_correctable\":\"%s\",\"remapped_uncorrectable\":\"%s\",\"remap_pending\":\"%s\",\"remap_failure\":\"%s\"}", g, v("mode"), v("cv"), v("uv"), v("ca"), v("ua"), v("sbe"), v("dbe"), v("rpend"), v("rrc"), v("rru"), v("rrp"), v("rrf"))
    }
    END {
      if (f != "") s = "FAIL|" f (w != "" ? "; also " w : ""); else if (w != "") s = "WARN|" w
      else if (!on) s = "N/A|no ECC reported by these GPUs"
      else s = "PASS|ECC on " on "/" gpus " GPUs, no uncorrected error, nothing pending, " cv + 0 " corrected since the driver loaded"
      print "V|ECC / memory|" s "|[" js "]"
    }' "$TMPD/ecc0.csv" "$TMPD/ecc1.csv")
}

# klog FILE: kernel log with [seconds since boot] stamps: dmesg, else the journal, else sudo -n
klog() {
  KLOG_SRC=""
  # shellcheck disable=SC2024  # sudo only reads the log; the file stays the user's
  if dmesg > "$1" 2>/dev/null && grep -qE '^\[ *[0-9]+\.[0-9]+\]' "$1"; then KLOG_SRC=dmesg
  elif journalctl -k -b --no-pager -o short-monotonic > "$1" 2>/dev/null && grep -qE '^\[ *[0-9]+\.[0-9]+\]' "$1"; then KLOG_SRC=journalctl
  elif sudo -n dmesg > "$1" 2>/dev/null && grep -qE '^\[ *[0-9]+\.[0-9]+\]' "$1"; then KLOG_SRC="sudo -n dmesg"
  fi
  [ -n "$KLOG_SRC" ]
}

xid_verdict() {
  if [ "$HAVE_GPU" = 0 ]; then add 5 "Kernel log (Xid)" N/A "no GPU"; return; fi
  if ! klog "$TMPD/klog"; then add 5 "Kernel log (Xid)" SKIP "kernel log not readable (container, or not root): Xid errors not checked"; return; fi
  verdicts 5 < <(awk -v t0="$T0" -v hw="$XID_HW" -v src="$KLOG_SRC" '
    /NVRM: Xid/ {
      t = $0; sub(/^\[ */, "", t); sub(/\].*/, "", t)
      x = $0; sub(/.*NVRM: Xid \(/, "", x); pci = x; sub(/\).*/, "", pci); sub(/^[^)]*\): */, "", x); code = x; sub(/[^0-9].*/, "", code)
      item = "Xid " code " on " pci; during = (t + 0 >= t0 + 0)
      if (!seen[item, during]++) {
        if (during && index(hw, " " code " ")) f = f (f ? ", " : "") item
        else if (during) w = w (w ? ", " : "") item
        else if (index(hw, " " code " ")) e = e (e ? ", " : "") item " at " int(t) " s after boot"
        else o++
      }
      if (++lines <= 20) print "D|    [" t "] " item (during ? " (during the test)" : "")
      js = js (js == "" ? "" : ",") sprintf("{\"uptime_s\":%s,\"xid\":%s,\"pci\":\"%s\",\"during_test\":%s}", t + 0, code == "" ? "null" : code, pci, during ? "true" : "false")
    }
    END {
      if (f != "") s = "FAIL|during the test: " f (w != "" ? ", " w : "") (e != "" ? "; earlier in this boot: " e : "")
      else if (w != "") s = "WARN|during the test: " w (e != "" ? "; earlier in this boot: " e : "")
      else if (e != "") s = "WARN|none during the test; earlier in this boot: " e
      else s = "PASS|no Xid during the test" (o ? " (" o " earlier in this boot, not hardware errors)" : "") ", read with " src
      print "V|Kernel log (Xid)|" s "|[" js "]"
    }' "$TMPD/klog")
}

# ── 6. Hygiene: what a previous tenant left, who can log in, virtualization ───
found() {   # KIND PATH: one leftover, with its date (never its content)
  local m
  m=$(date -u -d "@$(stat -c %Y "$2" 2>/dev/null || echo 0)" '+%Y-%m-%d %H:%M UTC' 2>/dev/null)
  HYG="$HYG$1|${2#"$FS"}|$m"$'\n'
}

hygiene() {
  local d f homes=() unread="" n nc js="" kind path m keys=0 kl=""
  for d in "$FS"/root "$FS"/home/*; do [ -d "$d" ] && homes+=("$d"); done
  for d in "${homes[@]}"; do
    if [ ! -r "$d" ] || [ ! -x "$d" ]; then unread="$unread ${d#"$FS"}"; continue; fi
    for f in .netrc .git-credentials .aws/credentials .config/gcloud/application_default_credentials.json \
        .config/gcloud/credentials.db .cache/huggingface/token .huggingface/token .kaggle/kaggle.json .pgpass \
        .config/rclone/rclone.conf .ssh/id_rsa .ssh/id_ecdsa .ssh/id_ed25519; do
      [ -s "$d/$f" ] && found credentials "$d/$f"
    done
    if [ -s "$d/.docker/config.json" ] && grep -q '"auth' "$d/.docker/config.json" 2>/dev/null; then found credentials "$d/.docker/config.json"; fi
    for f in .bash_history .zsh_history .python_history .local/share/fish/fish_history .psql_history .mysql_history \
        .sqlite_history .node_repl_history .ipython/profile_default/history.sqlite; do
      [ -s "$d/$f" ] && found history "$d/$f"
    done
    for f in .cache/huggingface .cache/torch .triton .nv/ComputeCache .cache/wandb wandb; do
      [ -n "$(ls -A "$d/$f" 2>/dev/null)" ] && found cache "$d/$f"
    done
    for f in "$d"/*; do   # a fresh image has only dotfiles in a home directory
      [ -e "$f" ] || continue
      case "${f##*/}" in hwcheck*|acceptance*.sh|snap) ;; *) found file "$f" ;; esac
    done
  done
  if [ -d "$FS/workspace" ]; then
    for f in "$FS"/workspace/* "$FS"/workspace/.[!.]*; do [ -e "$f" ] && found file "$f"; done
  fi
  if [ "$BOOT" -gt 0 ] && touch -d "@$BOOT" "$TMPD/boot.ref" 2>/dev/null; then
    while IFS= read -r f; do found "tmp, older than this boot" "$f"; done < <(find "$FS"/tmp "$FS"/var/tmp -mindepth 1 -maxdepth 1 \
      ! -newer "$TMPD/boot.ref" ! -name 'systemd-private-*' ! -name '.*-unix' ! -name 'snap-private-tmp' ! -name 'hwcheck-acceptance.*' 2>/dev/null)
  fi
  while IFS= read -r f; do found model "$f"; done < <(timeout 60 find "${homes[@]}" "$FS"/tmp "$FS"/var/tmp "$FS"/workspace -maxdepth 4 \
    -type f -size +10M \( -name '*.safetensors' -o -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' -o -name '*.gguf' -o -name '*.onnx' -o -name '*.bin' \) 2>/dev/null | head -n 20)
  n=$(printf '%s' "$HYG" | grep -c .); nc=$(printf '%s' "$HYG" | grep -c '^credentials|')
  while IFS='|' read -r kind path m; do
    [ -n "$kind" ] || continue
    detail 6 "    $kind: $path (modified $m)"
    js="$js${js:+,}{\"kind\":$(jstr "$kind"),\"path\":$(jstr "$path"),\"modified\":$(jstr "$m")}"
  done <<< "$HYG"
  local s="nothing found in the home directories, /tmp, /var/tmp and /workspace" st=PASS
  if [ "$nc" -gt 0 ]; then st=FAIL; s="$nc credential file(s) left on the machine (path and date in the details, content never read)"
  elif [ "$n" -gt 0 ]; then st=WARN; s="$n leftover(s): history, caches or files (details: anything older than your rental is not yours)"; fi
  [ -n "$unread" ] && s="$s; not readable:$unread (run as root to include it)"
  add 6 "Previous tenant" "$st" "$s" "[$js]"

  for d in "${homes[@]}"; do
    for f in "$d"/.ssh/authorized_keys "$d"/.ssh/authorized_keys2; do
      [ -r "$f" ] || continue
      n=$(grep -cvE '^[[:space:]]*(#|$)' "$f"); [ "$n" -gt 0 ] || continue
      keys=$((keys + n)); kl="$kl${kl:+, }${d##*/}: $n"
      detail 6 "    ${f#"$FS"}: $n key(s)"
      if command -v ssh-keygen >/dev/null 2>&1; then detail 6 "$(ssh-keygen -lf "$f" 2>/dev/null | awk '{ print "      " $1 " " $2 " " $NF }')"; fi
    done
  done
  if [ "$keys" -gt 1 ]; then add 6 "SSH keys" WARN "$keys keys can log in ($kl): check that each one is yours"
  elif [ "$keys" = 1 ]; then add 6 "SSH keys" PASS "1 key can log in ($kl)"
  else add 6 "SSH keys" INFO "no readable authorized_keys"; fi
}

gpus_idle() {
  if [ "$HAVE_GPU" = 0 ]; then add 6 "GPUs idle at start" N/A "no GPU"; return; fi
  local apps busy
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | grep -cE '^ *[0-9]')
  busy=$(paste -d, <(col "$TMPD/inv.csv" idx) <(col "$TMPD/inv.csv" used) | awk -F, -v m="$MEM_BUSY" '$2 + 0 > m { printf "%sGPU %s %s MiB", s, $1, $2; s = ", " }')
  if [ "$apps" -gt 0 ] || [ -n "$busy" ]; then add 6 "GPUs idle at start" WARN "in use before the test: $apps compute process(es)${busy:+, memory used on $busy}"
  else add 6 "GPUs idle at start" PASS "no compute process, no GPU memory in use"; fi
}

virt_verdict() {
  local kind="virtual machine"
  case "$VIRT" in docker|podman|lxc*|container*|systemd-nspawn|openvz|rkt|wsl|proot) kind=container ;; esac
  if [ "$VIRT" = none ]; then
    if [ "$BARE" = 1 ]; then add 6 "Virtualization" PASS "none detected ($VIRT_SRC): bare metal, as listed"
    else add 6 "Virtualization" INFO "none detected ($VIRT_SRC): bare metal"; fi
  elif [ "$VIRT" = unknown ]; then
    if [ "$BARE" = 1 ]; then add 6 "Virtualization" WARN "could not tell (no systemd-detect-virt, no DMI)"
    else add 6 "Virtualization" INFO "could not tell (no systemd-detect-virt, no DMI)"; fi
  elif [ "$BARE" = 1 ]; then add 6 "Virtualization" FAIL "$VIRT detected ($kind, $VIRT_SRC): the listing says bare metal"
  else add 6 "Virtualization" INFO "$VIRT ($kind, $VIRT_SRC)"; fi
}

# ── 7. Report: text and JSON in --out, with their SHA-256 ────────────────────
report() {
  local i sec worst=PASS fails="" warns="" label os kernel sysnccl txt json sums code sep
  for sec in 1 2 3 4 5 6; do
    for ((i = 0; i < N; i++)); do
      [ "${C_SEC[i]}" = "$sec" ] || continue
      case "${C_ST[i]}" in
        FAIL) worst=FAIL; fails="$fails${fails:+, }${C_NAME[i]}" ;;
        WARN) warns="$warns${warns:+, }${C_NAME[i]}" ;;
        SKIP) [ "${C_CNT[i]}" = 1 ] && warns="$warns${warns:+, }${C_NAME[i]} not run" ;;
      esac
    done
  done
  [ "$worst" = PASS ] && [ -n "$warns" ] && worst=WARN
  case "$worst" in PASS) code=0 ;; WARN) code=1 ;; *) code=2 ;; esac
  os=$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release 2>/dev/null | tr -d '"'); kernel=$(uname -r 2>/dev/null)
  sysnccl=$(dpkg-query -W -f='${Version}' libnccl2 2>/dev/null)
  txt="$OUT/hwcheck-acceptance-$STAMP.txt"; json="$OUT/hwcheck-acceptance-$STAMP.json"; sums="$OUT/hwcheck-acceptance-$STAMP.sha256"
  {
    echo "VoltageGPU hwcheck acceptance test $VERSION (read-only, nothing about this machine is sent anywhere)"
    echo "  started $STARTED, finished $(date -u '+%Y-%m-%d %H:%M:%S UTC') · $HOST"
    if [ "$HAVE_GPU" = 1 ]; then echo "  ${NGPU}x $FIRST · driver ${DRIVER:-?} · CUDA ${CUDA:-?} · virtualization: ${VIRT:-?}"
    else echo "  no NVIDIA GPU · virtualization: ${VIRT:-?}"; fi
    echo "  PyTorch ${TORCH:-not used}${TORCH_CUDA:+ (CUDA $TORCH_CUDA, $TORCH_SRC)} · NCCL ${NCCL:-not used}${sysnccl:+ (system libnccl2 $sysnccl)} · ${os:-?} · kernel ${kernel:-?}"
    [ -n "$NOTE" ] && echo "  note: $NOTE"
    for sec in 1 2 3 4 5 6; do
      [ -n "${DET[$sec]:-}" ] || continue
      echo; echo "── $sec $(section "$sec") ──"; printf '%s' "${DET[$sec]}"
    done
    echo
    for sec in 1 2 3 4 5 6; do
      label="$sec $(section "$sec")"
      for ((i = 0; i < N; i++)); do
        [ "${C_SEC[i]}" = "$sec" ] || continue
        printf '  %-11s %-19s %-5s %s\n' "$label" "${C_NAME[i]}" "${C_ST[i]}" "${C_SUM[i]}"; label=""
      done
    done
    echo
    case "$worst" in
      PASS) echo "RESULT: PASS" ;;
      WARN) echo "RESULT: WARN ($warns)"; echo "Read the WARN lines before starting a long job." ;;
      *) echo "RESULT: FAIL ($fails)"; echo "Do not start a long job on this machine: ask for another one and send this report." ;;
    esac
  } > "$txt"
  {
    printf '{\n  "tool": "VoltageGPU hwcheck acceptance test",\n  "version": %s,\n' "$(jstr "$VERSION")"
    printf '  "started_utc": %s,\n  "host": %s,\n' "$(jstr "$STARTED")" "$(jstr "$HOST")"
    printf '  "options": {"quick": %s, "burn_seconds": %s, "gpus_listed": %s, "bare_metal": %s, "install": %s},\n' \
      "$([ "$QUICK" = 1 ] && echo true || echo false)" "$BURN" "$(jnum "$WANT")" "$([ "$BARE" = 1 ] && echo true || echo false)" "$([ "$INSTALL" = 1 ] && echo true || echo false)"
    printf '  "versions": {"driver": %s, "cuda": %s, "torch": %s, "torch_cuda": %s, "torch_source": %s, "nccl": %s, "system_libnccl2": %s, "os": %s, "kernel": %s},\n' \
      "$(jstr "$DRIVER")" "$(jstr "$CUDA")" "$(jstr "$TORCH")" "$(jstr "$TORCH_CUDA")" "$(jstr "$TORCH_SRC")" "$(jstr "$NCCL")" "$(jstr "$sysnccl")" "$(jstr "$os")" "$(jstr "$kernel")"
    printf '  "virtualization": %s,\n  "gpus": [' "$(jstr "$VIRT")"
    if [ "$HAVE_GPU" = 1 ]; then
      paste -d'|' <(col "$TMPD/inv.csv" idx) <(col "$TMPD/inv.csv" name) <(col "$TMPD/inv.csv" uuid) <(col "$TMPD/inv.csv" serial) \
        <(col "$TMPD/inv.csv" bus) <(col "$TMPD/inv.csv" vbios) <(col "$TMPD/inv.csv" mem) | {
        sep=""
        while IFS='|' read -r a b c d e f g; do
          printf '%s\n    {"index": %s, "name": %s, "uuid": %s, "serial": %s, "pci_bus_id": %s, "vbios": %s, "memory_mib": %s}' \
            "$sep" "$(jnum "$a")" "$(jstr "$b")" "$(jstr "$c")" "$(jstr "$d")" "$(jstr "$e")" "$(jstr "$f")" "$(jnum "$g")"
          sep=","
        done
      }
    fi
    printf '\n  ],\n  "checks": [\n'
    sep=""
    for sec in 1 2 3 4 5 6; do
      for ((i = 0; i < N; i++)); do
        [ "${C_SEC[i]}" = "$sec" ] || continue
        printf '%s    {"section": %s, "name": %s, "status": %s, "summary": %s, "data": %s}' "$sep" "$(jstr "$sec $(section "$sec")")" \
          "$(jstr "${C_NAME[i]}")" "$(jstr "${C_ST[i]}")" "$(jstr "${C_SUM[i]}")" "${C_DATA[i]}"; sep=$',\n'
      done
    done
    printf '\n  ],\n  "raw": {"nvidia_smi_topo": %s, "lspci_nvidia": %s},\n' "$(jstr "$RAW_TOPO")" "$(jstr "$RAW_LSPCI")"
    printf '  "result": %s,\n  "exit_code": %d\n}\n' "$(jstr "$worst")" "$code"
  } > "$json"
  ( cd "$OUT" && sha256sum "${txt##*/}" "${json##*/}" ) > "$sums"
  cat "$txt"
  echo
  echo "Report: $txt and ${json##*/} (SHA-256 below and in ${sums##*/})"
  sed 's/^/  /' "$sums"
  return "$code"
}
section() { case "$1" in 1) echo Topology ;; 2) echo P2P ;; 3) echo NCCL ;; 4) echo Load ;; 5) echo Errors ;; 6) echo Hygiene ;; esac; }

# ── Run ────────────────────────────────────────────────────────────────────
if [ "$QUICK" = 1 ]; then say "VoltageGPU hwcheck acceptance test $VERSION, about 2 minutes (--quick); nothing is sent anywhere"
else say "VoltageGPU hwcheck acceptance test $VERSION, about $((BURN / 60 + 5)) minutes; nothing is sent anywhere"; fi
detect_virt
inventory
gpus_line
[ "$HAVE_GPU" = 1 ] && snapshot "$TMPD/ecc0.csv" "${ECC_FIELDS[@]}"
hygiene
gpus_idle
topology
: > "$TMPD/samples.csv"
gpu_tests
pcie
ecc_verdict
xid_verdict
virt_verdict
report
exit $?
