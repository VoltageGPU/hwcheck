#!/usr/bin/env bash
# Builds the constructed fixtures of tests/fixtures (README.md there says which files are real
# captures and which ones this script writes). Run it again after changing a value here.
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")"

GPU_COLS="index,name,uuid,serial,pci.bus_id,vbios_version,memory.total,memory.used,driver_version,ecc.mode.current,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit,power.default_limit,fabric.state,fabric.status,mig.mode.current,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total,retired_pages.single_bit_ecc.count,retired_pages.double_bit.count,retired_pages.pending,remapped_rows.correctable,remapped_rows.uncorrectable,remapped_rows.pending,remapped_rows.failure,temperature.gpu,temperature.memory,clocks.current.sm,clocks.max.sm,power.draw,utilization.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown,clocks_event_reasons.sw_thermal_slowdown"
LOAD_COLS="index,temperature.gpu,temperature.memory,clocks.current.sm,power.draw,utilization.gpu,pcie.link.gen.current,pcie.link.width.current,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_thermal_slowdown"
NA="Not Active"

# ── 8x H100 SXM, healthy (driver and CUDA of a real H100 run of 17/10/2026) ──
h=h100-8x; mkdir -p "$h/py"
{
  echo "$GPU_COLS"
  for i in 0 1 2 3 4 5 6 7; do
    echo "$i,NVIDIA H100 80GB HBM3,GPU-00000000-0000-0000-0000-00000000000$i,165000000000$i,00000000:0$((i + 1)):00.0,96.00.CF.00.01,81559,1,580.173.02,Enabled,1,5,16,16,700.00,700.00,Completed,Success,Disabled,0,0,0,0,[N/A],[N/A],[N/A],0,0,No,No,$((33 + i % 3)),29,345,1980,71.20,0,$NA,$NA,$NA,$NA,$NA"
  done
} > "$h/gpus.csv"
{
  echo "$LOAD_COLS"
  for i in 0 1 2 3 4 5 6 7; do echo "$i,$((62 + i % 4)),71,$((1755 + i * 4)),695.30,100,5,16,Active,$NA"; done
} > "$h/gpus-load.csv"
for i in 0 1 2 3 4 5 6 7; do echo "GPU $i: NVIDIA H100 80GB HBM3 (UUID: GPU-00000000-0000-0000-0000-00000000000$i)"; done > "$h/L.txt"
cat > "$h/header.txt" <<'EOF'
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------------------------------------------------------+
EOF
cat > "$h/dmesg.txt" <<'EOF'
[    0.000000] Linux version 6.8.0-45-generic (buildd@lcy02-amd64-075) (x86_64-linux-gnu-gcc-13) #45-Ubuntu SMP PREEMPT_DYNAMIC
[    6.214523] nvidia: loading out-of-tree module taints kernel.
[    6.402117] NVRM: loading NVIDIA UNIX Open Kernel Module for x86_64  580.173.02
EOF
printf 'TORCH=2.14.1+cu130\nTORCH_CUDA=13.0\nDEVICES=8\nNCCL=2.30.7\n' > "$h/py/probe.out"
p2p() {  # p2p FILE VALUE_FOR_PAIR_FUNCTION: writes a p2p.py output for 8 GPUs
  awk -v f="$2" 'BEGIN {
    n = 8; min = 1e9
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) if (i != j) {
      v = (f == "sick" && (i == 7 || j == 7)) ? 38 + (i + j) % 3 : (f == "pcie" ? 24 + (int(i / 4) == int(j / 4) ? 2 : 0) : 389 + (i * 3 + j) % 5)
      m[i, j] = v; all[++k] = v; if (v < min) { min = v; mi = i; mj = j }
    }
    for (a = 2; a <= k; a++) { t = all[a]; for (b = a - 1; b >= 1 && all[b] > t; b--) all[b + 1] = all[b]; all[b + 1] = t }
    print "H2D=" (f == "pcie" ? "25.1,25.0,24.9,25.2,24.8,25.1,25.0,24.9" : "53.1,53.0,52.9,53.2,52.8,53.1,53.0,52.9")
    printf "P2P_MIN=%d\nP2P_MIN_PAIR=%d>%d\nP2P_MEDIAN=%d\n", min, mi, mj, (all[k / 2] + all[k / 2 + 1]) / 2
    print "P2P_NOPEER="; print "P2P_BAD="
    for (i = 0; i < n; i++) { r = "ROW=" i; for (j = 0; j < n; j++) r = r " " (i == j ? "-" : m[i, j]); print r }
    js = "JSON={\"gb_per_s\": ["
    for (i = 0; i < n; i++) { js = js (i ? ", " : "") "["; for (j = 0; j < n; j++) js = js (j ? ", " : "") (i == j ? "null" : m[i, j] ".0"); js = js "]" }
    print js "], \"peer_access\": [], \"host_to_gpu_gb_per_s\": [], \"bytes\": 268435456, \"iterations\": 20, \"corrupted_pairs\": []}"
  }' > "$1"
}
p2p "$h/py/p2p.out" healthy
printf 'ALGBW=273.3\nBUSBW=478.3\nTIME_US=15715\nWRONG=0\nRANKS=8\n' > "$h/py/nccl.out"
printf 'TFLOPS=652.4\nMISMATCH=0\nFINITE=1\nCHECKSUM=-1234567\nDTYPE=bfloat16\n' > "$h/py/burn.out"

# ── The same node with faults: GPU 7 off NVLink (SYS), an NVSwitch missing, GPU 2 at x8,
#    GPU 5 hot and throttled, uncorrected ECC on GPU 6, Xid 79 during the test... ──
s=h100-8x-sick; mkdir -p "$s/py"
cp "$h/L.txt" "$h/header.txt" "$h/py/probe.out" "$s/"; mv "$s/probe.out" "$s/py/"
sed -E '/^GPU7/ s/NV18/SYS/g; /^GPU[0-6]/ s/^((GPU[0-6]\t)([^\t]*\t){7})NV18/\1SYS/' "$h/topo.txt" > "$s/topo.txt"
grep -v '^0c:00.0' "$h/lspci.txt" > "$s/lspci.txt"
awk -F, -v OFS=, 'NR > 1 {
    if ($1 == 1) $29 = "Yes"; if ($1 == 2) $13 = 8; if ($1 == 3) $8 = 40961; if ($1 == 4) $15 = "500.00"
    if ($1 == 6) $21 = 2; if ($1 == 7) { $17 = "In Progress"; $18 = "N/A" } } { print }' "$h/gpus.csv" > "$s/gpus.csv"
awk -F, -v OFS=, 'NR > 1 {
    if ($1 == 0) $4 = 1400; if ($1 == 2) $8 = 8; if ($1 == 5) { $2 = 88; $7 = 4; $10 = "Active" } } { print }' "$h/gpus-load.csv" > "$s/gpus-load.csv"
echo "4242, python3, 40960" > "$s/apps.csv"
{ cat "$h/dmesg.txt"
  echo "[    7.345678] NVRM: Xid (PCI:0000:03:00): 13, pid=2210, name=python3, Graphics Exception: ESR 0x514648=0x1000f00 0x514650=0x0"
  echo "[@NOW@] NVRM: Xid (PCI:0000:08:00): 79, pid=3107, name=python3, GPU has fallen off the bus."
} > "$s/dmesg.txt"
p2p "$s/py/p2p.out" sick
printf 'ALGBW=143.1\nBUSBW=250.4\nTIME_US=30013\nWRONG=0\nRANKS=8\n' > "$s/py/nccl.out"
cp "$h/py/burn.out" "$s/py/burn.out"
printf 'TFLOPS=301.2\nMISMATCH=0\nFINITE=1\nCHECKSUM=-1234567\nDTYPE=bfloat16\n' > "$s/py/burn-4.out"
printf 'TFLOPS=650.9\nMISMATCH=2\nFINITE=1\nCHECKSUM=-1234567\nDTYPE=bfloat16\n' > "$s/py/burn-6.out"

# ── 8x PCIe GPUs without NVLink (two sockets): every NVLink verdict is N/A ──
p=pcie-8x; mkdir -p "$p/py"
{
  echo "$GPU_COLS"
  for i in 0 1 2 3 4 5 6 7; do
    echo "$i,NVIDIA L40S,GPU-00000000-0000-0000-0000-00000000010$i,132000000000$i,00000000:$((16 + i * 16)):00.0,95.02.66.00.02,46068,1,580.173.02,Enabled,1,4,16,16,350.00,350.00,[N/A],[N/A],[N/A],0,0,0,0,[N/A],[N/A],[N/A],0,0,No,No,30,[N/A],210,2520,36.10,0,$NA,$NA,$NA,$NA,$NA"
  done
} > "$p/gpus.csv"
{
  echo "$LOAD_COLS"
  for i in 0 1 2 3 4 5 6 7; do echo "$i,58,[N/A],2490,348.20,100,4,16,Active,$NA"; done
} > "$p/gpus-load.csv"
for i in 0 1 2 3 4 5 6 7; do echo "GPU $i: NVIDIA L40S (UUID: GPU-00000000-0000-0000-0000-00000000010$i)"; done > "$p/L.txt"
cp "$h/header.txt" "$h/dmesg.txt" "$p/"
awk 'BEGIN {
  printf "\t\033[4mGPU0\tGPU1\tGPU2\tGPU3\tGPU4\tGPU5\tGPU6\tGPU7\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\033[0m\n"
  for (i = 0; i < 8; i++) {
    r = "GPU" i
    for (j = 0; j < 8; j++) r = r "\t" (i == j ? " X " : int(i / 4) != int(j / 4) ? "SYS" : int(i / 2) == int(j / 2) ? "PIX" : "NODE")
    print r "\t" (i < 4 ? "0-47,96-143\t0" : "48-95,144-191\t1") "\t\tN/A"
  }
  print ""; print "Legend:"; print ""; print "  X    = Self"
}' > "$p/topo.txt"
for i in 0 1 2 3 4 5 6 7; do printf '%02x:00.0 3D controller [0302]: NVIDIA Corporation AD102GL [L40S] [10de:26b9] (rev a1)\n' $((16 + i * 16)); done > "$p/lspci.txt"
printf 'TORCH=2.14.1+cu130\nTORCH_CUDA=13.0\nDEVICES=8\nNCCL=2.30.7\n' > "$p/py/probe.out"
p2p "$p/py/p2p.out" pcie
printf 'ALGBW=12.2\nBUSBW=21.3\nTIME_US=352049\nWRONG=0\nRANKS=8\n' > "$p/py/nccl.out"

# ── 1x RTX 3060 Ti: gpus.csv assembled from real nvidia-smi answers (driver 610.88, WSL2,
#    24/10/2026); only the UUID is zeroed. L.txt, header.txt, topo.txt, q-ecc.txt are raw. ──
r=rtx3060ti-1x
{
  echo "$GPU_COLS"
  echo "0,NVIDIA GeForce RTX 3060 Ti,GPU-00000000-0000-0000-0000-000000000000,[N/A],00000000:29:00.0,94.04.6b.00.4a,8192,1408,610.88,[N/A],3,3,16,16,200.00,200.00,N/A,N/A,[N/A],[N/A],[N/A],[N/A],[N/A],[N/A],[N/A],[N/A],[N/A],[N/A],[N/A],[N/A],44,N/A,1215,2100,38.45,20,$NA,$NA,$NA,$NA,$NA"
} > "$r/gpus.csv"
