#!/usr/bin/env bash
# Tests of hwcheck.sh and acceptance.sh, without a GPU: fake nvidia-smi, lspci, dmesg and PyTorch
# replay tests/fixtures (README.md there), and a fake root directory holds the leftovers.
#   bash tests/run.sh          # needs bash, coreutils, python3 (JSON check), shellcheck
# AWK=mawk (Ubuntu's default awk) or AWK=gawk picks the awk used by the scripts. Exit 0 = all pass.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(dirname "$HERE")
WORK=$(mktemp -d)
if [ -n "${KEEP:-}" ]; then echo "work directory kept: $WORK"; else trap 'rm -rf "$WORK"' EXIT; fi
PASSED=0; FAILED=0
ok() { PASSED=$((PASSED + 1)); printf '  ok    %s\n' "$1"; }
ko() { FAILED=$((FAILED + 1)); printf '  FAIL  %s\n' "$1"; }
check() { if "${@:2}"; then ok "$1"; else ko "$1"; fi; }   # check NAME COMMAND...

# Fake commands first in PATH (wrappers: the fakes need no exec bit in git)
BIN="$WORK/bin"; mkdir -p "$BIN"
for f in nvidia-smi python3 dmesg lspci systemd-detect-virt curl; do
  printf '#!/bin/sh\nexec bash "%s" "$@"\n' "$HERE/fake/$f" > "$BIN/$f"
done
for f in journalctl sudo; do printf '#!/bin/sh\nexec bash "%s" "$@"\n' "$HERE/fake/denied" > "$BIN/$f"; done
if [ -n "${AWK:-}" ]; then ln -s "$(command -v "$AWK")" "$BIN/awk"; fi
chmod +x "$BIN"/*
echo "awk: $(readlink -f "$(PATH="$BIN:/usr/bin:/bin" command -v awk)")"

# Fake root directories: a clean one (one SSH key) and one with a previous tenant's leftovers
mkfs() {
  local fs="$WORK/$1"; mkdir -p "$fs/root/.ssh" "$fs/home/ubuntu" "$fs/tmp" "$fs/var/tmp"
  printf 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHhwY2hlY2stZml4dHVyZS1rZXktb25lLWF0LWxlYXN0 me@laptop\n' > "$fs/root/.ssh/authorized_keys"
  printf '# bash defaults\n' > "$fs/root/.bashrc"
}
mkfs fs-clean
mkfs fs-dirty
D="$WORK/fs-dirty"
printf 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHhwY2hlY2stZml4dHVyZS1rZXktdHdvLWF0LWxlYXN0 someone@else\n' >> "$D/root/.ssh/authorized_keys"
printf 'fixture: not a credential\n' > "$D/root/.git-credentials"
printf 'ls\ncd /workspace\n' > "$D/root/.bash_history"
mkdir -p "$D/home/ubuntu/checkpoints" "$D/workspace" "$D/root/.cache/huggingface/hub"
truncate -s 11M "$D/home/ubuntu/checkpoints/model.safetensors"
printf 'print(1)\n' > "$D/workspace/train.py"
printf 'x\n' > "$D/root/.cache/huggingface/hub/x"
printf 'old\n' > "$D/tmp/job-1234.log"; touch -d '2020-01-01' "$D/tmp/job-1234.log"

# run NAME FIXTURE FSROOT [VAR=value...] -- [acceptance options]: one run in a clean environment
run() {
  local name=$1 fx=$2 fs=$3 envs=() out; shift 3
  while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done; shift
  out="$WORK/$name"; mkdir -p "$out/report" "$out/tmp" "$out/state"
  env -i HOME="$out" PATH="$BIN:/usr/local/bin:/usr/bin:/bin" TMPDIR="$out/tmp" FAKE_DIR="$HERE/fixtures/$fx" \
    FAKE_STATE="$out/state" HWCHECK_FS_ROOT="${fs:+$WORK/$fs}" ${envs[@]+"${envs[@]}"} \
    bash "$REPO/acceptance.sh" --out "$out/report" "$@" > "$out/stdout" 2> "$out/stderr"
  echo $? > "$out/code"
}
code() { [ "$(cat "$WORK/$1/code")" = "$2" ]; }
line() { grep -qE "^  .{12}$2 +$3( |$)" "$WORK/$1/stdout"; }   # line RUN "NAME" STATUS
has() { grep -qE "$2" "$WORK/$1/stdout"; }
clean_stderr() { ! grep -qE 'unbound variable|syntax error|command not found|No such file|awk: |integer expression' "$WORK/$1/stderr"; }
json_ok() {
  local j; j=$(ls "$WORK/$1/report"/*.json 2>/dev/null) || return 1
  /usr/bin/python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); assert d["result"] == sys.argv[2], d["result"]; assert d["exit_code"] == int(sys.argv[3]); assert all(c["status"] in ("PASS", "WARN", "FAIL", "N/A", "INFO", "SKIP") for c in d["checks"])' "$j" "$2" "$3"
}
sums_ok() { ( cd "$WORK/$1/report" && sha256sum -c --quiet ./*.sha256 ); }
tmp_clean() { [ -z "$(ls -A "$WORK/$1/tmp")" ] && [ -z "$(ls -A "$WORK/$1/state")" ]; }
common() {   # common RUN RESULT EXIT: what every run must satisfy
  check "$1: exit code $3" code "$1" "$3"
  check "$1: RESULT: $2" has "$1" "^RESULT: $2"
  check "$1: JSON report valid, result $2" json_ok "$1" "$2" "$3"
  check "$1: SHA-256 file matches the reports" sums_ok "$1"
  check "$1: temporary directory removed" tmp_clean "$1"
  check "$1: no shell or awk error" clean_stderr "$1"
}

echo "shellcheck"
SC=${SHELLCHECK:-shellcheck}
if command -v "$SC" >/dev/null 2>&1; then
  check "shellcheck: hwcheck.sh acceptance.sh tests" "$SC" -x "$REPO/hwcheck.sh" "$REPO/acceptance.sh" "$HERE/run.sh" "$HERE/fixtures/build.sh" "$HERE"/fake/*
else ko "shellcheck not installed (set SHELLCHECK=path)"; fi

echo "acceptance.sh, healthy 8x H100 SXM, bare metal, 10 s load"
run healthy h100-8x fs-clean -- --burn-seconds 10 --gpus 8 --bare-metal
common healthy PASS 0
for l in "GPUs PASS" "NVLink mesh PASS" "NVSwitch PASS" "Fabric state PASS" "PCIe links PASS" "CUDA init PASS" \
    "P2P copies PASS" "NCCL all-reduce PASS" "Throughput PASS" "Thermal / power PASS" "Clocks PASS" \
    "ECC / memory PASS" "Kernel log \(Xid\) PASS" "Previous tenant PASS" "SSH keys PASS" "GPUs idle at start PASS" "Virtualization PASS"; do
  check "healthy: ${l//\\/}" line healthy "${l% *}" "${l##* }"
done
check "healthy: 56 pairs on NV18" has healthy "56/56 GPU pairs on NV18 \(NV18 expected on H100 SXM\)"
check "healthy: PCIe Gen5 x16 seen under load" has healthy "Gen5 x16 on 8/8 GPUs under load"
check "healthy: the power cap alone is not a warning" has healthy "at the power cap in 100% of the samples"
check "healthy: bus bandwidth compared with H100 SXM" has healthy "bus bandwidth 478.3 GB/s on 8 GPUs \(4 GiB"

echo "acceptance.sh, 8x H100 with faults, in a VM sold as bare metal"
run sick h100-8x-sick fs-dirty FAKE_VIRT=kvm -- --burn-seconds 10 --bare-metal
common sick FAIL 2
for l in "NVLink mesh FAIL" "NVSwitch WARN" "Fabric state WARN" "PCIe links FAIL" "P2P copies FAIL" "NCCL all-reduce WARN" \
    "Throughput FAIL" "Thermal / power FAIL" "Clocks WARN" "ECC / memory FAIL" "Kernel log \(Xid\) FAIL" \
    "Previous tenant FAIL" "SSH keys WARN" "GPUs idle at start WARN" "Virtualization FAIL"; do
  check "sick: ${l//\\/}" line sick "${l% *}" "${l##* }"
done
check "sick: GPU 7 off NVLink" has sick "42/56 GPU pairs on NVLink, the others: SYS x14"
check "sick: GPU 2 at x8 under load" has sick "GPU 2 at x8 \(x16 possible\)"
check "sick: GPU 5 at Gen4 under load" has sick "GPU 5 at Gen4 under load"
check "sick: slow pair under the NVLink floor" has sick "under 112 GB/s \(25% of the 450 GB/s line rate"
check "sick: GPU 4 throughput halved" has sick "GPU 4 at 301 TFLOPS, under half"
check "sick: GPU 6 results changed" has sick "GPU 6: the same product changed 2 times"
check "sick: GPU 5 thermal slowdown" has sick "GPU 5: hardware thermal slowdown or power brake in [0-9]+ of [0-9]+ samples"
check "sick: GPU 4 power limit" has sick "GPU 4 power limit 500 W, under its default 700 W"
check "sick: GPU 5 hotter than the others" has sick "GPU 5 reached 88 C"
check "sick: GPU 0 clock low" has sick "GPU 0 median SM clock 1400 MHz"
check "sick: uncorrected ECC on GPU 6" has sick "GPU 6: 2 uncorrected ECC errors"
check "sick: row remapping pending on GPU 1" has sick "GPU 1: row remapping pending"
check "sick: Xid 79 during the test" has sick "during the test: Xid 79 on PCI:0000:08:00"
check "sick: Xid 13 before the test not counted" bash -c "! grep -q 'Xid 13 on PCI:0000:03:00 (during' '$WORK/sick/stdout'"
check "sick: credentials found, content not printed" bash -c "grep -q 'credentials: /root/.git-credentials' '$WORK/sick/stdout' && ! grep -rq 'not a credential' '$WORK/sick/stdout' '$WORK/sick/report'"
check "sick: history, model file, old /tmp file listed" bash -c "grep -q 'history: /root/.bash_history' '$WORK/sick/stdout' && grep -q 'model: /home/ubuntu/checkpoints/model.safetensors' '$WORK/sick/stdout' && grep -q 'tmp, older than this boot: /tmp/job-1234.log' '$WORK/sick/stdout'"
check "sick: two SSH keys" has sick "2 keys can log in \(root: 2\)"
check "sick: 40961 MiB in use on GPU 3" has sick "memory used on GPU 3 40961 MiB"

echo "acceptance.sh, 8x H100 SXM with slow P2P copies (thresholds measured on 8x H100 SXM, 2026-10-24)"
run slowp2p h100-8x fs-clean "FAKE_P2P=$HERE/fixtures/h100-8x/py/p2p-slow.out" -- --quick --gpus 8 --bare-metal
common slowp2p WARN 1
check "slowp2p: P2P copies WARN" line slowp2p "P2P copies" WARN
check "slowp2p: median under 75% of the line rate" has slowp2p "median under 338 GB/s \(75% of the H100 SXM line rate\)"
run onepair h100-8x fs-clean "FAKE_P2P=$HERE/fixtures/h100-8x/py/p2p-onepair.out" -- --quick --gpus 8 --bare-metal
common onepair WARN 1
check "onepair: P2P copies WARN" line onepair "P2P copies" WARN
check "onepair: one pair under 90% of the median" has onepair "slowest pair 340 GB/s \(GPU 0>5\), median 391 GB/s; a pair under 90% of the median"
run measured h100-8x fs-clean "FAKE_P2P=$HERE/fixtures/h100-8x/py/p2p-measured-2026-10-24.out" -- --quick --gpus 8 --bare-metal
common measured PASS 0
check "measured: the real 8x H100 SXM copies of 2026-10-24 pass the new thresholds" has measured "P2P copies +PASS +slowest pair 387 GB/s \(GPU 0>2\), median 389 GB/s \(FAIL under 112, WARN under 338"
check "measured: their host-to-GPU copies pass too" has measured "PCIe links +PASS .*host-to-GPU copies 53.1 to 55.5 GB/s"

echo "acceptance.sh, 8x PCIe GPUs without NVLink, --quick"
run pcie pcie-8x fs-clean FAKE_VIRT=kvm -- --quick
common pcie PASS 0
for l in "NVLink mesh N/A" "NVSwitch N/A" "Fabric state N/A" "P2P copies N/A" "NCCL all-reduce N/A" "Load test SKIP" "Virtualization INFO"; do
  check "pcie: ${l//\\/}" line pcie "${l% *}" "${l##* }"
done
check "pcie: says no NVLink" has pcie "no NVLink: the GPUs talk over PCIe \(NODE x16, PIX x8, SYS x32\)"

echo "acceptance.sh, real RTX 3060 Ti answers, no PyTorch, --no-install"
run rtx rtx3060ti-1x fs-clean FAKE_VIRT=wsl -- --no-install
common rtx WARN 1
for l in "GPUs PASS" "NVLink mesh N/A" "NVSwitch N/A" "P2P copies SKIP" "NCCL all-reduce SKIP" "Load test SKIP" \
    "ECC / memory N/A" "Kernel log \(Xid\) SKIP" "GPUs idle at start PASS" "Virtualization INFO"; do
  check "rtx: ${l//\\/}" line rtx "${l% *}" "${l##* }"
done
check "rtx: PCIe at idle only, not judged on generation" has rtx "PCIe links +PASS +Gen3 x16 on 1/1 GPUs \(seen at idle only"

echo "acceptance.sh, no GPU at all"
run nogpu h100-8x fs-clean FAKE_NO_GPU=1 -- --quick
common nogpu FAIL 2
check "nogpu: GPUs FAIL" line nogpu "GPUs" FAIL
check "nogpu: GPU tests N/A" bash -c "grep -qE 'P2P copies +N/A' '$WORK/nogpu/stdout' && grep -qE 'ECC / memory +N/A' '$WORK/nogpu/stdout'"
check "nogpu: hygiene still runs" line nogpu "Previous tenant" PASS
check "nogpu: GPUs on the PCI bus without a driver are named" has nogpu "8 NVIDIA GPUs on the PCI bus but no working driver: install the NVIDIA driver"

echo "acceptance.sh, NVSwitch board without its fabric manager"
run nofm h100-8x fs-clean "FAKE_CUDA_FAIL=CUDA error: system not yet initialized" -- --quick
common nofm FAIL 2
check "nofm: P2P FAIL with the fabric manager hint" has nofm "P2P copies +FAIL .*system not yet initialized \(is nvidia-fabricmanager running\?\)"
check "nofm: NCCL FAIL with the fabric manager hint" has nofm "NCCL all-reduce +FAIL .*\(is nvidia-fabricmanager running\?\)"

echo "acceptance.sh, options"
check "bad option: exit 64" bash -c "bash '$REPO/acceptance.sh' --nope >/dev/null 2>&1; [ \$? = 64 ]"
check "missing value: exit 64, no loop" bash -c "timeout 5 bash '$REPO/acceptance.sh' --gpus >/dev/null 2>&1; [ \$? = 64 ]"
check "--burn-seconds out of range: exit 64" bash -c "bash '$REPO/acceptance.sh' --burn-seconds 5 >/dev/null 2>&1; [ \$? = 64 ]"
check "--help: exit 0" bash -c "bash '$REPO/acceptance.sh' --help | grep -q -- '--burn-seconds'"

echo "hwcheck.sh --acceptance"
PIN=$(sed -n 's/^ACCEPTANCE_SHA256="\([0-9a-f]*\)"$/\1/p' "$REPO/hwcheck.sh")
check "the pinned SHA-256 is the one of acceptance.sh" [ "$PIN" = "$(sha256sum "$REPO/acceptance.sh" | cut -d' ' -f1)" ]
check "from a clone: runs the local acceptance.sh" bash -c "bash '$REPO/hwcheck.sh' --acceptance --help | grep -q 'usage: acceptance.sh'"
mkdir -p "$WORK/clone"; cp "$REPO/hwcheck.sh" "$WORK/clone/"; printf 'echo tampered\n' > "$WORK/clone/acceptance.sh"; cp "$WORK/clone/acceptance.sh" "$WORK/evil.sh"
check "tampered local copy and download: refused, exit 70" bash -c "out=\$(PATH='$BIN:/usr/bin:/bin' FAKE_CURL_SRC='$WORK/evil.sh' bash '$WORK/clone/hwcheck.sh' --acceptance 2>&1); [ \$? = 70 ] && ! printf '%s' \"\$out\" | grep -q tampered"
check "piped hwcheck.sh: downloads, checks the hash, runs" bash -c "PATH='$BIN:/usr/bin:/bin' FAKE_CURL_SRC='$REPO/acceptance.sh' bash -s -- --acceptance --help < '$REPO/hwcheck.sh' | grep -q 'usage: acceptance.sh'"
check "--acceptance elsewhere than first: exit 64" bash -c "bash '$REPO/hwcheck.sh' --gpu H100 --acceptance >/dev/null 2>&1; [ \$? = 64 ]"

echo
echo "$PASSED passed, $FAILED failed"
[ "$FAILED" = 0 ]
