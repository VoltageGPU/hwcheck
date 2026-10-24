# hwcheck

One read-only command that checks whether the GPU machine you rented matches its listing. Works on any Linux box with an NVIDIA GPU, on any provider. For multi-GPU nodes, a second mode runs a [10-minute acceptance test](#acceptance-test-is-this-8-gpu-node-healthy) before a long job.

```
curl -fsSL https://voltagegpu.com/hwcheck.sh | bash -s -- --gpu "RTX 4090" --gpus 1 --vram 24 --ram 128 --disk 500
```

You pass what the listing promised; the script measures the machine and prints one line per item, then a verdict:

```
RESULT: MATCHES THE LISTING
```

Exit code `0` means everything matches, `2` means something does not (the failing line says what). Put it at the top of a startup script and stop paying the moment a machine is wrong.

## What it checks

- **GPU model** — every digit-bearing token of the listed name (`4090`, `h100`, `80gb`) must appear in the name the driver reports. Mixed GPUs are flagged.
- **GPU count** — what the driver sees vs. the listing.
- **VRAM per GPU** — smallest card vs. the listing, with a small tolerance.
- **Real CUDA work** — if the image ships PyTorch, a matrix multiplication runs on every GPU.
- **CPU, RAM, disk** — measured against what you passed.
- **Download speed** — one 25 MB fetch from Cloudflare (`--no-net` skips it).

Nothing about the machine is sent anywhere. `hwcheck.sh` is about 180 lines of bash, `acceptance.sh` about 900: read them before you run them, as you should with any piped script.

## Real output

Two real runs on the same workstation (an NVIDIA GeForce RTX 3060 Ti), in [`examples/`](examples/):

- [`rtx3060ti-match.txt`](examples/rtx3060ti-match.txt): listing says RTX 3060 Ti, 8 GB → `RESULT: MATCHES THE LISTING`, exit 0.
- [`rtx3060ti-mismatch.txt`](examples/rtx3060ti-mismatch.txt): the same card listed as an RTX 4090, 24 GB → `RESULT: DOES NOT MATCH THE LISTING (GPU model, VRAM)`, exit 2.

That second case is the classic one: a cheaper card sold under a bigger name.

Four real runs on rented machines (16 October 2026), one per model, each run over SSH exactly as a customer would right after launch, all `RESULT: MATCHES THE LISTING`, exit 0:

- [`rtx-4090.txt`](examples/rtx-4090.txt): 1x RTX 4090, 24 GB.
- [`rtx-5090.txt`](examples/rtx-5090.txt): 1x RTX 5090, 32 GB.
- [`h200.txt`](examples/h200.txt): 1x H200, the driver reports 140 GB.
- [`b300.txt`](examples/b300.txt): 1x B300, the driver reports 269 GB.

Memory is shown as the driver reports it, which can differ from the round figure on a spec sheet.

One real acceptance test (24 October 2026) on a production 8x H100 SXM bare metal node, driver 595.112.07 and Fabric Manager installed a few minutes earlier, run as the `ubuntu` user with `--gpus 8 --bare-metal`, 7 minutes including the PyTorch download, `RESULT: PASS`, exit 0: [`h100-8x-acceptance.txt`](examples/h100-8x-acceptance.txt) (the report, P2P matrix and per-GPU load table included) and [`h100-8x-acceptance.json`](examples/h100-8x-acceptance.json), byte for byte as the tool wrote them (SHA-256 `bb285156…` and `a656867b…`). The verdict lines:

```
  1 Topology  GPUs                PASS  8x NVIDIA H100 80GB HBM3
              NVLink mesh         PASS  56/56 GPU pairs on NV18 (NV18 expected on H100 SXM)
              NVSwitch            PASS  4 NVSwitch chips visible (4 on an 8-GPU H100 SXM board)
              Fabric state        PASS  Completed / Success on 8/8 GPUs
              PCIe links          PASS  Gen5 x16 on 8/8 GPUs under load, host-to-GPU copies 53.1 to 55.5 GB/s
  2 P2P       CUDA init           PASS  8/8 GPUs in PyTorch 2.14.1+cu130 (installed for this run, cu130)
              P2P copies          PASS  slowest pair 387 GB/s (GPU 0>2), median 389 GB/s (FAIL under 112, WARN under 270: H100 SXM line rate 450 GB/s)
  3 NCCL      NCCL all-reduce     PASS  bus bandwidth 478.5 GB/s on 8 GPUs (4 GiB, NCCL 2.30.7); H100 SXM: about 370 GB/s in ring, about 480 with NVLink SHARP, PASS from 340
  4 Load      Throughput          PASS  8/8 GPUs ran it, median 684 TFLOPS (bfloat16), slowest GPU 2 at 669
              Thermal / power     PASS  max 69 C (GPU 2), no thermal or hardware slowdown; at the power cap in 95% of the samples (normal under full load)
              Clocks              PASS  median SM clock 1368 MHz, every GPU between 1335 and 1380 MHz, at least 90% busy
  5 Errors    ECC / memory        PASS  ECC on 8/8 GPUs, no uncorrected error, nothing pending, 0 corrected since the driver loaded
              Kernel log (Xid)    PASS  no Xid during the test, read with journalctl
  6 Hygiene   Previous tenant     PASS  nothing found in the home directories, /tmp, /var/tmp and /workspace; not readable: /root (run as root to include it)
              SSH keys            PASS  1 key can log in (ubuntu: 1)
              GPUs idle at start  PASS  no compute process, no GPU memory in use
              Virtualization      PASS  none detected (systemd-detect-virt): bare metal, as listed

RESULT: PASS
```

That run was made with the thresholds of the time ("WARN under 270" above); the P2P thresholds were raised after it, from its own measurements (see [Where the reference values come from](#where-the-reference-values-come-from)): with the current ones the same node still passes every line.

## What it does not prove

It catches mislabeled machines, wrong counts and undersized nodes — what actually goes wrong in practice. It does **not** defend against a host determined to lie: whoever controls the machine can, in principle, make the driver report another name. Proof against the operator itself needs hardware attestation (Intel TDX + NVIDIA attestation); that is a different tool: [voltage-verify](https://github.com/VoltageGPU/voltage-verify).

## Options

| Flag | Meaning |
|---|---|
| `--gpu "<model>"` | listed GPU model |
| `--gpus <n>` | listed GPU count |
| `--vram <GB>` | listed VRAM per GPU |
| `--ram <GB>` | listed system RAM |
| `--disk <GB>` | listed disk size |
| `--no-net` | skip the download-speed test |
| `--acceptance [...]` | first option only: run the acceptance test below instead |

## Acceptance test: is this 8-GPU node healthy?

Before a training run of several days, run this once on the machine, bare metal or VM, on any provider:

```
curl -fsSL https://voltagegpu.com/hwcheck.sh | bash -s -- --acceptance
```

`hwcheck.sh` hands over to [`acceptance.sh`](acceptance.sh), and only to the copy whose SHA-256 it pins: from a clone, the file next to it; piped, it downloads `https://voltagegpu.com/hwcheck-acceptance.sh` and refuses to run it if the hash differs. From a clone, `bash acceptance.sh` works too.

It prints one line per measure with a verdict, then `RESULT: PASS`, `RESULT: WARN (...)` or `RESULT: FAIL (...)`, and writes three files in the current directory: `hwcheck-acceptance-<UTC time>.txt` (what you saw), `.json` (every value) and `.sha256` (the SHA-256 of both, to show later that the report was not edited).

Exit code `0` PASS, `1` WARN (read those lines before a long job), `2` FAIL (do not start a long job there), `64` bad option. About 10 minutes, 2 with `--quick`, plus the PyTorch download if the image has none.

| Option | Meaning |
|---|---|
| `--quick` | skip the 5-minute load test |
| `--burn-seconds N` | length of the load test, 10 to 3600 seconds (default 300) |
| `--gpus N` | GPU count of the listing: FAIL if fewer are visible |
| `--bare-metal` | the listing says bare metal: FAIL if a hypervisor is detected |
| `--no-install` | never download PyTorch; the tests that need it are skipped, which makes the result WARN |
| `--out DIR` | where to write the report files |

### What it measures, and how each verdict is decided

Each line reads PASS, WARN or FAIL; N/A when the measure does not apply (no NVLink on a PCIe machine is "not applicable", not a failure); INFO when it is measured but there is no reference to judge it; SKIP when it could not run, which makes the result WARN (except the load test skipped with `--quick`).

| | Measure | How | Verdict |
|---|---|---|---|
| 1 | GPUs | `nvidia-smi --query-gpu` | FAIL if fewer than `--gpus`; WARN on mixed models or MIG enabled |
| 1 | NVLink mesh | `nvidia-smi topo -m`, every GPU pair | On SXM boards (H100, H200, A100, B200, B300) every pair must read `NV<n>`: FAIL if a pair goes over PCIe (`SYS`, `NODE`, `PHB`, `PXB`, `PIX`); WARN if a pair has fewer links than the model has (NV18 on H100, H200, B200, B300, NV12 on A100): a link is down. PCIe GPUs: N/A |
| 1 | NVSwitch | `lspci -nn`, NVIDIA devices of class 0680 | PASS with 4 on H100 and H200 boards, 6 on A100 boards; WARN on another count; INFO in a VM that shows none (the host may keep them; the NVLink lines judge the fabric) |
| 1 | Fabric state | `fabric.state`, `fabric.status` | WARN unless `Completed` / `Success` on every GPU that reports it |
| 1 | PCIe links | `pcie.link.gen` and `pcie.link.width`, current against max, the best seen during the tests (an idle GPU drops to Gen1); host-to-GPU copies from pinned memory | FAIL if a GPU under load runs narrower than both ends allow (x8 in an x16 slot halves the bandwidth: usually a badly seated card or riser); WARN if its generation stays under the maximum under load, or its host-to-GPU copies are under 70% of the median GPU |
| 2 | P2P copies | 256 MiB from every GPU to every other one, 20 times, bytes compared after; the matrix is printed | FAIL if bytes differ, if a pair lacks peer access on an NVLink mesh, or if a pair is under 25% of the model's NVLink line rate per direction (PCIe Gen5 x16 tops out at 64 GB/s, 14% of H100's 450: under 25%, the pair is not really on NVLink); WARN if the median pair is under 75% of the line rate (measured: 86% on 8x H100 SXM, 2026-10-24), or a pair under 90% of the median (measured: the slowest pair at 99% of it). Models without a known line rate: the relative rules only |
| 3 | NCCL all-reduce | `torch.distributed` with NCCL, one process per GPU, 4 GiB of float32 per GPU (1 GiB under 40 GB of memory), 20 iterations, values checked; bus bandwidth as nccl-tests computes it | H100 SXM and H200: PASS from 340 GB/s, FAIL under 185 (measured: 478.5 GB/s on 8x H100 SXM, 2026-10-24). A100 SXM: PASS from 210, FAIL under 115. WARN in between. Other NVLink models: INFO (no reference). PCIe machines: N/A |
| 4 | Throughput | 5 minutes of bf16 8192x8192 matrix products on every GPU at once | FAIL if a GPU does not run, returns non-finite values, returns other bits for the same product (silent data corruption), or runs under half the median GPU; WARN under 85% of the median, or if a GPU computes another result than an identical GPU |
| 4 | Thermal / power | `nvidia-smi` every 5 s: temperature, power, slowdown reasons (`clocks_event_reasons`) | FAIL if hardware thermal slowdown or power brake is active in 5% of the samples or more on a GPU; WARN on any thermal slowdown, hardware slowdown, a power limit set under the GPU's default, or a GPU 10 C hotter than the median GPU |
| 4 | Clocks | the same samples: SM clock, utilisation | WARN if a GPU's median SM clock is under 85% of the median GPU, or a GPU is under 90% busy |
| 5 | ECC / memory | ECC counters (volatile: since the driver loaded; aggregate: lifetime), retired pages, remapped rows, before and after the tests | FAIL on any uncorrected error since the driver loaded, a pending row remap or page retirement (the GPU needs a reset), a failed remap; WARN on ECC disabled, corrected errors during the test, lifetime uncorrected errors, retired or remapped rows |
| 5 | Kernel log (Xid) | `dmesg`, else `journalctl -k`, else `sudo -n dmesg` | FAIL on a hardware Xid during the test (48, 62, 64, 74, 79, 92, 94, 95, 119, 120, 140 in NVIDIA's Xid catalog); WARN on another Xid during the test, or a hardware Xid earlier in this boot |
| 6 | Previous tenant | home directories, `/tmp`, `/var/tmp`, `/workspace`: credential files, shell histories, ML caches, model files, files in a home, `/tmp` files older than this boot. Paths and dates only: contents are never read | FAIL on credential files (`.netrc`, `.git-credentials`, cloud and registry credentials, Hugging Face token, private SSH keys): whoever comes after you would find yours the same way. WARN on anything else |
| 6 | SSH keys | `authorized_keys` of root and of every home | WARN if more than one key can log in (their fingerprints are in the report) |
| 6 | GPUs idle at start | compute processes, memory in use | WARN if a process, or more than 2 GiB of memory, is already on a GPU |
| 6 | Virtualization | `systemd-detect-virt` | INFO; with `--bare-metal`, FAIL if a hypervisor is detected |

Reaching the power cap is not a warning: under a full matrix load an H100 runs at its power limit, as designed. Thermal and hardware slowdowns are the warning signs.

### Where the reference values come from

- NVLink line rate per direction: half the two-way figure of NVIDIA's datasheets: 900 GB/s for H100 SXM and H200 SXM (450 per direction), 600 GB/s for A100 SXM (300), 1.8 TB/s for B200 and B300 (900). Links per GPU: 18 for NVLink 4 and 5, 12 for NVLink 3.
- NVSwitch chips on 8-GPU boards: 4 on H100 and H200 boards, 6 on A100 boards (NVIDIA DGX H100 and DGX A100 datasheets).
- NCCL all-reduce on 8 H100 SXM: NCCL maintainers at NVIDIA put NVLink's effective rate at about 370 GB/s (20% protocol overhead on 450) and the all-reduce peak with NVLink SHARP at about 480 GB/s on large sizes ([nccl-tests#212](https://github.com/NVIDIA/nccl-tests/issues/212)); 476 GB/s "looks to be in the expected range for H100" ([nccl-tests#290](https://github.com/NVIDIA/nccl-tests/issues/290)). PASS from 340 accepts a node without NVLink SHARP; FAIL under 185 is half the effective rate. H200 has the same NVLink and NVSwitch, hence the same values.
- NCCL all-reduce on 8 A100 SXM: about 230 GB/s at 8 GB, same source ([nccl-tests#149](https://github.com/NVIDIA/nccl-tests/issues/149)); PASS from 210, FAIL under 115.
- Bus bandwidth: `busbw = algbw × 2(n−1)/n` for an all-reduce, as defined in [nccl-tests' PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md).

- Measured on 8x H100 SXM, 2026-10-24 ([`examples/h100-8x-acceptance.txt`](examples/h100-8x-acceptance.txt), a production bare metal node, driver 595.112.07, PyTorch 2.14.1, NCCL 2.30.7): GPU pairs from 387 to 391 GB/s, median 389, that is 86% of the 450 GB/s line rate (NVIDIA's NCCL team puts effective NVLink at about 80% of it); NCCL bus bandwidth 478.5 GB/s, the NVLink SHARP figure above; 669 to 689 TFLOPS bf16 per GPU, at the power cap (up to 702 W) in 95% of the samples; SM clocks 1335 to 1380 MHz; hottest GPU 69 C for a median of 63.5 C; host-to-GPU copies 53.1 to 55.5 GB/s, Gen5 x16. From it, the P2P thresholds were raised: median pair WARN under 75% of the line rate (was 60%), one pair WARN under 90% of the median (was 80%). The NCCL PASS threshold stays at 340, so that a node without NVLink SHARP (about 370) passes.

The H100 SXM thresholds come from that run; one node is one sample, and further runs will be added to `examples/` with their date, any threshold change noted in the [changelog](CHANGELOG.md). Still provisional, from NVIDIA's figures only: H200 (same NVLink and NVSwitch, hence the same values), A100 SXM, and B200 and B300, which have a line rate from the datasheet but no NCCL reference yet (INFO).

### What it shows, and what it does not

It shows that during these ten minutes every GPU is visible and computes correctly at full load without thermal or power slowdown, that the NVLink mesh is complete and fast between every pair, that NCCL collectives reach the expected bandwidth, that no memory or Xid error appeared, and that the machine holds no obvious trace of a previous tenant. That covers what usually ruins a long job in its first hours: a GPU off NVLink, a link down, a card at x8, a GPU that overheats or computes wrong, ECC errors, a machine still holding someone else's files.

It does not show:

- that the machine will hold for days: failures that come after hours (an Xid 79 at hour 30, slowly growing ECC counts) need monitoring during the job;
- anything between machines: no InfiniBand or Ethernet test, only inside the node;
- storage, CPU or RAM performance;
- that the disk was wiped: the leftover search looks in the usual places only, deleted files can survive in unwiped blocks, and other volumes are not searched;
- anything against a host determined to lie: the driver reports what it is told to report (same limit as `hwcheck.sh`, see above);
- exact figures: bandwidths come from PyTorch (copy engines, the NCCL bundled with PyTorch) and can differ by a few percent from cuda-samples or nccl-tests.

### What it touches

Read-only, no account, nothing about the machine sent anywhere. It writes the three report files and a temporary directory in `$TMPDIR` (removed at exit, Ctrl-C included). When the image has no PyTorch that sees the GPUs, it installs a pinned build into that temporary directory from `download.pytorch.org`, about 3 GB, matched to the driver's CUDA version: 13.x → torch 2.14.1 (cu130), 12.8 and 12.9 → 2.11.0 (cu128), 12.0 to 12.7 → 2.14.1 (cu126), 11.x → 2.7.1 (cu118). That download is the only network traffic; pip sends its usual user agent (Python version, OS, CPU architecture), ignores any `pip.conf` and keeps no cache. `--no-install` forbids it. Root is best (kernel log, `/root`); as another user, the lines it could not check say so.

Written for Ubuntu 22.04 and 24.04 with the NVIDIA driver: bash, coreutils, awk, python3; `lspci` (pciutils) is optional.

## Tests

`bash tests/run.sh` checks both scripts without a GPU: fake `nvidia-smi`, `lspci`, `dmesg` and PyTorch replay the fixtures of [`tests/fixtures/`](tests/fixtures/) (real outputs of an 8x H100 SXM node and of an RTX 3060 Ti, and constructed ones, listed in its README) on a healthy node, a faulty one, a PCIe machine without NVLink, one without GPU, plus shellcheck. `AWK=mawk` or `AWK=gawk` chooses the awk. After editing `acceptance.sh`, put its new SHA-256 (`sha256sum acceptance.sh`) in `ACCEPTANCE_SHA256` at the top of `hwcheck.sh`; the tests check that it matches.

## License

MIT. Maintained by [VoltageGPU](https://voltagegpu.com); the canonical copies served at `https://voltagegpu.com/hwcheck.sh` and `https://voltagegpu.com/hwcheck-acceptance.sh` are `hwcheck.sh` and `acceptance.sh`. Changes: [CHANGELOG.md](CHANGELOG.md).
