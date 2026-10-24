# Test fixtures

What the fake `nvidia-smi`, `lspci`, `dmesg` and PyTorch of `tests/fake/` replay. Real captures are copied as they are (tabs and escape codes included); everything else is written by [`build.sh`](build.sh), in the format of the real outputs, with typical values that are not measurements.

| File | Origin |
|---|---|
| `help-query-gpu.txt` | real: `nvidia-smi --help-query-gpu`, driver 610.88, RTX 3060 Ti, WSL2 Ubuntu 24.04, 24 October 2026. The fake rejects any field it does not list, as the real tool does |
| `rtx3060ti-1x/topo.txt`, `header.txt`, `q-ecc.txt` | real, same machine and day |
| `rtx3060ti-1x/L.txt` | real, same machine, UUID zeroed |
| `rtx3060ti-1x/gpus.csv` | real values of `nvidia-smi --query-gpu` on that machine, put in one file by `build.sh` (UUID zeroed) |
| `h100-8x/topo.txt`, `h100-8x/lspci.txt` | real: `nvidia-smi topo -m` and `lspci -nn` NVIDIA lines of an 8x H100 SXM virtual machine, 10 September 2026 (4 NVSwitch visible) |
| `h100-8x/*` (the rest) | constructed: a healthy node; driver 580.173.02 and CUDA 13.0 as seen on a real H100 on 17 October 2026 |
| `h100-8x-sick/*` | constructed from `h100-8x`: GPU 7 off NVLink (SYS), one NVSwitch missing, GPU 2 at x8, GPU 5 at Gen4 and hot with hardware thermal slowdown, GPU 0 at a low clock, GPU 4 capped at 500 W and at half throughput, GPU 6 with uncorrected ECC errors and changing results, GPU 1 with a pending row remap, GPU 7 fabric in progress, a process on GPU 3, Xid 79 during the test |
| `pcie-8x/*` | constructed: 8 PCIe GPUs on two sockets, no NVLink |

`gpus.csv` has one column per `nvidia-smi` field; while a fake PyTorch script "runs", the columns of `gpus-load.csv` replace them (GPUs under load). In `dmesg.txt`, `@NOW@` becomes the current uptime: that line falls during the test. The files under `py/` are what `acceptance.sh`'s own PyTorch scripts would print.
