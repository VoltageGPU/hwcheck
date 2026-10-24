# Changelog

## 1.1 (2026-10-24)

- New `acceptance.sh`: a 10-minute acceptance test of a multi-GPU machine before a long job (NVLink mesh, NVSwitch, fabric state, PCIe links, P2P copies, NCCL all-reduce, 5-minute load with thermal, power and clock samples, ECC, Xid, leftovers of a previous tenant, SSH keys, virtualization), with a PASS, WARN or FAIL verdict per measure and a text, JSON and SHA-256 report. Thresholds and their sources: README.md. Run on a production 8x H100 SXM bare metal node on 2026-10-24 (`examples/h100-8x-acceptance.txt`, `RESULT: PASS`); its measurements raised the P2P thresholds (median pair WARN under 75% of the NVLink line rate instead of 60%, one pair WARN under 90% of the median instead of 80%). H200, A100, B200 and B300 values are still provisional.
- `hwcheck.sh --acceptance` (first option) runs it, and only the copy whose SHA-256 `hwcheck.sh` pins.
- Tests without a GPU: `tests/run.sh`, with fake `nvidia-smi`, `lspci`, `dmesg` and PyTorch replaying `tests/fixtures/`.
- `.gitattributes`: the scripts keep LF line endings in a Windows checkout.

## 1.0 (2026-10-16)

- First release: GPU model, GPU count, VRAM, a CUDA matmul on every GPU, CPU, RAM, disk and download speed, compared with the listing.
