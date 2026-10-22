# hwcheck

One read-only command that checks whether the GPU machine you rented matches its listing. Works on any Linux box with an NVIDIA GPU, on any provider.

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

Nothing about the machine is sent anywhere. It is ~150 lines of bash: read it before you run it, as you should with any piped script.

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

## License

MIT. Maintained by [VoltageGPU](https://voltagegpu.com); the canonical copy served at `https://voltagegpu.com/hwcheck.sh` is this file.
