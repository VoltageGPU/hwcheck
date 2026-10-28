# PantheonGPU load tests on an 8x H100 SXM node (28 October 2026)

Raw JSON reports from three PantheonGPU tests (pantheon-gpu 1.2.4), run back to back on all
eight GPUs of a bare-metal 8x H100 SXM node, 150 s each, after `hwcheck` passed on the same
node (see `../h100-8x-acceptance.txt` for an earlier acceptance run on this node type):

```
pantheon --test omni_virus --gpu all --duration 150
pantheon --test transformer_train_step --gpu all --duration 150
pantheon --test memory_hammer --gpu all --duration 150
```

Node: Supermicro SYS-821GE-TNHR, 2x Xeon Platinum 8558, Ubuntu 24.04, NVIDIA driver
595.112.07, CUDA 13.4.

| Test | Node median | Range over 8 GPUs |
|---|---|---|
| omni_virus | 399.7 TFLOPS | 392.9 to 426.1 |
| transformer_train_step | 2.85e10 ai-ops/s | 2.826e10 to 2.869e10 |
| memory_hammer | 4.78e11 aggressor reads/s | 4.653e11 to 4.817e11 |

Verdict: HEALTHY on all 8 GPUs for all three tests; the largest gap to the node median is
-2.7 % (memory_hammer, GPU 3).

GPU UUIDs and serial numbers are replaced by `GPU-redacted` and `redacted`; nothing else in the
reports is changed.
