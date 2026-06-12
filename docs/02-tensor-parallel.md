# 02 — Tensor-parallel (TP=2 / TP=4) across multiple B70 cards

Multi-GPU tensor-parallel on B70 was the hard blocker for months. It now works.
It requires **two independent fixes in series** — with only one, TP still fails,
just at a different layer.

## Prerequisites (both required)

1. **Kernel ≥ `7.1.0-rc6`** *masks* the defect (the OOM no longer reproduces even
   on the old container NEO 26.09, which still lacks the fix); **host NEO ≥
   `26.14`** additionally *carries the actual fix* (compute-runtime commit
   `028e23e576`). On the bare-metal stack here both are present. This is the
   compute-runtime multi-root Level-Zero USM host-pool defect that OOM'd TP at
   `xpu_worker.py:202` `init_device`. See `error-reports/multiroot-usm-oom.md`.
2. **The Triton `init_devices` fix** — the in-tree patch
   (`scripts/patch_triton_driver_noopenclsel.py` + `rm -rf ~/.triton/cache`), or a
   stack where triton-xpu ≥ 3.7.0's files actually land on disk. See
   `error-reports/triton-init-devices.md`.

Validated stack: kernel `7.1.0-rc6`/`-rc7`, host NEO `26.18.38308.1`, vLLM
`0.20.2`, torch `2.11.0+xpu`, pytorch-triton-xpu `3.6.0+git` (patched), oneAPI
`2025.3`.

> **Valid TP sizes are model-dependent.** For Gemma 4 31B (32 attention heads, 16
> KV heads) only **TP=2 and TP=4** divide evenly — `--tensor-parallel-size 3` is
> invalid and will fail. Pick a TP size that divides the model's KV-head count.

With only fix 1, TP loads the model on all cards then dies at the first cross-rank
collective with `No device of requested type available` (that's the Triton bug).
With only fix 2, TP dies even earlier with `OUT_OF_HOST_MEMORY` at init (that's the
USM bug). Apply both.

## Multi-root + oneCCL environment

On top of the base env from `01-baremetal-vllm.md`, TP needs the oneCCL transport
configured. This combination is validated:

```bash
export ZE_AFFINITY_MASK=0,1            # or 0,1,2,3 for TP=4 — list the real cards
export ONEAPI_DEVICE_SELECTOR='level_zero:*'
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export CCL_ENABLE_SYCL_KERNELS=0
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=pidfd
export FI_PROVIDER=shm
export CCL_ATL_SHM=1
export CCL_WORKER_COUNT=2               # = TP size
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export ZES_ENABLE_SYSMAN=1
export TRITON_INTEL_DEVICE_ARCH=bmg     # mandatory under spawn workers (auto-parse fails)
unset SYCL_CACHE_PERSISTENT
```

## Launch

```bash
vllm serve RedHatAI/gemma-4-31B-it-FP8-Dynamic \
  --tensor-parallel-size 2 \            # or 4
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \       # TP=2: 0.90  |  TP=4: 0.80 (0.90 crashes on KV alloc)
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --attention-backend TRITON_ATTN \
  --served-model-name gemma-4-31B-it \
  --host 127.0.0.1 --port 8003
```

`scripts/serve_gemma4-31b-fp8-tpN.sh` parameterizes TP size, card mask, window,
and gpu-util via env vars and applies the Triton patch if missing.

> Remember to `source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force` and activate
> the venv first (the wrapper does both). Without 2025.3 you get `icx not on PATH`
> or silent kernel failures (`01-baremetal-vllm.md`).

### The gpu-util trap

- **TP=2** is fine at `--gpu-memory-utilization 0.90`.
- **TP=4** must use **`0.80`** for FP8 (`0.90` crashes on the KV-cache allocation).
- For **int4** quant on TP=4, go lower still (~`0.65`) — int4 weights leave a
  larger KV pool that OOMs at higher util.

## Bring-up reference (Gemma 4 31B FP8, max-model-len 16384)

| Config | Cards | Weights/card | KV pool | Max concurrency |
|--------|-------|--------------|---------|-----------------|
| TP=2 @ util 0.90 | 0,1 | 16.56 GiB | 190,218 tok | 11.6× |
| TP=4 @ util 0.80 | 0–3 | 9.07 GiB  | 535,003 tok | 32.7× |

## Decision rule: TP=2 vs TP=4

Measured with llama-benchy (`pp=512 tg=128 depth={0,2048} concurrency={1,4,8}`):

| Scenario | Winner | Why |
|----------|--------|-----|
| Single user, short prompts | **TP=2** | 12.16 vs 10.17 tok/s single-stream (+20%). Each extra card boundary adds a PCIe sync per token; with no batch to amortize it, fewer cards win. |
| Many users / RAG / long context | **TP=4** | +10% aggregate decode at shallow depth, but **1.55× at deep context** (depth 2048 / conc 8: 14.93 vs 9.62 tok/s), −37% TTFT at depth, and a **2.8× larger KV pool** (535k vs 190k tok) → ~2.8× more simultaneous sessions before KV thrashing (max concurrency 32.65× vs 11.61×). |

Rule of thumb: **one user / short prompts → TP=2; many users / deep context → TP=4.**

> This inter-card-sync law is general on this stack — the same pattern appears on
> llama.cpp Vulkan layer-split, where single-request throughput is *inversely*
> proportional to card count. For a single request, the optimum is the **fewest
> cards that fit the model**; use the remaining cards for a second parallel
> instance rather than widening one.

## Operational gotchas for TP

- **Orphaned workers survive `kill -9`.** A `kill -9` / `pkill -f 'vllm serve'`
  does **not** match the TP workers (they carry a distinct setproctitle:
  `VLLM::EngineCore`, `VLLM::Worker_TP*`). Orphans squat ~27–30 GB per card, and
  the **next** launch then startup-OOMs with `Free memory on device xpu:0 (5.0/31.89
  GiB)`. Full teardown:
  ```bash
  pkill -9 -f 'vllm serve'; pkill -9 -f 'VLLM::EngineCore'; pkill -9 -f 'VLLM::Worker'
  pkill -9 -f 'multiprocessing.resource_tracker'
  ```
  (Run this from a script file, not interactively — `pkill -f` can self-match.)
- **Card topology / the desktop card.** The validated TP=4 run used all four
  cards — including the card driving the KDE/Wayland desktop — with correct
  output and tool calling. Just leave headroom on the display card: it already
  holds ~1–1.5 GB for the compositor, so a too-high `--gpu-memory-utilization` is
  what bites (as `UR_RESULT_ERROR_OUT_OF_RESOURCES` on KV allocation — see the
  gpu-util trap above), not the desktop card per se.
- **Don't co-serve two multi-root TP instances.** Two concurrent TP serves trip
  `UR_RESULT_ERROR_OUT_OF_RESOURCES` (Level-Zero error 40). Run them sequentially,
  or split the cards into single-card instances.

## MoE models: mostly work; one specific model is blocked

MoE is **not** universally blocked on this stack. **Qwen3-30B-A3B** (FP8) is an
MoE and is the model the whole TP fix was validated on — its FP8-MoE path runs
through the same Triton kernels the `init_devices` patch enables (container-proven;
online `--quantization fp8` needs TP≥2 because it materializes bf16 first).

The one MoE that is **blocked** is **Gemma 4 26B-A4B**: its Triton `fused_moe`
kernel fails to build on the bare-metal toolchain with
`ZE_RESULT_ERROR_MODULE_BUILD_FAILURE` — a *separate* defect from the two TP
fixes. Tracked as open; for that specific model, use a dense model or the
container path.
