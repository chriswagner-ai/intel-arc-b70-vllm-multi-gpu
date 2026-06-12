# 01 — Bare-metal vLLM on B70 (the coupled wheel set)

This is the "no Docker" path: vLLM running directly in a Python 3.12 venv against
torch+xpu. It is what gives Gemma 4 31B + native tool calling + Open WebUI on a
single B70, and is the base for the tensor-parallel path (`02-tensor-parallel.md`).

> Prerequisite: the host GPU **driver + runtime** must already be installed
> (compute-runtime / Level Zero / oneAPI). That is a host install, **not** a
> container — see `00-system-setup.md` §"Userspace GPU runtime". No Docker image
> and no init/bootstrap script is needed to bring up the GPU; the Intel container
> images that appear in this project's history were only for TP *debugging*.

> **Read this before you `pip install` anything:** the XPU stack is a *coupled
> nightly wheel set*. torch+xpu, pytorch-triton-xpu, vllm, and vllm-xpu-kernels
> are version-matched. `pip install -U vllm` will pull CUDA wheels and destroy the
> +xpu build. A real upgrade is a matched rebuild of all four, not an in-place
> bump. (Why these exact versions, and what breaks if you move them, is in
> `05-why-these-versions.md`.)

## The validated wheel set (Python 3.12 venv)

These exact versions are known-good together on the baseline stack:

```
vllm                    0.20.2          (0.20.2rc1.dev345+g768f4a6f2)
torch                   2.11.0+xpu
torchvision             0.26.0+xpu
torchaudio              2.11.0+xpu
pytorch-triton-xpu      3.6.0+git225cdbde   (provides the driver.c that needs the patch — see below)
triton-xpu              3.7.0               (co-installed, but 3.6.0's files win on disk — see below)
vllm-xpu-kernels        0.1.8
oneccl / oneccl-devel   2021.17.2
transformers            5.8.1
intel-*-rt (cmplr/sycl/opencl/openmp)   2025.3.2
intel-pti               0.16.0
```

Note the split: the Intel runtime wheels are the **2025.3** series. That is not an
accident — see the oneAPI note below.

### Where the non-PyPI wheels come from

This set is **not** reconstructable from `pip install vllm` — several pieces are
off-index nightlies:

- **torch / torchvision / torchaudio `+xpu`** and **pytorch-triton-xpu
  `3.6.0+git…`** come from Intel's XPU nightly index, not PyPI (the clean public
  xpu index tops out at an older triton). Install torch+xpu first and let it pull
  its matched `pytorch-triton-xpu`.
- **vllm-xpu-kernels 0.1.8** is a GitHub release wheel:
  `https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.8/…manylinux…whl`.
- **vllm 0.20.2** is installed `--no-deps` against the above so it does not drag in
  CUDA wheels.

Because it is hand-assembled, **snapshot it** after it works: a `pip freeze`
manifest plus a filesystem-level copy of the venv. See
`05-why-these-versions.md` for the pip-freeze + backup discipline.

## The oneAPI 2025.3 requirement (critical, easy to miss)

The torch+xpu wheels in this set were built against **oneAPI 2025.3**, even though
the host system default is oneAPI 2026.0. Run the venv against the **matching
2025.3** toolchain to avoid a SYCL-runtime ABI mismatch — a known footgun, because
high-level checks (`sycl-ls`, `torch.xpu.device_count()`) still report success, so
a mismatch tends to fail late and confusingly. Always source 2025.3 first, with
`--force` to override the host environment:

```bash
source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force
command -v icx   # should now resolve to the 2025.3 toolchain
```

Install both oneAPI versions side by side; 2025.3 is for running vLLM, 2026.0 is
the host default for everything else.

## The Triton fix (required even single-card if any Triton kernel runs)

pytorch-triton-xpu 3.6.0's `init_devices()` has an OpenCL-selector block that
throws on Battlemage. Patch it once (idempotent):

```bash
python3 scripts/patch_triton_driver_noopenclsel.py \
  <venv>/lib/python3.12/site-packages/triton/backends/intel/driver.c
rm -rf ~/.triton/cache      # force the compiled spirv_utils.so to rebuild
```

Full root-cause writeup: `error-reports/triton-init-devices.md`.

> **About "just pin triton-xpu 3.7.0 and skip the patch":** it does not work on
> *this* wheel set. Although `triton-xpu 3.7.0` is co-installed, the
> `pytorch-triton-xpu 3.6.0` files are what actually land in
> `triton/backends/intel/driver.c` (the installed `triton.__version__` reports
> `3.6.0` and `driver.c` keeps the OpenCL block). So the patch **is required and
> applied here.** "Pin 3.7.0 → no patch" only holds on a stack where 3.7.0's files
> overwrite 3.6.0's on disk — which they do not in this venv.

> The patch is **ephemeral**: any fresh venv or triton reinstall loses it and you
> hit the `profile_run` throw again. The serve wrapper re-applies it
> automatically; if you hand-run `vllm serve`, re-apply it yourself first.

## Environment for a B70 vLLM launch

The serve scripts export the full set below. It is a superset that works for both
single-card and TP:

```bash
export ONEAPI_DEVICE_SELECTOR='level_zero:*'
export SYCL_UR_USE_LEVEL_ZERO_V2=0           # V1 UR adapter; V2 (=1) throws UR_RESULT_ERROR_UNKNOWN in xpu device_count on multi-root
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1   # for KV cache > 4 GiB
export HF_HUB_OFFLINE=1                       # use local HF cache for gated tokenizers (e.g. gemma)
unset SYCL_CACHE_PERSISTENT                   # NEVER set this on BMG (poisons the cache across restarts)
# --- the following are strictly required only for TP (multi-card), harmless single-card ---
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZES_ENABLE_SYSMAN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TRITON_INTEL_DEVICE_ARCH=bmg          # under spawn TP workers, arch auto-parse returns 'unknown' -> empty ocloc -device -> ZEBIN error
```

Pin **one** card for a single-card serve with `ZE_AFFINITY_MASK=0` (or
equivalently `ONEAPI_DEVICE_SELECTOR=level_zero:<n>`). For TP, the full oneCCL
block and the card mask are in `02-tensor-parallel.md`.

> Minimal single-card set: you only strictly need `SYCL_UR_USE_LEVEL_ZERO_V2=0`,
> `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`, `HF_HUB_OFFLINE=1`,
> `unset SYCL_CACHE_PERSISTENT`, and a single-card pin. `TRITON_INTEL_DEVICE_ARCH=bmg`
> and `VLLM_WORKER_MULTIPROC_METHOD=spawn` are only *required* under TP — the
> single-card GEMM path auto-detects the arch — but setting them is harmless.

## Quick start — Gemma 4 31B, single card, with tool calling

```bash
# int4-AutoRound is the validated single-card model (~18 GiB weights fit a 32 GB card)
ZE_AFFINITY_MASK=0 vllm serve Intel/gemma-4-31B-it-int4-AutoRound \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --attention-backend TRITON_ATTN \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --served-model-name gemma-4-31B-it \
  --host 127.0.0.1 --port 8003
```

**Why int4 for single-card:** it is the only Gemma 4 31B path validated at TP=1
on a 32 GB B70 (~18 GiB weights, serves at util 0.90; ~20% faster than FP8 at
matched TP — see `06-benchmarks.md`).
`--enforce-eager` is **required** for int4 (W4A16 on XPU). The **FP8-Dynamic**
checkpoint (`RedHatAI/gemma-4-31B-it-FP8-Dynamic`) is ~31 GiB of pre-compressed
weights — it does **not** fit a single 32 GB card with any KV slack, so run FP8 on
**TP≥2** instead (see `02-tensor-parallel.md`). See `04-quant-formats.md` for the
format trade-offs.

Smoke-test (prove the kernels actually run, not just that the endpoint is up):

```bash
# 1-token generation smoke
curl -s localhost:8003/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-31B-it","prompt":"2+2=","max_tokens":3}'

# tool-calling smoke
curl -s localhost:8003/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"gemma-4-31B-it",
  "messages":[{"role":"user","content":"Weather in Tokyo? Use the tool."}],
  "tools":[{"type":"function","function":{"name":"get_weather",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "tool_choice":"auto"}'
```

A ready-to-run wrapper that sets all of the above (and applies the Triton patch if
missing) is in `scripts/serve_gemma4-31b-fp8-tpN.sh`.

## Performance reference

Measured with **llama-benchy** (warmup + multi-run; depth 0, concurrency 1) — the
honest decode rate, not a single-shot curl:

- **Gemma 4 31B int4-AutoRound, single B70 (TP=1): ~16.5 tok/s** single-stream
  decode. (An early single-shot curl read ~21.8 tok/s, but the rigorous
  warmup+multi-run rate is ~15–16.5 — use that.)
- **Gemma 4 31B FP8: ~12.2 tok/s** single-stream, measured on **TP=2** (two cards,
  llama-benchy depth 0/conc 1). The ~12.2 tok/s figure is the **TP=2** rate. FP8
  single-card (TP=1) was **not** benchmarked and is too tight to serve with KV
  slack on a 32 GB card (~31 GiB weights) — FP8 is a **TP≥2** model here (see above
  and `04-quant-formats.md`).

See `04-quant-formats.md` for why int4-AutoRound is ~20% faster than FP8 here, and
`02-tensor-parallel.md` for the full TP=2/TP=4 matrix.
