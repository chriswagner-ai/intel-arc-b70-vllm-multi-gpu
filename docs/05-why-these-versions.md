# 05 — Why this stack is pinned (and what regressed)

This page is the *reasoning layer*. The other docs tell you **what** to install;
this one tells you **why** each pin exists, so you (human or AI) don't "helpfully
upgrade" into a regression that already cost this project days to diagnose.

> One-line rule: **on B70, newer is not safer.** The working stack is a
> hand-assembled set of version-matched nightlies. Every pin below is load-bearing.

## Why bare-metal vLLM (not Intel's container image)

Intel ships container images (`intel/llm-scaler-vllm:0.14`, `intel/vllm:0.17`) —
the obvious, easier starting point. **They are not broken for TP**: the
`intel/vllm:0.17` image served TP=2 and TP=4 on this stack. So this is a
*trade-off*, not a "the container can't do it" situation. We went bare-metal for
three real reasons:

- **Newer vLLM.** `0.20.2` brings the XPU code paths and `vllm-xpu-kernels 0.1.8`
  that make Gemma 4, int4-AutoRound, and TurboQuant usable on Battlemage — the
  container's `0.14`/`0.17` lines lack these.
- **Control of the GPU runtime.** The containers pin an old NEO
  (`26.09.37435.12`) that *lacks* the multi-root USM fix (`028e23e576`, first in
  NEO 26.14). That NEO only OOMs TP on **kernels older than `7.1.0-rc6`** — a
  recent kernel **masks** the defect, which is why the container does TP anyway
  (case-file Finding 55). Bare-metal runs against the **host** NEO
  (`26.18.38308.1`), which *carries* the fix natively — so you are not depending on
  the kernel to paper over an old runtime. (`intel/llm-scaler-vllm:1.4` is a
  byte-identical re-tag of `0.14` — same old NEO.)
- **Host integration.** No Docker layer between vLLM and host tooling (Open WebUI).

**When the container is the better call:** single-card use, wanting a reproducible
pinned image instead of the hand-built (fragile) wheel set, or the one model
bare-metal can't serve yet (the Gemma 4 26B-A4B MoE, `fused_moe` build failure).
The honest summary: bare-metal is the more powerful, more fiddly path; the
container is the easier, official one that also works. Pick by need.

## Why vLLM 0.20.2 specifically

- It is recent enough to ship with **`vllm-xpu-kernels` 0.1.8** (the separate XPU
  operator package) and the XPU code paths that make Gemma 4, int4-AutoRound, and
  TurboQuant-style fp8-KV usable on Battlemage — capabilities the older 0.14/0.17
  container lines lack.
- It is *not* upgraded past 0.20.2 because the XPU wheel set is coupled (below) and
  a bump breaks it.

## The coupled wheel set — pin as a whole, never `pip -U`

`torch 2.11.0+xpu`, `pytorch-triton-xpu 3.6.0+git`, `vllm 0.20.2`, and
`vllm-xpu-kernels 0.1.8` are a **version-matched nightly set**. They must move
together or not at all. Concretely, `pip install -U vllm` (0.20.2 → 0.22.x) against
the default index pulls the **CUDA** build — `nvidia-cuda-runtime/nvcc/nvrtc`,
`nvidia-cutlass-dsl[cu13]`, `flashinfer-python`, and a generic CUDA `triton 3.7.0`
— all of which **replace and break** the `+xpu` build. A real upgrade is a matched
rebuild of all four against the XPU index, not a pip bump.

**Discipline that makes this safe to live with:**

- Snapshot a `pip freeze` manifest of the known-good set *before* touching it.
- Keep a filesystem-level copy of the venv (a reflink/`cp` backup) so a botched
  upgrade is a one-command rollback.
- The environment is deliberately **not** committed to git — it is a fragile build
  artifact, not a reproducible-from-requirements set.

## Why oneAPI 2025.3 at runtime (decoupled from the host's 2026.0)

The `torch+xpu` wheels were built against **oneAPI 2025.3**. The host default was
later upgraded to **2026.0**, which shifts the SYCL-runtime ABI. To avoid running
the venv against a mismatched toolchain, install 2025.3 *alongside* 2026.0 and have
every serve wrapper `source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force`; the
rest of the system stays on 2026.0. This is a **build-provenance precaution** — and
worth being precise about: the one `profile_run` failure actually observed on this
stack ("No device of requested type available") was traced to **triton-xpu 3.6.0**,
*not* the oneAPI ABI (see regression-log row 2). Do not "upgrade the venv to
2026.0" without a matched torch+xpu rebuild.

## Regression log

Real events that produced the pins above. Each is grounded in this project's
experiment records / script headers; host-specific ones are flagged.

| Date | What broke | Trigger | Resolution / pin |
|------|-----------|---------|------------------|
| 2026-05-17 | (precautionary pin — no independently confirmed kernel failure) | host oneAPI upgraded 2025.3 → 2026.0; the venv's torch+xpu was built against 2025.3, so run it against a parallel 2025.3 to avoid SYCL-runtime ABI skew | install 2025.3 in parallel; serve wrappers `source …2025.3… --force` |
| 2026-05-25 → 06-04 | Gemma 4 31B died at `profile_run` "No device of requested type available"; first **mis**-attributed to the oneAPI ABI skew | actual cause: triton-xpu 3.6.0 `init_devices` OpenCL selector | patch `driver.c` + `rm -rf ~/.triton/cache`; oneAPI-ABI hypothesis falsified |
| 2026-06-04 (TP bring-up) | TP serves, then first request errors with empty `ocloc -device` / ZEBIN `stoul` | Triton arch auto-parse returns `unknown` under spawn workers | export `TRITON_INTEL_DEVICE_ARCH=bmg` (TP only) |
| 2026-06-04 | TP=4 crashes on KV allocation at `--gpu-memory-utilization 0.90` | util too high for the 4-card KV plan (worse for int4's tiny weights) | TP=4 → 0.80 (FP8) / 0.65 (int4); TP=2 stays 0.90 |
| pre-7.1.0-rc6 → 06-04 | TP≥2 OOM-2 MiB at `xpu_worker.py:202` the moment ≥2 root devices are visible | container NEO 26.09 lacks USM fix `028e23e576` | kernel ≥ 7.1.0-rc6 masks it; host NEO ≥ 26.14 carries the fix |
| 2026-06-04/10 | next vLLM launch startup-OOMs (`5.0/31.89 GiB`) | orphan `VLLM::EngineCore` / `VLLM::Worker_TP*` survive `kill -9` (distinct setproctitle) | also `pkill -9 -f 'VLLM::EngineCore'`/`'VLLM::Worker'` (see `02`) |
| 2026-06-04 | `pip install -U vllm` would replace the +xpu build with a CUDA build | naive `pip -U` against the default (PyPI/CUDA) index | not upgraded; pinned + pip-freeze snapshot + venv backup |
| 2026-06-10 | AWQ / GPTQ / NVFP4 checkpoints fail to load on XPU | running NVIDIA-format quants on the Intel build | use FP8 or int4-AutoRound (see `04`) |
| 2026-05-22 (llama.cpp, related) | SYCL layer-split across ≥2 cards in one process throws `UR_RESULT_ERROR_UNKNOWN` | V2 UR adapter sets all visible L0 devices "resident" (same N=2 boundary, different layer); `028e23e576` does **not** cure it | use Vulkan for multi-card llama.cpp; per-card fleet for SYCL |

## The meta-lesson

Almost every pin here traces to the same root tension: **B70 is new, so the
software stack is a moving target of nightly builds where the pieces are only
mutually compatible at specific versions.** The forensic record exists because, on
this hardware, "update to fix it" is more often the *cause* of the next failure
than the cure. When in doubt, reproduce the pinned state first, then change one
thing at a time with a rollback ready.
