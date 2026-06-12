# Error report: triton-xpu `init_devices` OpenCL-selector throws on BMG multi-root

**Status:** ROOT-CAUSED and FIXED. Made fail-safe upstream and shipped in the
triton-xpu 3.7.0 line; patchable on 3.6.0.
**Component:** `intel/intel-xpu-backend-for-triton` (pytorch-triton-xpu 3.6.0).
**Affected:** vLLM tensor-parallel (and any Triton kernel) on Intel Arc Pro B70
(BMG-G31) when ≥2 root devices are visible to one process.

> **Joint requirement:** this Triton fix alone does **not** make TP work. TP≥2
> also needs the kernel/NEO fix (`multiroot-usm-oom.md`). With a good kernel but no
> Triton fix you hit the throw below; with the Triton fix but an old kernel you hit
> the USM OOM first. Apply both.
>
> **The in-place patch is ephemeral** — a fresh venv/container or a triton
> reinstall loses it and the throw returns. Re-apply after any such change (the
> serve wrappers do this automatically).

## Symptom

`vllm serve` with `--tensor-parallel-size ≥ 2` loads the model on all cards, then
dies at the first cross-rank collective — inside `profile_run` /
`determine_available_memory` — with an uncaught native exception:

```
RuntimeError: No device of requested type available
  ... sycl::_V1::exception ...
```

(`std::terminate`, not a clean Python traceback.) It fires because vLLM's
FP8-MoE / Triton-attention path runs a Triton kernel whose device-init executes in
the first forward pass.

## Root cause (pinned via gdb-oneapi backtrace)

`spirv_utils.init_devices()` in triton-xpu's `driver.c` contains a *"workaround to
get opencl extensions"* block. For each Level-Zero device it constructs a
`sycl::device(selector)` that requires an **OpenCL-backend** device whose name
matches the Level-Zero B70 *by name*:

```cpp
// workaround to get opencl extensions
const auto &name = sycl_devices[i].get_info<sycl::info::device::name>();
sycl::device opencl_device([&](const sycl::device &dev) -> int {
  return (dev.get_backend() == sycl::backend::opencl &&
          dev.get_info<sycl::info::device::name>() == name)
             ? 1 : -1;
});
sycl_opencl_device_list.push_back(opencl_device);
```

When the OpenCL ICD is CPU-only (or no name-matching OpenCL device exists for the
B70), `select_device(predicate)` rejects all GPUs and throws "No device of
requested type available". The throw is uncaught → terminate.

This is a **separate defect** from the compute-runtime multi-root USM OOM (see
`multiroot-usm-oom.md`). UR / Level-Zero enumeration is healthy here — the failure
is purely the SYCL selector above UR. Stack-independent: reproduced across vLLM
0.14/0.17/0.20, oneCCL 2021.15/2021.17, NEO 26.09/25.48/26.18 — all bundling the
same triton-xpu 3.6.0.

## Fix

**Upstream:** the throwing path was made fail-safe in
**[PR #5745](https://github.com/intel/intel-xpu-backend-for-triton/pull/5745)**
("Use ocloc for querying OpenCL extensions if OpenCL backend isn't available",
commit `e2086237ee43`, merged 2025-12-25): instead of constructing a device
selector that throws when no OpenCL B70 exists, it falls back to `ocloc`. This is
in the 3.7.0 line. (A related multi-GPU vLLM-startup hardening landed in
[PR #6767](https://github.com/intel/intel-xpu-backend-for-triton/pull/6767).)
Note the nuance: the block was not simply *deleted* — it was *guarded* so it
degrades safely. **On this wheel set, pinning 3.7.0 is not enough** because
3.6.0's `driver.c` still wins on disk (see `01-baremetal-vllm.md`) — so patch it.

**On 3.6.0:** wrap the block in `try/catch(sycl::exception)` so a missing OpenCL
device degrades safely (`has_opencl_extension()` already guards on list size and
returns false for a short list). Apply with the included patcher:

```bash
python3 scripts/patch_triton_driver_noopenclsel.py \
  <venv>/lib/python3.12/site-packages/triton/backends/intel/driver.c
rm -rf ~/.triton/cache     # force spirv_utils.so to recompile
```

The patcher is idempotent and refuses to double-patch. After patching + cache
clear, TP=2 and TP=4 serve correctly (validated on Qwen3-30B-A3B FP8 and Gemma 4
31B FP8).

## A second Triton wall on bare-metal TP: empty `ocloc -device`

After the `init_devices` patch, bare-metal TP hits a *second*, distinct Triton
issue: under `VLLM_WORKER_MULTIPROC_METHOD=spawn` workers, Triton's device-arch
auto-parse returns `unknown`, so `ocloc` is invoked with an **empty** `-device`
arg (`-device  -options …`) → a `stoul`/ZEBIN codegen error on the first forward.
The model serves, then dies the moment it is queried. Fix: export
**`TRITON_INTEL_DEVICE_ARCH=bmg`** (Triton maps `intel_gpu_bmg_g21` → `bmg`).
Single-card never hits this because its GEMM path auto-detects the arch. This is
separate from the opencl-selector defect above, but a reader following only this
report for bare-metal TP would still fail without it.

## Falsified workarounds

- `OCL_ICD_FILENAMES` pointed at the GPU ICD alone — did not cure it.
- Intel's documented XPU-TP oneCCL vars (`CCL_ZE_IPC_EXCHANGE=sockets`,
  `FI_PROVIDER=shm`, `CCL_ATL_SHM=1`, `CCL_WORKER_COUNT=N`) — transport inits
  cleanly but the SYCL throw still fires later. (These vars are still set in the
  working config; they just don't address this defect.)
