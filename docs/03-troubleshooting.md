# 03 — Troubleshooting

Start with the symptom → fix table in `AGENTS.md`. This page adds the diagnostic
commands and the longer-form gotchas behind that table.

## Diagnostic commands

```bash
# Driver binding — every card should say "Kernel driver in use: xe"
lspci -nnk -d 8086:e223

# Module loaded
lsmod | grep '^xe'

# Kernel version (>= 7.1.0-rc6 needed for TP)
uname -r

# Live link state / utilization (the reliable telemetry on BMG)
intel_gpu_top

# Intel discovery (use xpumcli, NOT xpu-smi, on this stack)
xpumcli discovery

# vLLM version actually importable in the venv
# (activate the venv AND source oneAPI 2025.3 first, or this hits system python / fails to import)
source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force >/dev/null
python -c 'import vllm; print(vllm.__version__)'
```

## Forensics for any crash

Save these together so a later session (human or AI) can diagnose without rerunning:

- the full stderr/stdout of the failing process
- `dmesg` slice from the crash window (note: `kernel.dmesg_restrict=1` on many
  systems blocks non-root `dmesg`; use `sudo dmesg` or set
  `kernel.dmesg_restrict=0`)
- `sycl-ls` output
- `xpumcli discovery` snapshot

## Gotchas (full list)

1. **`GGML_SYCL_DISABLE_OPT=1` is mandatory on B70** (llama.cpp). Without it,
   llama-server SEGVs at slot-init on MoE and corrupts output on dense. Costs ~5%
   on dense; non-negotiable on MoE. (llama.cpp issues #21893, #15580.)
2. **Never set `SYCL_CACHE_PERSISTENT=1`.** Cross-restart kernel-cache persistence
   poisons the cache on BMG. Let JIT recompile each load (~30s warm cost).
3. **`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` for KV > 4 GiB.** Level Zero caps
   single allocations at 4 GB by default; required for large contexts on 30B-class
   models.
4. **`lspci` x1 Gen1 at idle is a display quirk.** Trains to x16 Gen5 under load.
   Verify with `intel_gpu_top`, not idle `lspci`.
5. **`xpumcli` PCIe / Xe-Link telemetry shows N/A on BMG-G31.** Gap in Intel's
   tooling. Use `intel_gpu_top` for live link state.
6. **`GGML_SYCL_DEVICE_ARCH=bmg_g21` for B70 llama.cpp builds.** The card is
   BMG-G31 but the SYCL arch id is `bmg_g21` (different naming convention). For
   vLLM/Triton the value is `TRITON_INTEL_DEVICE_ARCH=bmg`.
7. **TP needs BOTH the kernel/NEO fix AND the Triton fix.** See
   `error-reports/multiroot-usm-oom.md` + `error-reports/triton-init-devices.md`.
   One alone is not enough; they fail at different layers.
8. **vLLM's torch+xpu is built against oneAPI 2025.3, not the host 2026.0.**
   `source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force` before launching, or
   kernels fail at `profile_run`.
9. **TP=4 FP8 needs `--gpu-memory-utilization 0.80`** (0.90 crashes on KV alloc).
   TP=2 is fine at 0.90. int4 on TP=4 needs ~0.65.
10. **Serialize multi-card fleet launches.** Concurrent vLLM shard-loads stack
    host RSS and OOM-kill the desktop. Start card N, poll `/v1/models` until 200,
    then start N+1. Build readiness-gating into any orchestrator from day 1.
11. **`SYCL_UR_USE_LEVEL_ZERO_V2=0` (use the V1 UR adapter).** On the vLLM venv
    (oneAPI 2025.3) V1 is the working multi-root path and TP serves. The V2
    adapter (`=1`) breaks earlier — `torch._C._xpu_getDeviceCount()` throws
    `UR_RESULT_ERROR_UNKNOWN`. (Separately, on the *host* oneAPI 2026.0 toolchain
    V1 itself enumerates 0 devices, so this flag is specific to the 2025.3 venv.)
12. **The Gemma 4 26B-A4B MoE hits `ZE_RESULT_ERROR_MODULE_BUILD_FAILURE`** when
    its Triton `fused_moe` kernel builds on the bare-metal toolchain. This is
    **model-specific** — MoE is not universally blocked (Qwen3-30B-A3B FP8 is the
    canonical proven MoE; the model behind the TP fix). Open issue; for 26B-A4B
    specifically, use a dense model or the container path.
13. **Don't run NVIDIA-format quants (AWQ / GPTQ / NVFP4) on B70** — their dequant
    kernels are CUDA-only. Use FP8 or int4-AutoRound. See `04-quant-formats.md`.
14. **"Served" ≠ "working."** The endpoint answering `/v1/models` only proves the
    process is up. Always run a 1-token generation smoke test to prove the kernels
    actually execute.
15. **Orphaned vLLM workers survive `kill -9` and OOM the next launch.** `pkill -f
    'vllm serve'` does not match `VLLM::EngineCore` / `VLLM::Worker_TP*` (distinct
    setproctitle); orphans hold ~27–30 GB/card. Full teardown: also
    `pkill -9 -f 'VLLM::EngineCore'` and `pkill -9 -f 'VLLM::Worker'`. See
    `02-tensor-parallel.md` §Operational gotchas.
16. **`int4-AutoRound` requires `--enforce-eager`.** W4A16 on XPU will not run
    without it. See `04-quant-formats.md`.
