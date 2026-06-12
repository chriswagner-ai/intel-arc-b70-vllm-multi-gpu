# Intel Arc Pro B70 (Battlemage) — bare-metal vLLM, the working configuration
<img width="1760" height="982" alt="E_1760_webp_q90" src="https://github.com/user-attachments/assets/93e48284-4880-4a48-bb57-2f22012bc773" />

Field-tested record of running modern LLMs — including **Gemma 4 31B** with native
**tool calling**, and **tensor-parallel across 4 GPUs** — on a 4× Intel Arc Pro
B70 (BMG-G31 / Xe2, `8086:e223`) workstation, via **bare-metal vLLM**.

These cards are new and the ML stack is young. Most of this repo is the set of
non-obvious fixes that took long forensic debugging to find — written so both
humans and AI coding assistants can act on them directly.

> **For AI assistants:** read [`AGENTS.md`](AGENTS.md) first. It has a
> symptom → fix lookup table and the two load-bearing fixes. Don't apply generic
> "Intel GPU" advice here — several fixes on Battlemage are the *opposite* of the
> common guidance.

> **Tip:** install + authenticate the GitHub CLI (`gh auth login`) before letting
> an AI assistant help. It can then verify upstream fixes directly from git
> history (compute-runtime, triton-xpu, the kernel) instead of guessing from web
> search — the grounding is far more reliable. See `AGENTS.md`.

## What "bare-metal" means here (and the Intel container alternative)

This guide runs vLLM **bare-metal** — directly in a host Python venv against the
host's GPU driver/runtime — **not** inside a container. The alternative is Intel's
official **Docker images** (`intel/llm-scaler-vllm`, `intel/vllm`), which bundle
vLLM + the XPU runtime and are the easier, reproducible, "blessed" starting point.

**Both can do tensor-parallel on this hardware** — the Intel `intel/vllm:0.17`
image served TP=2/TP=4 here too. We chose bare-metal for three reasons:

- **Newer vLLM** (`0.20.2`) → Gemma 4 31B, int4-AutoRound, and TurboQuant KV — XPU
  capabilities the container's older `0.14`/`0.17` lines lack.
- **Control of the GPU runtime** — the host NEO carries the multi-root USM fix
  (`028e23e576`) natively, rather than *relying* on a recent kernel to **mask** the
  older NEO the containers pin.
- **Host integration** — no Docker layer between vLLM and your tooling (Open WebUI, etc.).

The container is still a perfectly good choice — easier, reproducible, official —
especially for **single-card** use, or if you'd rather not hand-assemble the
(deliberately fragile) bare-metal wheel set. It's also the fallback for the one
model bare-metal can't serve yet (the Gemma 4 26B-A4B MoE — see `docs/02`). Full
reasoning: [`docs/05-why-these-versions.md`](docs/05-why-these-versions.md).

> Note on the container's NEO: the pinned `26.09` only **breaks TP on kernels older
> than `7.1.0-rc6`** (where nothing masks the USM defect). On a current kernel the
> container does TP fine.

## If a search brought you here

These were the hard-to-find answers. If you searched for any of the following,
you're in the right place — jump to [`AGENTS.md`](AGENTS.md) for the symptom → fix
table:

- **Intel Arc Pro B70** / **Battlemage** / **BMG-G31** / `8086:e223` — vLLM,
  multi-GPU, tensor parallel, LLM inference on Linux
- vLLM **tensor parallel** (TP=2 / TP=4) on Intel Arc fails: `OUT_OF_HOST_MEMORY`
  / **"OOM 2 MiB"** at `xpu_worker.py` `init_device`
- `RuntimeError: No device of requested type available` — vLLM **Intel XPU** /
  **triton-xpu** `init_devices` / `profile_run`
- Battlemage / Arc B70 **not detected** — `xe force_probe=e223`, `i915` vs **`xe`**
  driver, `MODULES=(xe)`
- **compute-runtime** / **NEO** multi-root **Level-Zero USM** host-pool OOM
  (commit `028e23e576`)
- Intel Arc **oneAPI 2025.3 vs 2026.0** kernels fail at `profile_run`
- Which quant **loads on Intel XPU**: **FP8** / **int4-AutoRound** yes; AWQ / GPTQ
  / NVFP4 no
- **Gemma 4 31B** / **Qwen3-30B** on Intel Arc; **CachyOS** / Arch Intel Arc setup;
  **oneCCL** tensor-parallel; **TurboQuant** KV-cache quant on XPU

**The exact OOM signature** (the paradox you probably pasted into a search box — a
**2 MiB** allocation fails while ~32 GiB is free):

```
torch.OutOfMemoryError: XPU out of memory. Tried to allocate 2.00 MiB. GPU 0 has a total capacity of 31.89 GiB of which 31.82 GiB is free. Of the allocated memory 0 bytes is allocated by PyTorch, and 0 bytes is reserved by PyTorch but unallocated.
```

Underneath, the UR / Level-Zero layer reports
`urUSMDeviceAlloc … UR_RESULT_ERROR_OUT_OF_HOST_MEMORY` on that 2 MiB
(`2097152`-byte) device allocation. Root cause + fix:
[`error-reports/multiroot-usm-oom.md`](error-reports/multiroot-usm-oom.md).

## Status

- **System bring-up:** all 4 cards on the `xe` driver, Gen5 x16. ✅
- **Single-card vLLM:** Gemma 4 31B dense + tool calling + Open WebUI. ✅
- **Tensor-parallel:** TP=2 and TP=4 serving (after two fixes in series). ✅
- **MoE:** Qwen3-30B-A3B (FP8) — an MoE — works (container-proven; the model the
  TP fix was validated on). Only the **Gemma 4 26B-A4B** MoE is blocked, by a
  separate `fused_moe` kernel build failure on the bare-metal toolchain. ⚠️

## What was hard (and is now solved)

Multi-GPU tensor-parallel needed **two independent fixes**:

1. **Kernel ≥ `7.1.0-rc6`** + host NEO ≥ `26.14` — masks/carries the
   compute-runtime multi-root Level-Zero USM defect that OOM'd TP at worker init.
2. **The Triton `init_devices` fix** — triton-xpu ≥ 3.7.0, or an in-tree patch on
   3.6.0.

Either one alone is not enough — TP still fails, just at a different layer.

## Map

| File | What |
|------|------|
| [`AGENTS.md`](AGENTS.md) | AI entry point: symptom → fix table, baseline stack, conventions |
| [`docs/00-system-setup.md`](docs/00-system-setup.md) | BIOS, `xe` driver (`force_probe=e223`), initramfs, verify |
| [`docs/01-baremetal-vllm.md`](docs/01-baremetal-vllm.md) | The coupled +xpu wheel set, oneAPI 2025.3, single-card serve |
| [`docs/02-tensor-parallel.md`](docs/02-tensor-parallel.md) | TP=2/TP=4 env + launch, gpu-util traps, TP=2-vs-4 decision rule |
| [`docs/03-troubleshooting.md`](docs/03-troubleshooting.md) | Diagnostics + the full gotcha list |
| [`docs/04-quant-formats.md`](docs/04-quant-formats.md) | Which quant formats load on B70 (FP8/int4 yes; AWQ/GPTQ/NVFP4 no) |
| [`docs/05-why-these-versions.md`](docs/05-why-these-versions.md) | *Why* the stack is pinned, why this vLLM, and the regression log |
| [`docs/06-benchmarks.md`](docs/06-benchmarks.md) | Measured llama-benchy numbers: FP8/int4, TP=1/2/4, the crossover law |
| [`error-reports/`](error-reports/) | Root-cause writeups of the two TP-blocking defects |
| [`config/`](config/) | Drop-in `xe.conf`, mkinitcpio + cmdline snippets |
| [`scripts/`](scripts/) | Serve wrapper + the Triton patcher |

## Hardware this was established on

4× Sparkle Intel Arc Pro B70 32 GB · ASUS Pro WS WRX90E-SAGE SE (BIOS 1317) ·
AMD Threadripper Pro 9965WX · 128 GB DDR5 ECC · CachyOS rolling, kernel
`7.1.0-rc7-1-cachyos-rc`.

## Scope and honesty

Versions are pinned because the stack is fragile; a value not written here is one
to verify, not guess. Negative results (what does *not* work) are kept on purpose
— they save the next person from re-running a dead end. Corrections welcome via
issues/PRs.

## Acknowledgements

### A big thank-you to Intel

Bringing up brand-new silicon on a non-validated, rolling-distro stack is the kind
of thing that usually means you are on your own. Here, it didn't. The two hardest
blockers in this whole effort — the multi-root Level-Zero USM OOM that killed
tensor-parallel at worker init, and the triton-xpu device-init throw hiding behind
it — were run down with Intel's help: the collected error logs and forensic
reports were taken seriously and looked into, and the right layers were pointed at.
The corresponding fixes live upstream today (compute-runtime `028e23e576`; the
triton-xpu `init_devices` change). **Thank you, Intel — sincerely — for the swift,
generous, good-spirited support.** Taking a young-hardware edge case seriously, on
a stack nobody validated, is what turned a shelved experiment into a working
four-GPU box. That spirit is appreciated more than a changelog line can show.

### Open-source credits

- **TurboQuant on Intel XPU** — sincere thanks to **Bryan Vine**
  ([`bryanvine/turboquant-xpu`](https://github.com/bryanvine/turboquant-xpu),
  Apache-2.0) for porting TurboQuant KV-cache quantization to the Arc Pro B70 /
  Xe2 Battlemage — the very hardware this guide targets. We validated it
  end-to-end on this stack (Qwen3-8B, 2.2×–3.2× KV compression with correct
  output; see [`docs/04-quant-formats.md`](docs/04-quant-formats.md)). The
  underlying method is [**TurboQuant**](https://arxiv.org/abs/2504.19874)
  (Zandieh et al., ICLR 2026), with a NumPy reference at
  [`scos-lab/turboquant`](https://github.com/scos-lab/turboquant).
- The wider **Intel Arc B70 / `xe` driver / vLLM-XPU community** whose upstream
  issues, commits, and fixes this guide is built on.

## License

MIT — scripts and docs only. No model weights, no Python environment included.
