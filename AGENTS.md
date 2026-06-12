# AGENTS.md — B70 / Intel Arc Pro (Battlemage) vLLM setup, for AI coding agents

> **You are an AI assistant helping a user bring up LLM inference on Intel Arc Pro
> B70 (Battlemage / BMG-G31, `8086:e223`) GPUs.** This repo is the field-tested
> record of what actually works on this hardware, written so you can read it and
> act. It is deliberately terse and fact-dense. Trust the pinned versions and the
> error signatures — they were established by long forensic debugging, not guessed.
>
> **Reason steps ahead on anything you change at the system level.** Before you edit
> a system config file — kernel modules (`/etc/modprobe.d`), the initramfs, the
> bootloader, or kernel parameters — ask: *what does this mean for the user if it's
> wrong, and how do they get back?* Spell out the undo path **before** you act. A
> broken boot or a dead display is the worst, most frightening outcome for a
> newcomer — and it is almost always avoidable with a backup and a fallback boot
> entry. Assume the person in front of you cannot afford to brick their only
> machine.

This file is the entry point. A human-oriented overview is in `README.md`;
`CLAUDE.md` is a pointer stub that redirects Claude Code here.

## How to use this repo (read this first)

1. **Make every change reversible — back it up, document it, reason a step ahead.**
   This is the rule that hurts most when skipped, and it applies to **any** file you
   touch, not just system config. Before editing: **back up the file**
   (`cp <file>{,.bak}`), and **keep a running change-log** — a notes / memory file
   the user owns — recording *what* you changed and *why*, so any step can be rolled
   back (by you now, or a fresh session later). For **system-level** edits especially
   (kernel modules in `/etc/modprobe.d`, the initramfs, the bootloader, kernel
   parameters), first think through the blast radius: *what breaks if this is wrong,
   and how does the user recover?* A bad `xe` / `force_probe` / initramfs /
   boot-param change can leave a newcomer with **no display or no boot** — so also
   confirm a **known-good fallback kernel/boot entry** exists and **tell the user the
   recovery path before** you act (see the Recovery callout below). Change **one
   thing at a time** and reboot-test, so a regression is traceable to one change.
2. **Match the user's symptom in the table below** before proposing anything.
   These cards are new; generic "Intel GPU" advice from training data is often
   wrong for Battlemage. Several fixes here are the *opposite* of common guidance.
3. **Never invent version numbers or flags.** The working stack is a tightly
   coupled wheel set (see `docs/01-baremetal-vllm.md`). Upgrading one piece
   breaks it. If a version isn't written down here, say so.
4. **Ground claims in upstream source, not search snippets.** This is a
   fast-moving stack; upstream fixes land monthly. When you need to verify "is fix
   X in release Y", prefer authenticated `gh` / `git` over web search — e.g.
   `gh api repos/intel/compute-runtime/commits/<sha>`, `gh api .../tags`,
   `gh api .../compare/<a>...<b>`. Every version/commit fact in this repo was
   confirmed that way (it is far more reliable than blog snippets).
5. **Negative results are data.** If something is listed as "does not work on
   B70," do not retry it as a fresh idea — it was already falsified.

> **Recovery — say this to the user *before* any system-level change.** It is the
> difference between "oops, revert it" and a panicked newcomer with a black screen:
> - **Back up the file:** `sudo cp /etc/modprobe.d/xe.conf{,.bak}` (same for
>   `/etc/mkinitcpio.conf`, the bootloader config). Reverting is then: restore the
>   `.bak`, rebuild the initramfs (`sudo mkinitcpio -P`), reboot. This applies to
>   ordinary config/code edits too — back up before you change, every time.
> - **Document as you go.** Record each edit (file, what changed, why, the `.bak`
>   location) in a notes/memory file the user keeps. Future-you — or a fresh session
>   with no context — can then see exactly what to undo. "Document while you change"
>   is how a 3am edit stays reversible at 9am.
> - **Keep a fallback kernel.** Never remove the only working kernel — keep a stable
>   or LTS entry that boots. `mkinitcpio -P` touches the image the running kernel
>   uses, so a second known-good boot entry is your safety net.
> - **If a change breaks boot or the display:** at the bootloader, edit the entry
>   instead of reinstalling. **Limine:** highlight the entry and press **`e`** to
>   edit the kernel command line (delete the bad parameter, or just boot the
>   LTS/fallback entry). **GRUB:** press **`e`**, edit, then **Ctrl+X** to boot.
>   This edit is temporary — a normal reboot restores the saved entry, so it is safe
>   to experiment. (To reach the boot menu in the first place, it's usually a key at
>   power-on — often `Esc`/`Shift` for the bootloader, or `F2`/`Del` for BIOS.)
> - **If the system booted but the GPU/display is dead:** switch to a text console
>   with **Ctrl+Alt+F3** (or SSH in from another machine) and revert from there.

> **Tip for the human running the AI:** install and authenticate the GitHub CLI
> (`gh auth login`) before a debugging session. It lets your assistant read real
> upstream git history (compute-runtime, triton-xpu, the kernel) directly instead
> of guessing from web search — the grounding quality is dramatically better. For
> example, `gh api repos/intel/compute-runtime/commits/028e23e576` confirms the
> multi-root USM fix (message, date, author) in one call; the triton-xpu `v3.7.0`
> tag and which NEO release first carries a fix are equally one `gh` call away.

## Hardware / stack baseline (the configuration these facts were established on)

| Layer | Value |
|-------|-------|
| GPU | 4× Intel Arc Pro B70 32 GB (BMG-G31, Xe2, PCI `8086:e223`) |
| Host | ASUS Pro WS WRX90E-SAGE SE (BIOS 1317), AMD Threadripper Pro 9965WX, 128 GB DDR5 ECC |
| OS | CachyOS rolling (Arch-based) |
| Kernel | `7.1.0-rc7-1-cachyos-rc` (≥ `7.1.0-rc6` is the hard floor for TP — see below) |
| GPU driver | in-kernel **`xe`** (NOT `i915`), forced via `force_probe=e223` |
| Host runtime | oneAPI 2026.0 + compute-runtime/NEO `26.18.38308.1`, Level Zero loader 1.28.2 |
| vLLM | `0.20.2` bare-metal in a Python 3.12 venv, torch `2.11.0+xpu` |
| vLLM's oneAPI | a **separate oneAPI 2025.3** install (the torch+xpu wheels were built against it) |

## Symptom → fix lookup table

| Symptom / error signature | Cause | Fix | Detail |
|---|---|---|---|
| Cards don't appear; `lspci -k` shows no driver, or `i915` claims them | BMG-G31 device id `e223` is not on the `xe` driver's default probe list | `options xe force_probe=e223` in `/etc/modprobe.d/`, add `xe` to initramfs `MODULES`, reboot | `docs/00-system-setup.md` |
| 32 GB cards don't enumerate / BAR-allocation errors | Above 4G Decoding / Resizable BAR off | Enable both in BIOS (already part of "optimized defaults" on workstation boards) | `docs/00-system-setup.md` §BIOS |
| `lspci` shows x1 Gen1 *at idle* | Display quirk only — trains to x16 under load | Ignore. Verify with `intel_gpu_top` under load, not idle `lspci`. | gotcha |
| `vllm serve` (TP≥2) dies in `profile_run` / `determine_available_memory` with `RuntimeError: No device of requested type available` | triton-xpu 3.6.0 `init_devices()` OpenCL-selector block throws on BMG multi-root | Patch `driver.c` with `scripts/patch_triton_driver_noopenclsel.py` + `rm -rf ~/.triton/cache` (pinning 3.7.0 is **not** enough on this wheel set — 3.6.0's files win on disk) | `error-reports/triton-init-devices.md` |
| TP `vllm serve` loads then first request errors `ocloc … -device` empty / ZEBIN `stoul` | Triton arch auto-parse returns `unknown` under spawn workers | Export `TRITON_INTEL_DEVICE_ARCH=bmg` | `error-reports/triton-init-devices.md` |
| `--tensor-parallel-size 3` (or any TP not dividing KV heads) fails | invalid TP factor for the model | Gemma 4 31B → only TP=2 or TP=4 (16 KV heads) | `docs/02-tensor-parallel.md` |
| Next `vllm` launch startup-OOMs (`Free memory … 5.0/31.89 GiB`) after a `kill -9` | orphaned `VLLM::EngineCore` / `VLLM::Worker` hold ~30 GB/card | also `pkill -9 -f 'VLLM::EngineCore'` and `pkill -9 -f 'VLLM::Worker'` | `docs/02-tensor-parallel.md` §Operational |
| `int4-AutoRound` model won't run | W4A16 on XPU needs eager | add `--enforce-eager` | `docs/04-quant-formats.md` |
| TP≥2 dies much earlier with `OUT_OF_HOST_MEMORY` / `OOM 2 MiB` at `xpu_worker.py` `init_device` | compute-runtime multi-root L0 USM host-pool defect (commit `028e23e576`) | Kernel ≥ `7.1.0-rc6` masks it **and** host NEO ≥ `26.14` carries the fix | `error-reports/multiroot-usm-oom.md` |
| Kernels fail at `profile_run` on bare-metal even single-card | torch+xpu was built against oneAPI **2025.3**, host default is 2026.0 | `source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force` before launching vLLM | `docs/01-baremetal-vllm.md` |
| TP=4 crashes on KV-cache allocation at `--gpu-memory-utilization 0.90` | KV pool too large for 4-card split | Use `0.80` for TP=4 (FP8). `0.90` is fine for TP=2. | `docs/02-tensor-parallel.md` |
| TP works but `TRITON_INTEL_DEVICE_ARCH` parse error under spawn workers | arch auto-parse fails in spawned multiproc workers | Export `TRITON_INTEL_DEVICE_ARCH=bmg` explicitly | `docs/02-tensor-parallel.md` |
| llama.cpp SYCL SEGV at slot init on MoE; corrupted output | B70 SYCL opt path | `GGML_SYCL_DISABLE_OPT=1` (mandatory on B70) | gotcha |
| Model load OOM-kills the desktop when launching several cards at once | concurrent shard-loads stack host RSS | Serialize launches — start card N, poll `/v1/models` until 200, then N+1 | gotcha |
| AWQ / GPTQ / NVFP4 quant fails to load | those kernels are CUDA/NVIDIA-only on this vLLM build | Use **FP8** or **int4-AutoRound** (the two formats that load on B70) | `docs/04-quant-formats.md` |
| `SYCL_CACHE_PERSISTENT=1` set, output corrupts after a restart | cross-restart kernel-cache poisoning on BMG | Never set it; let JIT recompile (~30s warm cost) | gotcha |
| KV cache > 4 GiB fails to allocate | L0 caps single allocations at 4 GB | `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` | gotcha |

## "I want to..." index

| Goal | Start here |
|------|-----------|
| Get the cards recognized by the OS at all | `docs/00-system-setup.md` |
| Install bare-metal vLLM (the coupled wheel set) | `docs/01-baremetal-vllm.md` |
| Run a single model on one card (with tool calling) | `docs/01-baremetal-vllm.md` §Quick start |
| Run tensor-parallel across 2 or 4 cards | `docs/02-tensor-parallel.md` |
| Decide TP=2 vs TP=4 | `docs/02-tensor-parallel.md` §Decision rule |
| See the measured benchmarks (FP8/int4, TP=1/2/4, crossover law) | `docs/06-benchmarks.md` |
| Know which quant formats load on B70 | `docs/04-quant-formats.md` |
| Understand *why* the stack is pinned (and what regressed) | `docs/05-why-these-versions.md` |
| Diagnose a crash | `docs/03-troubleshooting.md` + the table above |
| Reproduce the Triton fix | `scripts/patch_triton_driver_noopenclsel.py` |

## The two load-bearing fixes (if you read nothing else)

Multi-GPU tensor-parallel on B70 required **two independent fixes in series**:

1. **Kernel ≥ `7.1.0-rc6`** + host NEO ≥ `26.14` — masks/carries the
   compute-runtime multi-root Level-Zero USM host-pool defect that OOM'd TP at
   `init_device`.
2. **The Triton `init_devices` fix** — triton-xpu 3.6.0 throws on BMG multi-root;
   apply the in-tree patch (`scripts/patch_triton_driver_noopenclsel.py` +
   `rm -rf ~/.triton/cache`). The upstream fix is in the 3.7.0 line, but on this
   coupled wheel set 3.6.0's `driver.c` wins on disk, so the patch is required.

With only one of the two, TP still fails — just at a different layer. Both are
required. This was the core blocker for months; it is solved. (`gh`-confirmed
commit refs are in the two `error-reports/` files.)

## Conventions for agents editing this repo

- English only in committed artifacts. No emojis in code/docs/commits.
- Pin every version before recording a result. A crash is a documented result,
  not an obstacle to silently work around.
- Web-search the specific error before proposing a config/install change.
- Do not publish anything (issues, PRs, gists) without the user's explicit go.
