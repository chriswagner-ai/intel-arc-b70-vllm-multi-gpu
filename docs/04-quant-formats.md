# 04 — Quantization formats that load on B70

Hard-won compatibility matrix. **Do not run NVIDIA-format quants on Intel** — web
search the format's kernel backend first. Several formats that are common on CUDA
silently fail or error on B70 because their dequant kernels are CUDA-only.

| Format | Loads on B70? | Known-good model (validated) | Notes |
|--------|---------------|------------------------------|-------|
| **FP8** (`*-FP8-Dynamic`) | **Yes** (probed) | `RedHatAI/gemma-4-31B-it-FP8-Dynamic` | The reliable baseline. Native KV `--kv-cache-dtype fp8`. The listed checkpoint is **pre-quantized** (compressed-tensors, ~31 GB, loads directly) — ~31 GB on a 32 GB card leaves almost no KV slack, so all validated runs used **TP≥2** (TP=1 is too tight). Separately, *online* `--quantization fp8` is a different path that materializes **bf16 first** (~2× RAM) — also TP≥2 for 30B-class. |
| **int4-AutoRound** (Intel `*-int4-AutoRound`) | **Yes** (probed) | `Intel/gemma-4-31B-it-int4-AutoRound` | ~20% faster than FP8 (at matched TP), ~18 GiB vs ~31 GiB. **Requires `--enforce-eager`** (W4A16 on XPU). Preferred single-stream. On TP=4 needs `--gpu-memory-utilization ~0.65` (KV pool OOMs higher). |
| **bf16 / fp16** (unquantized) | Yes | — | Fits only smaller models per card (32 GB). |
| **AWQ** | **No** (probed) | — | `torch.ops._C.awq_dequantize` is a CUDA-only custom op, absent from the XPU build → `AttributeError` at load. |
| **GPTQ-Int4** | **No** (probed) | — | `gptq_shuffle`/gptq op missing from the XPU build. |
| **NVFP4** (`RedHatAI/*-NVFP4`) | **No** (carried) | — | NVIDIA Blackwell-only format; no XPU path. Not load-probed here — verdict carried from format knowledge. |

> Provenance: "probed" = directly load-tested on this stack (`vllm serve` smoke);
> "carried" = verdict from format/web knowledge, not a fresh in-repo probe.

## KV-cache quantization (TurboQuant / fp8 KV)

- Native **`--kv-cache-dtype fp8`** works and is the default recommendation.
- **TurboQuant** KV quant (`k8v4`, `k3v4_nc`) is validated on bare-metal vLLM for
  **uniform-attention** models (e.g. Qwen3-8B: 2.2×–3.2× KV compression, correct
  output). It is **blocked on hybrid/sliding-window attention** models (e.g.
  Gemma) — use native fp8 KV there instead. (TurboQuant is now merged natively in
  vLLM via PR #38479, so on 0.20.2 the `turboquant_*` presets are built in.)
  **Credit:** the Intel-XPU port is
  [Bryan Vine's `turboquant-xpu`](https://github.com/bryanvine/turboquant-xpu)
  (Apache-2.0), developed on the **same Arc Pro B70 / Xe2 hardware** as this
  guide; the underlying method is
  [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026).
  Thank you — see the Acknowledgements in the README.

## MXFP4 note

MXFP4 is an OCP Microscaling standard, **not** Intel-exclusive. On B70, MXFP4
*storage* works but the kernels are software-decompressed (no native tensor-core
acceleration — that's primarily NVIDIA Hopper/Blackwell today). Don't expect a
speedup from MXFP4 on Battlemage.

## Practical guidance for an agent picking a model

1. Prefer an **int4-AutoRound** build if one exists for the target model (fastest
   single-stream).
2. Otherwise use an **FP8** build.
3. If the user points at an AWQ/GPTQ/NVFP4 repo, stop and tell them it won't load
   on B70 — find the FP8 or int4-AutoRound equivalent instead.
4. Always confirm with a 1-token generation smoke test after load; "served" alone
   (the endpoint answering `/v1/models`) does not prove the kernels work:
   ```bash
   curl -s localhost:8003/v1/completions -H 'Content-Type: application/json' \
     -d '{"model":"gemma-4-31B-it","prompt":"2+2=","max_tokens":3}'
   ```
