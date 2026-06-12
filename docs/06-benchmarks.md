# 06 — Benchmarks (Gemma 4 31B on B70, measured)

All numbers below are **client-side, measured with [llama-benchy]** against the
running OpenAI endpoint — warmup + multi-run means (not single-shot curls). Same
matrix for every run, so the rows are comparable:

```
pp=512   tg=128   depth={0, 2048}   concurrency={1, 4, 8}   runs=3
```

Stack: 4× Intel Arc Pro B70 32 GB, kernel `7.1.0-rc6`, bare-metal vLLM `0.20.2`
(torch `2.11.0+xpu`), oneAPI `2025.3`. `--max-model-len 16384`, `--kv-cache-dtype
fp8`, `--enforce-eager`, `--attention-backend TRITON_ATTN`.

Metrics: **dec/req** = per-request decode tok/s · **agg** = aggregate decode tok/s
across the concurrent batch · **TTFT** = end-to-end time to first token (ms).

## Single-stream headline (depth 0, concurrency 1)

| Model | Quant | TP | Cards | decode tok/s |
|-------|-------|----|-------|-------------:|
| `Intel/gemma-4-31B-it-int4-AutoRound` | int4 AutoRound (~18 GiB) | 1 | 1 | **16.54** |
| `RedHatAI/gemma-4-31B-it-FP8-Dynamic` | FP8 (~31 GB) | 2 | 2 | **12.16** |

int4 single-card (16.54 tok/s, TP=1) is the fastest single-stream path. At
**matched TP**, int4 is ~20% faster than FP8 (TP=2: 14.68 vs 12.16, +20.7%) — note
these are two separate sweeps, and `16.54 vs 12.16` is **not** the speedup (those
are different TP sizes / different runs). FP8 (~31 GiB) does not fit one 32 GB
card, so its smallest config is TP=2 — see `04-quant-formats.md` /
`01-baremetal-vllm.md`.

## Gemma 4 31B **FP8** — TP=2 vs TP=4 (sweep `tpperf-1`)

TP=2: cards 0,1 @ util 0.90, KV pool **190,218 tok** (max concurrency 11.6×).
TP=4: cards 0–3 @ util 0.80, KV pool **535,003 tok** (max concurrency 32.7×).

| depth | conc | TP2 dec/req | TP2 agg | TP2 TTFT | TP4 dec/req | TP4 agg | TP4 TTFT |
|------:|-----:|------------:|--------:|---------:|------------:|--------:|---------:|
| 0     | 1    | 12.16 | 12.16 | 1613  | 10.17 | 10.17 | 661   |
| 0     | 4    | 9.36  | 32.80 | 2853  | 11.47 | 42.47 | 2876  |
| 0     | 8    | 6.73  | 42.42 | 4541  | 7.24  | 46.78 | 3978  |
| 2048  | 1    | 8.48  | 8.48  | 11322 | 9.69  | 9.69  | 7355  |
| 2048  | 4    | 4.07  | 9.70  | 29943 | 5.06  | 14.15 | 18730 |
| 2048  | 8    | 2.36  | 9.62  | 53429 | 3.44  | 14.93 | 33499 |

## Gemma 4 31B **int4-AutoRound** — TP=1 / TP=2 / TP=4 (sweep `tpperf-2`)

TP=1: card 0 @ 0.90, KV **38,913 tok**. TP=2: cards 1,2 @ 0.90, KV **155,209 tok**.
TP=4: cards 0–3 @ **0.65**, KV **242,776 tok** (int4's tiny weights force a lower
util — at 0.85 the KV allocation OOMs; see `02-tensor-parallel.md` §gpu-util trap).

| depth | conc | TP1/req | TP2/req | TP4/req | TP1 agg | TP2 agg | TP4 agg | TP4 TTFT |
|------:|-----:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|
| 0     | 1    | 16.54 | 14.68 | 14.42 | 16.54 | 14.68 | 14.42 | 753   |
| 0     | 4    | 9.18  | 11.09 | 11.94 | 32.73 | 36.56 | 43.83 | 1917  |
| 0     | 8    | 5.54  | 7.61  | 9.58  | 35.73 | 50.09 | 59.66 | 3213  |
| 2048  | 1    | 9.49  | 10.37 | 12.85 | 9.49  | 10.37 | 12.85 | 5989  |
| 2048  | 4    | 3.30  | 4.82  | 6.01  | 7.36  | 12.09 | 16.77 | 16158 |
| 2048  | 8    | 2.45  | 2.75  | 4.01  | 6.10  | 11.31 | 17.50 | 28725 |

Peak aggregate decode: **TP1 35.7 · TP2 50.1 · TP4 59.7 tok/s.**

## Prefill (prompt processing) — pp tok/s

Prompt-processing throughput (mean over 3 runs), pulled from the same llama-benchy
JSON. Prefill **parallelizes across cards** (more cards → faster), and depth-2048
prefill is ~2–3× slower than depth-0 (the 512-token prompt is attended against a
full 2048-token KV):

| Config | d0 c1 | d0 c4 | d0 c8 | d2048 c1 | d2048 c4 | d2048 c8 |
|--------|------:|------:|------:|---------:|---------:|---------:|
| FP8 TP2  | 468 | 658 | 669 | 226 | 228 | 227 |
| FP8 TP4  | 777 | 723 | 756 | 350 | 364 | 362 |
| int4 TP1 | 416 | 441 | 445 | 141 | 141 | 115 |
| int4 TP2 | 692 | 702 | 721 | 260 | 261 | 261 |
| int4 TP4 | 700 | 879 | 911 | 428 | 425 | 424 |

This is the other half of the TP=4 story: it trades a little single-stream *decode*
speed for markedly faster *prefill* — so it both serves more concurrent users and
returns the first token sooner on long prompts (lowest deep-context TTFT above).

## The crossover law (what the numbers teach)

1. **Shallow context + single stream → fewer cards win.** Each inter-card boundary
   adds a PCIe sync per token; with no batch to amortize it, TP1 > TP2 > TP4
   (int4: 16.54 > 14.68 > 14.42, monotonic).
2. **Concurrency → more cards win.** Bigger KV pool + more compute sustain more
   simultaneous sessions: peak aggregate TP1 35.7 < TP2 50.1 < TP4 59.7.
3. **Deep context inverts rule 1.** At depth 2048, TP4 single-stream (12.85) beats
   TP1 (9.49) — long-context attention is memory-bandwidth-bound, so more cards
   help even one stream, and TP4 has the lowest deep-context TTFT (4-way prefill).

**Decision rule:** one user + short prompts → **TP=1/TP=2**; many users **or** long
context/RAG → **TP=4**.

## KV-cache capacity (TurboQuant)

On **uniform-attention** models, TurboQuant KV quant stretches the KV pool further
(measured on Qwen3-8B, single B70, same VRAM budget): baseline fp16 90,704 tok →
`k8v4` **200,944 tok (2.22×)** → `k3v4_nc` **287,072 tok (3.16×)**, output correct.
Blocked on hybrid/sliding-window models (Gemma) — see `04-quant-formats.md`.

## Honesty notes

- FP8 (`tpperf-1`) and int4 (`tpperf-2`) both use llama-benchy with the **identical
  matrix**, so the int4-vs-FP8 comparison is apples-to-apples. Throughput only —
  int4-vs-FP8 *accuracy* was not measured in these runs.
- KV-pool sizes differ across runs because util and weights differ (e.g. int4 TP=4
  @ 0.65 vs FP8 TP=4 @ 0.80) — compare KV pools only within the same column.
- Do **not** cross-compare these against older single-shot `curl` numbers (e.g. the
  ~21.8 tok/s int4 figure) — different instrument, not comparable.

[llama-benchy]: https://github.com/alexziskind1/llama-benchy
