# Error report: compute-runtime multi-root Level-Zero USM host-pool OOM

**Status:** ROOT-CAUSED. Masked by kernel ≥ `7.1.0-rc6`; fix carried by host NEO
≥ `26.14` (first release with commit `028e23e576`; reference host runs `26.18.38308.1`).
**Component:** `intel/compute-runtime` — the UR Level-Zero adapter's multi-device
host-USM-pool path.
**Affected:** any Python process whose Level-Zero context spans **≥2 distinct root
devices** on B70 (BMG-G31). This is what blocked vLLM TP≥2 for months on the
6.x BKC / 7.0.x kernels.

## Symptom (on 6.x / 7.0.x kernels)

`vllm serve --tensor-parallel-size ≥ 2` dies during worker init
(`xpu_worker.py` `init_device`) with an out-of-host-memory error on a tiny (2 MiB)
allocation, long before model weights load. The tell-tale paradox: a **2 MiB**
allocation fails while **~32 GiB is free**:

```
torch.OutOfMemoryError: XPU out of memory. Tried to allocate 2.00 MiB. GPU 0 has a total capacity of 31.89 GiB of which 31.82 GiB is free. Of the allocated memory 0 bytes is allocated by PyTorch, and 0 bytes is reserved by PyTorch but unallocated.
```

Underneath, the UR / Level-Zero layer reports the real failure on that 2 MiB
(`2097152`-byte) device allocation:

```
... urUSMDeviceAlloc ... UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
```

## The N=2 boundary

The trigger is exactly **≥2 root devices visible to one process** — not "many".
Affinity-mask sweep: `0,1`, `0,1,2`, `0,1,2,3` all fail identically; only a
single-device mask (`0`) bypasses. Even `TP=1` with `ZE_AFFINITY_MASK=0,1,2,3`
reproduces. "Multi-root" means literally ≥2.

## Root cause (two independent inside-call channels)

The fix is compute-runtime commit
**[`028e23e57673bd02fe2f6bbefa1142fc573b256c`](https://github.com/intel/compute-runtime/commit/028e23e57673bd02fe2f6bbefa1142fc573b256c)**
(2026-03-20, *"fix: disable l0 usm host pool on multi device"*, Related-To:
GSD-12391, author Dominik Dabek). It makes `initHostUsmAllocPool` take a
`multiDevice` flag and disables the L0 USM host pool when >1 device is present.
The underlying regression is tracked as **[intel/compute-runtime issue
#916](https://github.com/intel/compute-runtime/issues/916)** (GSD-12641),
introduced by an earlier "enable l0 device usm growing pools" change.

Release provenance (confirmed via `gh api .../compare`): the fix is **absent** from
the `26.09.37435.x` branch that older Intel containers pin, **first appears in
`26.14.37833.4`** (release published 2026-04-20; commit dated 2026-04-14), and is present in the host's **`26.18.38308.1`**
(the reference bare-metal stack). So: container NEO 26.09 reproduces the bug; host
NEO ≥ 26.14 carries the fix.

Evidence the alloc fails *inside* UR despite NEO succeeding:

- **NEO `LogAllocations=1`** emits `Created Graphics Allocation ... Size: 3145728`
  *inside* the `urUSMDeviceAlloc` window — NEO succeeds at the device alloc, yet
  UR's adapter returns `OUT_OF_HOST_MEMORY`. Each failure leaks exactly 3 MiB of
  device LocalMemory (orphaned BUFFER never released).
- **`UMF_LOG='level:debug;output:stderr'`** prints
  `create_slab: allocation of slab data failed!` → `bucket_create_slab failed!` →
  `enqueueUSMAllocHelper: allocation from the UMF pool ... failed` → UR's OOM
  return. (Note: UMF/UR log grammar uses **semicolons**, not commas, as field
  separators.)

## What was ruled out

Kernel layer (strace of `DRM_IOCTL_XE_*` — the EINVAL events are NEO's normal
probe-and-retry, also present in the working single-device control); NEO USM pool
manager; `RLIMIT_MEMLOCK`; oneCCL IPC race (no oneCCL needed to trigger);
`ZE_FLAT_DEVICE_HIERARCHY` (FLAT and COMPOSITE both fail); `docker --privileged`.
The V2 UR adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=1`) fails at a *different* surface
(`_xpu_getDeviceCount` returns `UR_RESULT_ERROR_UNKNOWN`) — both adapters fail
under multi-root visibility.

## The kernel side: why ≥ 7.1.0-rc6 *masks* it (and why there is no single "kernel fix" commit)

A common misremembering is that "a kernel patch fixed this." That is not accurate,
and it matters for anyone trying to cherry-pick a fix:

- The real, pinned root cause is the **user-space** compute-runtime defect above
  (`028e23e576`). A kernel bump does **not** fix the allocator — it changes the
  multi-root device/memory behaviour enough that the broken host-USM-pool path is
  **no longer exercised**. This was **never bisected** to a specific kernel commit.
- The most symptom-matched candidate in the 7.1 `drm/xe` merge window is
  **Transparent Huge Pages (2 MB folios) for device pages in `drm_pagemap`** —
  same 2 MB granularity as the failing `urUSMDeviceAlloc(2 MiB)`
  ([THP-for-device-SVM](https://www.phoronix.com/news/Intel-Xe-THP-For-Device-SVM),
  patch-series writeup [LWN 1053533](https://lwn.net/Articles/1053533/)).
  Alternates: drm/xe purgeable-BO / improved VRAM-pressure behaviour (7.1), and
  the foundational **multi-device SVM** that landed one cycle earlier in **Linux
  7.0** — the latter is the most likely origin of the "mainline kernel gained a
  multi-GPU fix" recollection.
- **Practical guidance:** treat "kernel ≥ 7.1.0-rc6" as an empirical masking
  boundary, not a cherry-pickable patch. The *correctness-preserving* fix for any
  kernel is host NEO ≥ 26.14 (carries `028e23e576`).

## Fix / current status

On the baseline stack (kernel `7.1.0-rc7`, host NEO `26.18.38308.1`) this **no
longer reproduces** — the kernel change masks it and the host NEO carries
`028e23e576`. After also applying the Triton fix (`triton-init-devices.md`), TP=2
and TP=4 serve correctly.

> A per-process `ZE_AFFINITY_MASK=<single-card>` set *before* `import torch`
> bypasses the OOM, but it breaks every cross-rank collective — so it is a
> diagnostic, not a correctness-preserving workaround. Use the kernel+NEO fix.

## Note for a fresh AI session

This failure mode is **kernel-conditional**. If a user is on kernel < `7.1.0-rc6`
they will hit this; the fix is to update the kernel and host NEO, not to fight the
allocator. If they are on ≥ `7.1.0-rc6` and TP still fails, the cause is almost
certainly the Triton defect (`triton-init-devices.md`), which presents *later* in
the run.
