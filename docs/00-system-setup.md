# 00 — System setup: getting the B70 cards recognized

Goal: a fresh CachyOS / Arch-family install where all four Intel Arc Pro B70 cards
are bound to the `xe` driver and trained to Gen5 x16.

This is the part that the user (and prior AI sessions) found hardest to remember,
because it is "set once and forget." Two small config files are load-bearing.

> **Before you edit any system file below — back up and know your way out.** The
> changes here (modprobe, initramfs, boot config) are what bind the GPU; a wrong
> one can leave you with **no display or no boot**. So: back up each file first
> (`sudo cp <file>{,.bak}`), keep a **fallback / LTS kernel entry** that boots, and
> know that a bad boot entry is fixable **non-destructively** at the bootloader —
> highlight the entry and press **`e`** (Limine/GRUB) to edit or remove a bad
> kernel parameter; a reboot restores the saved entry. Change **one thing, reboot,
> verify**, then the next. (Full recovery checklist: `AGENTS.md` §Recovery.)

## Step 1 — BIOS: enable Above 4G Decoding + Resizable BAR

The one thing in BIOS that genuinely matters is that **Above 4G Decoding** and
**Resizable BAR** are enabled — large-VRAM GPUs (32 GB each here) need their BAR
mapped into 64-bit address space above the 4 GB boundary. On a workstation board
like the ASUS Pro WS WRX90E-SAGE SE these are already part of **"optimized
defaults"**, so in practice loading optimized defaults was enough — **no extensive
BIOS tuning was needed.**

```
Advanced -> PCI Subsystem Settings
  Above 4G Decoding         = Enabled   <- load-bearing for 32 GB GPUs
  Re-Size BAR Support       = Enabled   <- load-bearing for 32 GB GPUs
```

The author's full config also carried the settings below (mostly inherited from
optimized defaults). They are listed for completeness but were **not** deliberately
tuned and are **not** known to be required:

```
SR-IOV Support  = Disabled
AMD CBS -> NBIO -> PCIE:  Ten Bit Tag = Enabled,  ACS Enable = Disabled
AMD CBS -> IOMMU:         IOMMU = Disabled
Boot:                     CSM = Disabled
```

> **Do not chase a "Gen1 x1" reading — it is a display artifact, not a problem to
> fix in BIOS.** With the `xe` driver, `lspci` reports the PCIe link as **x1 Gen1
> when the cards are idle**; the link trains up to **x16 Gen5 under load**. So a
> `LnkSta: ... Width x1` at idle does **not** mean link training failed, and there
> is nothing to "fix" in BIOS for it. To check, compare *capable* vs *current*
> width — `lspci -vvv -s 03:00.0 | grep -E 'LnkCap|LnkSta'` (`LnkCap` shows the
> x16 Gen5 the slot supports) — and confirm the live width **under load** with
> `intel_gpu_top` / `xpumcli`. (See "Known display/telemetry quirks" below; this
> is the same quirk.)

> Folklore to ignore: kernel params `xe.max_vfs=0`, `pci=realloc=on`,
> `pcie_aspm=off` were tried during early debugging and are **not** needed.
> (Intel's Data Center GPU docs suggest `pci=realloc=off` for *some* multi-card
> enumeration problems — not needed here, but worth knowing if cards fail to
> appear in `/dev/dri`.)

## Step 2 — Force the `xe` driver to claim the cards

Battlemage needs the modern **`xe`** kernel driver, not the legacy **`i915`**.
The kernel will not bind `xe` to device id `e223` by default because the id is too
new for the probe list. Force it.

`/etc/modprobe.d/xe.conf` (see `config/etc-modprobe.d-xe.conf`):

```
options xe force_probe=e223
options i915 enable_guc=3
```

`force_probe=e223` is the magic line — it tells `xe` to claim PCI `8086:e223`.
Without it, the cards either get no driver or fall to `i915` (which does not
properly support Battlemage compute).

## Step 3 — Load `xe` early via the initramfs

Add `xe` to the initramfs `MODULES` so it is present before userspace.

`/etc/mkinitcpio.conf` (Arch/CachyOS default initramfs generator):

```
MODULES=(xe)
```

Then rebuild and reboot:

```bash
sudo mkinitcpio -P
sudo reboot
```

(On dracut-based distros the equivalent is `force_drivers+=" xe "` in
`/etc/dracut.conf.d/`.)

## Step 4 — Verify

```bash
# All four cards should show: Kernel driver in use: xe
lspci -nnk -d 8086:e223

# The xe module is loaded
lsmod | grep '^xe'

# Running kernel is >= 7.1.0-rc6 (required for multi-GPU TP)
uname -r
```

Expected: four `VGA compatible controller ... [Arc Pro B70] [8086:e223]` lines,
each `Kernel driver in use: xe`. On the reference host the four cards enumerate at
PCI `03:00.0`, `23:00.0`, `c3:00.0`, `f3:00.0` (Sparkle subsystem `172f:0105`),
32 GB each. Note: `i915` should **not** be loaded (`lsmod | grep i915` empty) —
the `enable_guc=3` line in `xe.conf` only matters if an Intel iGPU is present; it
does not affect the dGPUs, which `force_probe` routes to `xe`.

> Debugging tip: many hosts set `kernel.dmesg_restrict=1`, so a non-root `dmesg`
> returns 0 lines. Use `sudo dmesg | grep -i xe` to inspect driver binding.

## Step 5 — Userspace GPU runtime (driver libraries + oneAPI)

Steps 1–4 bind the cards at the **kernel** level. vLLM also needs the **userspace**
GPU runtime: the Level-Zero/OpenCL driver (Intel "compute-runtime" / NEO) and the
oneAPI toolchain. These are **host packages + an installer — not a container.** No
Docker image and no init/bootstrap script is required to bring the GPU up.

On Arch / CachyOS:

```bash
# Level-Zero + OpenCL userspace driver (provides libze_intel_gpu.so + libigdrcl.so)
sudo pacman -S intel-compute-runtime level-zero-loader level-zero-headers \
               intel-graphics-compiler intel-gmmlib
# xpumcli (telemetry) is AUR-only — install separately with an AUR helper:
paru -S intel-xpumanager-bin

# Confirm the NEO version carries the multi-root USM fix (commit 028e23e576);
# you want >= 26.14, and the reference host runs 26.18.38308.1:
pacman -Qi intel-compute-runtime | grep Version
```

Install **both** oneAPI toolkits side by side:

- **2026.0** — the host default. On Arch/CachyOS this is a distro package
  (`intel-oneapi-toolkit`, which provides `/opt/intel/oneapi/2026.0`).
- **2025.3** — the toolchain the venv's `torch+xpu` wheels were built against
  (load-bearing; see `01-baremetal-vllm.md`). This one is **not** packaged —
  install it with Intel's standalone offline installer into `/opt/intel/oneapi`.

They coexist as `/opt/intel/oneapi/2025.3` and `/opt/intel/oneapi/2026.0`.

Verify the runtime actually sees the cards **before** launching vLLM:

```bash
clinfo -l                 # should list 4x "Intel(R) Arc(TM) Pro B70 Graphics"
source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force && sycl-ls   # 4x level_zero:gpu
```

## Kernel version matters for multi-GPU

Single-card inference works on older kernels. **Tensor-parallel (TP≥2) requires
kernel ≥ `7.1.0-rc6`** — earlier kernels hit a compute-runtime multi-root USM
defect that OOMs TP at worker init. See `error-reports/multiroot-usm-oom.md`.
On CachyOS the rc kernel line is `linux-cachyos-rc`; keep a stable kernel
installed as a fallback boot entry. The reference host now runs
`7.1.0-rc7-1-cachyos-rc`.

> **The kernel is necessary but not sufficient for TP.** A correct kernel only
> *masks* the USM defect; TP also needs the **Triton `init_devices` fix** (a second,
> independent fix). With a new-enough kernel but no Triton fix, TP loads the model
> then dies at the first cross-rank collective. See `02-tensor-parallel.md`.

## Bootloader note

This host boots via **Limine** (CachyOS's rolling default bootloader), not GRUB.
The kernel cmdline is minimal — nothing GPU-specific is required there:

```
quiet nowatchdog splash rw rootflags=subvol=/@ root=UUID=<your-root-uuid>
```

If your distro uses GRUB, the equivalent goes in `GRUB_CMDLINE_LINUX_DEFAULT`;
but again, no GPU-specific kernel parameter is needed once BIOS + modprobe are set.

## Known display/telemetry quirks (not bugs)

- `lspci` shows **x1 Gen1 at idle** on BMG-G31 with the `xe` driver. Display quirk
  only; under load the cards train to x16 Gen5. Verify with `intel_gpu_top`.
- `xpumcli` (use `xpumcli`, not `xpu-smi`) shows **N/A for PCIe / Xe-Link
  telemetry** on BMG-G31 — a gap in Intel's tooling, not a fault.
- If you have a desktop on one card (e.g. KDE/Wayland), that card will always show
  ~1.5 GB used. Not a leak.
