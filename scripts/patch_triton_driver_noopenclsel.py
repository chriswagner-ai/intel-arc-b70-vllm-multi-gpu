#!/usr/bin/env python3
# What: Patches intel-xpu-backend-for-triton 3.6.0 driver.c to make the
#       "workaround to get opencl extensions" block in init_devices
#       non-fatal. The block builds sycl::device(selector) requiring an
#       OpenCL-backend device whose name matches each L0 B70; on BMG
#       multi-root no such OpenCL device exists -> throws
#       "No device of requested type available" -> std::terminate (see the public
#       error report error-reports/triton-init-devices.md). Wrapping it in try/catch mirrors
#       what upstream 3.7.0 did (removed the block); has_opencl_extension()
#       already guards on list size and returns False, so a short opencl list
#       degrades safely.
# Usage: python3 patch_triton_driver_noopenclsel.py /path/to/driver.c
# Idempotent: refuses to double-patch.
import sys, pathlib

path = pathlib.Path(sys.argv[1])
src = path.read_text()

ORIG = """    // workaround to get opencl extensions
    const auto &name = sycl_devices[i].get_info<sycl::info::device::name>();
    sycl::device opencl_device([&](const sycl::device &dev) -> int {
      return (dev.get_backend() == sycl::backend::opencl &&
              dev.get_info<sycl::info::device::name>() == name)
                 ? 1
                 : -1;
    });
    sycl_opencl_device_list.push_back(opencl_device);
"""

PATCHED = """    // workaround to get opencl extensions (PATCHED: tolerate missing OpenCL
    // device on BMG multi-root; upstream 3.7.0 removed this block entirely).
    try {
      const auto &name = sycl_devices[i].get_info<sycl::info::device::name>();
      sycl::device opencl_device([&](const sycl::device &dev) -> int {
        return (dev.get_backend() == sycl::backend::opencl &&
                dev.get_info<sycl::info::device::name>() == name)
                   ? 1
                   : -1;
      });
      sycl_opencl_device_list.push_back(opencl_device);
    } catch (const sycl::exception &) {
      // No matching OpenCL device. Leave list short; has_opencl_extension()
      // guards on size() and returns False (safe degradation).
    }
"""

if "PATCHED: tolerate missing OpenCL" in src:
    print("already patched; nothing to do")
    sys.exit(0)
if ORIG not in src:
    print("ERROR: target block not found verbatim; aborting", file=sys.stderr)
    sys.exit(2)

path.write_text(src.replace(ORIG, PATCHED, 1))
print("patched OK:", path)
