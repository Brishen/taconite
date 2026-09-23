#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
# SPDX-License-Identifier: Apache-2.0
#
# Package a binary built against taconite-adaface into a self-contained
# portable directory that runs with NO XRT installation on the target:
#
#   <out>/<binary>          rpath $ORIGIN/lib (set by the app's build.rs)
#   <out>/lib/*.so.2        the three XRT userspace libraries
#   <out>/<bundle-name>/    the exported model bundle (optional 3rd arg)
#
# XRT locates its root via dladdr() on libxrt_coreutil.so and then scans that
# same lib/ directory for driver plugins, so this layout needs no environment
# variables or /opt/xilinx install. The target still needs the amdxdna kernel
# driver + NPU firmware, access to /dev/accel/accel0, and a memlock limit
# high enough for buffer allocation.
#
# Run where the XRT userspace is installed (default /opt/xilinx/xrt; override
# with XRT_ROOT), e.g. inside the build container.
#
# Usage: package_portable.sh <binary> <outdir> [bundle_dir]

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: $0 <binary> <outdir> [bundle_dir]" >&2
    exit 2
fi

bin=$1
out=$2
bundle=${3:-}
xrt=${XRT_ROOT:-/opt/xilinx/xrt}

[ -f "$bin" ] || { echo "no such binary: $bin" >&2; exit 1; }
[ -d "$xrt/lib" ] || { echo "no XRT libs at $xrt/lib (set XRT_ROOT)" >&2; exit 1; }

mkdir -p "$out/lib"
cp "$bin" "$out/"
# coreutil = the API library the binary links; driver_xdna = the NPU plugin
# (discovered by directory scan + dlopen); xrt_core = the plugin's own dep.
for lib in libxrt_coreutil.so.2 libxrt_driver_xdna.so.2 libxrt_core.so.2; do
    cp "$xrt/lib/$lib" "$out/lib/"
done
if [ -n "$bundle" ]; then
    cp -r "$bundle" "$out/$(basename "$bundle")"
fi

echo "packaged $(basename "$bin") -> $out"
echo "run on the target: $out/$(basename "$bin") <bundle_dir> [reps]"
