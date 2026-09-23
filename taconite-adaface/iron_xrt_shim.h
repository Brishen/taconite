/* SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

/*
 * Flat C ABI over XRT for replaying an iron-compiled network on the NPU.
 *
 * This exposes a persistent session: one xrt::device shared across many kernels,
 * with each (xclbin, kernel) loaded and cached under a key. Unlike the DETR
 * shim (which keeps exactly one hw_context live and reloads on every kernel
 * change), this keeps an LRU cache of up to TACONITE_CTX_CACHE resident
 * hardware contexts (default 16 -- the NPU2 concurrent-context limit measured
 * for these col8 kernels). A network whose distinct kernels all fit reloads
 * nothing after the first forward, so repeated inferences amortize the
 * hw-context creation cost. The NPU call convention matches mlir-aie's XRT
 * hostruntime:
 *
 *   kernel(opcode=3, insts_bo, insts_nbytes, in_bo, weights_bo, out_bo)
 */

#ifndef IRON_XRT_SHIM_H
#define IRON_XRT_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iron_xrt_session iron_xrt_session;

/* Open the NPU (device_index, usually 0). NULL on error (message in err). */
iron_xrt_session *iron_xrt_open(int device_index, char *err, size_t err_len);

/* Load an (xclbin, kernel_name) with its instruction buffer, cached under
 * `key` (any stable string, e.g. the xclbin basename). A key already resident
 * in the LRU cache is a no-op (just promoted to most-recently-used); otherwise
 * the context is created, evicting the least-recently-used kernel(s) if the
 * cache is full or the driver rejects another concurrent context. Returns 0 on
 * success. */
int iron_xrt_load(iron_xrt_session *s, const char *key, const char *xclbin_path,
                  const char *insts_path, const char *kernel_name, char *err,
                  size_t err_len);

/* Run the kernel cached under `key` on three host buffers (input, weights,
 * output) matching the Conv2d ABI. `in`/`weights` are copied to the device;
 * `out` receives the result. Returns 0 on success. */
int iron_xrt_run_conv(iron_xrt_session *s, const char *key, const void *in,
                      size_t in_bytes, const void *weights, size_t wt_bytes,
                      void *out, size_t out_bytes, char *err, size_t err_len);

void iron_xrt_close(iron_xrt_session *s);

#ifdef __cplusplus
}
#endif

#endif /* IRON_XRT_SHIM_H */
