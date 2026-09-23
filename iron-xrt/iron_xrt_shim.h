// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* A C surface over the parts of XRT's C++ API a Rust runtime needs to replay
 * kernels compiled by IRON (mlir-aie) on an AMD XDNA NPU: open the device,
 * keep hardware contexts resident, allocate host-visible buffer objects the
 * host writes into directly, and launch a kernel over N of them.
 *
 * Every function that can fail takes an `err` buffer and returns NULL / a
 * non-zero code; nothing here prints or aborts. Handles are owned by the
 * caller and freed with the matching `iron_*_free`; a session must outlive
 * the kernels and buffers created from it.
 */
#ifndef IRON_XRT_SHIM_H
#define IRON_XRT_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iron_session iron_session;
typedef struct iron_kernel iron_kernel;
typedef struct iron_buffer iron_buffer;

/* Open NPU `device_index` (0 on a laptop). NULL on failure, message in err. */
iron_session *iron_open(int device_index, char *err, size_t err_len);
void iron_close(iron_session *s);

/* Register `xclbin_path`, create a hardware context for it (resident until
 * freed), look up `kernel_name` (NULL: the xclbin's first kernel) and load the
 * instruction stream at `insts_path` into a cacheable buffer.
 *
 * `gops` is the work one run of the kernel performs, in GOP (10^9
 * operations, a multiply-add counting two), declared to the driver as the
 * context's QoS `gops` so `xrt-smi examine -r aie-partitions` reports it and
 * a completion rate can be turned into ops per second (the VitisAI provider
 * declares the same field from the compiler's op count). It is a label, not
 * a request: the amdxdna solver only lets QoS pick a DPM level when `fps` or
 * `latency` is set as well, and neither is here, so the context runs at the
 * same (top) level it would without it. 0 declares nothing. */
iron_kernel *iron_kernel_load(iron_session *s,
                              const char *xclbin_path,
                              const char *insts_path,
                              const char *kernel_name,
                              uint32_t gops,
                              char *err,
                              size_t err_len);
void iron_kernel_free(iron_kernel *k);

/* A host-only buffer object of `bytes` bytes, usable as an argument of any
 * kernel of the session. `iron_buffer_map` is its host address for the
 * buffer's lifetime; the caller syncs around kernel runs. */
iron_buffer *iron_buffer_alloc(iron_session *s, size_t bytes, char *err, size_t err_len);
void *iron_buffer_map(iron_buffer *b);
size_t iron_buffer_size(iron_buffer *b);
int iron_buffer_sync_to_device(iron_buffer *b, char *err, size_t err_len);
int iron_buffer_sync_from_device(iron_buffer *b, char *err, size_t err_len);
void iron_buffer_free(iron_buffer *b);
/* A view of `bytes` bytes of `parent` from `offset`, usable as a kernel
 * argument like a buffer of its own (an XRT sub-buffer: the same allocation,
 * a device address `offset` further on). It keeps the allocation alive on
 * its own, so it may outlive `parent`; syncs cover its bytes only. */
iron_buffer *iron_buffer_sub(iron_buffer *parent, size_t offset, size_t bytes, char *err, size_t err_len);

/* Launch `k` with `n` argument buffers (in the kernel's argument order,
 * after the fixed opcode / instructions / instruction-size triple) and wait
 * for completion. Returns 0 on ERT_CMD_STATE_COMPLETED; otherwise non-zero
 * with the state or exception in err. `elapsed_ns`, when non-NULL, receives
 * the wall time of launch + wait.
 *
 * The kernel keeps the XRT run object it built for each distinct argument
 * tuple and restarts it on the next launch over the same buffers, so a
 * caller that cycles through a fixed set of tuples pays the command setup
 * once per tuple, not per launch. Freeing a buffer drops the runs that
 * referenced it. */
int iron_kernel_run(iron_kernel *k, iron_buffer **bufs, size_t n, uint64_t *elapsed_ns, char *err, size_t err_len);

/* The same launch split in two, so the host can work while the array does:
 * `iron_kernel_start` submits and returns a run handle (NULL on failure);
 * `iron_run_wait` blocks until it completes, reports the state as
 * `iron_kernel_run` does, and frees the handle either way. */
typedef struct iron_run iron_run;
iron_run *iron_kernel_start(iron_kernel *k, iron_buffer **bufs, size_t n, char *err, size_t err_len);
int iron_run_wait(iron_run *r, uint64_t *elapsed_ns, char *err, size_t err_len);

#ifdef __cplusplus
}
#endif

#endif /* IRON_XRT_SHIM_H */
