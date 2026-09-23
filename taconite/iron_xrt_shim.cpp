// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

// See iron_xrt_shim.h. Mirrors what IRON's Python host runtime does per
// kernel launch (`kernel(3, insts_bo, insts_bytes, *buffers)`, instruction
// buffer `cacheable` in the kernel's group 1, argument buffers `host_only`)
// — the same calls, with the buffers mapped once and reused.
#include "iron_xrt_shim.h"

#include "xrt/detail/ert.h"
#include "xrt/experimental/xrt_xclbin.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

// IRON's kernels take (opcode, instructions, instruction bytes, args...).
constexpr int kOpcode = 3;

void set_err(char *err, size_t err_len, const std::string &msg)
{
    if (!err || err_len == 0)
        return;
    std::strncpy(err, msg.c_str(), err_len - 1);
    err[err_len - 1] = '\0';
}

} // namespace

struct iron_kernel;

// A loaded xclbin's hardware context and kernel object, shared by every
// iron_kernel that names that xclbin (keyed by path + kernel name). Two
// kernels on one context run with their own instruction streams and never
// switch contexts between them; a switch reconfigures every tile of the
// incoming design, measured ~90 us a tile on NPU2 (2.9 ms for an 8-column
// GEMM). The QoS is the first loader's: the context is one resource.
struct loaded_xclbin {
    xrt::hw_context context;
    xrt::kernel kernel;
};

struct iron_session {
    xrt::device device;
    // Buffers get a session-unique id, so a run cached against one can
    // never match a later buffer that happens to reuse its address.
    uint64_t next_buffer_id = 1;
    // Every live kernel, so freeing a buffer can drop the runs holding it.
    std::vector<iron_kernel *> kernels;
    std::map<std::string, loaded_xclbin> xclbins;
};

// An xrt::run is restartable once it has completed, and building one (a
// command buffer from XRT's pool plus one set_arg per argument) is the
// fixed cost of every launch. The tagger launches ~900 times a forward over
// a fixed set of argument tuples, so each kernel keeps the run it built for
// a tuple and restarts it. `in_flight` covers the one case a cached run
// can't serve: the same tuple queued again before its first launch is
// waited — that launch gets a one-off run instead.
struct cached_run {
    xrt::run run;
    bool in_flight = false;
};

struct tuple_hash {
    size_t operator()(const std::vector<uint64_t> &v) const noexcept
    {
        size_t h = v.size();
        for (uint64_t x : v)
            h ^= std::hash<uint64_t>{}(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct iron_kernel {
    iron_session *session = nullptr;
    // The `xclbins` entry this kernel runs on; freed with its last kernel.
    std::string xclbin_key;
    xrt::hw_context context;
    xrt::kernel kernel;
    xrt::bo insts_bo;
    size_t insts_bytes = 0;
    // Node-based, so a pointer to an entry survives rehashing.
    std::unordered_map<std::vector<uint64_t>, cached_run, tuple_hash> runs;
};

struct iron_buffer {
    xrt::bo bo;
    void *host = nullptr;
    size_t bytes = 0;
    uint64_t id = 0;
    iron_session *session = nullptr;
};

struct iron_run {
    // Points into `iron_kernel::runs` when the launch reused a cached run
    // (released on wait); otherwise `fresh` holds a one-off run.
    cached_run *cached = nullptr;
    xrt::run fresh;
    std::chrono::steady_clock::time_point started;
    xrt::run &run()
    {
        return cached ? cached->run : fresh;
    }
};

namespace
{

xrt::run build_run(iron_kernel *k, iron_buffer **bufs, size_t n)
{
    // xrt::kernel's operator() is variadic over the argument count, so the
    // argument buffers are set positionally on an xrt::run instead.
    xrt::run run(k->kernel);
    run.set_arg(0, static_cast<uint64_t>(kOpcode));
    run.set_arg(1, k->insts_bo);
    run.set_arg(2, static_cast<uint32_t>(k->insts_bytes));
    for (size_t i = 0; i < n; ++i) {
        run.set_arg(static_cast<int>(3 + i), bufs[i]->bo);
    }
    return run;
}

// The cached run for this argument tuple, built on first use and marked in
// flight; NULL when it already is (the caller then builds a one-off).
cached_run *acquire_run(iron_kernel *k, iron_buffer **bufs, size_t n)
{
    std::vector<uint64_t> key(n);
    for (size_t i = 0; i < n; ++i)
        key[i] = bufs[i]->id;
    auto it = k->runs.find(key);
    if (it == k->runs.end()) {
        it = k->runs.emplace(std::move(key), cached_run{build_run(k, bufs, n), false}).first;
    } else if (it->second.in_flight) {
        return nullptr;
    }
    it->second.in_flight = true;
    return &it->second;
}

} // namespace

extern "C" {

iron_session *iron_open(int device_index, char *err, size_t err_len)
{
    try {
        auto *s = new iron_session;
        s->device = xrt::device(static_cast<unsigned int>(device_index));
        return s;
    } catch (const std::exception &e) {
        set_err(err, err_len, std::string("xrt::device: ") + e.what());
        return nullptr;
    }
}

void iron_close(iron_session *s)
{
    delete s;
}

iron_kernel *iron_kernel_load(iron_session *s,
                              const char *xclbin_path,
                              const char *insts_path,
                              const char *kernel_name,
                              uint32_t gops,
                              char *err,
                              size_t err_len)
{
    if (!s) {
        set_err(err, err_len, "null session");
        return nullptr;
    }
    try {
        std::ifstream f(insts_path, std::ios::binary | std::ios::ate);
        if (!f) {
            set_err(err, err_len, std::string("cannot open instructions ") + insts_path);
            return nullptr;
        }
        std::streamsize n = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<char> insts(static_cast<size_t>(n));
        if (n > 0 && !f.read(insts.data(), n)) {
            set_err(err, err_len, std::string("failed reading ") + insts_path);
            return nullptr;
        }

        xrt::xclbin xclbin(std::string{xclbin_path});
        std::string name;
        if (kernel_name && *kernel_name) {
            name = kernel_name;
        } else {
            auto kernels = xclbin.get_kernels();
            if (kernels.empty()) {
                set_err(err, err_len, std::string("no kernels in ") + xclbin_path);
                return nullptr;
            }
            name = kernels.front().get_name();
        }

        std::string key = std::string{xclbin_path} + "\n" + name;
        auto found = s->xclbins.find(key);
        if (found == s->xclbins.end()) {
            auto uuid = s->device.register_xclbin(xclbin);
            loaded_xclbin lx;
            if (gops > 0) {
                xrt::hw_context::cfg_param_type qos{{"gops", gops}};
                lx.context = xrt::hw_context(s->device, uuid, qos);
            } else {
                lx.context = xrt::hw_context(s->device, uuid);
            }
            lx.kernel = xrt::kernel(lx.context, name);
            found = s->xclbins.emplace(key, std::move(lx)).first;
        }
        auto *k = new iron_kernel;
        k->session = s;
        k->xclbin_key = key;
        k->context = found->second.context;
        k->kernel = found->second.kernel;
        k->insts_bytes = static_cast<size_t>(n);
        k->insts_bo = xrt::bo(s->device, k->insts_bytes, xrt::bo::flags::cacheable, k->kernel.group_id(1));
        std::memcpy(k->insts_bo.map<char *>(), insts.data(), k->insts_bytes);
        k->insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s->kernels.push_back(k);
        return k;
    } catch (const std::exception &e) {
        set_err(err, err_len, std::string("loading ") + xclbin_path + ": " + e.what());
        return nullptr;
    }
}

void iron_kernel_free(iron_kernel *k)
{
    if (!k)
        return;
    auto *s = k->session;
    std::string key = k->xclbin_key;
    auto &ks = s->kernels;
    ks.erase(std::remove(ks.begin(), ks.end(), k), ks.end());
    delete k;
    // The last kernel on a context releases it, so a caller can make room
    // for another (NPU2 holds at most 16 contexts, across all processes).
    bool used = std::any_of(ks.begin(), ks.end(), [&](const iron_kernel *o) { return o->xclbin_key == key; });
    if (!used)
        s->xclbins.erase(key);
}

iron_buffer *iron_buffer_alloc(iron_session *s, size_t bytes, char *err, size_t err_len)
{
    if (!s) {
        set_err(err, err_len, "null session");
        return nullptr;
    }
    try {
        auto *b = new iron_buffer;
        // Group 0, like IRON's XRTTensor, so any kernel can take it, and
        // `host_only` like it too: `cacheable` BOs come from a small
        // device-side pool (a few of these 3 MB buffers and CREATE_BO fails
        // with ENOSPC). A host_only mapping is *uncached* on the host, so
        // callers should move rows in and out of it with wide copies rather
        // than element-wise loads — see the tagger's iron_backend.
        b->bo = xrt::bo(s->device, bytes, xrt::bo::flags::host_only, 0);
        b->host = b->bo.map<void *>();
        b->bytes = bytes;
        b->id = s->next_buffer_id++;
        b->session = s;
        std::memset(b->host, 0, bytes);
        b->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return b;
    } catch (const std::exception &e) {
        set_err(err, err_len, std::string("xrt::bo(") + std::to_string(bytes) + "): " + e.what());
        return nullptr;
    }
}

iron_buffer *iron_buffer_sub(iron_buffer *parent, size_t offset, size_t bytes, char *err, size_t err_len)
{
    if (!parent) {
        set_err(err, err_len, "null parent buffer");
        return nullptr;
    }
    if (offset + bytes > parent->bytes) {
        set_err(err,
                err_len,
                "sub-buffer [" + std::to_string(offset) + ", +" + std::to_string(bytes) + ") outside a buffer of " +
                    std::to_string(parent->bytes) + " bytes");
        return nullptr;
    }
    try {
        auto *b = new iron_buffer;
        // xrt::bo's sub-buffer constructor shares the parent's allocation and
        // holds it by reference, so the view stands on its own; the host
        // pointer is the parent's mapping (owned by that allocation) offset.
        b->bo = xrt::bo(parent->bo, bytes, offset);
        b->host = static_cast<char *>(parent->host) + offset;
        b->bytes = bytes;
        b->id = parent->session->next_buffer_id++;
        b->session = parent->session;
        return b;
    } catch (const std::exception &e) {
        set_err(err,
                err_len,
                std::string("xrt::bo sub-buffer(") + std::to_string(offset) + ", " + std::to_string(bytes) +
                    "): " + e.what());
        return nullptr;
    }
}

void *iron_buffer_map(iron_buffer *b)
{
    return b ? b->host : nullptr;
}
size_t iron_buffer_size(iron_buffer *b)
{
    return b ? b->bytes : 0;
}

int iron_buffer_sync_to_device(iron_buffer *b, char *err, size_t err_len)
{
    try {
        b->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return 0;
    } catch (const std::exception &e) {
        set_err(err, err_len, std::string("sync to device: ") + e.what());
        return 1;
    }
}

int iron_buffer_sync_from_device(iron_buffer *b, char *err, size_t err_len)
{
    try {
        b->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        return 0;
    } catch (const std::exception &e) {
        set_err(err, err_len, std::string("sync from device: ") + e.what());
        return 1;
    }
}

void iron_buffer_free(iron_buffer *b)
{
    if (!b)
        return;
    // A cached run holds a reference to the BO: drop those runs so the
    // memory goes with the buffer rather than lingering until the kernel
    // is freed. (Freeing a buffer a launch is still using is the caller's
    // bug; the Rust side's borrows rule it out.)
    for (iron_kernel *k : b->session->kernels) {
        for (auto it = k->runs.begin(); it != k->runs.end();) {
            bool uses = false;
            for (uint64_t id : it->first)
                uses |= id == b->id;
            it = uses ? k->runs.erase(it) : std::next(it);
        }
    }
    delete b;
}

int iron_kernel_run(iron_kernel *k, iron_buffer **bufs, size_t n, uint64_t *elapsed_ns, char *err, size_t err_len)
{
    if (!k) {
        set_err(err, err_len, "null kernel");
        return 1;
    }
    try {
        cached_run *cached = acquire_run(k, bufs, n);
        xrt::run fresh;
        if (!cached)
            fresh = build_run(k, bufs, n);
        xrt::run &run = cached ? cached->run : fresh;
        auto t0 = std::chrono::steady_clock::now();
        run.start();
        auto state = run.wait();
        auto t1 = std::chrono::steady_clock::now();
        if (cached)
            cached->in_flight = false;
        if (elapsed_ns) {
            *elapsed_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        }
        if (state != ERT_CMD_STATE_COMPLETED) {
            set_err(err, err_len, "kernel finished in state " + std::to_string(static_cast<int>(state)));
            return 2;
        }
        return 0;
    } catch (const std::exception &e) {
        set_err(err, err_len, std::string("kernel run: ") + e.what());
        return 1;
    }
}

iron_run *iron_kernel_start(iron_kernel *k, iron_buffer **bufs, size_t n, char *err, size_t err_len)
{
    if (!k) {
        set_err(err, err_len, "null kernel");
        return nullptr;
    }
    try {
        auto *r = new iron_run;
        r->cached = acquire_run(k, bufs, n);
        if (!r->cached)
            r->fresh = build_run(k, bufs, n);
        r->started = std::chrono::steady_clock::now();
        r->run().start();
        return r;
    } catch (const std::exception &e) {
        set_err(err, err_len, std::string("kernel start: ") + e.what());
        return nullptr;
    }
}

int iron_run_wait(iron_run *r, uint64_t *elapsed_ns, char *err, size_t err_len)
{
    if (!r) {
        set_err(err, err_len, "null run");
        return 1;
    }
    int rc = 0;
    try {
        auto state = r->run().wait();
        auto t1 = std::chrono::steady_clock::now();
        if (r->cached)
            r->cached->in_flight = false;
        if (elapsed_ns) {
            *elapsed_ns =
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - r->started).count());
        }
        if (state != ERT_CMD_STATE_COMPLETED) {
            set_err(err, err_len, "kernel finished in state " + std::to_string(static_cast<int>(state)));
            rc = 2;
        }
    } catch (const std::exception &e) {
        set_err(err, err_len, std::string("kernel wait: ") + e.what());
        rc = 1;
    }
    delete r;
    return rc;
}

} // extern "C"
