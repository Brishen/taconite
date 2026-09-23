// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

#include "iron_xrt_shim.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <list>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"
#include "xrt/detail/ert.h"

static constexpr int kOpcode = 3;
static constexpr size_t kDefaultCtxCache = 16;  // NPU2 concurrent col8 limit

// One cached, loaded kernel: its hw_context, kernel, and staged instruction BO.
struct LoadedKernel {
    std::string key;
    xrt::hw_context context;
    xrt::kernel kernel;
    xrt::bo insts_bo;
    uint32_t insts_nbytes = 0;
};

// The session keeps an LRU cache of resident hardware contexts. Front of the
// list is most-recently-used; the back is evicted first. Keeping every kernel
// resident (when they all fit within TACONITE_CTX_CACHE) means a repeated
// forward pays the hw-context creation cost only once.
struct iron_xrt_session {
    xrt::device device;
    std::list<LoadedKernel> cache;  // front = MRU
    size_t cap = kDefaultCtxCache;
    // Instrumentation (printed on close when TACONITE_TIMING is set).
    double t_reg = 0, t_ctx = 0, t_ker = 0, t_run = 0;
    double t_bo = 0, t_wait = 0;  // sub-breakdown of run
    int n_load = 0, n_run = 0, n_hit = 0, n_evict = 0;
};

using Clock = std::chrono::steady_clock;
static double secs_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static void set_err(char *err, size_t err_len, const std::string &msg) {
    if (err && err_len) {
        std::strncpy(err, msg.c_str(), err_len - 1);
        err[err_len - 1] = '\0';
    }
}

// Find a resident kernel by key, promoting it to most-recently-used.
static LoadedKernel *find_promote(iron_xrt_session *s, const char *key) {
    for (auto it = s->cache.begin(); it != s->cache.end(); ++it) {
        if (it->key == key) {
            if (it != s->cache.begin())
                s->cache.splice(s->cache.begin(), s->cache, it);
            return &s->cache.front();
        }
    }
    return nullptr;
}

extern "C" iron_xrt_session *iron_xrt_open(int device_index, char *err,
                                           size_t err_len) {
    try {
        auto *s = new iron_xrt_session();
        s->device = xrt::device(static_cast<unsigned int>(device_index));
        if (const char *env = std::getenv("TACONITE_CTX_CACHE")) {
            long v = std::strtol(env, nullptr, 10);
            if (v >= 1) s->cap = static_cast<size_t>(v);
        }
        return s;
    } catch (const std::exception &e) {
        set_err(err, err_len, e.what());
        return nullptr;
    }
}

extern "C" int iron_xrt_load(iron_xrt_session *s, const char *key,
                             const char *xclbin_path, const char *insts_path,
                             const char *kernel_name, char *err,
                             size_t err_len) {
    if (!s) {
        set_err(err, err_len, "null session");
        return 1;
    }
    try {
        if (find_promote(s, key)) {
            s->n_hit += 1;
            return 0;  // already resident
        }

        // Read the instruction buffer first, so we only touch the cache once we
        // know we can build the replacement.
        std::ifstream f(insts_path, std::ios::binary | std::ios::ate);
        if (!f) {
            set_err(err, err_len, std::string("cannot open insts ") + insts_path);
            return 2;
        }
        std::streamsize n = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<char> insts(static_cast<size_t>(n));
        if (!f.read(insts.data(), n)) {
            set_err(err, err_len, "failed reading insts");
            return 2;
        }

        xrt::xclbin xclbin(std::string{xclbin_path});
        auto tr = Clock::now();
        auto uuid = s->device.register_xclbin(xclbin);
        s->t_reg += secs_since(tr);

        // Make room, then create the context. A multi-column context can hold
        // the whole array, so the driver caps concurrent contexts; if creation
        // fails (or the cache is full), evict the LRU kernel and retry, down to
        // an empty cache.
        LoadedKernel lk;
        lk.key = key;
        while (true) {
            while (s->cache.size() >= s->cap && !s->cache.empty()) {
                s->cache.pop_back();
                s->n_evict += 1;
            }
            try {
                auto tc = Clock::now();
                lk.context = xrt::hw_context(s->device, uuid);
                s->t_ctx += secs_since(tc);
                break;
            } catch (const std::exception &) {
                if (s->cache.empty()) throw;  // cannot free any more room
                s->cache.pop_back();
                s->n_evict += 1;
            }
        }
        auto tk = Clock::now();
        lk.kernel = xrt::kernel(lk.context, kernel_name);
        s->t_ker += secs_since(tk);
        s->n_load += 1;
        lk.insts_nbytes = static_cast<uint32_t>(n);
        lk.insts_bo = xrt::bo(s->device, static_cast<size_t>(n),
                              xrt::bo::flags::cacheable, lk.kernel.group_id(1));
        std::memcpy(lk.insts_bo.map<char *>(), insts.data(),
                    static_cast<size_t>(n));
        lk.insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        s->cache.push_front(std::move(lk));
        return 0;
    } catch (const std::exception &e) {
        set_err(err, err_len, e.what());
        return 3;
    }
}

extern "C" int iron_xrt_run_conv(iron_xrt_session *s, const char *key,
                                 const void *in, size_t in_bytes,
                                 const void *weights, size_t wt_bytes, void *out,
                                 size_t out_bytes, char *err, size_t err_len) {
    if (!s) {
        set_err(err, err_len, "null session");
        return 1;
    }
    LoadedKernel *lkp = find_promote(s, key);
    if (!lkp) {
        set_err(err, err_len, std::string("kernel not loaded: ") + key);
        return 1;
    }
    LoadedKernel &lk = *lkp;
    try {
        auto trun = Clock::now();
        auto tbo = Clock::now();
        xrt::bo in_bo(s->device, in_bytes, xrt::bo::flags::host_only,
                      lk.kernel.group_id(3));
        xrt::bo wt_bo(s->device, wt_bytes, xrt::bo::flags::host_only,
                      lk.kernel.group_id(4));
        xrt::bo out_bo(s->device, out_bytes, xrt::bo::flags::host_only,
                       lk.kernel.group_id(5));
        std::memcpy(in_bo.map<char *>(), in, in_bytes);
        std::memcpy(wt_bo.map<char *>(), weights, wt_bytes);
        in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        wt_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s->t_bo += secs_since(tbo);

        auto tw = Clock::now();
        auto run = lk.kernel(kOpcode, lk.insts_bo, lk.insts_nbytes, in_bo, wt_bo,
                             out_bo);
        auto state = run.wait();
        s->t_wait += secs_since(tw);
        if (state != ERT_CMD_STATE_COMPLETED) {
            set_err(err, err_len, std::string("kernel state ") +
                                      std::to_string(static_cast<int>(state)));
            return 2;
        }
        out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(out, out_bo.map<char *>(), out_bytes);
        s->t_run += secs_since(trun);
        s->n_run += 1;
        return 0;
    } catch (const std::exception &e) {
        set_err(err, err_len, e.what());
        return 3;
    }
}

extern "C" void iron_xrt_close(iron_xrt_session *s) {
    if (s && std::getenv("TACONITE_TIMING")) {
        std::fprintf(stderr,
                     "[xrt] loads=%d hits=%d evict=%d runs=%d  register=%.2fs "
                     "hw_context=%.2fs kernel=%.2fs run=%.2fs (bo=%.2fs wait=%.2fs)\n",
                     s->n_load, s->n_hit, s->n_evict, s->n_run, s->t_reg,
                     s->t_ctx, s->t_ker, s->t_run, s->t_bo, s->t_wait);
    }
    delete s;
}
