// SGP4 on CUDA. The kernel calls the same sgp4() as the CPU build (include/kessler/sgp4.hpp).
//
// Work item k = (object i, time j) with time fastest, so the threads of a warp usually share
// one object: they read the same Elset fields (a broadcast) and differ only in t.
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "kessler/screen.hpp"
#include "kessler/sgp4.hpp"

using kessler::Candidate;
using kessler::Elset;

namespace {

__global__ void propagate_kernel(const Elset* __restrict__ s, int64_t n,
                                 const double* __restrict__ tsince, int64_t m,
                                 double* __restrict__ r, double* __restrict__ v,
                                 int32_t* __restrict__ err) {
    int64_t k = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (k >= n * m) return;
    int64_t i = k / m;
    int64_t j = k - i * m;
    double rr[3], vv[3];
    int32_t e = kessler::sgp4(s[i], tsince[j], rr, vv);
    r[3 * k] = rr[0];
    r[3 * k + 1] = rr[1];
    r[3 * k + 2] = rr[2];
    if (v) {
        v[3 * k] = vv[0];
        v[3 * k + 1] = vv[1];
        v[3 * k + 2] = vv[2];
    }
    err[k] = e;
}

// One screening step: every object's state at time t (seconds after the start).
__global__ void step_states(const Elset* __restrict__ s, const double* __restrict__ off,
                            int64_t n, double t, double cell, double* __restrict__ r,
                            double* __restrict__ v, uint64_t* __restrict__ keys,
                            int32_t* __restrict__ ids) {
    int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    double rr[3], vv[3];
    int32_t e = kessler::sgp4(s[i], off[i] + t / 60.0, rr, vv);
    for (int c = 0; c < 3; ++c) {
        r[3 * i + c] = rr[c];
        v[3 * i + c] = vv[c];
    }
    ids[i] = static_cast<int32_t>(i);
    keys[i] = e == kessler::kOk
                  ? kessler::cell_key(kessler::cell_coord(rr[0], cell), kessler::cell_coord(rr[1], cell),
                                      kessler::cell_coord(rr[2], cell))
                  : ~uint64_t(0);  // invalid objects sort last and are skipped
}

__device__ int64_t lower_bound_key(const uint64_t* keys, int64_t n, uint64_t key) {
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        if (keys[mid] < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// Pairs from the 27 neighbouring cells of each object, tested as on the CPU (src/screen.cpp).
__global__ void step_pairs(const uint64_t* __restrict__ keys, const int32_t* __restrict__ ids,
                           int64_t n, const double* __restrict__ r, const double* __restrict__ v,
                           const double* __restrict__ vmax, double cell, double h, double dmax,
                           double margin, int32_t k, Candidate* __restrict__ out,
                           unsigned long long* __restrict__ count, int64_t cap) {
    int64_t q = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (q >= n || keys[q] == ~uint64_t(0)) return;
    int32_t i = ids[q];
    const double* ri = r + 3 * i;
    const double* vi = v + 3 * i;
    int64_t cx = kessler::cell_coord(ri[0], cell), cy = kessler::cell_coord(ri[1], cell),
            cz = kessler::cell_coord(ri[2], cell);
    for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dz = -1; dz <= 1; ++dz) {
                uint64_t key = kessler::cell_key(cx + dx, cy + dy, cz + dz);
                for (int64_t p = lower_bound_key(keys, n, key); p < n && keys[p] == key; ++p) {
                    int32_t j = ids[p];
                    if (j <= i) continue;
                    const double* rj = r + 3 * j;
                    double ex = rj[0] - ri[0], ey = rj[1] - ri[1], ez = rj[2] - ri[2];
                    double reach = dmax + margin + (vmax[i] + vmax[j]) * h;
                    if (ex * ex + ey * ey + ez * ez > reach * reach) continue;
                    double dl = kessler::window_min_distance(ri, vi, rj, v + 3 * j, h);
                    if (dl < dmax + margin) {
                        unsigned long long slot = atomicAdd(count, 1ULL);
                        if (static_cast<int64_t>(slot) < cap) out[slot] = {i, j, k, static_cast<float>(dl)};
                    }
                }
            }
}

bool check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kessler cuda: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

}  // namespace

extern "C" {

int kessler_cuda_device_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

// Propagates n host element sets to m times on the GPU and copies the results back.
// v may be null. Writes the kernel time (ms, averaged over `repeat` launches) to kernel_ms.
// Returns 0 on success.
int kessler_cuda_propagate(const Elset* s, int64_t n, const double* tsince, int64_t m, double* r,
                           double* v, int32_t* err, int repeat, float* kernel_ms) {
    Elset* d_s = nullptr;
    double *d_t = nullptr, *d_r = nullptr, *d_v = nullptr;
    int32_t* d_err = nullptr;
    int64_t total = n * m;
    bool ok = check(cudaMalloc(&d_s, n * sizeof(Elset)), "malloc elsets") &&
              check(cudaMalloc(&d_t, m * sizeof(double)), "malloc times") &&
              check(cudaMalloc(&d_r, total * 3 * sizeof(double)), "malloc r") &&
              (v == nullptr || check(cudaMalloc(&d_v, total * 3 * sizeof(double)), "malloc v")) &&
              check(cudaMalloc(&d_err, total * sizeof(int32_t)), "malloc err") &&
              check(cudaMemcpy(d_s, s, n * sizeof(Elset), cudaMemcpyHostToDevice), "copy elsets") &&
              check(cudaMemcpy(d_t, tsince, m * sizeof(double), cudaMemcpyHostToDevice), "copy times");

    if (ok) {
        const int threads = 256;
        int64_t blocks = (total + threads - 1) / threads;
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        propagate_kernel<<<blocks, threads>>>(d_s, n, d_t, m, d_r, d_v, d_err);  // warm-up
        ok = check(cudaGetLastError(), "launch") && check(cudaDeviceSynchronize(), "warm-up");
        if (ok) {
            if (repeat < 1) repeat = 1;
            cudaEventRecord(a);
            for (int q = 0; q < repeat; ++q)
                propagate_kernel<<<blocks, threads>>>(d_s, n, d_t, m, d_r, d_v, d_err);
            cudaEventRecord(b);
            ok = check(cudaEventSynchronize(b), "timed launches");
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, a, b);
            if (kernel_ms) *kernel_ms = ms / repeat;
        }
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    if (ok) {
        ok = check(cudaMemcpy(r, d_r, total * 3 * sizeof(double), cudaMemcpyDeviceToHost), "copy r") &&
             (v == nullptr ||
              check(cudaMemcpy(v, d_v, total * 3 * sizeof(double), cudaMemcpyDeviceToHost), "copy v")) &&
             check(cudaMemcpy(err, d_err, total * sizeof(int32_t), cudaMemcpyDeviceToHost), "copy err");
    }
    cudaFree(d_s);
    cudaFree(d_t);
    cudaFree(d_r);
    cudaFree(d_v);
    cudaFree(d_err);
    return ok ? 0 : 1;
}

// GPU version of kessler_screen_cpu (same arguments and result; candidate order differs).
// Writes the time spent on the GPU, in ms, to gpu_ms. Returns -1 on a CUDA error.
int64_t kessler_cuda_screen(const Elset* s, const double* off, const double* vmax, int64_t n,
                            double t0, double step, int64_t nsteps, double dmax, Candidate* out,
                            int64_t cap, float* gpu_ms) {
    const double h = 0.5 * step;
    const double margin = kessler::window_margin(h);
    double vbig = 0.0;
    for (int64_t i = 0; i < n; ++i) vbig = std::max(vbig, vmax[i]);
    const double cell = dmax + margin + 2.0 * vbig * h;

    Elset* d_s = nullptr;
    double *d_off = nullptr, *d_vmax = nullptr, *d_r = nullptr, *d_v = nullptr;
    uint64_t* d_keys = nullptr;
    int32_t* d_ids = nullptr;
    Candidate* d_out = nullptr;
    unsigned long long* d_count = nullptr;
    int64_t dcap = std::max<int64_t>(cap, 1);
    bool ok = check(cudaMalloc(&d_s, n * sizeof(Elset)), "malloc elsets") &&
              check(cudaMalloc(&d_off, n * sizeof(double)), "malloc off") &&
              check(cudaMalloc(&d_vmax, n * sizeof(double)), "malloc vmax") &&
              check(cudaMalloc(&d_r, n * 3 * sizeof(double)), "malloc r") &&
              check(cudaMalloc(&d_v, n * 3 * sizeof(double)), "malloc v") &&
              check(cudaMalloc(&d_keys, n * sizeof(uint64_t)), "malloc keys") &&
              check(cudaMalloc(&d_ids, n * sizeof(int32_t)), "malloc ids") &&
              check(cudaMalloc(&d_out, dcap * sizeof(Candidate)), "malloc candidates") &&
              check(cudaMalloc(&d_count, sizeof(unsigned long long)), "malloc count") &&
              check(cudaMemcpy(d_s, s, n * sizeof(Elset), cudaMemcpyHostToDevice), "copy elsets") &&
              check(cudaMemcpy(d_off, off, n * sizeof(double), cudaMemcpyHostToDevice), "copy off") &&
              check(cudaMemcpy(d_vmax, vmax, n * sizeof(double), cudaMemcpyHostToDevice), "copy vmax") &&
              check(cudaMemset(d_count, 0, sizeof(unsigned long long)), "zero count");

    unsigned long long total = 0;
    if (ok) {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a);
        const int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        thrust::device_ptr<uint64_t> kp(d_keys);
        thrust::device_ptr<int32_t> ip(d_ids);
        for (int64_t k = 0; k < nsteps && ok; ++k) {
            step_states<<<blocks, threads>>>(d_s, d_off, n, t0 + k * step, cell, d_r, d_v, d_keys, d_ids);
            thrust::sort_by_key(thrust::device, kp, kp + n, ip);
            step_pairs<<<blocks, threads>>>(d_keys, d_ids, n, d_r, d_v, d_vmax, cell, h, dmax, margin,
                                            static_cast<int32_t>(k), d_out, d_count, dcap);
            ok = check(cudaGetLastError(), "screen step");
        }
        cudaEventRecord(b);
        ok = ok && check(cudaEventSynchronize(b), "screen");
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        if (gpu_ms) *gpu_ms = ms;
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        ok = ok && check(cudaMemcpy(&total, d_count, sizeof(total), cudaMemcpyDeviceToHost), "copy count");
        int64_t got = std::min<int64_t>(static_cast<int64_t>(total), cap);
        if (ok && got > 0)
            ok = check(cudaMemcpy(out, d_out, got * sizeof(Candidate), cudaMemcpyDeviceToHost), "copy candidates");
    }
    cudaFree(d_s);
    cudaFree(d_off);
    cudaFree(d_vmax);
    cudaFree(d_r);
    cudaFree(d_v);
    cudaFree(d_keys);
    cudaFree(d_ids);
    cudaFree(d_out);
    cudaFree(d_count);
    return ok ? static_cast<int64_t>(total) : -1;
}

}  // extern "C"
