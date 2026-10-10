// SGP4 on CUDA. The kernel calls the same sgp4() as the CPU build (include/kessler/sgp4.hpp).
//
// Work item k = (object i, time j) with time fastest, so the threads of a warp usually share
// one object: they read the same Elset fields (a broadcast) and differ only in t.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#include "kessler/sgp4.hpp"

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

}  // extern "C"
