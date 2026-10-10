// C API over the CPU propagator, for Python (ctypes) and tests.
#include <cstddef>
#include <cstdint>

#include "kessler/sgp4.hpp"

using kessler::Elset;
using kessler::MeanElements;

extern "C" {

size_t kessler_elset_size() { return sizeof(Elset); }

// Initialises n element sets. err[i] is nonzero if object i cannot be propagated.
void kessler_init(const MeanElements* m, int64_t n, Elset* out, int32_t* err) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        out[i] = kessler::sgp4init(m[i]);
        err[i] = out[i].error;
    }
}

// Propagates n objects to the same m times (minutes since each object's epoch).
// r and v are [n][m][3] in km and km/s (TEME); err is [n][m].
void kessler_propagate(const Elset* s, int64_t n, const double* tsince, int64_t m, double* r,
                       double* v, int32_t* err) {
#pragma omp parallel for schedule(dynamic, 64)
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < m; ++j) {
            int64_t k = i * m + j;
            err[k] = kessler::sgp4(s[i], tsince[j], r + 3 * k, v + 3 * k);
        }
    }
}

// Propagates each object to its own time: pairs (s[idx[k]], tsince[k]).
void kessler_propagate_at(const Elset* s, const int64_t* idx, const double* tsince, int64_t k,
                          double* r, double* v, int32_t* err) {
#pragma omp parallel for schedule(static)
    for (int64_t q = 0; q < k; ++q) {
        err[q] = kessler::sgp4(s[idx[q]], tsince[q], r + 3 * q, v + 3 * q);
    }
}

}  // extern "C"
