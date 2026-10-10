// CPU conjunction screener, exact refinement of candidates, and a brute-force reference.
//
// Times here are seconds after the screening start T0. Object i's SGP4 time is
// off[i] + t / 60 minutes, where off[i] is T0 minus its epoch, in minutes.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "kessler/screen.hpp"
#include "kessler/sgp4.hpp"

using namespace kessler;

namespace {

struct State {
    double r[3], v[3];
    bool ok;
};

inline State state_at(const Elset& s, double off_min, double t_s) {
    State st;
    st.ok = sgp4(s, off_min + t_s / 60.0, st.r, st.v) == kOk;
    return st;
}

inline double dist2(const State& a, const State& b) {
    double dx = b.r[0] - a.r[0], dy = b.r[1] - a.r[1], dz = b.r[2] - a.r[2];
    return dx * dx + dy * dy + dz * dz;
}

// Brent's method: minimum of f on [a, b]. Returns the abscissa; fmin receives f there.
template <class F>
double brent_min(F f, double a, double b, double tol, double& fmin) {
    const double cgold = 0.3819660112501051;
    double x = a + cgold * (b - a), w = x, v = x;
    double fx = f(x), fw = fx, fv = fx;
    double d = 0.0, e = 0.0;
    for (int iter = 0; iter < 200; ++iter) {
        double xm = 0.5 * (a + b);
        double tol1 = tol * std::fabs(x) + 1e-10, tol2 = 2.0 * tol1;
        if (std::fabs(x - xm) <= tol2 - 0.5 * (b - a)) break;
        bool golden = true;
        if (std::fabs(e) > tol1) {
            double r = (x - w) * (fx - fv);
            double q = (x - v) * (fx - fw);
            double p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p;
            q = std::fabs(q);
            double etemp = e;
            e = d;
            if (!(std::fabs(p) >= std::fabs(0.5 * q * etemp) || p <= q * (a - x) || p >= q * (b - x))) {
                d = p / q;
                double u = x + d;
                if (u - a < tol2 || b - u < tol2) d = (xm - x >= 0.0) ? tol1 : -tol1;
                golden = false;
            }
        }
        if (golden) {
            e = (x >= xm) ? a - x : b - x;
            d = cgold * e;
        }
        double u = (std::fabs(d) >= tol1) ? x + d : x + ((d >= 0.0) ? tol1 : -tol1);
        double fu = f(u);
        if (fu <= fx) {
            if (u >= x) a = x; else b = x;
            v = w; fv = fw;
            w = x; fw = fx;
            x = u; fx = fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) {
                v = w; fv = fw;
                w = u; fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u; fv = fu;
            }
        }
    }
    fmin = fx;
    return x;
}

struct Grid {
    std::vector<uint64_t> keys;   // sorted
    std::vector<int32_t> ids;     // object index for each sorted key
};

void build_grid(const std::vector<State>& st, double cell, Grid& g) {
    std::vector<std::pair<uint64_t, int32_t>> kv;
    kv.reserve(st.size());
    for (size_t i = 0; i < st.size(); ++i) {
        if (!st[i].ok) continue;
        kv.emplace_back(cell_key(cell_coord(st[i].r[0], cell), cell_coord(st[i].r[1], cell),
                                 cell_coord(st[i].r[2], cell)),
                        static_cast<int32_t>(i));
    }
    std::sort(kv.begin(), kv.end());
    g.keys.resize(kv.size());
    g.ids.resize(kv.size());
    for (size_t q = 0; q < kv.size(); ++q) {
        g.keys[q] = kv[q].first;
        g.ids[q] = kv[q].second;
    }
}

}  // namespace

extern "C" {

// Candidate (pair, sample) for every sample k in [0, nsteps) at t_k = t0 + k * step seconds.
// vmax[i] bounds object i's speed (km/s). Writes up to cap candidates; returns how many there
// are in total (call again with a larger buffer if that exceeds cap).
int64_t kessler_screen_cpu(const Elset* s, const double* off, const double* vmax, int64_t n,
                           double t0, double step, int64_t nsteps, double dmax, Candidate* out,
                           int64_t cap) {
    const double h = 0.5 * step;
    const double margin = window_margin(h);
    double vbig = 0.0;
    for (int64_t i = 0; i < n; ++i) vbig = std::max(vbig, vmax[i]);
    const double cell = dmax + margin + 2.0 * vbig * h;

    int64_t total = 0;
#pragma omp parallel
    {
        std::vector<State> st(n);
        Grid g;
        std::vector<Candidate> local;
#pragma omp for schedule(dynamic, 4)
        for (int64_t k = 0; k < nsteps; ++k) {
            double t = t0 + k * step;
            for (int64_t i = 0; i < n; ++i) st[i] = state_at(s[i], off[i], t);
            build_grid(st, cell, g);
            for (size_t q = 0; q < g.ids.size(); ++q) {
                int32_t i = g.ids[q];
                const State& a = st[i];
                int64_t cx = cell_coord(a.r[0], cell), cy = cell_coord(a.r[1], cell),
                        cz = cell_coord(a.r[2], cell);
                for (int dx = -1; dx <= 1; ++dx)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dz = -1; dz <= 1; ++dz) {
                            uint64_t key = cell_key(cx + dx, cy + dy, cz + dz);
                            auto lo = std::lower_bound(g.keys.begin(), g.keys.end(), key);
                            for (auto it = lo; it != g.keys.end() && *it == key; ++it) {
                                int32_t j = g.ids[it - g.keys.begin()];
                                if (j <= i) continue;
                                const State& b = st[j];
                                double reach = dmax + margin + (vmax[i] + vmax[j]) * h;
                                if (dist2(a, b) > reach * reach) continue;
                                double dl = window_min_distance(a.r, a.v, b.r, b.v, h);
                                if (dl < dmax + margin)
                                    local.push_back({i, j, static_cast<int32_t>(k), static_cast<float>(dl)});
                            }
                        }
            }
        }
#pragma omp critical
        {
            for (const Candidate& c : local) {
                if (total < cap) out[total] = c;
                ++total;
            }
        }
    }
    return total;
}

// Refines each candidate to its exact closest approach within the sample's window (widened by
// `pad` seconds so a minimum on a window edge is not lost). Keeps local minima closer than dmax,
// one per pair and approach. Returns the number of events written to out (at most nc).
int64_t kessler_refine(const Elset* s, const double* off, const Candidate* c, int64_t nc,
                       double t0, double step, double pad, double dmax, Event* out) {
    const double h = 0.5 * step;
    std::vector<Event> ev(nc);
    std::vector<char> keep(nc, 0);
#pragma omp parallel for schedule(dynamic, 64)
    for (int64_t q = 0; q < nc; ++q) {
        const Elset& a = s[c[q].i];
        const Elset& b = s[c[q].j];
        double oa = off[c[q].i], ob = off[c[q].j];
        double tk = t0 + c[q].k * step;
        double lo = tk - h - pad, hi = tk + h + pad;
        bool bad = false;
        auto f = [&](double t) {
            State sa = state_at(a, oa, t), sb = state_at(b, ob, t);
            if (!sa.ok || !sb.ok) {
                bad = true;
                return 1e30;
            }
            return dist2(sa, sb);
        };
        double fmin;
        double t = brent_min(f, lo, hi, 1e-12, fmin);
        if (bad) continue;
        // a minimum on the bracket's edge is not a local minimum inside it
        if (t - lo < 1e-3 || hi - t < 1e-3) continue;
        double miss = std::sqrt(fmin);
        if (miss >= dmax) continue;
        State sa = state_at(a, oa, t), sb = state_at(b, ob, t);
        double vx = sb.v[0] - sa.v[0], vy = sb.v[1] - sa.v[1], vz = sb.v[2] - sa.v[2];
        ev[q] = {c[q].i, c[q].j, t, miss, std::sqrt(vx * vx + vy * vy + vz * vz)};
        keep[q] = 1;
    }
    std::vector<Event> kept;
    for (int64_t q = 0; q < nc; ++q)
        if (keep[q]) kept.push_back(ev[q]);
    std::sort(kept.begin(), kept.end(), [](const Event& x, const Event& y) {
        if (x.i != y.i) return x.i < y.i;
        if (x.j != y.j) return x.j < y.j;
        return x.tca < y.tca;
    });
    // the same approach found from neighbouring windows: keep one
    int64_t m = 0;
    for (size_t q = 0; q < kept.size(); ++q) {
        if (m > 0 && out[m - 1].i == kept[q].i && out[m - 1].j == kept[q].j &&
            kept[q].tca - out[m - 1].tca < 1.0) {
            if (kept[q].miss < out[m - 1].miss) out[m - 1] = kept[q];
            continue;
        }
        out[m++] = kept[q];
    }
    return m;
}

// Brute force: every pair at every sample t0 + k * step, k in [0, nsteps). Each sampled local
// minimum of the distance is refined exactly between its two neighbouring samples, and kept
// if closer than dmax. Returns the total number of events (writes up to cap).
int64_t kessler_bruteforce(const Elset* s, const double* off, int64_t n, double t0, double step,
                           int64_t nsteps, double dmax, Event* out, int64_t cap) {
    std::vector<double> pos(static_cast<size_t>(n) * nsteps * 3);
    std::vector<char> ok(static_cast<size_t>(n) * nsteps);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i)
        for (int64_t k = 0; k < nsteps; ++k) {
            State st = state_at(s[i], off[i], t0 + k * step);
            size_t q = static_cast<size_t>(i) * nsteps + k;
            ok[q] = st.ok;
            pos[3 * q] = st.r[0];
            pos[3 * q + 1] = st.r[1];
            pos[3 * q + 2] = st.r[2];
        }
    int64_t total = 0;
#pragma omp parallel
    {
        std::vector<Event> local;
        std::vector<double> d(nsteps);
#pragma omp for schedule(dynamic, 8)
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = i + 1; j < n; ++j) {
                for (int64_t k = 0; k < nsteps; ++k) {
                    size_t a = static_cast<size_t>(i) * nsteps + k, b = static_cast<size_t>(j) * nsteps + k;
                    if (!ok[a] || !ok[b]) {
                        d[k] = 1e30;
                        continue;
                    }
                    double dx = pos[3 * b] - pos[3 * a], dy = pos[3 * b + 1] - pos[3 * a + 1],
                           dz = pos[3 * b + 2] - pos[3 * a + 2];
                    d[k] = std::sqrt(dx * dx + dy * dy + dz * dz);
                }
                for (int64_t k = 1; k + 1 < nsteps; ++k) {
                    if (!(d[k] <= d[k - 1] && d[k] < d[k + 1])) continue;
                    // a sample can sit up to (relative speed) x step from the true minimum
                    if (d[k] > dmax + kMaxRelSpeed * step + 1.0) continue;
                    double tlo = t0 + (k - 1) * step, thi = t0 + (k + 1) * step;
                    bool bad = false;
                    auto f = [&](double t) {
                        State sa = state_at(s[i], off[i], t), sb = state_at(s[j], off[j], t);
                        if (!sa.ok || !sb.ok) {
                            bad = true;
                            return 1e30;
                        }
                        return dist2(sa, sb);
                    };
                    double fmin;
                    double t = brent_min(f, tlo, thi, 1e-12, fmin);
                    // at the bracket's edge it is not a local minimum (nearly constant distance)
                    if (bad || t - tlo < 1e-3 || thi - t < 1e-3 || std::sqrt(fmin) >= dmax) continue;
                    local.push_back({static_cast<int32_t>(i), static_cast<int32_t>(j), t, std::sqrt(fmin), 0.0});
                }
            }
#pragma omp critical
        {
            for (const Event& e : local) {
                if (total < cap) out[total] = e;
                ++total;
            }
        }
    }
    return total;
}

}  // extern "C"
