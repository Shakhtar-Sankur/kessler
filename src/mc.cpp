// Monte Carlo probability of collision with two-body motion, the method of NASA CARA's SDMC.
//
// Each trial draws both objects' states at TCA from their 6x6 covariances, moves both along
// exact Kepler orbits over a time window, and counts a hit when they come within the hard-body
// radius. This needs no short-encounter assumption, so it is right where the 2D method is not
// (slow encounters, long ones, curved relative motion).
//
// Units: m, m/s, s. Trials are split into fixed blocks with their own random streams, so the
// result depends on the seed and not on the number of threads.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr double MU = 3.986004418e14;  // m^3/s^2, EGM-96 / WGS-84

struct Vec {
  double x, y, z;
};
inline Vec operator+(Vec a, Vec b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec operator-(Vec a, Vec b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec operator*(double s, Vec a) { return {s * a.x, s * a.y, s * a.z}; }
inline double dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Stumpff functions C(z), S(z), with series near zero.
inline void stumpff(double z, double& c, double& s) {
  if (z > 1e-6) {
    double sz = std::sqrt(z);
    c = (1.0 - std::cos(sz)) / z;
    s = (sz - std::sin(sz)) / (sz * z);
  } else if (z < -1e-6) {
    double sz = std::sqrt(-z);
    c = (std::cosh(sz) - 1.0) / (-z);
    s = (std::sinh(sz) - sz) / (sz * -z);
  } else {
    c = 0.5 - z / 24.0 + z * z / 720.0;
    s = 1.0 / 6.0 - z / 120.0 + z * z / 5040.0;
  }
}

// An orbit from one state; position and velocity at time t by universal variables.
struct Orbit {
  Vec r0, v0;
  double r0n, rv0, alpha, sqmu;
  double chi = 0.0;  // last solution, a warm start for the next nearby time

  void init(Vec r, Vec v) {
    r0 = r;
    v0 = v;
    r0n = std::sqrt(dot(r, r));
    rv0 = dot(r, v);
    alpha = 2.0 / r0n - dot(v, v) / MU;
    sqmu = std::sqrt(MU);
    chi = 0.0;
  }

  void at(double t, Vec& r, Vec& v) {
    if (t == 0.0) {
      r = r0;
      v = v0;
      chi = 0.0;
      return;
    }
    double x = (chi != 0.0) ? chi : sqmu * std::fabs(alpha) * t;
    double c = 0, s = 0, rn = r0n;
    for (int it = 0; it < 50; ++it) {
      double z = alpha * x * x;
      stumpff(z, c, s);
      double x2 = x * x, x3 = x2 * x;
      double f = rv0 / sqmu * x2 * c + (1.0 - alpha * r0n) * x3 * s + r0n * x - sqmu * t;
      rn = rv0 / sqmu * x * (1.0 - z * s) + (1.0 - alpha * r0n) * x2 * c + r0n;
      double dx = f / rn;
      x -= dx;
      if (std::fabs(dx) < 1e-12 * (1.0 + std::fabs(x))) break;
    }
    chi = x;
    double z = alpha * x * x;
    stumpff(z, c, s);
    double x2 = x * x;
    double f = 1.0 - x2 / r0n * c;
    double g = t - x2 * x / sqmu * s;
    r = f * r0 + g * v0;
    rn = std::sqrt(dot(r, r));
    double fd = sqmu / (rn * r0n) * (alpha * x2 * x * s - x);
    double gd = 1.0 - x2 / rn * c;
    v = fd * r0 + gd * v0;
  }
};


// Equinoctial elements (a, h, k, p, q, lambda) of a prograde orbit (Broucke and Cefola, 1972):
// h, k = e sin, cos(w + O); p, q = tan(i/2) sin, cos(O); lambda = mean longitude. They are
// nonsingular for circular and equatorial orbits, and sampling them keeps a large along-track
// uncertainty on the orbit rather than on a straight line off it.
inline void eq_frame(double p, double q, Vec& f, Vec& g) {
  double s = 1.0 + p * p + q * q;
  f = {(1.0 - p * p + q * q) / s, 2.0 * p * q / s, -2.0 * p / s};
  g = {2.0 * p * q / s, (1.0 + p * p - q * q) / s, 2.0 * q / s};
}

void cart_to_eq(Vec r, Vec v, double* el) {
  double rn = std::sqrt(dot(r, r));
  double a = 1.0 / (2.0 / rn - dot(v, v) / MU);
  Vec hv = {r.y * v.z - r.z * v.y, r.z * v.x - r.x * v.z, r.x * v.y - r.y * v.x};
  double hn = std::sqrt(dot(hv, hv));
  Vec w = (1.0 / hn) * hv;
  double p = w.x / (1.0 + w.z), q = -w.y / (1.0 + w.z);
  Vec f, g;
  eq_frame(p, q, f, g);
  Vec vxh = {v.y * hv.z - v.z * hv.y, v.z * hv.x - v.x * hv.z, v.x * hv.y - v.y * hv.x};
  Vec e = (1.0 / MU) * vxh - (1.0 / rn) * r;
  double k = dot(e, f), h = dot(e, g);
  double x1 = dot(r, f), y1 = dot(r, g);
  double sq = std::sqrt(1.0 - h * h - k * k), b = 1.0 / (1.0 + sq);
  double sf = h + ((1.0 - h * h * b) * y1 - h * k * b * x1) / (a * sq);
  double cf = k + ((1.0 - k * k * b) * x1 - h * k * b * y1) / (a * sq);
  double F = std::atan2(sf, cf);
  el[0] = a; el[1] = h; el[2] = k; el[3] = p; el[4] = q;
  el[5] = F + h * std::cos(F) - k * std::sin(F);
}

void eq_to_cart(const double* el, Vec& r, Vec& v) {
  double a = el[0], h = el[1], k = el[2], p = el[3], q = el[4], lam = el[5];
  double F = lam;
  for (int it = 0; it < 30; ++it) {
    double s = std::sin(F), c = std::cos(F);
    double dF = (F + h * c - k * s - lam) / (1.0 - h * s - k * c);
    F -= dF;
    if (std::fabs(dF) < 1e-15) break;
  }
  double s = std::sin(F), c = std::cos(F);
  double sq = std::sqrt(1.0 - h * h - k * k), b = 1.0 / (1.0 + sq);
  double n = std::sqrt(MU / (a * a * a));
  double x1 = a * ((1.0 - h * h * b) * c + h * k * b * s - k);
  double y1 = a * (h * k * b * c + (1.0 - k * k * b) * s - h);
  double rn = a * (1.0 - k * c - h * s);
  double xd = a * a * n / rn * (h * k * b * c - (1.0 - h * h * b) * s);
  double yd = a * a * n / rn * ((1.0 - k * k * b) * c - h * k * b * s);
  Vec f, g;
  eq_frame(p, q, f, g);
  r = x1 * f + y1 * g;
  v = xd * f + yd * g;
}

// xoshiro256** seeded by splitmix64, and Box-Muller normals.
struct Rng {
  uint64_t s[4];
  bool have = false;
  double spare = 0.0;

  static uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  explicit Rng(uint64_t seed) {
    for (auto& w : s) w = splitmix(seed);
  }
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  uint64_t next() {
    uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return r;
  }
  double uniform() { return ((next() >> 11) + 0.5) * 0x1.0p-53; }  // (0, 1)
  double normal() {
    if (have) {
      have = false;
      return spare;
    }
    double u = uniform(), w = uniform();
    double m = std::sqrt(-2.0 * std::log(u));
    spare = m * std::sin(2.0 * M_PI * w);
    have = true;
    return m * std::cos(2.0 * M_PI * w);
  }
};

inline void to_state(const double* x, int space, Vec& r, Vec& v) {
  if (space == 1) {
    eq_to_cart(x, r, v);
  } else if (space == 2) {  // the same elements with mean motion n (rad/s) in place of a
    double el[6] = {std::cbrt(MU / (x[0] * x[0])), x[1], x[2], x[3], x[4], x[5]};
    eq_to_cart(el, r, v);
  } else {
    r = {x[0], x[1], x[2]};
    v = {x[3], x[4], x[5]};
  }
}

// Draws one state, x = mean + a z with z standard normal, in Cartesian coordinates (space 0),
// equinoctial elements with a (space 1) or with mean motion n (space 2); returns |z|^2.
double sample(Rng& g, const double* mean, const double* a, int space, Vec& r, Vec& v) {
  double z[6], x[6], zz = 0.0;
  for (double& e : z) {
    e = g.normal();
    zz += e * e;
  }
  for (int i = 0; i < 6; ++i) {
    double acc = mean[i];
    for (int j = 0; j < 6; ++j) acc += a[6 * i + j] * z[j];
    x[i] = acc;
  }
  to_state(x, space, r, v);
  return zz;
}

inline Vec gravity(Vec r) {
  double rn2 = dot(r, r);
  return (-MU / (rn2 * std::sqrt(rn2))) * r;
}

// Smallest distance between the two orbits over [t0, t1]: a grid of step h finds every interval
// where the range rate goes from closing to opening, and a safeguarded Newton iteration on the
// range rate locates the minimum inside it. Returns early once the distance is below stop.
double min_distance(Orbit& o1, Orbit& o2, double t0, double t1, double h, double stop) {
  Vec r1, v1, r2, v2;
  auto rel = [&](double t, Vec& dr, Vec& dv) {
    o1.at(t, r1, v1);
    o2.at(t, r2, v2);
    dr = r2 - r1;
    dv = v2 - v1;
  };
  Vec dr, dv;
  rel(t0, dr, dv);
  double best = std::sqrt(dot(dr, dr));
  double ta = t0, fa = dot(dr, dv);
  int steps = static_cast<int>(std::ceil((t1 - t0) / h));
  for (int k = 1; k <= steps && best >= stop; ++k) {
    double tb = (k == steps) ? t1 : t0 + k * h;
    rel(tb, dr, dv);
    double fb = dot(dr, dv);
    best = std::fmin(best, std::sqrt(dot(dr, dr)));
    if (fa < 0.0 && fb >= 0.0) {
      double lo = ta, hi = tb, t = ta - fa * (tb - ta) / (fb - fa);
      for (int it = 0; it < 60; ++it) {
        rel(t, dr, dv);
        double f = dot(dr, dv);
        if (f < 0.0) lo = t; else hi = t;
        double fp = dot(dv, dv) + dot(dr, gravity(r2) - gravity(r1));
        double tn = t - f / fp;
        if (!(tn > lo && tn < hi)) tn = 0.5 * (lo + hi);
        bool done = std::fabs(tn - t) < 1e-9 || hi - lo < 1e-9;
        t = tn;
        if (done) break;
      }
      rel(t, dr, dv);
      best = std::fmin(best, std::sqrt(dot(dr, dr)));
    }
    ta = tb;
    fa = fb;
  }
  return best;
}


// Where in [t0, t1] a trial can come within hbr. A trial is the nominal states plus
// a1 z1 and a2 z2, with z standard normal; to first order its relative position at time t
// is the nominal one plus M(t) z, so it lies within |z| ||M(t)||_F of the nominal one. Cells of
// CELL seconds are kept where the nominal distance can fall within that bound (for |z| up to
// ZCAP, with a margin for second-order terms) of hbr; trials with |z| > ZCAP are searched over
// the whole window instead, so no trial is restricted on the strength of the bound alone being
// rare.
constexpr double ZCAP = 8.0;   // P(|z| > 8) for 12 normals is about 3e-9
constexpr double CELL = 1.0;   // s

struct Window {
  double a, b;
};

std::vector<Window> search_windows(const double* mean1, const double* a1, const double* mean2,
                                   const double* a2, int space, double hbr, double t0, double t1) {
  int cells = static_cast<int>(std::ceil((t1 - t0) / CELL));
  auto tc = [&](int k) { return std::fmin(t1, t0 + k * CELL); };
  // ||M(t)||_F at the cell edges, by central differences along each column of a1 and a2
  std::vector<double> fro(cells + 1, 0.0);
  for (int obj = 0; obj < 2; ++obj) {
    const double* mean = obj ? mean2 : mean1;
    const double* a = obj ? a2 : a1;
    for (int j = 0; j < 6; ++j) {
      double p[6], m[6];
      for (int i = 0; i < 6; ++i) {
        p[i] = mean[i] + a[6 * i + j];
        m[i] = mean[i] - a[6 * i + j];
      }
      Orbit op, om;
      Vec r, v;
      to_state(p, space, r, v);
      op.init(r, v);
      to_state(m, space, r, v);
      om.init(r, v);
      for (int k = 0; k <= cells; ++k) {
        Vec rp, vp, rm, vm;
        op.at(tc(k), rp, vp);
        om.at(tc(k), rm, vm);
        Vec d = 0.5 * (rp - rm);
        fro[k] += dot(d, d);
      }
    }
  }
  for (double& f : fro) f = std::sqrt(f);

  Orbit n1, n2;
  Vec r, v;
  to_state(mean1, space, r, v);
  n1.init(r, v);
  to_state(mean2, space, r, v);
  n2.init(r, v);
  std::vector<char> keep(cells, 0);
  for (int k = 0; k < cells; ++k) {
    double dmin = min_distance(n1, n2, tc(k), tc(k + 1), CELL, 0.0);
    double reach = ZCAP * std::fmax(fro[k], fro[k + 1]);
    keep[k] = dmin - 1.5 * reach - 100.0 < hbr;
  }
  std::vector<Window> out;
  for (int k = 0; k < cells; ++k) {
    if (!keep[k]) continue;
    int e = k;
    while (e + 1 < cells && keep[e + 1]) ++e;
    double a = tc(std::max(0, k - 1)), b = tc(std::min(cells, e + 2));
    if (!out.empty() && a <= out.back().b) out.back().b = b;
    else out.push_back({a, b});
    k = e;
  }
  return out;
}

}  // namespace

extern "C" {

// mean1/mean2: 6-vectors at TCA, (r, v) for space 0 or equinoctial elements for space 1;
// a1/a2: 6x6 row-major factors with a a^T = covariance in the same coordinates.
// Runs n trials and returns the number of hits. With restrict = 0 every trial is searched over
// the whole window (slow; for checking that the search windows change nothing).
int64_t kessler_pc_mc(const double* mean1, const double* a1, const double* mean2, const double* a2,
                      int space, double hbr, double t0, double t1, double h, int64_t n,
                      uint64_t seed, int restrict) {
  // Only the encounter at TCA: the window containing t = 0 (another window in [t0, t1] is a
  // different close approach, such as the second crossing of two orbits half a revolution
  // later). Trials beyond the bound's range are searched from halfway to the window before it
  // to halfway to the one after.
  std::vector<Window> windows{{t0, t1}};
  double full0 = t0, full1 = t1;
  if (restrict) {
    const auto all = search_windows(mean1, a1, mean2, a2, space, hbr, t0, t1);
    if (!all.empty()) {
      size_t k = 0;
      double best = 1e300;
      for (size_t i = 0; i < all.size(); ++i) {
        double d = all[i].a > 0 ? all[i].a : (all[i].b < 0 ? -all[i].b : 0.0);
        if (d < best) best = d, k = i;
      }
      windows = {all[k]};
      if (k > 0) full0 = 0.5 * (all[k - 1].b + all[k].a);
      if (k + 1 < all.size()) full1 = 0.5 * (all[k].b + all[k + 1].a);
    } else {
      windows.clear();
    }
  }
  constexpr int64_t BLOCK = 1 << 16;
  int64_t nblocks = (n + BLOCK - 1) / BLOCK, hits = 0;
#pragma omp parallel for schedule(dynamic, 4) reduction(+ : hits)
  for (int64_t b = 0; b < nblocks; ++b) {
    Rng g(seed * 0x2545F4914F6CDD1Dull + static_cast<uint64_t>(b));
    Orbit o1, o2;
    int64_t end = std::min(n, (b + 1) * BLOCK);
    for (int64_t i = b * BLOCK; i < end; ++i) {
      Vec r, v;
      double zz = sample(g, mean1, a1, space, r, v);
      o1.init(r, v);
      zz += sample(g, mean2, a2, space, r, v);
      o2.init(r, v);
      bool hit = false;
      if (zz > ZCAP * ZCAP) {
        hit = min_distance(o1, o2, full0, full1, h, hbr) < hbr;
      } else {
        for (const auto& w : windows)
          if (min_distance(o1, o2, w.a, w.b, std::fmin(h, w.b - w.a), hbr) < hbr) {
            hit = true;
            break;
          }
      }
      hits += hit;
    }
  }
  return hits;
}

// Every window search_windows finds in [t0, t1] (the Monte Carlo uses only the one at TCA),
// for inspection: writes up to cap (a, b) pairs and returns how many there are.
int64_t kessler_pc_mc_windows(const double* mean1, const double* a1, const double* mean2,
                              const double* a2, int space, double hbr, double t0, double t1,
                              double* out, int64_t cap) {
  const auto w = search_windows(mean1, a1, mean2, a2, space, hbr, t0, t1);
  for (int64_t i = 0; i < static_cast<int64_t>(w.size()) && i < cap; ++i) {
    out[2 * i] = w[i].a;
    out[2 * i + 1] = w[i].b;
  }
  return static_cast<int64_t>(w.size());
}

// Position and velocity after time t along a two-body orbit (for tests).
void kessler_kepler(const double* rv0, double t, double* rv) {
  Orbit o;
  o.init({rv0[0], rv0[1], rv0[2]}, {rv0[3], rv0[4], rv0[5]});
  Vec r, v;
  o.at(t, r, v);
  rv[0] = r.x; rv[1] = r.y; rv[2] = r.z;
  rv[3] = v.x; rv[4] = v.y; rv[5] = v.z;
}

// Equinoctial elements of a state (m, m/s), and back.
void kessler_cart_to_eq(const double* rv, double* el) {
  cart_to_eq({rv[0], rv[1], rv[2]}, {rv[3], rv[4], rv[5]}, el);
}

void kessler_eq_to_cart(const double* el, double* rv) {
  Vec r, v;
  eq_to_cart(el, r, v);
  rv[0] = r.x; rv[1] = r.y; rv[2] = r.z;
  rv[3] = v.x; rv[4] = v.y; rv[5] = v.z;
}

}  // extern "C"
