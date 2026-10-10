// Conjunction screening helpers shared by the CPU and CUDA screeners.
//
// Time is sampled every `step` seconds; sample k stands for the window [t_k - h, t_k + h] with
// h = step / 2. A pair is a candidate at sample k if, moving in a straight line with its
// sampled relative velocity, it comes within D + margin somewhere in that window. `margin`
// bounds how far the real (curved) relative motion can depart from the straight line within
// the window, so no approach closer than D is missed (see window_margin).
#pragma once

#include <cmath>
#include <cstdint>

#include "kessler/sgp4.hpp"

namespace kessler {

struct Candidate {
    int32_t i, j;   // object indices, i < j
    int32_t k;      // sample index
    float dlin;     // straight-line minimum distance in the window (km)
};

struct Event {
    int32_t i, j;
    double tca;     // time of closest approach, seconds after the screening start
    double miss;    // distance at closest approach, km
    double vrel;    // relative speed at closest approach, km/s
};

// Upper bound on the relative acceleration of two objects near Earth: twice surface gravity.
constexpr double kMaxRelAccel = 2.0 * 398600.8 / (6378.135 * 6378.135);  // km/s^2

// Upper bound on relative speed: two objects at Earth escape speed, head on.
constexpr double kMaxRelSpeed = 2.0 * 11.2;  // km/s

// How far the real relative motion can be from the straight line over half a window h (s).
KESSLER_HD inline double window_margin(double h) { return 0.5 * kMaxRelAccel * h * h + 0.01; }

// Cell coordinate packed into 21 bits per axis.
constexpr int64_t kCellBias = int64_t(1) << 20;

KESSLER_HD inline int64_t cell_coord(double x, double cell) {
    return static_cast<int64_t>(std::floor(x / cell)) + kCellBias;
}

KESSLER_HD inline uint64_t cell_key(int64_t cx, int64_t cy, int64_t cz) {
    return (static_cast<uint64_t>(cx) << 42) | (static_cast<uint64_t>(cy) << 21) |
           static_cast<uint64_t>(cz);
}

// Straight-line minimum distance between i and j within [-h, h] seconds of the sample.
KESSLER_HD inline double window_min_distance(const double ri[3], const double vi[3],
                                             const double rj[3], const double vj[3], double h) {
    double rx = rj[0] - ri[0], ry = rj[1] - ri[1], rz = rj[2] - ri[2];
    double vx = vj[0] - vi[0], vy = vj[1] - vi[1], vz = vj[2] - vi[2];
    double vv = vx * vx + vy * vy + vz * vz;
    double tau = vv > 0.0 ? -(rx * vx + ry * vy + rz * vz) / vv : 0.0;
    if (tau > h) tau = h;
    if (tau < -h) tau = -h;
    double dx = rx + vx * tau, dy = ry + vy * tau, dz = rz + vz * tau;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

}  // namespace kessler
