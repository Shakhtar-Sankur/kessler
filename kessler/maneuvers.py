"""Maneuver detection from a satellite's public element-set history.

A maneuver changes the orbit's size. For each pair of consecutive element sets of one object,
the mean semi-major axis jumps by the maneuver's effect plus noise; drag adds a slow, smooth
decay. kessler removes the decay with a robust local trend, estimates the object's own noise
from the median absolute deviation of its jumps, and flags a jump that is both large in
absolute terms and far outside that noise.

The detector is tested on synthetic histories with known maneuvers (tests/test_maneuvers.py)
and, on real data, on debris, which cannot maneuver: its detection rate there is the false
alarm rate.
"""
import math

import numpy as np

from .catalog import epoch_days

MU = 398600.8           # km^3/s^2 (WGS-72, as SGP4)
RE = 6378.135           # km
J2 = 0.001082616
XKE = 60.0 / math.sqrt(RE ** 3 / MU)


def brouwer_sma_km(no_kozai, ecc, incl):
    """Mean semi-major axis (km) from the Kozai mean motion (rad/min) of a TLE/OMM, recovering
    Brouwer's mean motion exactly as SGP4's initialisation does."""
    cosio2 = math.cos(incl) ** 2
    omeosq = 1.0 - ecc * ecc
    rteosq = math.sqrt(omeosq)
    ak = (XKE / no_kozai) ** (2.0 / 3.0)
    d1 = 0.75 * J2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq)
    de = d1 / (ak * ak)
    adel = ak * (1.0 - de * de - de * (1.0 / 3.0 + 134.0 * de * de / 81.0))
    de = d1 / (adel * adel)
    no = no_kozai / (1.0 + de)
    return (XKE / no) ** (2.0 / 3.0) * RE


def history(rows):
    """(epoch in days since 1949-12-31, mean SMA in km) for one object's OMM rows, sorted and
    with repeated epochs removed."""
    out = {}
    for r in rows:
        t = epoch_days(r["EPOCH"])
        a = brouwer_sma_km(float(r["MEAN_MOTION"]) / 720.0 * math.pi, float(r["ECCENTRICITY"]),
                           math.radians(float(r["INCLINATION"])))
        out[round(t, 8)] = (t, a)
    v = sorted(out.values())
    return np.array([x[0] for x in v]), np.array([x[1] for x in v])


def detect(t, a, floor_m=50.0, z=6.0, w=5, max_span_days=3.0, max_gap_days=4.0):
    """Indices k such that a maneuver happened between element sets k-1 and k, and the size of
    each (m).

    At each boundary k, the element sets within w of it on either side (and within
    max_span_days) are fitted with a(t) = level + rate * t + step * [t after the boundary]:
    drag is the shared rate, the maneuver is the step. A boundary is a maneuver if |step| >
    floor_m and |step| > z standard errors, where the error comes from the object's own noise
    (the median absolute deviation of its drag-corrected jumps). Of neighbouring boundaries
    that pass, only the strongest is kept, since a fit near a maneuver sees part of it too."""
    t = np.asarray(t, dtype=float)
    am = np.asarray(a, dtype=float) * 1e3                     # m
    n = len(t)
    if n < 4:
        return np.array([], dtype=int), np.array([])
    gaps, da = np.diff(t), np.diff(am)
    long_enough = gaps > 0.05
    rate = float(np.median(da[long_enough] / gaps[long_enough])) if long_enough.any() else 0.0
    d = da - rate * gaps
    sigma = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / math.sqrt(2.0)
    sigma = max(sigma, 1e-3)

    stat = np.zeros(n)
    step = np.zeros(n)
    for k in range(1, n):
        if not 0.0 < t[k] - t[k - 1] <= max_gap_days:
            continue
        before = [i for i in range(max(0, k - w), k) if t[i] >= t[k - 1] - max_span_days]
        after = [i for i in range(k, min(n, k + w)) if t[i] <= t[k] + max_span_days]
        if len(before) < 2 or len(after) < 2:
            continue
        idx = before + after
        tb = 0.5 * (t[k - 1] + t[k])
        x = np.column_stack([np.ones(len(idx)), t[idx] - tb, [0.0] * len(before) + [1.0] * len(after)])
        xtx_inv = np.linalg.inv(x.T @ x)
        beta = xtx_inv @ x.T @ am[idx]
        step[k] = beta[2]
        stat[k] = beta[2] / (sigma * math.sqrt(xtx_inv[2, 2]))
    cand = np.where((np.abs(step) > floor_m) & (np.abs(stat) > z))[0]
    keep = [k for k in cand
            if all(abs(stat[k]) >= abs(stat[j]) for j in cand if j != k and abs(j - k) <= w)]
    k = np.array(keep, dtype=int)
    return k, step[k]
