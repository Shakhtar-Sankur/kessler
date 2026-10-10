"""Which Starlink maneuvers followed a close approach predicted from public data?

For a day D, the public catalogue is rebuilt as it stood at the start of D (each object's
latest element set before D) and screened for the following days. Satellites that maneuvered
during D are compared with satellites that did not: if maneuvers are collision avoidance driven
by predictions like these, predicted close approaches should be much more common before them.
For a maneuver followed by a predicted approach, the approach is recomputed with the satellite's
first element set after the maneuver: avoidance should have moved it away, by much more than a
routine update of the elements moves the predictions of satellites that did not maneuver.
"""
import math

import numpy as np

from ._native import Elsets
from .catalog import Catalog, epoch_days
from .screen import max_speed, offsets, screen


def snapshot(by_id, t, max_age_days=3.0):
    """Each object's latest element set with epoch in [t - max_age_days, t), as {id: row}."""
    out = {}
    for norad, rows in by_id.items():
        best, best_t = None, -math.inf
        for r in rows:
            e = epoch_days(r["EPOCH"])
            if t - max_age_days <= e < t and e > best_t:
                best, best_t = r, e
        if best is not None:
            out[norad] = best
    return out


def closest_per_object(ev, n, t_lo=-math.inf, t_hi=math.inf):
    """For each of n objects, its closest approach (event index) with TCA in [t_lo, t_hi], or -1."""
    best = np.full(n, -1, dtype=np.int64)
    miss = np.full(n, np.inf)
    for q, e in enumerate(ev):
        if not t_lo <= e["tca"] <= t_hi:
            continue
        for k in (e["i"], e["j"]):
            if e["miss"] < miss[k]:
                miss[k], best[k] = e["miss"], q
    return best


def min_distance_near(row_a, row_b, tca_days, half_window_s=600.0, dmax=200.0):
    """Closest approach (miss km, vrel km/s, TCA days) of two element sets within
    half_window_s of tca_days, or None if they stay further apart than dmax km."""
    cat = Catalog([row_a, row_b])
    els = Elsets(cat.mean)
    start = tca_days - half_window_s / 86400.0
    ev, _ = screen(els, offsets(cat.mean, start), max_speed(cat.mean), 2.0 * half_window_s, dmax)
    if len(ev) == 0:
        return None
    e = ev[np.argmin(ev["miss"])]
    return float(e["miss"]), float(e["vrel"]), start + float(e["tca"]) / 86400.0


def wilson(k, n, z=1.96):
    """Fraction k/n with its 95% Wilson interval."""
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, max(0.0, c - h), min(1.0, c + h)]
