"""Building blocks of the maneuver attribution (bench/m4_attribution.py)."""
import gzip
import os
import sys
from datetime import datetime, timedelta

import numpy as np

from kessler._native import Elsets
from kessler.attribution import closest_per_object, min_distance_near, snapshot, wilson
from kessler.catalog import Catalog, epoch_days
from kessler.screen import EVENT_DTYPE, max_speed, offsets, screen

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "bench"))


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")


def test_snapshot_takes_latest_before_t():
    t0 = datetime(2026, 1, 15)
    rows = {1: [{"EPOCH": iso(t0 - timedelta(hours=h))} for h in (30, 5, -2)],
            2: [{"EPOCH": iso(t0 - timedelta(days=5))}]}
    snap = snapshot(rows, epoch_days(iso(t0)))
    assert list(snap) == [1] and snap[1]["EPOCH"] == iso(t0 - timedelta(hours=5))


def test_wilson():
    p, lo, hi = wilson(5, 10)
    assert p == 0.5 and abs(lo - 0.2366) < 1e-4 and abs(hi - 0.7634) < 1e-4
    assert wilson(0, 10)[1] == 0.0


def test_closest_per_object():
    ev = np.array([(0, 1, 10.0, 3.0, 1.0), (1, 2, 20.0, 0.5, 1.0), (0, 2, 99.0, 0.1, 1.0)], dtype=EVENT_DTYPE)
    assert closest_per_object(ev, 4).tolist() == [2, 1, 2, -1]
    assert closest_per_object(ev, 4, t_hi=50.0).tolist() == [0, 1, 1, -1]


def test_min_distance_near_matches_screen():
    cat = Catalog.from_csv(gzip.open(os.path.join(HERE, "active_2026-10-10.csv.gz"), "rt").read())
    idx = np.random.default_rng(1).choice(len(cat), size=1500, replace=False)
    sub = Catalog([cat.rows[k] for k in idx])
    start = float(np.max(sub.mean["epoch"]))
    ev, _ = screen(Elsets(sub.mean), offsets(sub.mean, start), max_speed(sub.mean), 6 * 3600.0, 5.0)
    ev = ev[ev["vrel"] > 1.0]
    e = ev[np.argmin(ev["miss"])]
    tca = start + e["tca"] / 86400.0
    miss, vrel, t = min_distance_near(sub.rows[e["i"]], sub.rows[e["j"]], tca + 60.0 / 86400.0)
    assert abs(miss - e["miss"]) < 1e-6 and abs(vrel - e["vrel"]) < 1e-6
    assert abs(t - tca) * 86400.0 < 1e-3


def _history(d0, step_at=None, dn=-0.0005):
    """Synthetic Starlink history every 8 h from d0-4 to d0+3 days; mean motion (rev/day) drops by
    |dn| (about 150 m higher) at step_at (days from d0)."""
    rng = np.random.default_rng(0)
    rows = []
    for h in range(-96, 72, 8):
        n = 15.06 + (dn if step_at is not None and h / 24.0 > step_at else 0.0) + 1e-6 * rng.standard_normal()
        rows.append({"EPOCH": iso(d0 + timedelta(hours=h, minutes=17)), "MEAN_MOTION": f"{n:.8f}",
                     "ECCENTRICITY": "0.0001", "INCLINATION": "53.05"})
    return rows


def test_classify():
    from m4_attribution import classify
    d0 = datetime(2026, 1, 15)
    t0 = epoch_days(iso(d0))
    g, during = classify(_history(d0, 0.5), t0)
    assert g == "maneuvered" and len(during) == 1 and during[0][2] > 100
    assert classify(_history(d0), t0)[0] == "quiet"
    assert classify(_history(d0, -0.5), t0)[0] is None     # maneuvered the day before
    assert classify(_history(d0, 1.5), t0)[0] is None      # maneuvered the day after
