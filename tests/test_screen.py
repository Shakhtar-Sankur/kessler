"""The screener against brute force: every pair, every second, nothing missed."""
import gzip
import os

import numpy as np
import pytest

from kessler._native import Elsets
from kessler.catalog import Catalog
from kessler.screen import bruteforce, max_speed, offsets, screen

HERE = os.path.dirname(__file__)


def subset(n=600, seed=0, shell=True):
    """n random objects; with shell=True, only orbits crossing 450-600 km altitude, where
    Starlink and most traffic live. Otherwise the whole catalogue, incl. deep-space orbits."""
    cat = Catalog.from_csv(gzip.open(os.path.join(HERE, "active_2026-10-10.csv.gz"), "rt").read())
    m = cat.mean
    a = (398600.8 / (m["no_kozai"] / 60.0) ** 2) ** (1 / 3)
    hp, ha = a * (1 - m["ecco"]) - 6378.135, a * (1 + m["ecco"]) - 6378.135
    idx = np.where((hp < 600) & (ha > 450))[0] if shell else np.arange(len(m))
    idx = np.sort(np.random.default_rng(seed).choice(idx, size=min(n, len(idx)), replace=False))
    return m[idx], float(np.max(m["epoch"]))


def match(a, b, dt=300.0, dd=1e-4):
    """Indices of events in a with no event in b for the same pair, approach (within dt s)
    and miss distance (within dd km)."""
    missing = []
    for q, e in enumerate(a):
        sel = b[(b["i"] == e["i"]) & (b["j"] == e["j"])]
        if not np.any((np.abs(sel["tca"] - e["tca"]) < dt) & (np.abs(sel["miss"] - e["miss"]) < dd)):
            missing.append(q)
    return missing


def max_tca_error(a, b, fast=1.0):
    """Largest TCA difference between matched events whose relative speed exceeds `fast` km/s
    (for slower pairs the minimum is flat and its time ill-defined)."""
    worst = 0.0
    for e in a[a["vrel"] > fast]:
        sel = b[(b["i"] == e["i"]) & (b["j"] == e["j"])]
        if len(sel):
            worst = max(worst, float(np.min(np.abs(sel["tca"] - e["tca"]))))
    return worst


CASES = [  # objects, seed, shell only, seconds, threshold km
    (600, 0, True, 1800.0, 50.0),
    (1000, 1, True, 3600.0, 10.0),
    (1000, 2, True, 3600.0, 200.0),
    (1500, 3, False, 3600.0, 500.0),
]


@pytest.mark.parametrize("n,seed,shell,duration,dmax", CASES)
def test_screen_matches_bruteforce(n, seed, shell, duration, dmax):
    mean, start = subset(n, seed, shell)
    els = Elsets(mean)
    off = offsets(mean, start)
    ev, cand = screen(els, off, max_speed(mean), duration, dmax)
    ref = bruteforce(els, off, duration, dmax)
    inside = lambda x: x[(x["tca"] >= 1.0) & (x["tca"] <= duration - 1.0)]
    ref, evi = inside(ref), inside(ev)
    missed = match(ref, ev)
    extra = match(evi, ref)
    terr = max_tca_error(evi, ref)
    print(f"{len(mean)} objects, {duration:.0f} s, dmax {dmax} km: {len(cand)} candidates, "
          f"{len(ev)} events, brute force {len(ref)}; missed {len(missed)}, extra {len(extra)}, "
          f"max TCA difference (vrel > 1 km/s) {terr * 1e3:.3f} ms")
    assert len(ref) > 10
    assert not missed, ref[missed][:5]
    assert not extra, ev[extra][:5]
    assert terr < 1e-3
