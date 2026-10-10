"""kessler's SGP4 against the reference (python `sgp4`, which wraps Vallado's C++)."""
import gzip
import os

import numpy as np
import sgp4
from sgp4 import omm
from sgp4.api import Satrec

from kessler._native import MEAN_DTYPE, Elsets
from kessler.catalog import Catalog

HERE = os.path.dirname(__file__)
VER_TLE = os.path.join(os.path.dirname(sgp4.__file__), "SGP4-VER.TLE")


def mean_from_satrec(sat):
    return (sat.jdsatepoch + sat.jdsatepochF - 2433281.5, sat.bstar, sat.ecco, sat.argpo,
            sat.inclo, sat.mo, sat.no_kozai, sat.nodeo)


def verification_cases():
    lines = [l.rstrip("\n") for l in open(VER_TLE) if not l.startswith("#")]
    cases = []
    for l1, l2 in zip(lines[0::2], lines[1::2]):
        start, stop, step = (float(x) for x in l2[69:].split())
        sat = Satrec.twoline2rv(l1, l2[:69])
        times = np.arange(start, stop + step / 2, step)
        cases.append((l1[2:7], sat, times))
    return cases


def compare(sats, mean, times_per_sat):
    """Max position (km) and velocity (km/s) difference, and error-code mismatches."""
    els = Elsets(np.array(mean, dtype=MEAN_DTYPE))
    idx = np.concatenate([np.full(len(t), i) for i, t in enumerate(times_per_sat)])
    tt = np.concatenate(times_per_sat)
    r, v, err = els.propagate_at(idx, tt)
    dr = dv = 0.0
    mismatches = []
    for k, (i, t) in enumerate(zip(idx, tt)):
        e_ref, r_ref, v_ref = sats[i].sgp4_tsince(t)
        if (e_ref != 0) != (err[k] != 0):
            mismatches.append((i, t, e_ref, int(err[k])))
            continue
        if e_ref == 0 or e_ref == 6:
            dr = max(dr, float(np.max(np.abs(r[k] - r_ref))))
            dv = max(dv, float(np.max(np.abs(v[k] - v_ref))))
    return dr, dv, mismatches, len(tt)


def test_verification_vectors():
    cases = verification_cases()
    dr, dv, mism, n = compare([c[1] for c in cases], [mean_from_satrec(c[1]) for c in cases],
                              [c[2] for c in cases])
    print(f"verification: {len(cases)} objects, {n} states, max |dr| {dr:.3e} km, "
          f"max |dv| {dv:.3e} km/s, error-code mismatches {len(mism)}")
    assert not mism, mism
    assert dr < 1e-6 and dv < 1e-9


def test_full_catalog():
    text = gzip.open(os.path.join(HERE, "active_2026-10-10.csv.gz"), "rt").read()
    cat = Catalog.from_csv(text)
    sats = []
    for row in cat.rows:
        s = Satrec()
        omm.initialize(s, row)
        sats.append(s)
    times = np.array([-1440.0, 0.0, 90.0, 1440.0, 4320.0, 10080.0])
    dr, dv, mism, n = compare(sats, cat.mean, [times] * len(sats))
    print(f"catalog: {len(sats)} objects, {n} states, max |dr| {dr:.3e} km, "
          f"max |dv| {dv:.3e} km/s, error-code mismatches {len(mism)}")
    assert not mism, mism[:10]
    assert dr < 1e-6 and dv < 1e-9
