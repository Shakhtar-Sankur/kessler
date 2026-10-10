"""Maneuver detection on synthetic element-set histories with known maneuvers."""
import math

import numpy as np

from kessler.maneuvers import brouwer_sma_km, detect


def synthetic(days=180.0, per_day=3.0, decay_m_per_day=-20.0, noise_m=8.0, maneuvers=(), seed=0):
    """A Starlink-like history: drag decay, measurement noise, and step changes at `maneuvers`
    ((time in days, size in m), ...)."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, days, int(days * per_day)))
    a = 6928.0 + decay_m_per_day * t / 1e3 + rng.normal(0.0, noise_m, len(t)) / 1e3
    for tm, da in maneuvers:
        a[t > tm] += da / 1e3
    return t, a


def test_brouwer_sma_matches_sgp4():
    """SGP4's own initialisation gives the same Brouwer mean motion."""
    from sgp4.api import Satrec, WGS72
    s = Satrec()
    no_kozai, ecc, inc = 15.06 / 720.0 * math.pi, 0.0001, math.radians(53.05)
    s.sgp4init(WGS72, "i", 1, 27000.0, 0.0, 0.0, 0.0, ecc, 0.0, inc, 0.0, no_kozai, 0.0)
    a_ref = s.a * 6378.135          # SGP4's semi-major axis from Brouwer's mean motion, in earth radii
    assert abs(brouwer_sma_km(no_kozai, ecc, inc) - a_ref) < 1e-9


def test_finds_known_maneuvers():
    truth = [(20.3, 150.0), (55.1, -120.0), (90.7, 300.0), (131.2, 90.0), (160.0, -200.0)]
    t, a = synthetic(maneuvers=truth)
    k, size = detect(t, a)
    found = [float(t[i]) for i in k]
    for tm, _ in truth:
        assert any(t_prev <= tm <= t_hit for t_prev, t_hit in zip(t[k - 1], found)), (tm, found)
    assert len(k) == len(truth), found


def test_no_false_alarms_without_maneuvers():
    """Debris-like histories: decay and noise only. Over 50 objects x 180 days, nothing."""
    hits = 0
    for seed in range(50):
        t, a = synthetic(seed=seed, decay_m_per_day=-5.0 - seed, noise_m=10.0)
        hits += len(detect(t, a)[0])
    assert hits == 0
