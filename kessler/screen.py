"""Conjunction screening: every pair of objects that comes closer than `dmax` km."""
import ctypes
import math

import numpy as np

from ._native import Elsets, _lib, _ptr

CANDIDATE_DTYPE = np.dtype([("i", "i4"), ("j", "i4"), ("k", "i4"), ("dlin", "f4")])
EVENT_DTYPE = np.dtype([("i", "i4"), ("j", "i4"), ("tca", "f8"), ("miss", "f8"), ("vrel", "f8")])

MU = 398600.8  # km^3/s^2, WGS-72

_p, _i64, _d = ctypes.c_void_p, ctypes.c_int64, ctypes.c_double
_lib.kessler_screen_cpu.argtypes = [_p, _p, _p, _i64, _d, _d, _i64, _d, _p, _i64]
_lib.kessler_screen_cpu.restype = _i64
_lib.kessler_refine.argtypes = [_p, _p, _p, _i64, _d, _d, _d, _d, _p]
_lib.kessler_refine.restype = _i64
_lib.kessler_bruteforce.argtypes = [_p, _p, _i64, _d, _d, _i64, _d, _p, _i64]
_lib.kessler_bruteforce.restype = _i64


def offsets(mean, start_epoch):
    """Minutes from each object's epoch to the screening start (days since 1949-12-31)."""
    return np.ascontiguousarray((start_epoch - mean["epoch"]) * 1440.0)


def max_speed(mean):
    """Upper bound on each object's speed (km/s): perigee speed from its mean elements, plus 5%."""
    n = mean["no_kozai"] / 60.0                     # rad/s
    a = (MU / (n * n)) ** (1.0 / 3.0)
    e = np.clip(mean["ecco"], 0.0, 0.999)
    vp = np.sqrt(MU * (1.0 + e) / (a * (1.0 - e)))
    return np.ascontiguousarray(1.05 * vp)


def _call_growing(fn, dtype, guess):
    cap = max(guess, 1024)
    while True:
        buf = np.empty(cap, dtype=dtype)
        total = fn(buf, cap)
        if total <= cap:
            return buf[:total]
        cap = total


def candidates(els: Elsets, off, vmax, t0, step, nsteps, dmax):
    return _call_growing(
        lambda buf, cap: _lib.kessler_screen_cpu(_ptr(els.buf), _ptr(off), _ptr(vmax), els.n, t0,
                                                 step, nsteps, dmax, _ptr(buf), cap),
        CANDIDATE_DTYPE, 4 * els.n)


def refine(els: Elsets, off, cand, t0, step, dmax, pad=1.0):
    cand = np.ascontiguousarray(cand, dtype=CANDIDATE_DTYPE)
    out = np.empty(len(cand), dtype=EVENT_DTYPE)
    m = _lib.kessler_refine(_ptr(els.buf), _ptr(off), _ptr(cand), len(cand), t0, step, pad, dmax,
                            _ptr(out))
    return out[:m]


def merge_approaches(ev, window=300.0):
    """One event per pair and approach: events of the same pair less than `window` seconds
    apart are the same approach (relative motion is nearly straight over a few minutes, so
    two real approaches of one pair are about half an orbit apart). Keeps the closest."""
    if len(ev) == 0:
        return ev
    ev = np.sort(ev, order=["i", "j", "tca"])
    out = []
    cur = ev[0]
    for e in ev[1:]:
        if e["i"] == cur["i"] and e["j"] == cur["j"] and e["tca"] - cur["tca"] < window:
            if e["miss"] < cur["miss"]:
                cur = e
            continue
        out.append(cur)
        cur = e
    out.append(cur)
    return np.array(out, dtype=EVENT_DTYPE)


def screen(els: Elsets, off, vmax, duration, dmax, step=10.0):
    """Close approaches below dmax km in [0, duration] seconds after the screening start."""
    nsteps = int(math.ceil(duration / step)) + 1
    cand = candidates(els, off, vmax, 0.0, step, nsteps, dmax)
    ev = merge_approaches(refine(els, off, cand, 0.0, step, dmax))
    return ev[(ev["tca"] >= 0.0) & (ev["tca"] <= duration)], cand


def bruteforce(els: Elsets, off, duration, dmax, step=1.0):
    """Every pair at every `step` seconds: the slow reference."""
    nsteps = int(math.ceil(duration / step)) + 1
    return merge_approaches(_call_growing(
        lambda buf, cap: _lib.kessler_bruteforce(_ptr(els.buf), _ptr(off), els.n, 0.0, step, nsteps,
                                                 dmax, _ptr(buf), cap),
        EVENT_DTYPE, 1024))
