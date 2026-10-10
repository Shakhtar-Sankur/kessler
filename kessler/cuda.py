"""GPU propagation (kessler/_libkessler_cuda.so, built with `make cuda`)."""
import ctypes
import os

import numpy as np

from ._native import Elsets, _ptr

_path = os.path.join(os.path.dirname(__file__), "_libkessler_cuda.so")
_lib = ctypes.CDLL(_path) if os.path.exists(_path) else None

if _lib is not None:
    _p, _i64 = ctypes.c_void_p, ctypes.c_int64
    _lib.kessler_cuda_device_count.restype = ctypes.c_int
    _lib.kessler_cuda_propagate.argtypes = [_p, _i64, _p, _i64, _p, _p, _p, ctypes.c_int,
                                            ctypes.POINTER(ctypes.c_float)]
    _lib.kessler_cuda_propagate.restype = ctypes.c_int


def available():
    return _lib is not None and _lib.kessler_cuda_device_count() > 0


def propagate(els: Elsets, tsince, with_velocity=False, repeat=1):
    """Same as Elsets.propagate, on the GPU. Also returns the mean kernel time in ms."""
    if not available():
        raise RuntimeError("no CUDA device, or kessler/_libkessler_cuda.so not built")
    t = np.ascontiguousarray(np.atleast_1d(tsince), dtype=np.float64)
    n, m = els.n, len(t)
    r = np.empty((n, m, 3))
    v = np.empty((n, m, 3)) if with_velocity else None
    err = np.empty((n, m), dtype=np.int32)
    ms = ctypes.c_float(0.0)
    rc = _lib.kessler_cuda_propagate(_ptr(els.buf), n, _ptr(t), m, _ptr(r),
                                     _ptr(v) if v is not None else None, _ptr(err), repeat,
                                     ctypes.byref(ms))
    if rc != 0:
        raise RuntimeError("kessler_cuda_propagate failed")
    return r, v, err, ms.value


if _lib is not None:
    _lib.kessler_cuda_screen.argtypes = [_p, _p, _p, _i64, ctypes.c_double, ctypes.c_double, _i64,
                                         ctypes.c_double, _p, _i64, ctypes.POINTER(ctypes.c_float)]
    _lib.kessler_cuda_screen.restype = _i64


def candidates(els: Elsets, off, vmax, t0, step, nsteps, dmax, cap=None):
    """GPU version of kessler.screen.candidates. Also returns the GPU time in ms."""
    from .screen import CANDIDATE_DTYPE
    if not available():
        raise RuntimeError("no CUDA device, or kessler/_libkessler_cuda.so not built")
    cap = cap or 64 * els.n
    while True:
        buf = np.empty(cap, dtype=CANDIDATE_DTYPE)
        ms = ctypes.c_float(0.0)
        total = _lib.kessler_cuda_screen(_ptr(els.buf), _ptr(off), _ptr(vmax), els.n, t0, step,
                                         nsteps, dmax, _ptr(buf), cap, ctypes.byref(ms))
        if total < 0:
            raise RuntimeError("kessler_cuda_screen failed")
        if total <= cap:
            return buf[:total], ms.value
        cap = total


def screen(els: Elsets, off, vmax, duration, dmax, step=10.0):
    """kessler.screen.screen with candidates found on the GPU. Returns events, candidates and
    the GPU time in ms."""
    import math
    from .screen import merge_approaches, refine
    nsteps = int(math.ceil(duration / step)) + 1
    cand, ms = candidates(els, off, vmax, 0.0, step, nsteps, dmax)
    ev = merge_approaches(refine(els, off, cand, 0.0, step, dmax))
    return ev[(ev["tca"] >= 0.0) & (ev["tca"] <= duration)], cand, ms
