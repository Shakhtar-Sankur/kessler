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
