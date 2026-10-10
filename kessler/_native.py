"""ctypes bindings to the C++ propagator (kessler/_libkessler.so)."""
import ctypes
import os

import numpy as np

_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "_libkessler.so"))

MEAN_DTYPE = np.dtype([
    ("epoch", "f8"),      # days since 1949-12-31 00:00 UT
    ("bstar", "f8"),
    ("ecco", "f8"),
    ("argpo", "f8"),      # rad
    ("inclo", "f8"),      # rad
    ("mo", "f8"),         # rad
    ("no_kozai", "f8"),   # rad/min
    ("nodeo", "f8"),      # rad
])

_lib.kessler_elset_size.restype = ctypes.c_size_t
ELSET_SIZE = _lib.kessler_elset_size()

_p = ctypes.c_void_p
_i64 = ctypes.c_int64
_lib.kessler_init.argtypes = [_p, _i64, _p, _p]
_lib.kessler_propagate.argtypes = [_p, _i64, _p, _i64, _p, _p, _p]
_lib.kessler_propagate_at.argtypes = [_p, _p, _p, _i64, _p, _p, _p]


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


class Elsets:
    """Initialised SGP4 element sets for n objects, held in native memory."""

    def __init__(self, mean):
        mean = np.ascontiguousarray(mean, dtype=MEAN_DTYPE)
        self.n = len(mean)
        self.buf = np.empty(self.n * ELSET_SIZE, dtype=np.uint8)
        self.init_error = np.empty(self.n, dtype=np.int32)
        _lib.kessler_init(_ptr(mean), self.n, _ptr(self.buf), _ptr(self.init_error))

    def propagate(self, tsince):
        """Positions and velocities of every object at every time (minutes since its epoch).

        Returns r, v of shape [n, m, 3] (km, km/s, TEME) and err of shape [n, m].
        """
        t = np.ascontiguousarray(np.atleast_1d(tsince), dtype=np.float64)
        m = len(t)
        r = np.empty((self.n, m, 3))
        v = np.empty((self.n, m, 3))
        err = np.empty((self.n, m), dtype=np.int32)
        _lib.kessler_propagate(_ptr(self.buf), self.n, _ptr(t), m, _ptr(r), _ptr(v), _ptr(err))
        return r, v, err

    def propagate_at(self, idx, tsince):
        """Object idx[k] at its own time tsince[k]. Returns r, v [k, 3] and err [k]."""
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        t = np.ascontiguousarray(tsince, dtype=np.float64)
        k = len(idx)
        r = np.empty((k, 3))
        v = np.empty((k, 3))
        err = np.empty(k, dtype=np.int32)
        _lib.kessler_propagate_at(_ptr(self.buf), _ptr(idx), _ptr(t), k, _ptr(r), _ptr(v), _ptr(err))
        return r, v, err
