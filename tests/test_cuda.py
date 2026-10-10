"""The CUDA kernel against the CPU build (skipped without a GPU)."""
import gzip
import os

import numpy as np
import pytest

from kessler import cuda
from kessler._native import Elsets
from kessler.catalog import Catalog

HERE = os.path.dirname(__file__)


@pytest.mark.skipif(not cuda.available(), reason="no CUDA device or library")
def test_cuda_matches_cpu():
    cat = Catalog.from_csv(gzip.open(os.path.join(HERE, "active_2026-10-10.csv.gz"), "rt").read())
    els = Elsets(cat.mean)
    times = np.array([-1440.0, 0.0, 90.0, 1440.0, 4320.0, 10080.0])
    r_cpu, v_cpu, e_cpu = els.propagate(times)
    r_gpu, v_gpu, e_gpu, _ = cuda.propagate(els, times, with_velocity=True)
    assert np.array_equal(e_cpu, e_gpu)
    ok = e_cpu == 0
    dr = np.max(np.abs(r_gpu - r_cpu)[ok])
    dv = np.max(np.abs(v_gpu - v_cpu)[ok])
    print(f"GPU vs CPU: max |dr| {dr:.3e} km, max |dv| {dv:.3e} km/s")
    assert dr < 1e-6 and dv < 1e-9
