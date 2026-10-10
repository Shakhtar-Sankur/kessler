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


@pytest.mark.skipif(not cuda.available(), reason="no CUDA device or library")
def test_cuda_screen_matches_cpu():
    from kessler.screen import max_speed, offsets, screen
    cat = Catalog.from_csv(gzip.open(os.path.join(HERE, "active_2026-10-10.csv.gz"), "rt").read())
    start = float(np.max(cat.mean["epoch"]))
    els = Elsets(cat.mean)
    off, vmax = offsets(cat.mean, start), max_speed(cat.mean)
    ev_cpu, cand_cpu = screen(els, off, vmax, 1800.0, 20.0)
    ev_gpu, cand_gpu, ms = cuda.screen(els, off, vmax, 1800.0, 20.0)
    key = lambda c: set(zip(c["i"].tolist(), c["j"].tolist(), c["k"].tolist()))
    print(f"{len(cat)} objects, 30 min, 20 km: CPU {len(cand_cpu)} candidates / {len(ev_cpu)} events, "
          f"GPU {len(cand_gpu)} / {len(ev_gpu)} in {ms:.0f} ms")
    assert key(cand_cpu) == key(cand_gpu)
    assert len(ev_cpu) == len(ev_gpu)
    a, b = np.sort(ev_cpu, order=["i", "j", "tca"]), np.sort(ev_gpu, order=["i", "j", "tca"])
    assert np.array_equal(a["i"], b["i"]) and np.array_equal(a["j"], b["j"])
    assert np.max(np.abs(a["tca"] - b["tca"])) < 1e-6 and np.max(np.abs(a["miss"] - b["miss"])) < 1e-9
