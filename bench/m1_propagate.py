"""M1 benchmark: propagate the catalogue to 1,440 times (one day, one-minute steps).

Methods, all checked against the reference (python `sgp4`, Vallado's C++):
  kessler-cuda    this repo, one GPU
  kessler-cpu     this repo, OpenMP on all CPU cores
  sgp4-array      python `sgp4` SatrecArray (Vallado's C++, one thread)
  dsgp4-cuda      ESA's PyTorch SGP4 (dsgp4) on the GPU, near-earth objects only
                  (its batched propagator does not support deep-space objects)

Usage: python bench/m1_propagate.py [--minutes 1440] [--out bench/results/m1.json]
"""
import argparse
import gzip
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sgp4 import omm  # noqa: E402
from sgp4.api import Satrec, SatrecArray  # noqa: E402

from kessler import cuda  # noqa: E402
from kessler._native import Elsets  # noqa: E402
from kessler.catalog import Catalog  # noqa: E402


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
    except Exception:
        return "none"


def reference(rows, sats, times, rows_idx):
    """Reference states for a sample of objects: [len(rows_idx), m, 3]."""
    out = np.empty((len(rows_idx), len(times), 3))
    err = np.zeros((len(rows_idx), len(times)), dtype=np.int32)
    for a, i in enumerate(rows_idx):
        for b, t in enumerate(times):
            e, r, _ = sats[i].sgp4_tsince(float(t))
            out[a, b] = r
            err[a, b] = e
    return out, err


def max_diff(r, ref, err_ref, err):
    ok = (err_ref == 0) & (err == 0)
    return float(np.max(np.abs(r - ref)[ok])) if ok.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=1440)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results", "m1.json"))
    args = ap.parse_args()

    text = gzip.open(os.path.join(ROOT, "tests", "active_2026-10-10.csv.gz"), "rt").read()
    cat = Catalog.from_csv(text)
    n = len(cat)
    times = np.arange(args.minutes, dtype=np.float64)
    m = len(times)
    states = n * m
    leo = np.where(2 * np.pi / (cat.mean["no_kozai"]) < 225.0)[0]

    sats = []
    for row in cat.rows:
        s = Satrec()
        omm.initialize(s, row)
        sats.append(s)
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(n, size=min(500, n), replace=False))
    ref, ref_err = reference(cat.rows, sats, times, sample)

    res = {"gpu": gpu_name(), "cpu": platform.processor() or platform.machine(),
           "cpu_count": os.cpu_count(), "objects": n, "near_earth_objects": int(len(leo)),
           "times": m, "states": states, "methods": {}}
    print(f"{n} objects ({len(leo)} near-earth) x {m} times = {states:,} states; GPU: {res['gpu']}")

    els = Elsets(cat.mean)

    # kessler-cpu
    t0 = time.perf_counter()
    r_cpu, _, e_cpu = els.propagate(times)
    dt = time.perf_counter() - t0
    res["methods"]["kessler-cpu"] = {"seconds": dt, "states_per_s": states / dt,
                                     "max_dr_km_vs_reference": max_diff(r_cpu[sample], ref, ref_err, e_cpu[sample])}

    # sgp4-array (reference library, vectorised C++, one thread)
    arr = SatrecArray(sats)
    # SatrecArray propagates every object to every date in (jd, fr): give it one common
    # one-day grid, the same amount of work as m times per object.
    epochs = np.array([s.jdsatepoch + s.jdsatepochF for s in sats])
    base_jd = float(np.floor(np.median(epochs)))
    jd_grid = np.full(m, base_jd)
    fr_grid = times / 1440.0
    t0 = time.perf_counter()
    e_arr, r_arr, _ = arr.sgp4(jd_grid, fr_grid)
    dt = time.perf_counter() - t0
    res["methods"]["sgp4-array"] = {"seconds": dt, "states_per_s": states / dt}

    # kessler-cuda
    if cuda.available():
        t0 = time.perf_counter()
        r_gpu, _, e_gpu, kernel_ms = cuda.propagate(els, times, repeat=5)
        dt = time.perf_counter() - t0
        ok = (e_gpu == 0) & (e_cpu == 0)
        res["methods"]["kessler-cuda"] = {
            "kernel_ms": kernel_ms, "states_per_s_kernel": states / (kernel_ms / 1e3),
            "seconds_incl_copies": dt,
            "max_dr_km_vs_kessler_cpu": float(np.max(np.abs(r_gpu - r_cpu)[ok])),
            "max_dr_km_vs_reference": max_diff(r_gpu[sample], ref, ref_err, e_gpu[sample]),
            "error_code_mismatches_vs_cpu": int(np.sum(e_gpu != e_cpu)),
        }
    else:
        res["methods"]["kessler-cuda"] = {"skipped": "no CUDA device or library"}

    # dsgp4 on the GPU, near-earth objects, 60 of the times (it propagates one time per call)
    try:
        import torch
        import dsgp4
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_default_device(dev)
        objs = [dsgp4.omm.OMM(cat.rows[i]) for i in leo]
        t0 = time.perf_counter()
        _, batch = dsgp4.initialize_tle(objs, gravity_constant_name="wgs-72")
        init_s = time.perf_counter() - t0
        k = min(60, m)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = []
        for t in times[:k]:
            ts = torch.full((len(leo),), float(t))
            outs.append(dsgp4.sgp4_batched(batch, ts)[:, 0, :])
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        r_ds = torch.stack(outs, dim=1).cpu().numpy()          # [n_leo, k, 3]
        pos = {int(i): a for a, i in enumerate(leo)}
        rows_in = [a for a, i in enumerate(sample) if int(i) in pos]
        sel = np.array([pos[int(sample[a])] for a in rows_in])
        dr = max_diff(r_ds[sel], ref[rows_in][:, :k], ref_err[rows_in][:, :k],
                      np.zeros((len(rows_in), k), dtype=np.int32)) if rows_in else float("nan")
        res["methods"]["dsgp4-" + dev] = {"init_seconds": init_s, "seconds": dt,
                                          "states": len(leo) * k,
                                          "states_per_s": len(leo) * k / dt,
                                          "max_dr_km_vs_reference": dr}
    except Exception as e:  # report and carry on
        res["methods"]["dsgp4"] = {"skipped": f"{type(e).__name__}: {e}"}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
