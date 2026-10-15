"""kessler's Monte Carlo against NASA CARA's own SDMC library, same inputs, same machine.

CARA ships its two-body Monte Carlo as a compiled library (libsdmc_linux.so, built with Intel
Fortran) plus a small C++ entry point (sdmcEntry.cpp). This script builds a driver around them,
runs both NASA's library and kessler.pc.pc_mc on CARA's test conjunctions with the same states,
covariances, hard-body radius and time window (a quarter of the shorter orbital period either side
of TCA), and compares hit rates and time per trial on one core each.

Needs a checkout of github.com/nasa/CARA_Analysis_Tools (--cara) and Intel's Fortran runtime:
  pip install intel-fortran-rt intel-cmplr-lib-rt
then run with LD_LIBRARY_PATH pointing at the runtime's lib directory (see --help).

Usage: OMP_NUM_THREADS=1 python bench/m3_vs_nasa_lib.py --cara PATH [--cases 1,2,25] [--trials N]
"""
import argparse
import ctypes
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))

from kessler.pc import CDM_COV6, cdm_states, orbit_period, pc_mc_from_cdm, read_cdm  # noqa: E402
from m3_sdmc import cara_table, fetch, zscore  # noqa: E402

DRIVER = r'''
#include <cstring>
#include <vector>
#include "SdmcInterface.h"
extern "C" int nasa_sdmc(const double* p1, const double* c1, const double* p2, const double* c2,
                         double tca, double span, int seed, int trials, double radius, int* hits) {
  SdmcInputs in;
  std::memset(&in, 0, sizeof(in));
  in.priSat.satno = 1; in.secSat.satno = 2;
  in.priSat.epoch = in.secSat.epoch = in.tca = tca;
  for (int i = 0; i < 6; ++i) { in.priSat.posVel[i] = p1[i]; in.secSat.posVel[i] = p2[i]; }
  for (int i = 0; i < 21; ++i) { in.priSat.cov[i] = c1[i]; in.secSat.cov[i] = c2[i]; }
  in.trajectoryMode = 0; in.span = span; in.seed = seed; in.numTrials = trials;
  in.radius = radius; in.maxRadius = radius * 1.001; in.maxNumOutputTrials = 1;
  std::vector<SdmcOutputRecord> rec(1);
  SdmcOutputs out; out.numHits = 0; out.numOutputTrials = 0; out.data = rec.data();
  int rc = RunSdmc(&in, &out);
  *hits = out.numHits;
  return rc;
}
'''


def build(cara, rt_libs):
    sd = os.path.join(cara, "DistributedMatlab", "ProbabilityOfCollision", "SDMC_Utils")
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "driver.cpp")
    with open(src, "w") as f:
        f.write(DRIVER)
    lib = os.path.join(tmp, "libnasa_sdmc.so")
    cmd = ["g++", "-O2", "-shared", "-fPIC", f"-I{sd}/include", f"-I{sd}", src,
           f"{sd}/sdmcEntry.cpp", f"{sd}/lib/libsdmc_linux.so", f"-L{rt_libs}", "-lifcore",
           "-lirc", "-limf", "-lintlc", f"-Wl,-rpath,{sd}/lib:{rt_libs}", "-o", lib]
    subprocess.run(cmd, check=True)
    nasa = ctypes.CDLL(lib)
    p = ctypes.c_void_p
    nasa.nasa_sdmc.argtypes = [p, p, p, p, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                               ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
    return nasa


def run_nasa(nasa, cdm, hbr, trials, seed):
    head, objs = read_cdm(cdm)
    pv, cv = [], []
    for o in objs:
        r, v, _ = cdm_states(o)
        pv.append(np.ascontiguousarray(np.concatenate([r, v]) / 1e3))
        cv.append(np.ascontiguousarray(
            [float(o[f"C{a}_{b}"]) for i, a in enumerate(CDM_COV6) for b in CDM_COV6[: i + 1]]))
    (r1, v1, _), (r2, v2, _) = cdm_states(objs[0]), cdm_states(objs[1])
    half = min(orbit_period(r1, v1), orbit_period(r2, v2)) / 4.0
    tca = datetime.datetime.fromisoformat(head["TCA"])
    tca_days = (tca - datetime.datetime(1970, 1, 1)).total_seconds() / 86400.0 + 1.0
    hits = ctypes.c_int(0)
    ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    rc = nasa.nasa_sdmc(ptr(pv[0]), ptr(cv[0]), ptr(pv[1]), ptr(cv[1]), tca_days,
                        half / 86400.0, seed, trials, hbr, ctypes.byref(hits))
    return hits.value, rc


def half_window(cdm):
    """The span run_nasa gives CARA's library (a quarter period either side), for kessler too."""
    _, objs = read_cdm(cdm)
    (r1, v1, _), (r2, v2, _) = cdm_states(objs[0]), cdm_states(objs[1])
    return min(orbit_period(r1, v1), orbit_period(r2, v2)) / 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cara", required=True, help="checkout of nasa/CARA_Analysis_Tools")
    ap.add_argument("--rt-libs", required=True,
                    help="directory holding libifcore.so, libirc.so, libimf.so, libintlc.so")
    ap.add_argument("--cases", default="1,2,5,8,14,25")
    ap.add_argument("--trials", type=int, default=1_000_000)
    ap.add_argument("--cache", default=os.path.join(ROOT, "bench", ".cara_cache"))
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results", "m3_vs_nasa_lib.json"))
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    nasa = build(args.cara, args.rt_libs)
    table = cara_table(args.cache)
    out = []
    for c in (int(x) for x in args.cases.split(",")):
        row = table[c - 1]
        cdm = fetch(row["Conjunction_ID"] + ".cdm", args.cache).decode()
        t = time.time()
        nh, rc = run_nasa(nasa, cdm, row["HBR_m"], args.trials, -1000 - c)
        tn = time.time() - t
        t = time.time()
        k = pc_mc_from_cdm(cdm, args.trials, hbr=row["HBR_m"], seed=c, window=half_window(cdm))
        tk = time.time() - t
        res = {"case": c, "vrel_mps": row["Vrel_mps"], "trials": args.trials,
               "nasa_hits": nh, "nasa_rc": rc, "nasa_s": tn, "kessler_hits": k["hits"],
               "kessler_s": tk, "speedup": tn / tk,
               "z": zscore(k["hits"], args.trials, nh, args.trials)}
        out.append(res)
        print(f"case {c:2d} v={row['Vrel_mps']:8.1f}: NASA {nh} hits in {tn:.1f}s, kessler "
              f"{k['hits']} hits in {tk:.2f}s ({tn / tk:.0f}x), z={res['z']:+.2f}", flush=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
