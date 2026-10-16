"""The three choices that decide whether a two-body Monte Carlo agrees with NASA CARA's, each
varied alone on the CARA test conjunctions where it matters (results: bench/results/m3_ablations.json).

  * sampling coordinates: Cartesian states, equinoctial elements with the semi-major axis, or
    with the mean motion (CARA's), on cases 25 (53.6 m/s) and 45 (15.2 km/s, Pc 7.4e-7), with
    case 1 as a control where all three should agree;
  * window length and which encounter is counted: a quarter, a half or a whole shorter period
    either side of TCA, counting the encounter at TCA only (or, over the whole period, every
    close approach in the window), on the two drifting cases 52 and 53 (9 and 11 m/s);
  * how many of the 53 conjunctions have a second close approach within half a period of TCA
    (the search windows of src/mc.cpp, which cost no trials).

Usage: python bench/m3_ablations.py [--cache DIR] [--scale 1.0]
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))

import numpy as np  # noqa: E402

from kessler._native import pc_mc_windows  # noqa: E402
from kessler.pc import (_factor, cdm_states6, orbit_period, pc_mc, read_cdm,  # noqa: E402
                        to_equinoctial)
from m3_sdmc import cara_table, fetch, zscore  # noqa: E402


def states(cdm):
    _, objs = read_cdm(cdm)
    return cdm_states6(objs[0]) + cdm_states6(objs[1])


def run(row, cdm, trials, seed, **kw):
    r1, v1, c1, r2, v2, c2 = states(cdm)
    t = time.time()
    k = pc_mc(r1, v1, c1, r2, v2, c2, row["HBR_m"], trials, seed=seed, **kw)
    k["seconds"] = time.time() - t
    k["z_vs_nasa"] = zscore(k["hits"], k["trials"], row["NhitSDMC"], row["NtotSDMC"])
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "bench", ".cara_cache"))
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every trial count")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results", "m3_ablations.json"))
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    rows = cara_table(args.cache)
    cdms = [fetch(r["Conjunction_ID"] + ".cdm", args.cache).decode() for r in rows]
    out = {"sampling": [], "window": [], "second_encounter": {}}

    def save():
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)

    def nasa(row):
        return {"pc": row["PcSDMC"], "hits": row["NhitSDMC"], "trials": row["NtotSDMC"]}

    # 1. sampling coordinates, at the default window (half a period, encounter at TCA only)
    for case, trials in ((1, 5_000_000), (25, 4_000_000), (45, 100_000_000)):
        row = rows[case - 1]
        for space in ("cartesian", "equinoctial", "equinoctial-n"):
            k = run(row, cdms[case - 1], int(trials * args.scale), seed=100 + case, space=space)
            out["sampling"].append({"case": case, "space": space, "kessler": k, "nasa_sdmc": nasa(row)})
            print(f"case {case:2d} {space:14s} {k['pc']:.3e} ({k['hits']} hits / {k['trials']:.2e}, "
                  f"{k['seconds']:.0f} s)  NASA {row['PcSDMC']:.3e}  z={k['z_vs_nasa']:+.2f}", flush=True)
            save()

    # 2. window length and which encounter is counted
    for case, trials in ((52, 8_000_000), (53, 8_000_000)):
        row = rows[case - 1]
        r1, v1, _, r2, v2, _ = states(cdms[case - 1])
        period = min(orbit_period(r1, v1), orbit_period(r2, v2))
        for frac, restrict in ((0.25, True), (0.5, True), (1.0, True), (1.0, False)):
            k = run(row, cdms[case - 1], int(trials * args.scale), seed=200 + case,
                    window=frac * period, restrict=restrict)
            what = "encounter at TCA" if restrict else "every encounter in the window"
            out["window"].append({"case": case, "half_window_periods": frac, "counted": what,
                                  "period_s": period, "kessler": k, "nasa_sdmc": nasa(row)})
            print(f"case {case:2d} ±{frac:.2f} P, {what:30s} {k['pc']:.3e} ({k['hits']} hits, "
                  f"{k['seconds']:.0f} s)  NASA {row['PcSDMC']:.3e}  z={k['z_vs_nasa']:+.2f}", flush=True)
            save()

    # 3. second close approaches within half a period of TCA
    counts = []
    for i, (row, cdm) in enumerate(zip(rows, cdms), 1):
        r1, v1, c1, r2, v2, c2 = states(cdm)
        w = min(orbit_period(r1, v1), orbit_period(r2, v2)) / 2.0
        m1, e1 = to_equinoctial(r1, v1, c1, True)
        m2, e2 = to_equinoctial(r2, v2, c2, True)
        wins = pc_mc_windows(m1, _factor(e1), m2, _factor(e2), row["HBR_m"], -w, w, space=2)
        counts.append({"case": i, "windows": len(wins)})
    out["second_encounter"] = {"cases_with_more_than_one_window": sum(c["windows"] > 1 for c in counts),
                               "per_case": counts}
    print(f"{out['second_encounter']['cases_with_more_than_one_window']} of {len(rows)} conjunctions "
          "have more than one search window within half a period of TCA")
    save()


if __name__ == "__main__":
    main()
