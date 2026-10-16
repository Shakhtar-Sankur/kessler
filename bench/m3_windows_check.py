"""Do the search windows change any hit? For every CARA test conjunction with a single
encounter in +-P/2 of TCA (so that a full search and the restricted one look for the same
approach), the same random draws are searched both ways and the hit counts compared
(results: bench/results/m3_windows_check.json).

Usage: python bench/m3_windows_check.py [--trials 200000]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))

from m3_ablations import states  # noqa: E402
from m3_sdmc import cara_table, fetch  # noqa: E402

from kessler._native import pc_mc_windows  # noqa: E402
from kessler.pc import _factor, orbit_period, pc_mc, to_equinoctial  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "bench", ".cara_cache"))
    ap.add_argument("--trials", type=int, default=200_000)
    ap.add_argument("--hbr-scale", type=float, default=20.0,
                    help="radius multiplier, so that every case has hits to compare")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results", "m3_windows_check.json"))
    args = ap.parse_args()
    out = []
    for i, row in enumerate(cara_table(args.cache), 1):
        cdm = fetch(row["Conjunction_ID"] + ".cdm", args.cache).decode()
        r1, v1, c1, r2, v2, c2 = states(cdm)
        w = min(orbit_period(r1, v1), orbit_period(r2, v2)) / 2.0
        hbr = row["HBR_m"] * args.hbr_scale
        m1, e1 = to_equinoctial(r1, v1, c1, True)
        m2, e2 = to_equinoctial(r2, v2, c2, True)
        if len(pc_mc_windows(m1, _factor(e1), m2, _factor(e2), hbr, -w, w, space=2)) != 1:
            continue
        h = [pc_mc(r1, v1, c1, r2, v2, c2, hbr, args.trials, seed=700 + i, restrict=rs)["hits"]
             for rs in (True, False)]
        out.append({"case": i, "hbr_m": hbr, "trials": args.trials, "restricted": h[0], "full": h[1]})
        print(f"case {i:2d}  hbr {hbr:7.1f} m  restricted {h[0]:7d}  full {h[1]:7d}"
              f"  {'same' if h[0] == h[1] else 'DIFFERENT'}", flush=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    same = sum(x["restricted"] == x["full"] for x in out)
    print(f"{same} of {len(out)} single-encounter conjunctions: identical hit counts")


if __name__ == "__main__":
    main()
