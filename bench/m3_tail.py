"""CARA's case 45 (ICESat-2 and SpaceBEE-114, Pc 7.4e-7), decided by the far tail of the
uncertainty: equinoctial elements with the semi-major axis against the same elements with the
mean motion, a billion trials each (results: bench/results/m3_tail.json). At the 1e8 trials of
bench/m3_ablations.py the two forms differ by 2-3 standard errors; this settles which, if
either, departs from CARA's 2,952 hits in 4 billion trials.

Usage: python bench/m3_tail.py [--trials 1e9] [--chunks 10]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))

from m3_ablations import states  # noqa: E402
from m3_sdmc import cara_table, fetch, zscore  # noqa: E402

from kessler.pc import pc_mc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "bench", ".cara_cache"))
    ap.add_argument("--trials", type=float, default=1e9)
    ap.add_argument("--chunks", type=int, default=10, help="independent seeds the trials are split into")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results", "m3_tail.json"))
    args = ap.parse_args()
    row = cara_table(args.cache)[44]
    cdm = fetch(row["Conjunction_ID"] + ".cdm", args.cache).decode()
    r1, v1, c1, r2, v2, c2 = states(cdm)
    per = int(args.trials // args.chunks)
    out = []
    for space in ("equinoctial", "equinoctial-n"):
        hits, trials, chunks, t = 0, 0, [], time.time()
        for i in range(args.chunks):
            k = pc_mc(r1, v1, c1, r2, v2, c2, row["HBR_m"], per, seed=4500 + i, space=space)
            hits += k["hits"]
            trials += k["trials"]
            chunks.append(k["hits"])
        z = zscore(hits, trials, row["NhitSDMC"], row["NtotSDMC"])
        out.append({"case": 45, "space": space, "hits": hits, "trials": trials, "pc": hits / trials,
                    "chunk_hits": chunks, "seconds": time.time() - t, "z_vs_nasa": z,
                    "nasa_sdmc": {"pc": row["PcSDMC"], "hits": row["NhitSDMC"], "trials": row["NtotSDMC"]}})
        print(f"case 45 {space:14s} {hits / trials:.3e} ({hits} hits / {trials:.2e})  "
              f"NASA {row['PcSDMC']:.3e}  z={z:+.2f}", flush=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
