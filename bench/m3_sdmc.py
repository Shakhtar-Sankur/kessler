"""kessler's Monte Carlo probability of collision against NASA CARA's, on CARA's 53 test conjunctions.

CARA publishes, for each conjunction, the result of its own two-body Monte Carlo (SDMC: hits and
trials) next to its 2D and 3D estimates. For each one this script runs kessler.pc.pc_mc with
enough trials for about 3,000 hits (within a time budget), and compares:
  * kessler with CARA's Monte Carlo: a two-proportion z-score from both hit counts;
  * the 2D method with the Monte Carlo, to show where the short-encounter assumption fails.

The CDMs and CARA's table are fetched from github.com/nasa/CARA_Analysis_Tools at a pinned commit
(cached in --cache, not redistributed). Results go to bench/results/m3_sdmc.json.

Usage: python bench/m3_sdmc.py [--hits 3000] [--budget 120] [--cache DIR]
"""
import argparse
import io
import json
import math
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kessler.pc import pc2d_from_cdm, pc_mc_from_cdm  # noqa: E402

COMMIT = "ce0f00722c56808755ee0adbe47a4e3d56439484"
BASE = f"https://raw.githubusercontent.com/nasa/CARA_Analysis_Tools/{COMMIT}/DataFiles/PcTestCaseCDMs/"


def fetch(name, cache):
    path = os.path.join(cache, name)
    if not os.path.exists(path):
        with urllib.request.urlopen(BASE + name, timeout=60) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
    with open(path, "rb") as f:
        return f.read()


def cara_table(cache):
    import openpyxl
    data = fetch("CARA_PcMethod_Test_Conjunctions.xlsx", cache)
    rows = list(openpyxl.load_workbook(io.BytesIO(data), read_only=True).worksheets[0]
                .iter_rows(values_only=True))
    idx = {h: i for i, h in enumerate(rows[0])}
    return [{k: r[i] for k, i in idx.items()} for r in rows[1:]]


def zscore(h1, n1, h2, n2):
    p = (h1 + h2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (h1 / n1 - h2 / n2) / se if se > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", type=int, default=3000, help="target number of hits per case")
    ap.add_argument("--budget", type=float, default=120.0, help="seconds per case at most")
    ap.add_argument("--cache", default=os.path.join(ROOT, "bench", ".cara_cache"))
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results", "m3_sdmc.json"))
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    results = []
    for i, row in enumerate(cara_table(args.cache), 1):
        cdm = fetch(row["Conjunction_ID"] + ".cdm", args.cache).decode()
        hbr = row["HBR_m"]
        t = time.time()
        pc_mc_from_cdm(cdm, 200_000, hbr=hbr, seed=1000 + i)
        rate = 200_000 / (time.time() - t)
        trials = int(min(args.hits / row["PcSDMC"], rate * args.budget))
        trials = max(1_000_000, (trials // 65536 + 1) * 65536)
        t = time.time()
        k = pc_mc_from_cdm(cdm, trials, hbr=hbr, seed=i)
        dt = time.time() - t
        pc2d = pc2d_from_cdm(cdm, hbr)
        res = {
            "case": i, "id": row["Conjunction_ID"], "comment": row["Comment"],
            "vrel_mps": row["Vrel_mps"], "hbr_m": hbr,
            "kessler": k, "seconds": dt, "trials_per_s": trials / dt,
            "nasa_sdmc": {"pc": row["PcSDMC"], "lo": row["PcSDMCLo"], "hi": row["PcSDMCHi"],
                          "hits": row["NhitSDMC"], "trials": row["NtotSDMC"]},
            "nasa_pc2d": row["Pc2D"], "nasa_nc3d": row["Nc3D"], "kessler_pc2d": pc2d,
            "z_vs_nasa": zscore(k["hits"], k["trials"], row["NhitSDMC"], row["NtotSDMC"]),
            "pc2d_over_mc": pc2d / row["PcSDMC"],
        }
        results.append(res)
        print(f"{i:2d} v={row['Vrel_mps']:8.1f}  kessler {k['pc']:.4e} ({k['hits']} hits, "
              f"{trials:.2e} trials, {dt:.0f}s)  NASA {row['PcSDMC']:.4e}  z={res['z_vs_nasa']:+.2f}"
              f"  2D/MC={res['pc2d_over_mc']:.2e}", flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)

    z = [abs(r["z_vs_nasa"]) for r in results]
    print(f"\n{len(results)} conjunctions: |z| < 2 for {sum(v < 2 for v in z)}, "
          f"< 3 for {sum(v < 3 for v in z)}; largest |z| {max(z):.2f}")


if __name__ == "__main__":
    main()
