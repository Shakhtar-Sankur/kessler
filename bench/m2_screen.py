"""M2: screen the catalogue for close approaches (all pairs), on the GPU and on the CPU.

Catalogue: the CelesTrak 'active' snapshot in tests/ by default, or with --spacetrack the full
public catalogue from Space-Track (needs SPACETRACK_USER / SPACETRACK_PASS, e.g. Kaggle
secrets). Space-Track data is used in memory only and never written to the repository.

Usage: python bench/m2_screen.py [--hours 24] [--dmax 5] [--spacetrack] [--out bench/results/m2.json]
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kessler import cuda  # noqa: E402
from kessler._native import Elsets  # noqa: E402
from kessler.catalog import Catalog  # noqa: E402
from kessler.screen import max_speed, offsets, screen  # noqa: E402


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
    except Exception:
        return "none"


def spacetrack_csv():
    """Every on-orbit object with elements from the last 30 days, from Space-Track."""
    import csv
    import io
    from kessler.spacetrack import Client
    rows = Client().query("class/gp/decay_date/null-val/epoch/%3Enow-30/orderby/norad_cat_id")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def summarize(ev, cat, idx, start_iso, top=15):
    names = [cat.name[k] for k in idx]
    # docked or co-located objects (e.g. a cargo ship on the ISS, two satellites sharing a
    # geostationary slot) stay together: below 10 m/s relative speed they are not conjunctions
    together = ev[ev["vrel"] < 0.01]
    ev = ev[ev["vrel"] >= 0.01]
    star = np.array(["STARLINK" in n.upper() for n in names])
    both = star[ev["i"]] & star[ev["j"]]
    one = star[ev["i"]] ^ star[ev["j"]]
    order = np.argsort(ev["miss"])[:top]
    return {
        "events": int(len(ev)),
        "docked_or_colocated_pairs": int(len(set(zip(together["i"].tolist(), together["j"].tolist())))),
        "events_below_1km": int(np.sum(ev["miss"] < 1.0)),
        "events_below_200m": int(np.sum(ev["miss"] < 0.2)),
        "starlink_vs_starlink": int(np.sum(both)),
        "starlink_vs_other": int(np.sum(one)),
        "neither_starlink": int(np.sum(~both & ~one)),
        "closest": [{"a": names[e["i"]], "a_norad": int(cat.norad[idx[e["i"]]]),
                     "b": names[e["j"]], "b_norad": int(cat.norad[idx[e["j"]]]),
                     "tca_hours_after_start": round(float(e["tca"]) / 3600, 4),
                     "miss_km": round(float(e["miss"]), 4), "vrel_km_s": round(float(e["vrel"]), 3)}
                    for e in ev[order]],
        "start": start_iso,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--dmax", type=float, default=5.0)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--spacetrack", action="store_true")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results", "m2.json"))
    a = ap.parse_args()

    if a.spacetrack:
        text, source = spacetrack_csv(), "space-track gp (decay_date null, epoch > now-30)"
    else:
        text, source = gzip.open(os.path.join(ROOT, "tests", "active_2026-10-10.csv.gz"), "rt").read(), \
            "celestrak active 2026-10-10"
    cat = Catalog.from_csv(text)
    # one screening start for everyone: when the catalogue was current. A few objects carry
    # predicted epochs days ahead (e.g. XMM-Newton, Chandra), so use the 99.9th percentile.
    start = float(np.quantile(cat.mean["epoch"], 0.999))
    from datetime import datetime, timedelta
    start_iso = (datetime(1949, 12, 31) + timedelta(days=start)).isoformat(timespec="seconds") + "Z"
    idx = np.arange(len(cat))
    mean = cat.mean[idx]
    els = Elsets(mean)
    ok = els.init_error == 0
    off, vmax = offsets(mean, start), max_speed(mean)
    duration = a.hours * 3600.0
    res = {"gpu": gpu_name(), "cpu_count": os.cpu_count(), "source": source, "objects": int(len(cat)),
           "objects_init_ok": int(ok.sum()), "hours": a.hours, "dmax_km": a.dmax, "step_s": a.step,
           "start": start_iso}
    print(f"{source}: {len(cat)} objects, {a.hours} h from {start_iso}, dmax {a.dmax} km, GPU {res['gpu']}")

    if cuda.available():
        t0 = time.perf_counter()
        ev_gpu, cand_gpu, gpu_ms = cuda.screen(els, off, vmax, duration, a.dmax, a.step)
        res["gpu_run"] = {"seconds_total": time.perf_counter() - t0, "gpu_candidate_seconds": gpu_ms / 1e3,
                          "candidates": int(len(cand_gpu))}
        res["gpu_run"].update(summarize(ev_gpu, cat, idx, start_iso))
        print(json.dumps(res["gpu_run"], indent=2))
    else:
        ev_gpu = None

    if not a.skip_cpu:
        t0 = time.perf_counter()
        ev_cpu, cand_cpu = screen(els, off, vmax, duration, a.dmax, a.step)
        res["cpu_run"] = {"seconds_total": time.perf_counter() - t0, "candidates": int(len(cand_cpu)),
                          "events": int(len(ev_cpu))}
        print(json.dumps(res["cpu_run"], indent=2))
        if ev_gpu is not None:
            x, y = np.sort(ev_cpu, order=["i", "j", "tca"]), np.sort(ev_gpu, order=["i", "j", "tca"])
            same = len(x) == len(y) and np.array_equal(x["i"], y["i"]) and np.array_equal(x["j"], y["j"])
            res["gpu_equals_cpu"] = {
                "same_events": bool(same),
                "max_tca_diff_s": float(np.max(np.abs(x["tca"] - y["tca"]))) if same and len(x) else None,
                "max_miss_diff_km": float(np.max(np.abs(x["miss"] - y["miss"]))) if same and len(x) else None,
            }
            print(json.dumps(res["gpu_equals_cpu"]))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
