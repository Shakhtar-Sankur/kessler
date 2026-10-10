"""Reproduce CelesTrak SOCRATES with kessler.

SOCRATES (celestrak.org/SOCRATES) screens active satellites against the public catalogue with
commercial software (STK/CAT). For its N closest conjunctions this script:
  1. downloads SOCRATES's raw results (sort-minRange.csv);
  2. downloads each object's element set from CelesTrak (GP, or SupGP when GP does not match)
     and keeps a conjunction only if both element sets are the ones SOCRATES used, i.e. the
     days from epoch to TCA agree with SOCRATES's DSE to within 0.001 day;
  3. finds kessler's closest approach near SOCRATES's TCA and compares time, range and speed;
  4. compares SOCRATES's maximum probability with kessler's (Alfano's method), reporting the
     hard-body radius that would reproduce SOCRATES's value, since SOCRATES does not publish it.

Run where CelesTrak is reachable (e.g. Kaggle). Usage: python bench/m2_socrates.py [--n 100]
"""
import argparse
import csv
import io
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kessler._native import Elsets  # noqa: E402
from kessler.catalog import Catalog, epoch_days  # noqa: E402
from kessler.pc import max_pc  # noqa: E402
from kessler.screen import max_speed, offsets, screen  # noqa: E402

UA = {"User-Agent": "kessler (github.com/Shakhtar-Sankur/kessler)"}
SOCRATES = "https://celestrak.org/SOCRATES/sort-minRange.csv"
GP = "https://celestrak.org/NORAD/elements/gp.php?CATNR={}&FORMAT=csv"
SUPGP = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?CATNR={}&FORMAT=csv"


def get(url, pause=0.5):
    time.sleep(pause)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read().decode()


def parse_time(s):
    s = s.strip().replace(" ", "T")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(s)


def days_since_1949(dt):
    return (dt - datetime(1949, 12, 31)).total_seconds() / 86400.0


def matching(rows, tca_days, dse):
    for row in rows:
        if "EPOCH" in row and abs((tca_days - epoch_days(row["EPOCH"])) - dse) < 1e-3:
            return row
    return None


def element_sets(norad, tca_days, dse, cache):
    """The element set SOCRATES used for this object, or None: CelesTrak GP, CelesTrak SupGP, then
    (if credentials are set) Space-Track's history, which keeps superseded element sets."""
    if ("history", norad) in cache:
        row = matching(cache[("history", norad)], tca_days, dse)
        if row is not None:
            return row, "space-track gp_history"
    for kind, url in (("gp", GP), ("supgp", SUPGP)):
        key = (kind, norad)
        if key not in cache:
            try:
                rows = list(csv.DictReader(io.StringIO(get(url.format(norad)))))
            except Exception:
                rows = []
            cache[key] = rows
        row = matching(cache[key], tca_days, dse)
        if row is not None:
            return row, kind
    return None, None


def fetch_history(soc, cache):
    """One Space-Track query for every object of every conjunction, around SOCRATES's epochs."""
    try:
        from kessler.spacetrack import Client
        st = Client()
    except Exception as e:
        print(f"Space-Track history not used: {e}")
        return
    from collections import defaultdict
    from datetime import timedelta
    ids, epochs = set(), []
    for c in soc:
        tca = parse_time(c["TCA"])
        for k in ("1", "2"):
            ids.add(int(c["NORAD_CAT_ID_" + k]))
            epochs.append(tca - timedelta(days=float(c["DSE_" + k])))
    start = (min(epochs) - timedelta(days=1)).strftime("%Y-%m-%d")
    end = (max(epochs) + timedelta(days=2)).strftime("%Y-%m-%d")
    by = defaultdict(list)
    for r in st.gp_history(sorted(ids), start, end):
        by[int(r["NORAD_CAT_ID"])].append(r)
    for norad, rows in by.items():
        cache[("history", norad)] = rows
    print(f"Space-Track history: {sum(len(v) for v in by.values())} element sets for {len(by)} objects")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results", "m2_socrates.json"))
    a = ap.parse_args()

    soc = list(csv.DictReader(io.StringIO(get(SOCRATES, 0))))[: a.n]
    print(f"SOCRATES: {len(soc)} conjunctions (closest first); columns {list(soc[0])}")
    cache, rows = {}, []
    fetch_history(soc, cache)
    for c in soc:
        tca = parse_time(c["TCA"])
        tca_days = days_since_1949(tca)
        o1, k1 = element_sets(int(c["NORAD_CAT_ID_1"]), tca_days, float(c["DSE_1"]), cache)
        o2, k2 = element_sets(int(c["NORAD_CAT_ID_2"]), tca_days, float(c["DSE_2"]), cache)
        row = {"a": c["OBJECT_NAME_1"], "b": c["OBJECT_NAME_2"], "tca": c["TCA"],
               "socrates_range_km": float(c["TCA_RANGE"]), "socrates_speed_km_s": float(c["TCA_RELATIVE_SPEED"]),
               "socrates_max_prob": float(c["MAX_PROB"]), "socrates_dilution_km": float(c["DILUTION"])}
        if o1 is None or o2 is None:
            row["status"] = "element set used by SOCRATES no longer available"
            rows.append(row)
            continue
        row["elements"] = [k1, k2]
        cat = Catalog([o1, o2])
        els = Elsets(cat.mean)
        start = tca_days - 300.0 / 86400.0
        dmax = max(2.0 * row["socrates_range_km"], 1.0)
        ev, _ = screen(els, offsets(cat.mean, start), max_speed(cat.mean), 600.0, dmax)
        if len(ev) == 0:
            row["status"] = "kessler found no approach within 5 minutes of SOCRATES's TCA"
            rows.append(row)
            continue
        e = ev[np.argmin(np.abs(ev["tca"] - 300.0))]
        row.update(status="compared", kessler_range_km=float(e["miss"]), kessler_speed_km_s=float(e["vrel"]),
                   tca_diff_ms=float((e["tca"] - 300.0) * 1e3),
                   range_diff_m=float((e["miss"] - row["socrates_range_km"]) * 1e3))
        # states at kessler's TCA for the maximum probability
        r, v, _ = els.propagate_at(np.array([0, 1]), offsets(cat.mean, start) + e["tca"] / 60.0)
        r, v = r * 1e3, v * 1e3

        def mp(hbr):
            return max_pc(r[0], v[0], r[1], v[1], hbr)[0]

        row["kessler_max_prob_hbr_5m"] = mp(5.0)
        sp = row["socrates_max_prob"]
        if 0.0 < sp < 1.0:
            lo, hi = 1e-3, 1e3        # hard-body radius (m) that reproduces SOCRATES's value
            for _ in range(80):
                mid = math.sqrt(lo * hi)
                lo, hi = (mid, hi) if mp(mid) < sp else (lo, mid)
            row["implied_hbr_m"] = math.sqrt(lo * hi)
        rows.append(row)
        print(f"{row['a'][:24]:>24} x {row['b'][:24]:<24} range {row['socrates_range_km']*1e3:8.1f} m  "
              f"dTCA {row['tca_diff_ms']:8.3f} ms  drange {row['range_diff_m']:8.3f} m  "
              f"SOCRATES maxP {sp:.3e}  implied HBR {row.get('implied_hbr_m', float('nan')):.2f} m")

    done = [r for r in rows if r["status"] == "compared"]
    sources = {}
    for r in done:
        for k in r["elements"]:
            sources[k] = sources.get(k, 0) + 1
    summary = {"socrates_rows": len(soc), "compared": len(done), "element_set_sources": sources,
               "skipped": {s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows} - {"compared"}}}
    if done:
        summary.update(
            max_abs_tca_diff_ms=max(abs(r["tca_diff_ms"]) for r in done),
            max_abs_range_diff_m=max(abs(r["range_diff_m"]) for r in done),
            median_abs_range_diff_m=float(np.median([abs(r["range_diff_m"]) for r in done])),
            max_abs_speed_diff_m_s=max(abs(r["kessler_speed_km_s"] - r["socrates_speed_km_s"]) * 1e3 for r in done),
            implied_hbr_m=sorted(round(r["implied_hbr_m"], 3) for r in done if "implied_hbr_m" in r),
        )
    print(json.dumps(summary, indent=2))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
