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


ELEMENTS = ("EPOCH", "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE", "ARG_OF_PERICENTER",
            "MEAN_ANOMALY", "BSTAR")


def matching(rows, tca_days, dse):
    """The element set whose days-from-epoch is closest to SOCRATES's DSE (within 0.001 day, the
    precision SOCRATES publishes), and how many distinct element sets were within it."""
    hits = {}
    for row in rows:
        if "EPOCH" in row:
            diff = abs((tca_days - epoch_days(row["EPOCH"])) - dse)
            if diff < 1e-3:
                hits.setdefault(tuple(row.get(k) for k in ELEMENTS), (diff, row))
    if not hits:
        return None, 0
    return min(hits.values(), key=lambda x: x[0])[1], len(hits)


def element_sets(norad, tca_days, dse, cache):
    """The element set SOCRATES used for this object, its source and the number of candidates:
    Space-Track's history (if credentials are set; it keeps superseded element sets), then
    CelesTrak GP and SupGP."""
    if ("history", norad) in cache:
        row, n = matching(cache[("history", norad)], tca_days, dse)
        if row is not None:
            return row, "space-track gp_history", n
    for kind, url in (("gp", GP), ("supgp", SUPGP)):
        key = (kind, norad)
        if key not in cache:
            try:
                rows = list(csv.DictReader(io.StringIO(get(url.format(norad)))))
            except Exception:
                rows = []
            cache[key] = rows
        row, n = matching(cache[key], tca_days, dse)
        if row is not None:
            return row, kind, n
    return None, None, 0


def fetch_history(soc, cache):
    """One Space-Track query for every object of every conjunction, around SOCRATES's epochs."""
    try:
        _fetch_history(soc, cache)
    except Exception as e:
        print(f"Space-Track history not used ({type(e).__name__}: {e}); CelesTrak only")


def _fetch_history(soc, cache):
    from kessler.spacetrack import Client, credentials
    credentials()
    st = Client()
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
    # radar cross-section class (SMALL < 0.1 m2 < MEDIUM < 1 m2 < LARGE), to see what SOCRATES's
    # hard-body radius depends on
    try:
        part = ",".join(str(x) for x in sorted(ids))
        for r in st.query(f"class/satcat/NORAD_CAT_ID/{part}"):
            cache[("rcs", int(r["NORAD_CAT_ID"]))] = r.get("RCS_SIZE") or "unknown"
    except Exception as e:
        print(f"Space-Track satcat not used ({type(e).__name__}: {e})")


def size_class(name, norad, cache):
    return "STARLINK" if "STARLINK" in name.upper() else cache.get(("rcs", norad), "unknown")


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
        n1, n2 = int(c["NORAD_CAT_ID_1"]), int(c["NORAD_CAT_ID_2"])
        o1, k1, c1 = element_sets(n1, tca_days, float(c["DSE_1"]), cache)
        o2, k2, c2 = element_sets(n2, tca_days, float(c["DSE_2"]), cache)
        row = {"a": c["OBJECT_NAME_1"], "b": c["OBJECT_NAME_2"], "tca": c["TCA"],
               "pair": " + ".join(sorted([size_class(c["OBJECT_NAME_1"], n1, cache),
                                          size_class(c["OBJECT_NAME_2"], n2, cache)])),
               "socrates_range_km": float(c["TCA_RANGE"]), "socrates_speed_km_s": float(c["TCA_RELATIVE_SPEED"]),
               "socrates_max_prob": float(c["MAX_PROB"]), "socrates_dilution_km": float(c["DILUTION"])}
        if o1 is None or o2 is None:
            row["status"] = "element set used by SOCRATES no longer available"
            rows.append(row)
            continue
        row["elements"] = [k1, k2]
        # more than one element set within SOCRATES's 0.001-day precision: which one it used is
        # uncertain, so these comparisons are reported apart
        row["ambiguous"] = c1 > 1 or c2 > 1
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
              f"SOCRATES maxP {sp:.3e}  implied HBR {row.get('implied_hbr_m', float('nan')):.2f} m  "
              f"{row['pair']}{'  (ambiguous element set)' if row['ambiguous'] else ''}")

    done = [r for r in rows if r["status"] == "compared"]
    sources = {}
    for r in done:
        for k in r["elements"]:
            sources[k] = sources.get(k, 0) + 1
    summary = {"socrates_rows": len(soc), "compared": len(done), "element_set_sources": sources,
               "skipped": {s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows} - {"compared"}}}
    def stats(rs):
        if not rs:
            return {}
        return {"n": len(rs),
                "max_abs_tca_diff_ms": max(abs(r["tca_diff_ms"]) for r in rs),
                "max_abs_range_diff_m": max(abs(r["range_diff_m"]) for r in rs),
                "median_abs_range_diff_m": float(np.median([abs(r["range_diff_m"]) for r in rs])),
                "range_within_0.5m": sum(abs(r["range_diff_m"]) <= 0.5 for r in rs),
                "max_abs_speed_diff_m_s": max(abs(r["kessler_speed_km_s"] - r["socrates_speed_km_s"]) * 1e3
                                              for r in rs)}

    if done:
        summary.update(stats(done))
        summary["unambiguous"] = stats([r for r in done if not r["ambiguous"]])
        summary["ambiguous"] = stats([r for r in done if r["ambiguous"]])
        by_pair = {}
        for r in done:
            if "implied_hbr_m" in r:
                by_pair.setdefault(r["pair"], []).append(r["implied_hbr_m"])
        summary["implied_hbr_m_by_pair"] = {
            k: {"n": len(v), "median": round(float(np.median(v)), 2), "min": round(min(v), 2), "max": round(max(v), 2)}
            for k, v in sorted(by_pair.items())}
    print(json.dumps(summary, indent=2))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
