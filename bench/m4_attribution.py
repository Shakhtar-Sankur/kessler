"""M4b: do Starlink maneuvers follow close approaches predicted from public data?

For each chosen day D (default: the 15th of each month of the FCC report's period):
  1. download from Space-Track every element set with epoch in [D-4, D+3] of the objects whose
     orbits reach the Starlink shells (perigee < 650 km, apogee > 300 km); kept in memory and in
     the local cache only;
  2. rebuild the catalogue as it stood at the start of D and screen it for 2 days (5 km);
  3. take the Starlink satellites in a stable shell with a fresh element set (< 1 day old), and
     split them into those that maneuvered during D (and not the day before) and those that did
     not maneuver from D-1 to D+2 (kessler.maneuvers.detect on their history);
  4. compare how often each group had a predicted approach closer than 0.2 / 0.5 / 1 / 2 / 5 km;
  5. for predicted approaches under 1 km after the maneuver, recompute the distance with the
     satellite's first element set after it, and for the quiet group with its latest element set
     before the approach (the change a routine update makes).

Usage: python bench/m4_attribution.py [--days 2025-12-15,2026-01-15] [--out ...]
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date, timedelta

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kessler import cuda  # noqa: E402
from kessler._native import Elsets  # noqa: E402
from kessler.attribution import closest_per_object, min_distance_near, snapshot, wilson  # noqa: E402
from kessler.catalog import Catalog, epoch_days  # noqa: E402
from kessler.maneuvers import detect, history  # noqa: E402
from kessler.screen import max_speed, offsets, screen  # noqa: E402
from kessler.spacetrack import Client  # noqa: E402

DAYS = ["2025-12-15", "2026-01-15", "2026-02-15", "2026-03-15", "2026-04-15", "2026-05-15"]
FIELDS = ("NORAD_CAT_ID,OBJECT_NAME,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,"
          "ARG_OF_PERICENTER,MEAN_ANOMALY,BSTAR")
ID_RANGES = ((0, 44999), (45000, 54999), (55000, 61999), (62000, 99999))
THRESHOLDS_KM = (0.2, 0.5, 1.0, 2.0, 5.0)
HOURS = 48.0


def download(st, d):
    lo, hi = (d - timedelta(days=4)).isoformat(), (d + timedelta(days=3)).isoformat()
    by_id = defaultdict(list)
    for a, b in ID_RANGES:
        for r in st.query(f"class/gp_history/NORAD_CAT_ID/{a}--{b}/EPOCH/{lo}--{hi}/PERIAPSIS/%3C650/"
                          f"APOAPSIS/%3E300/orderby/NORAD_CAT_ID,EPOCH/predicates/{FIELDS}"):
            by_id[int(r["NORAD_CAT_ID"])].append(r)
    return by_id


def classify(rows, d0, shell_km=10.0):
    """'maneuvered' (with the maneuver intervals during day d0), 'quiet', or None (unusable)."""
    t, a = history(rows)
    if len(t) < 8 or np.ptp(a) > shell_km:      # not in a stable shell: raising, lowering
        return None, []
    k, step = detect(t, a)
    iv = [(t[q - 1], t[q], float(s)) for q, s in zip(k, step)]
    during = [x for x in iv if x[0] >= d0 and 0.5 * (x[0] + x[1]) < d0 + 1]
    if during and not any(x[1] > d0 - 1 and x[0] < d0 for x in iv):
        return "maneuvered", during
    covered = all(np.any((t >= d0 + j) & (t < d0 + j + 1)) for j in (-1, 0, 1, 2))
    if not iv or all(x[1] <= d0 - 1 or x[0] >= d0 + 2 for x in iv):
        return ("quiet", []) if covered else (None, [])
    return None, []


def analyse_day(st, day):
    d = date.fromisoformat(day)
    d0 = epoch_days(f"{day}T00:00:00.000000")
    t0 = time.time()
    by_id = download(st, d)
    snap = snapshot(by_id, d0)
    ids = sorted(snap)
    cat = Catalog([snap[i] for i in ids])
    els = Elsets(cat.mean)
    off, vmax = offsets(cat.mean, d0), max_speed(cat.mean)
    if cuda.available():
        ev, _, _ = cuda.screen(els, off, vmax, HOURS * 3600.0, 5.0)
    else:
        ev, _ = screen(els, off, vmax, HOURS * 3600.0, 5.0)
    ev = ev[ev["vrel"] >= 0.01]                   # docked / co-located are not conjunctions
    best = closest_per_object(ev, len(ids))
    print(f"{day}: {len(by_id)} objects downloaded, {len(ids)} in the catalogue at 00:00, "
          f"{len(ev)} approaches < 5 km in {HOURS:.0f} h ({time.time() - t0:.0f} s)")

    out = {"day": day, "objects": len(ids), "approaches": int(len(ev)), "maneuvered": [], "quiet": []}
    for q, norad in enumerate(ids):
        if "STARLINK" not in cat.name[q].upper() or d0 - cat.mean["epoch"][q] > 1.0:
            continue
        group, during = classify(by_id[norad], d0)
        if group is None:
            continue
        rec = {"norad": norad, "closest_km": None}
        b = best[q]
        if b >= 0:
            e = ev[b]
            other = ids[e["j"] if e["i"] == q else e["i"]]
            tca = d0 + float(e["tca"]) / 86400.0
            rec.update(closest_km=float(e["miss"]), tca_hours=float(e["tca"]) / 3600.0,
                       vrel_km_s=float(e["vrel"]), partner=snap[other]["OBJECT_NAME"])
            if group == "maneuvered":
                rec["steps_m"] = [x[2] for x in during]
                rec["maneuver_hours"] = [[(x[0] - d0) * 24, (x[1] - d0) * 24] for x in during]
            # does the predicted approach survive the next element set?
            if e["miss"] < 1.0:
                rows = sorted(by_id[norad], key=lambda r: r["EPOCH"])
                if group == "maneuvered":
                    after = [r for r in rows if epoch_days(r["EPOCH"]) >= during[0][1]]
                    new = after[0] if after and epoch_days(after[0]["EPOCH"]) < tca else None
                else:
                    before = [r for r in rows if d0 <= epoch_days(r["EPOCH"]) < tca]
                    new = before[-1] if before else None
                if new is not None:
                    m = min_distance_near(new, snap[other], tca)
                    rec["recomputed_km"] = m[0] if m else ">200"
        out[group].append(rec)
    print(f"  Starlink: {len(out['maneuvered'])} maneuvered during the day, {len(out['quiet'])} quiet")
    return out


def summarise(days):
    groups = {g: [r for d in days for r in d[g]] for g in ("maneuvered", "quiet")}
    s = {"days": [d["day"] for d in days], "screening_hours": HOURS,
         "satellites": {g: len(v) for g, v in groups.items()}, "predicted_closer_than": {}}
    for thr in THRESHOLDS_KM:
        row = {}
        for g, v in groups.items():
            k = sum(r["closest_km"] is not None and r["closest_km"] < thr for r in v)
            row[g] = {"count": k, "fraction_and_95ci": wilson(k, len(v))}
        qm, qq = row["maneuvered"]["fraction_and_95ci"][0], row["quiet"]["fraction_and_95ci"][0]
        row["ratio"] = qm / qq if qq > 0 else None
        s["predicted_closer_than"][f"{thr} km"] = row
    s["recomputed_with_next_element_set"] = {}
    for g, v in groups.items():
        pairs = [(r["closest_km"], r["recomputed_km"]) for r in v if "recomputed_km" in r]
        moved = [(200.0 if n == ">200" else n) - p for p, n in pairs]
        s["recomputed_with_next_element_set"][g] = {
            "approaches": len(pairs),
            "median_change_km": float(np.median(moved)) if moved else None,
            "moved_further_by_more_than_1km": int(sum(m > 1.0 for m in moved)),
        }
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default=",".join(DAYS))
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results", "m4_attribution.json"))
    a = ap.parse_args()
    st = Client()
    days = []
    for day in a.days.split(","):
        try:
            days.append(analyse_day(st, day))
        except Exception as e:
            print(f"{day}: skipped ({type(e).__name__}: {e})")
    s = summarise(days)
    print(json.dumps(s, indent=2))
    for d in days:                   # keep the file small: quiet satellites only with an approach
        d["quiet_satellites"] = len(d["quiet"])
        d["quiet"] = [r for r in d["quiet"] if r["closest_km"] is not None]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:      # derived numbers only, no Space-Track data
        json.dump({"summary": s, "days": days}, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
