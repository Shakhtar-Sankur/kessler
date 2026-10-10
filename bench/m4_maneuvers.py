"""M4: how often do Starlink satellites maneuver, measured from public orbit history?

Compares with SpaceX's semiannual report to the FCC: 207,152 collision-avoidance maneuvers
from 1 December 2025 to 31 May 2026 (press coverage of the filing).

  * a random sample of Starlink satellites on orbit through the period (default 500);
  * a control group of debris (fragments of COSMOS 2251, IRIDIUM 33 and FENGYUN 1C), which
    cannot maneuver: what the detector finds there is its false alarm rate;
  * each object's element-set history from Space-Track (gp_history), kept in memory only.

Starlink satellites also thrust for station-keeping, orbit raising and lowering, so this counts
all detectable orbit changes, split by phase (operational shell vs raising/lowering). Telling
collision-avoidance maneuvers apart needs conjunction screening around each one (next step).

Usage: python bench/m4_maneuvers.py [--starlink 500] [--debris 300]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kessler.maneuvers import detect, history  # noqa: E402
from kessler.spacetrack import Client  # noqa: E402

START, END = "2025-12-01", "2026-06-01"
FCC_COUNT = 207152
DEBRIS = ("COSMOS 2251 DEB", "IRIDIUM 33 DEB", "FENGYUN 1C DEB")


def analyse(rows_by_id, operational_band_km=10.0):
    out = {}
    for norad, rows in rows_by_id.items():
        t, a = history(rows)
        if len(t) < 10:
            continue
        k, step = detect(t, a)
        alt = (a - 6378.135)
        shell = np.median(alt)
        # an element set is "operational" if within the band of the object's median altitude
        operational = np.abs(alt - shell) < operational_band_km
        days = t[-1] - t[0]
        oper_days = float(np.sum(np.diff(t)[operational[1:] & operational[:-1]]))
        out[norad] = {"days": float(days), "operational_days": oper_days, "elsets": int(len(t)),
                      "maneuvers": int(len(k)),
                      "operational_maneuvers": int(np.sum(operational[k] & operational[k - 1])),
                      "median_altitude_km": float(shell),
                      "steps_m": [round(float(x), 1) for x in step]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starlink", type=int, default=500)
    ap.add_argument("--debris", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results", "m4_maneuvers.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    st = Client()

    sat = st.query("class/satcat/OBJECT_NAME/~~STARLINK/OBJECT_TYPE/PAYLOAD")
    on_orbit = [r for r in sat if r.get("LAUNCH") and r["LAUNCH"] < START and (not r.get("DECAY") or r["DECAY"] >= END)]
    starlink_ids = rng.choice([int(r["NORAD_CAT_ID"]) for r in on_orbit], size=min(a.starlink, len(on_orbit)), replace=False)
    deb = []
    for name in DEBRIS:
        deb += [r for r in st.query(f"class/satcat/OBJECT_NAME/~~{name.replace(' ', '%20')}/DECAY/null-val")
                if r.get("PERIGEE") and float(r["PERIGEE"]) < 1000]
    debris_ids = rng.choice([int(r["NORAD_CAT_ID"]) for r in deb], size=min(a.debris, len(deb)), replace=False)
    print(f"Starlink on orbit all period: {len(on_orbit)}; sampled {len(starlink_ids)}. Debris pool {len(deb)}; sampled {len(debris_ids)}")

    groups = {}
    for label, ids in (("starlink", starlink_ids), ("debris", debris_ids)):
        by_id = defaultdict(list)
        for r in st.gp_history(ids, START, END):
            by_id[int(r["NORAD_CAT_ID"])].append(r)
        groups[label] = analyse(by_id)
        print(f"{label}: {len(groups[label])} objects with history")

    def rate(g, key="maneuvers", days="days"):
        m = sum(o[key] for o in g.values())
        d = sum(o[days] for o in g.values())
        return m, d, (m / d if d else float("nan"))

    sm, sd, srate = rate(groups["starlink"])
    om, od, orate = rate(groups["starlink"], "operational_maneuvers", "operational_days")
    dm, dd, drate = rate(groups["debris"])
    per_sat_period = srate * 182.0
    summary = {
        "period": [START, END], "fcc_reported_collision_avoidance": FCC_COUNT,
        "starlink_on_orbit_whole_period": len(on_orbit),
        "starlink_sampled": len(groups["starlink"]), "debris_sampled": len(groups["debris"]),
        "starlink_maneuvers": sm, "starlink_object_days": round(sd, 1),
        "starlink_maneuvers_per_sat_per_day": srate,
        "starlink_operational_maneuvers_per_sat_per_day": orate,
        "debris_detections": dm, "debris_object_days": round(dd, 1),
        "false_alarms_per_object_per_day": drate,
        "estimated_all_starlink_maneuvers_in_period": round(per_sat_period * len(on_orbit)),
        "fcc_collision_avoidance_per_sat_in_period": FCC_COUNT / max(len(on_orbit), 1),
        "detected_maneuvers_per_sat_in_period": per_sat_period,
    }
    print(json.dumps(summary, indent=2))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:      # derived numbers only, no Space-Track data
        json.dump({"summary": summary,
                   "starlink_per_object": {k: {kk: v[kk] for kk in ("days", "elsets", "maneuvers", "operational_maneuvers", "median_altitude_km")}
                                           for k, v in groups["starlink"].items()},
                   "debris_per_object": {k: {kk: v[kk] for kk in ("days", "elsets", "maneuvers")}
                                         for k, v in groups["debris"].items()}}, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
