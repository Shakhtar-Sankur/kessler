"""The daily screening: the whole public catalogue for the next 24 hours.

    python -m kessler.daily --out site_data [--history daily]

Catalogue: Space-Track's full catalogue (on-orbit objects with elements from the last 10 days)
when SPACETRACK_USER / SPACETRACK_PASS are set, otherwise CelesTrak's active satellites, or a
local OMM CSV with --csv. Writes derived results only, never element sets:

  <out>/latest.json        everything the public page shows: counts, the closest approaches with
                           their worst-case probability, Earth-fixed positions of every object at
                           the start and the tracks of the closest pairs around closest approach;
  <history>/YYYY/<date>.json   the day's counts and closest approaches (kept in the repository);
  <history>/history.csv        one line per day.
"""
import argparse
import csv
import io
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from . import cuda
from ._native import Elsets
from .catalog import Catalog, fetch
from .screen import max_speed, offsets, screen

EPOCH0 = datetime(1949, 12, 31, tzinfo=timezone.utc)
KINDS = ("starlink", "payload", "rocket body", "debris", "unknown")
HISTORY_FIELDS = ("date", "start", "source", "objects", "approaches", "below_1km", "below_200m",
                  "starlink_starlink", "starlink_other", "neither_starlink", "closest_m", "closest_pair",
                  "seconds")


def days(dt):
    return (dt - EPOCH0).total_seconds() / 86400.0


def iso(d):
    return (EPOCH0 + timedelta(days=float(d))).strftime("%Y-%m-%dT%H:%M:%SZ")


def gmst(d):
    """Greenwich mean sidereal time (rad) at d days since 1949-12-31 UT (IAU-82, as Vallado)."""
    t = (d + 2433281.5 - 2451545.0) / 36525.0
    s = -6.2e-6 * t ** 3 + 0.093104 * t ** 2 + (876600.0 * 3600 + 8640184.812866) * t + 67310.54841
    return math.fmod(math.radians(s / 240.0), 2.0 * math.pi)


def earth_fixed(r, d):
    """TEME positions [k, 3] at times d (days) to Earth-fixed (polar motion ignored: < 20 m)."""
    g = np.array([gmst(x) for x in np.atleast_1d(d)]) if np.ndim(d) else gmst(d)
    c, s = np.cos(g), np.sin(g)
    return np.stack([c * r[:, 0] + s * r[:, 1], -s * r[:, 0] + c * r[:, 1], r[:, 2]], axis=1)


def kind(row):
    if "STARLINK" in row["OBJECT_NAME"].upper():
        return 0
    t = (row.get("OBJECT_TYPE") or "PAYLOAD").upper()
    return {"PAYLOAD": 1, "ROCKET BODY": 2, "DEBRIS": 3}.get(t, 4)


def load(csv_path=None, max_age_days=10.0):
    if csv_path:
        import gzip
        op = gzip.open if csv_path.endswith(".gz") else open
        with op(csv_path, "rt") as f:
            return list(csv.DictReader(f)), f"file {os.path.basename(csv_path)}"
    try:
        from .spacetrack import Client, credentials
        credentials()
        rows = Client().query(f"class/gp/decay_date/null-val/epoch/%3Enow-{max_age_days:g}/orderby/norad_cat_id")
        return rows, "Space-Track full catalogue"
    except RuntimeError:
        pass
    return list(csv.DictReader(io.StringIO(fetch("active")))), "CelesTrak active satellites"


def worst_case_pc(els, off, e, hbr_m=10.0):
    from .pc import max_pc
    r, v, err = els.propagate_at(np.array([e["i"], e["j"]]), off[[e["i"], e["j"]]] + e["tca"] / 60.0)
    if np.any(err):
        return None
    return float(max_pc(r[0] * 1e3, v[0] * 1e3, r[1] * 1e3, v[1] * 1e3, hbr_m)[0])


def run(rows, source, start=None, hours=24.0, dmax=5.0, top=50, tracks=20):
    t_run = time.time()
    cat = Catalog(rows)
    start = start if start is not None else float(np.quantile(cat.mean["epoch"], 0.999))
    els = Elsets(cat.mean)
    off, vmax = offsets(cat.mean, start), max_speed(cat.mean)
    if cuda.available():
        ev, _, _ = cuda.screen(els, off, vmax, hours * 3600.0, dmax)
    else:
        ev, _ = screen(els, off, vmax, hours * 3600.0, dmax)
    together = ev[ev["vrel"] < 0.01]                   # docked or co-located: not conjunctions
    ev = np.sort(ev[ev["vrel"] >= 0.01], order="miss")
    kinds = np.array([kind(r) for r in rows], dtype=np.int8)
    star = kinds == 0
    both, one = star[ev["i"]] & star[ev["j"]], star[ev["i"]] ^ star[ev["j"]]

    def obj(q):
        return {"name": cat.name[q], "norad": int(cat.norad[q]), "kind": KINDS[kinds[q]]}

    closest = []
    for n, e in enumerate(ev[:top]):
        c = {"a": obj(e["i"]), "b": obj(e["j"]), "tca": iso(start + e["tca"] / 86400.0),
             "miss_m": round(float(e["miss"]) * 1e3, 1), "vrel_km_s": round(float(e["vrel"]), 3),
             "worst_case_pc_10m": worst_case_pc(els, off, e)}
        if n < tracks:                                  # both objects for 15 min either side
            ts = e["tca"] + np.arange(-900.0, 901.0, 30.0)
            idx = np.repeat([e["i"], e["j"]], len(ts))
            t = np.concatenate([off[e["i"]] + ts / 60.0, off[e["j"]] + ts / 60.0])
            r, _, err = els.propagate_at(idx, t)
            dd = start + np.tile(ts, 2) / 86400.0
            xyz = np.rint(earth_fixed(r, dd)).astype(int)
            c["track"] = [xyz[: len(ts)].ravel().tolist(), xyz[len(ts):].ravel().tolist()]
        closest.append(c)

    r, _, err = els.propagate_at(np.arange(len(rows)), off)
    ok = err == 0
    xyz = np.rint(earth_fixed(r[ok], start)).astype(int)
    summary = {
        "date": iso(start)[:10], "start": iso(start), "source": source, "hours": hours, "dmax_km": dmax,
        "objects": int(len(rows)), "objects_by_kind": {k: int(np.sum(kinds == q)) for q, k in enumerate(KINDS)},
        "approaches": int(len(ev)), "below_1km": int(np.sum(ev["miss"] < 1.0)),
        "below_200m": int(np.sum(ev["miss"] < 0.2)), "starlink_starlink": int(np.sum(both)),
        "starlink_other": int(np.sum(one)), "neither_starlink": int(np.sum(~both & ~one)),
        "docked_or_colocated_pairs": int(len(set(zip(together["i"].tolist(), together["j"].tolist())))),
        "gpu": cuda.available(), "seconds": round(time.time() - t_run, 1),
    }
    site = {"summary": summary, "closest": closest,
            "positions": {"xyz_km": xyz.ravel().tolist(), "kind": "".join(str(k) for k in kinds[ok])}}
    return summary, closest, site


def write(summary, closest, site, out, history):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "latest.json"), "w") as f:
        json.dump(site, f, separators=(",", ":"))
    if not history:
        return
    year = os.path.join(history, summary["date"][:4])
    os.makedirs(year, exist_ok=True)
    with open(os.path.join(year, summary["date"] + ".json"), "w") as f:
        json.dump({"summary": summary, "closest": [{k: v for k, v in c.items() if k != "track"} for c in closest]},
                  f, indent=1)
    path = os.path.join(history, "history.csv")
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = [r for r in csv.DictReader(f) if r["date"] != summary["date"]]
    c0 = closest[0] if closest else None
    row = dict(summary, closest_m=c0["miss_m"] if c0 else "",
               closest_pair=f"{c0['a']['name']} x {c0['b']['name']}" if c0 else "")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(old + [row], key=lambda r: r["date"]))
    with open(os.path.join(out, "history.csv"), "w") as f, open(path) as g:
        f.write(g.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site_data")
    ap.add_argument("--history", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--now", action="store_true", help="start at the current time, not the catalogue's")
    ap.add_argument("--hours", type=float, default=24.0)
    a = ap.parse_args()
    rows, source = load(a.csv or None)
    start = days(datetime.now(timezone.utc).replace(second=0, microsecond=0)) if a.now else None
    summary, closest, site = run(rows, source, start, a.hours)
    write(summary, closest, site, a.out, a.history)
    print(json.dumps(summary, indent=1))
    for c in closest[:10]:
        print(f"  {c['a']['name']} x {c['b']['name']}: {c['miss_m']} m at {c['vrel_km_s']} km/s, {c['tca']}")


if __name__ == "__main__":
    main()
