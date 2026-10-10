"""Prints the headline numbers of every bench/results/*.json that exists (for logs that get cut)."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def show(path):
    d = json.load(open(path))
    name = os.path.basename(path)
    if name.startswith("m2_socrates"):
        print(f"{name}: {json.dumps(d['summary'])}")
        for r in d["rows"][:15]:
            if r.get("status") == "compared":
                print(f"  {r['a'][:22]:>22} x {r['b'][:22]:<22} SOCRATES {r['socrates_range_km']*1e3:7.1f} m  "
                      f"kessler {r['kessler_range_km']*1e3:9.3f} m  dTCA {r['tca_diff_ms']:7.3f} ms  "
                      f"maxP {r['socrates_max_prob']:.3e}  implied HBR {r.get('implied_hbr_m', float('nan')):.2f} m")
            else:
                print(f"  {r['a'][:22]:>22} x {r['b'][:22]:<22} {r['status']}")
    elif name.startswith("m2"):
        g = d.get("gpu_run", {})
        print(f"{name}: {d['source']}, {d['objects']} objects, {d['hours']} h, {d['dmax_km']} km")
        if g:
            print(f"  GPU: candidates in {g['gpu_candidate_seconds']:.2f} s, total {g['seconds_total']:.1f} s; "
                  f"{g['events']} events ({g['events_below_1km']} < 1 km, {g['events_below_200m']} < 200 m); "
                  f"Starlink-Starlink {g['starlink_vs_starlink']}, Starlink-other {g['starlink_vs_other']}, "
                  f"neither {g['neither_starlink']}; docked/co-located pairs {g['docked_or_colocated_pairs']}")
            for c in g["closest"][:5]:
                print(f"    {c['a']} x {c['b']}: {c['miss_km']*1e3:.0f} m at {c['vrel_km_s']} km/s, "
                      f"{c['tca_hours_after_start']} h after start")
        if "cpu_run" in d:
            print(f"  CPU: {d['cpu_run']['seconds_total']:.1f} s; GPU equals CPU: {d.get('gpu_equals_cpu')}")
    else:
        print(f"{name}: {json.dumps(d.get('summary', d))[:2000]}")


for p in sorted(glob.glob(os.path.join(ROOT, "bench", "results", "*.json"))):
    try:
        show(p)
    except Exception as e:
        print(f"{os.path.basename(p)}: could not summarise ({e})")

# a step that crashed leaves its old .json (or none) behind: show the end of its log instead
for log in sorted(glob.glob(os.path.join(ROOT, "bench", "results", "*.log"))):
    text = open(log, errors="replace").read()
    if "Traceback" in text or "Error" in text:
        tail = "\n    ".join(text.strip().splitlines()[-6:])
        print(f"{os.path.basename(log)} FAILED:\n    {tail}")
