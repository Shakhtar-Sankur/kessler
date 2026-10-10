"""Archive operator-derived orbits that nobody else keeps.

CelesTrak's supplemental GP (SupGP) element sets are fitted to the operators' own ephemerides
(SpaceX for Starlink), so they are far more precise than the public catalogue, but CelesTrak
only serves the current ones. Run every few hours, this keeps each version, plus SOCRATES's
daily conjunction list (its 2,000 closest), so that kessler's audit can later be rerun with operator-quality orbits.

    python -m kessler.archive --dir archive

Files: <dir>/YYYY/MM/DD/<name>_<HHMM>Z.csv.gz, skipped when identical to the latest stored version.
"""
import argparse
import glob
import gzip
import hashlib
import os
import time
import urllib.request
from datetime import datetime, timezone

UA = {"User-Agent": "kessler archive (github.com/Shakhtar-Sankur/kessler)"}
SUPGP = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE={}&FORMAT=csv"
SOURCES = {f"supgp-{f}": SUPGP.format(f) for f in ("starlink", "oneweb", "kuiper", "planet", "iss")}
SOURCES["socrates-minrange"] = "https://celestrak.org/SOCRATES/sort-minRange.csv"
KEEP_ROWS = {"socrates-minrange": 2000}     # its full list is ~190,000 rows; keep the closest


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def latest_digest(root, name):
    files = sorted(glob.glob(os.path.join(root, "*", "*", "*", f"{name}_*.csv.gz")))
    if not files:
        return None
    return hashlib.sha256(gzip.open(files[-1]).read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="archive")
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    day = os.path.join(a.dir, now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"))
    saved = 0
    for name, url in SOURCES.items():
        try:
            body = get(url)
        except Exception as e:
            print(f"{name}: not fetched ({type(e).__name__}: {e})")
            continue
        if len(body) < 200 or not body.lstrip()[:1].isalpha():     # error page or empty answer
            print(f"{name}: unexpected answer ({len(body)} bytes), not saved")
            continue
        if name in KEEP_ROWS:
            body = b"\n".join(body.split(b"\n")[: KEEP_ROWS[name] + 1]) + b"\n"
        if hashlib.sha256(body).hexdigest() == latest_digest(a.dir, name):
            print(f"{name}: unchanged")
            continue
        os.makedirs(day, exist_ok=True)
        path = os.path.join(day, f"{name}_{now.strftime('%H%M')}Z.csv.gz")
        with gzip.open(path, "wb") as f:
            f.write(body)
        saved += 1
        rows = body.count(b"\n") - 1
        print(f"{name}: saved {path} ({rows} rows)")
        time.sleep(2)
    print(f"{saved} new files")


if __name__ == "__main__":
    main()
