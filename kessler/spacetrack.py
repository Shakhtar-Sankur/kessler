"""Minimal Space-Track.org client (https://www.space-track.org/documentation).

Credentials come from SPACETRACK_USER / SPACETRACK_PASS, or Kaggle secrets of those names.
Space-Track's user agreement limits redistribution of its data, so kessler only ever keeps it
in memory or in a local cache outside the repository, and publishes derived results only.
Requests are paced to stay well inside Space-Track's limits (30 per minute, 300 per hour).
"""
import csv
import gzip
import hashlib
import http.cookiejar
import io
import os
import time
import urllib.parse
import urllib.request

BASE = "https://www.space-track.org"
CACHE = os.environ.get("KESSLER_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "kessler", "spacetrack"))


def credentials():
    user, pw = os.environ.get("SPACETRACK_USER"), os.environ.get("SPACETRACK_PASS")
    if not (user and pw):
        try:
            from kaggle_secrets import UserSecretsClient
            sc = UserSecretsClient()
            user, pw = sc.get_secret("SPACETRACK_USER"), sc.get_secret("SPACETRACK_PASS")
        except Exception:
            pass
    if not (user and pw):
        raise RuntimeError("set SPACETRACK_USER and SPACETRACK_PASS (or Kaggle secrets of those names)")
    return user, pw


class Client:
    def __init__(self, pause=3.0):
        self.pause = pause
        self.last = 0.0
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.logged_in = False

    def _login(self):
        user, pw = credentials()
        self._open(BASE + "/ajaxauth/login", urllib.parse.urlencode({"identity": user, "password": pw}).encode())
        self.logged_in = True

    def _open(self, url, data=None):
        wait = self.pause - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        with self.op.open(url, data, timeout=600) as r:
            return r.read().decode()

    def query(self, path, cache=True):
        """Rows of a basicspacedata query, e.g. 'class/gp/NORAD_CAT_ID/25544'. Results are cached
        on disk (KESSLER_CACHE), so a rerun after a crash does not ask Space-Track again; queries
        relative to 'now' are never cached."""
        url = f"{BASE}/basicspacedata/query/{path.strip('/')}/format/csv"
        cache = cache and "now" not in path
        f = os.path.join(CACHE, hashlib.sha256(url.encode()).hexdigest() + ".csv.gz")
        if cache and os.path.exists(f):
            text = gzip.open(f, "rt").read()
        else:
            if not self.logged_in:
                self._login()
            text = self._open(url)
            if cache:
                os.makedirs(CACHE, exist_ok=True)
                with gzip.open(f + ".tmp", "wt") as out:
                    out.write(text)
                os.replace(f + ".tmp", f)
        return list(csv.DictReader(io.StringIO(text)))

    def gp_history(self, norad_ids, start, end, batch=100):
        """Every element set of these objects with epochs in [start, end) (YYYY-MM-DD)."""
        rows = []
        ids = sorted(set(int(x) for x in norad_ids))
        for k in range(0, len(ids), batch):
            part = ",".join(str(x) for x in ids[k:k + batch])
            rows += self.query(f"class/gp_history/NORAD_CAT_ID/{part}/EPOCH/{start}--{end}"
                               "/orderby/NORAD_CAT_ID,EPOCH")
        return rows
