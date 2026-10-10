"""Minimal Space-Track.org client (https://www.space-track.org/documentation).

Credentials come from SPACETRACK_USER / SPACETRACK_PASS, or Kaggle secrets of those names.
Space-Track's user agreement limits redistribution of its data, so kessler only ever keeps it
in memory or in a local cache outside the repository, and publishes derived results only.
Requests are paced to stay well inside Space-Track's limits (30 per minute, 300 per hour).
"""
import csv
import http.cookiejar
import io
import os
import time
import urllib.parse
import urllib.request

BASE = "https://www.space-track.org"


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
        user, pw = credentials()
        self._open(BASE + "/ajaxauth/login", urllib.parse.urlencode({"identity": user, "password": pw}).encode())

    def _open(self, url, data=None):
        wait = self.pause - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        with self.op.open(url, data, timeout=600) as r:
            return r.read().decode()

    def query(self, path):
        """Rows of a basicspacedata query, e.g. 'class/gp/NORAD_CAT_ID/25544'."""
        text = self._open(f"{BASE}/basicspacedata/query/{path.strip('/')}/format/csv")
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
