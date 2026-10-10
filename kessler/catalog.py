"""The public catalogue: CelesTrak General Perturbations (GP) data in OMM CSV form."""
import csv
import io
import math
import urllib.request
from datetime import datetime

import numpy as np

from ._native import MEAN_DTYPE

CELESTRAK_GP = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=csv"
_EPOCH0 = datetime(1949, 12, 31)
_DEG = math.pi / 180.0


def fetch(group="active", timeout=60):
    """Downloads one CelesTrak group as OMM CSV text ('active', 'starlink', 'last-30-days', ...)."""
    with urllib.request.urlopen(CELESTRAK_GP.format(group=group), timeout=timeout) as resp:
        return resp.read().decode()


def epoch_days(iso):
    """Days since 1949-12-31 00:00 UT, computed as the reference `sgp4.omm` does."""
    return (datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S.%f") - _EPOCH0).total_seconds() / 86400.0


class Catalog:
    """Mean elements plus identity for n objects."""

    def __init__(self, rows):
        self.rows = rows
        self.norad = np.array([int(r["NORAD_CAT_ID"]) for r in rows], dtype=np.int64)
        self.name = [r["OBJECT_NAME"] for r in rows]
        self.mean = np.empty(len(rows), dtype=MEAN_DTYPE)
        for i, r in enumerate(rows):
            self.mean[i] = (
                epoch_days(r["EPOCH"]),
                float(r["BSTAR"]),
                float(r["ECCENTRICITY"]),
                float(r["ARG_OF_PERICENTER"]) * _DEG,
                float(r["INCLINATION"]) * _DEG,
                float(r["MEAN_ANOMALY"]) * _DEG,
                float(r["MEAN_MOTION"]) / 720.0 * math.pi,
                float(r["RA_OF_ASC_NODE"]) * _DEG,
            )

    @classmethod
    def from_csv(cls, text):
        return cls(list(csv.DictReader(io.StringIO(text))))

    def __len__(self):
        return len(self.rows)
