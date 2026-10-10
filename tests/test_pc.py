"""Probability of collision against NASA CARA's own values for 53 real conjunctions.

The CDMs and CARA's results are fetched from github.com/nasa/CARA_Analysis_Tools at a pinned
commit (they are not redistributed here); the test is skipped without network access.
"""
import io
import urllib.request

import numpy as np
import pytest

from kessler.pc import pc2d_from_cdm

COMMIT = "ce0f00722c56808755ee0adbe47a4e3d56439484"
BASE = f"https://raw.githubusercontent.com/nasa/CARA_Analysis_Tools/{COMMIT}/DataFiles/PcTestCaseCDMs/"


def fetch(name):
    with urllib.request.urlopen(BASE + name, timeout=60) as r:
        return r.read()


def cara_table():
    openpyxl = pytest.importorskip("openpyxl")
    try:
        data = fetch("CARA_PcMethod_Test_Conjunctions.xlsx")
    except Exception as e:
        pytest.skip(f"cannot reach GitHub: {e}")
    rows = list(openpyxl.load_workbook(io.BytesIO(data), read_only=True).worksheets[0].iter_rows(values_only=True))
    idx = {h: i for i, h in enumerate(rows[0])}
    return [{k: r[i] for k, i in idx.items()} for r in rows[1:]]


def test_pc2d_matches_cara():
    rels = []
    for row in cara_table():
        cdm = fetch(row["Conjunction_ID"] + ".cdm").decode()
        for adjust, col in ((True, "Pc2D"), (False, "Pc2D_NoAdj")):
            p = pc2d_from_cdm(cdm, row["HBR_m"], adjust)
            rels.append(abs(p - row[col]) / row[col])
    rels = np.array(rels)
    print(f"{len(rels)} CARA values: max relative difference {rels.max():.2e}, "
          f"{(rels < 1e-8).sum()} within 1e-8, {(rels < 1e-12).sum()} within 1e-12")
    assert len(rels) == 106
    assert rels.max() < 1e-7
