"""Monte Carlo probability of collision (src/mc.cpp): orbits, elements, search windows, and NASA."""
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from kessler._native import cart_to_eq, eq_to_cart, kepler, pc_mc_hits
from kessler.pc import _factor, orbit_period, pc2d, pc_mc, to_equinoctial

MU = 3.986004418e14
STATES = [
    [7e6, 0, 0, 0, 7546, 0],                                     # circular, equatorial
    [6415116.6, 870305.4, 2418029.3, -1870.77, 6947.49, 2446.38],  # Hubble at a CARA TCA
    [7e6, 1e5, -3e5, 100, 7000, 2500],
    [7.2e6, -2e6, 4e5, 1500, 6500, -3000],                        # retrograde
    [-2.6e7, 3e5, 1e6, -200, -3500, 1500],                        # eccentric, high
]


@pytest.mark.parametrize("rv0", STATES)
def test_kepler_matches_numerical_integration(rv0):
    f = lambda t, y: np.concatenate([y[3:], -MU * y[:3] / np.linalg.norm(y[:3]) ** 3])
    for t in (-1425.0, 0.37, 600.0, 3000.0):
        ref = solve_ivp(f, (0, t), rv0, rtol=1e-13, atol=1e-9).y[:, -1]
        got = kepler(rv0, t)
        assert np.linalg.norm(got[:3] - ref[:3]) < 1e-4       # m
        assert np.linalg.norm(got[3:] - ref[3:]) < 1e-7       # m/s


@pytest.mark.parametrize("rv0", STATES)
def test_equinoctial_round_trip(rv0):
    rv0 = np.array(rv0, dtype=float)
    back = eq_to_cart(cart_to_eq(rv0))
    assert np.linalg.norm(back[:3] - rv0[:3]) < 1e-6
    assert np.linalg.norm(back[3:] - rv0[3:]) < 1e-9


def encounter(speed_mps, miss_m, sigma_m):
    """Two near-circular LEO objects meeting at TCA with the given relative speed and miss."""
    r = 7.0e6
    v = math.sqrt(MU / r)
    ang = 2.0 * math.asin(min(1.0, speed_mps / (2.0 * v)))
    r1, v1 = np.array([r, 0, 0]), np.array([0, v, 0])
    r2 = np.array([r, 0, miss_m])
    v2 = np.array([0, v * math.cos(ang), v * math.sin(ang)])
    c = np.diag([sigma_m ** 2] * 3 + [1e-4] * 3)
    return r1, v1, c, r2, v2, c


def test_fast_encounter_matches_2d():
    """For a fast encounter the 2D method is exact enough; the Monte Carlo must agree."""
    r1, v1, c1, r2, v2, c2 = encounter(10_000.0, 30.0, 40.0)
    p2 = pc2d(r1, v1, c1[:3, :3], r2, v2, c2[:3, :3], 10.0)
    for space in ("cartesian", "equinoctial", "equinoctial-n"):
        res = pc_mc(r1, v1, c1, r2, v2, c2, 10.0, 2_000_000, seed=3, space=space)
        assert res["lo"] < p2 < res["hi"], (space, res, p2)


@pytest.mark.parametrize("space", [0, 1, 2])
@pytest.mark.parametrize("speed", [10_000.0, 300.0, 5.0])
def test_search_windows_change_nothing(space, speed):
    """Restricting each trial to the windows where it can hit gives exactly the hits of a
    search over the whole window, trial for trial (same random draws)."""
    r1, v1, c1, r2, v2, c2 = encounter(speed, 300.0, 200.0)
    w = min(orbit_period(r1, v1), orbit_period(r2, v2)) / 4.0
    if space:
        m1, c1 = to_equinoctial(r1, v1, c1, mean_motion=space == 2)
        m2, c2 = to_equinoctial(r2, v2, c2, mean_motion=space == 2)
    else:
        m1, m2 = np.concatenate([r1, v1]), np.concatenate([r2, v2])
    a1, a2 = _factor(c1), _factor(c2)
    for hbr in (20.0, 1000.0):
        args = (m1, a1, m2, a2, hbr, -w, w, 60.0, 40_000, 5)
        full = pc_mc_hits(*args, restrict=False, space=space)
        assert pc_mc_hits(*args, restrict=True, space=space) == full
        if hbr == 1000.0:
            assert full > 100


def test_same_result_for_any_thread_count():
    import os
    import subprocess
    import sys
    code = ("import numpy as np; from kessler.pc import pc_mc;"
            "c=np.diag([150.0**2]*3+[1e-4]*3); r=7e6; v=(3.986004418e14/r)**0.5;"
            "print(pc_mc(np.array([r,0,0]),np.array([0,v,0]),c,np.array([r,0,100.0]),"
            "np.array([0,v,500.0]),c,20.0,300000,seed=9)['hits'])")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                          env={**os.environ, "OMP_NUM_THREADS": str(n)}, check=True).stdout
           for n in (1, 3)}
    assert len(out) == 1


def test_nasa_slow_encounter():
    """CARA's case 25 (WorldView-2 and a Fengyun-1C fragment, 53.6 m/s): the 2D method gives
    4.5e-23, CARA's Monte Carlo 1.51e-4 (9,937 hits in 66 million trials)."""
    from test_pc import cara_table, fetch
    from kessler.pc import pc_mc_from_cdm
    row = [r for r in cara_table() if r["Conjunction_ID"].startswith("000035946_conj_000030648")]
    if not row:
        pytest.skip("case not in CARA's table")
    row = row[0]
    cdm = fetch(row["Conjunction_ID"] + ".cdm").decode()
    res = pc_mc_from_cdm(cdm, 3_000_000, hbr=row["HBR_m"], seed=2)
    p, n = row["NhitSDMC"] / row["NtotSDMC"], res["trials"]
    z = (res["pc"] - p) / math.sqrt(p * (1 - p) / n)
    assert abs(z) < 4, (res, row["PcSDMC"])


def test_nasa_far_tail():
    """CARA's case 45 (ICESat-2 and SpaceBEE-114, Pc 7.4e-7 from 2,952 hits in 4 billion trials),
    decided by the far tail: with equinoctial elements the Monte Carlo agrees (with Cartesian
    states it finds no hit; bench/m3_ablations.py)."""
    from test_pc import cara_table, fetch
    from kessler.pc import pc_mc_from_cdm
    row = [r for r in cara_table() if r["Conjunction_ID"].startswith("000043613_conj_000052010")]
    if not row:
        pytest.skip("case not in CARA's table")
    row = row[0]
    cdm = fetch(row["Conjunction_ID"] + ".cdm").decode()
    res = pc_mc_from_cdm(cdm, 20_000_000, hbr=row["HBR_m"], seed=3)
    p, n = row["NhitSDMC"] / row["NtotSDMC"], res["trials"]
    assert abs(res["hits"] - p * n) < 4 * math.sqrt(p * n), (res, p * n)
