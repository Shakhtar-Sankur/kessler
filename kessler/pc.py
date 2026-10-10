"""Probability of collision for a short encounter (the 2D method of Foster, 1992).

At closest approach the two objects' combined position uncertainty is projected onto the
encounter plane (perpendicular to the relative velocity); the probability of collision is the
mass of that 2D Gaussian inside a circle of the combined hard-body radius, centred on the miss
vector. Conventions follow NASA CARA's PcCircle, against whose values kessler is tested
(tests/test_pc.py).
"""
import math
import re

import numpy as np
from scipy import integrate, special


def rtn_basis(r, v):
    """Columns R, T, N: radial, along-track, orbit normal."""
    rh = r / np.linalg.norm(r)
    h = np.cross(r, v)
    nh = h / np.linalg.norm(h)
    th = np.cross(nh, rh)
    return np.column_stack([rh, th, nh])


def rtn_to_inertial(cov_rtn, r, v):
    m = rtn_basis(r, v)
    return m @ cov_rtn @ m.T


def linear_closest_approach(r1, v1, r2, v2):
    """Moves both states along straight lines to their closest approach (as CARA does before
    the 2D integral, so the miss vector is perpendicular to the relative velocity)."""
    dv = v2 - v1
    vv = dv @ dv
    if vv == 0.0:
        return r1, r2, 0.0
    dt = -((r2 - r1) @ dv) / vv
    return r1 + dt * v1, r2 + dt * v2, dt


def encounter_plane(r1, v1, cov1, r2, v2, cov2):
    """Miss distance, the two standard deviations of the combined covariance in the encounter
    plane (largest first), and the miss vector's components along their axes."""
    r, v = r1 - r2, v1 - v2
    y = v / np.linalg.norm(v)
    h = np.cross(r, v)
    z = h / np.linalg.norm(h)
    x = np.cross(y, z)
    rot = np.vstack([x, y, z])
    c = rot @ (cov1 + cov2) @ rot.T
    a = np.array([[c[0, 0], c[0, 2]], [c[0, 2], c[2, 2]]])
    w, vec = np.linalg.eigh(a)                      # ascending
    l1, l2 = w[1], w[0]
    v1x = vec[:, 1]
    rm = float(np.linalg.norm(r))
    return rm, l1, l2, rm * abs(v1x[0]), rm * abs(v1x[1])


def _log_band(lo, hi):
    """log(Phi(hi) - Phi(lo)) for lo < hi, accurate far in either tail."""
    if lo > 0.0:   # both in the upper tail: Phi(-lo) - Phi(-hi)
        a, b = special.log_ndtr(-lo), special.log_ndtr(-hi)
    elif hi < 0.0:  # both in the lower tail
        a, b = special.log_ndtr(hi), special.log_ndtr(lo)
    else:
        return math.log(max(1.0 - special.ndtr(lo) - special.ndtr(-hi), 1e-300))
    return a + math.log1p(-math.exp(b - a)) if b < a else -math.inf


def pc_circle(xm, zm, sx, sz, hbr, rel_tol=1e-12):
    """Mass of N(0, diag(sx^2, sz^2)) inside the circle of radius hbr centred on (xm, zm).

    Written as an integral over the circle's angle (x = xm + hbr sin t), whose integrand is
    smooth. It is evaluated in logarithms and rescaled by its largest value, so probabilities
    far below 1e-300 of their peak still come out with full relative precision."""
    if hbr <= 0.0:
        return 0.0

    def logf(t):
        x = xm + hbr * math.sin(t)
        hz = hbr * math.cos(t)
        if hz <= 0.0:
            return -math.inf
        return (-0.5 * (x / sx) ** 2 + _log_band((zm - hz) / sz, (zm + hz) / sz)
                + math.log(hbr * math.cos(t)))

    grid = np.linspace(-math.pi / 2, math.pi / 2, 2001)[1:-1]
    logs = np.array([logf(t) for t in grid])
    k = int(np.argmax(logs))
    scale = logs[k]
    if not math.isfinite(scale):
        return 0.0
    f = lambda t: math.exp(logf(t) - scale)
    val, _ = integrate.quad(f, -math.pi / 2, math.pi / 2, epsabs=0.0, epsrel=rel_tol, limit=1000,
                            points=[float(grid[k])])
    return math.exp(math.log(val) + scale - math.log(math.sqrt(2.0 * math.pi) * sx)) if val > 0 else 0.0


def pc2d(r1, v1, cov1, r2, v2, cov2, hbr, adjust=True):
    """2D probability of collision. Positions in m, velocities in m/s, covariances in m^2
    (3x3, inertial). With adjust=True the states are first moved to their linear closest
    approach, as CARA's Pc2D does."""
    r1, v1, r2, v2 = (np.asarray(a, dtype=float) for a in (r1, v1, r2, v2))
    if adjust:
        r1, r2, _ = linear_closest_approach(r1, v1, r2, v2)
    rm, l1, l2, xm, zm = encounter_plane(r1, v1, np.asarray(cov1), r2, v2, np.asarray(cov2))
    clip = (1e-4 * hbr) ** 2                         # CARA's eigenvalue clipping level
    l1, l2 = max(l1, clip), max(l2, clip)
    return pc_circle(xm, zm, math.sqrt(l1), math.sqrt(l2), hbr)


def read_cdm(text):
    """Conjunction Data Message (CCSDS 508.0, keyword = value form) to a dict with a header
    and two objects. Units in [..] are dropped; the HBR comment CARA writes is kept."""
    head, objs, cur = {}, [], None
    for line in text.splitlines():
        m = re.match(r"^\s*COMMENT\s+HBR\s*=\s*([0-9.eE+-]+)", line)
        if m:
            head["HBR"] = float(m.group(1))
            continue
        if "=" not in line or line.strip().startswith("COMMENT"):
            continue
        k, val = (s.strip() for s in line.split("=", 1))
        val = re.sub(r"\s*\[.*\]\s*$", "", val)
        if k == "OBJECT":
            cur = {}
            objs.append(cur)
            continue
        (cur if cur is not None else head)[k] = val
    return head, objs


def cdm_states(obj):
    """Inertial position (m), velocity (m/s) and 3x3 position covariance (m^2) of one CDM object."""
    if obj["REF_FRAME"].upper() not in ("EME2000", "GCRF"):
        raise ValueError(f"unsupported frame {obj['REF_FRAME']}")
    r = np.array([float(obj[k]) for k in ("X", "Y", "Z")]) * 1e3
    v = np.array([float(obj[k]) for k in ("X_DOT", "Y_DOT", "Z_DOT")]) * 1e3
    g = lambda k: float(obj[k])
    c = np.array([[g("CR_R"), g("CT_R"), g("CN_R")],
                  [g("CT_R"), g("CT_T"), g("CN_T")],
                  [g("CN_R"), g("CN_T"), g("CN_N")]])
    return r, v, rtn_to_inertial(c, r, v)


def pc2d_from_cdm(text, hbr=None, adjust=True):
    head, objs = read_cdm(text)
    r1, v1, c1 = cdm_states(objs[0])
    r2, v2, c2 = cdm_states(objs[1])
    return pc2d(r1, v1, c1, r2, v2, c2, hbr if hbr is not None else head["HBR"], adjust)
