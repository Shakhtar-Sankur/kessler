# kessler

An independent, open-source space-traffic auditor.

SpaceX reports that Starlink satellites made **207,152 collision-avoidance maneuvers** between December 2025 and May 2026. Its satellites maneuver automatically whenever the probability of a collision passes 3 in 10 million. Nobody outside SpaceX can check those numbers today.

kessler is building the tools to do it from public data:

1. **SGP4 on CPU and GPU**: one C++ propagator, compiled for both, for the whole public catalogue.
2. **All-vs-all conjunction screening on a GPU**: every tracked object against every other.
3. **Collision probability** for the close approaches.
4. **Starlink maneuver detection** from public orbit history, compared with SpaceX's reports to the FCC.

Named after the Kessler syndrome: the runaway cascade of collisions that this kind of work exists to prevent.

## Status

| Milestone | State |
|---|---|
| M0: SGP4 (CPU), validated against the reference | done |
| M1: SGP4 on CUDA, full catalogue on a T4 | next |
| M2: GPU all-vs-all screening | planned |
| M3: collision probability | planned |
| M4: Starlink maneuver detection vs FCC reports | planned |
| M5: daily pipeline and public page | planned |

## M0: the propagator matches the reference

`include/kessler/sgp4.hpp` is a port of SGP4/SDP4 (near-earth and deep-space) from Vallado et al., *Revisiting Spacetrack Report #3* (AIAA 2006-6753), with WGS-72 constants and the "improved" operation mode. Propagation is a pure function of the initialised element set, so the same code runs in a CUDA kernel.

It is checked against the [python `sgp4`](https://pypi.org/project/sgp4/) package, which wraps Vallado's reference C++:

| Test | States | Max position difference | Max velocity difference | Error-code mismatches |
|---|---|---|---|---|
| 33 official verification cases (`SGP4-VER.TLE`), incl. deep-space, resonant and decaying orbits | 2,339 | 2.7e-12 km | 1.8e-15 km/s | 0 |
| Every active object on CelesTrak, 10 Oct 2026, at 6 times from -1 to +7 days | 100,170 (16,695 objects) | 7.1e-10 km | 7.3e-13 km/s | 0 |

```
python3 -m venv .venv && .venv/bin/pip install sgp4 numpy pytest
make
.venv/bin/python -m pytest -q -s tests/
```

## Limits

Public orbit data (TLE/OMM) is accurate to roughly a kilometre, much less precise than what operators use internally. kessler's results will be independent estimates from public data, and it will say so wherever it reports them.
