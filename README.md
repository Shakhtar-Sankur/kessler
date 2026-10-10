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
| M1: SGP4 on CUDA, full catalogue on a T4 | done |
| M2: GPU all-vs-all screening | next |
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

## M1: the whole catalogue, every minute of a day, on a free T4

16,695 objects (every active object on CelesTrak, 10 Oct 2026) propagated to 1,440 times, one minute apart: 24,040,800 states. Kaggle, one Tesla T4, 4 CPU cores ([raw results](bench/results/m1_t4.json), [script](bench/m1_propagate.py)).

| Method | States per second | vs reference library | Max position error vs reference |
|---|---|---|---|
| **kessler, CUDA kernel (T4, fp64)** | **125.7 M** | **73×** | 1.0e-10 km |
| kessler, CPU (4 cores, OpenMP) | 6.3 M | 3.7× | 9.2e-11 km |
| python `sgp4` SatrecArray (Vallado's C++, 1 thread) | 1.7 M | 1× | (the reference) |
| dsgp4 on the T4 (ESA's PyTorch SGP4), near-earth objects only | 2.7 M | 1.5× | 6.9e-9 km |

- The kernel's 191 ms covers the propagation alone. With the copies to and from the GPU it takes 1.68 s, mostly moving 577 MB of positions back to the host; screening (M2) keeps them on the GPU.
- The GPU and CPU builds agree to 2.2e-10 km on all 24 million states, with identical error codes.
- dsgp4 is built for differentiable propagation (gradients for orbit fitting), not for throughput, and its batched propagator handles near-earth objects only, so it was timed on 15,890 objects at 60 of the times, one call per time. It is a fair accuracy check and a rough speed reference, not a like-for-like contest.

## Limits

Public orbit data (TLE/OMM) is accurate to roughly a kilometre, much less precise than what operators use internally. kessler's results will be independent estimates from public data, and it will say so wherever it reports them.
