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
| M2: all-vs-all screening (CPU done and proven against brute force; GPU awaiting its T4 run) | in progress |
| M3: collision probability (matches NASA CARA on 53 real conjunctions) | in progress |
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
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
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

## M2: screening every pair, with nothing missed

Every 10 seconds of simulated time, each object is placed in a 3D grid whose cells are wider than the farthest two objects could travel towards each other in that time (from each object's perigee speed). Only objects in neighbouring cells are compared. A pair becomes a candidate if, moving in a straight line with its relative velocity, it would come within the threshold during that 10-second window, plus a margin that bounds how far real, curved motion can depart from a straight line over 5 seconds (half of twice surface gravity, times 5 s squared). Each candidate is then refined with SGP4 itself to the exact time and distance of closest approach. The GPU build sorts the grid with thrust and searches neighbours in a kernel; the CPU build does the same with OpenMP.

**Checked against brute force** (every pair, every second, each local minimum refined exactly), `tests/test_screen.py`:

| Objects | Window | Threshold | Approaches (brute force) | Missed | Extra | Max TCA difference (fast pairs) |
|---|---|---|---|---|---|---|
| 600 (450–600 km shell) | 30 min | 50 km | 262 | 0 | 0 | 0.06 ms |
| 1,000 (shell) | 1 h | 10 km | 127 | 0 | 0 | 0.01 ms |
| 1,000 (shell) | 1 h | 200 km | 8,739 | 0 | 0 | 0.36 ms |
| 1,500 (whole catalogue, incl. deep space) | 1 h | 500 km | 37,952 | 0 | 0 | 0.84 ms |

Two physical facts are built into the comparison, and both are stated rather than hidden. Objects flying in formation (a train of satellites, a cargo ship docked to the ISS) keep an almost constant distance, so their "closest approach" is flat to the millimetre and its time is ill-defined; events of the same pair less than 5 minutes apart are therefore one approach (two genuine approaches of one pair are about half an orbit apart). Docked and co-located objects (relative speed below 10 m/s) are reported separately, not as conjunctions.

First look (CPU, CelesTrak active catalogue of 10 Oct 2026, 1 hour, 5 km): 2,238 approaches, 1,675 of them between two Starlink satellites, the pairs that CelesTrak's SOCRATES service leaves out. The closest: STARLINK-2415 and TOMORROW-S4, 88 m apart at 4.6 km/s. These come from public orbits accurate to about a kilometre, so they are screening results, not predictions of collisions.

## M3: probability of collision, checked against NASA

`kessler/pc.py` computes the 2D probability of collision (Foster, 1992; the method NASA CARA uses operationally) from two states and covariances: the combined covariance is projected onto the encounter plane and the Gaussian is integrated over the hard-body circle, in logarithms, so even probabilities of 1e-168 keep full relative precision.

It reads standard Conjunction Data Messages (CCSDS 508.0) and is compared with NASA CARA's own published values for 53 real conjunctions (for example the Hubble Space Telescope and a 1966 Diamant rocket body; Terra and a fragment of Iridium 33 at 108 m, Pc = 0.021), each with and without CARA's closest-approach adjustment ([`tests/test_pc.py`](tests/test_pc.py); the data is fetched from [nasa/CARA_Analysis_Tools](https://github.com/nasa/CARA_Analysis_Tools) at a pinned commit and not redistributed):

| Values | Max relative difference | Within 1e-8 | Within 1e-12 |
|---|---|---|---|
| 106 | 1.6e-8 | 104 | 42 |

CARA's own quadrature (64-point Gauss–Chebyshev) is accurate to about 1e-8, so the residual difference is at the level of its precision.

## Limits

Public orbit data (TLE/OMM) is accurate to roughly a kilometre, much less precise than what operators use internally. kessler's results will be independent estimates from public data, and it will say so wherever it reports them.
