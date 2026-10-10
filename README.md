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
| M2: all-vs-all screening, proven against brute force; GPU = CPU on a T4; full 32,593-object catalogue in 28 s; SOCRATES's 100 closest conjunctions reproduced (median 0.24 m) | done |
| M3: collision probability (matches NASA CARA on 53 real conjunctions) | in progress |
| M4: Starlink maneuver detection vs FCC reports: public data shows about half as many orbit changes as the reported avoidance maneuvers; attributing them to conjunctions next | in progress |
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

### One day of the whole active catalogue on a free T4

All 16,695 active objects on CelesTrak (10 Oct 2026), 24 hours from 2026-10-10 00:00 UTC, every pair closer than 5 km (Kaggle, one Tesla T4, 4 CPU cores):

| | Time | Approaches found |
|---|---|---|
| **kessler, GPU** (candidates on the T4, refinement on the CPU) | **16.4 s** (candidate search 6.8 s) | 55,738 |
| kessler, CPU (4 cores) | 177.3 s | 55,738 |

The GPU and CPU runs find exactly the same approaches, with identical times and distances to the last digit (the unit test also checks this on a 30-minute run: 65,505 identical candidates).

Of the 55,738 approaches, **43,123 (77%) are between two Starlink satellites**, the pairs that CelesTrak's SOCRATES service leaves out; 10,464 are Starlink with something else, 2,151 involve no Starlink. 935 are closer than 1 km and 35 closer than 200 m; the closest are STARLINK-30334 and BOTSAT-1 (8 m at 14.3 km/s) and STARLINK-31070 and STARLINK-31188 (10 m at 7.8 km/s). 77 docked or co-located pairs are reported separately. These come from public orbits accurate to about a kilometre: they are screening results, not predictions of collisions.

### The full public catalogue, debris included

Space-Track's catalogue (every on-orbit object with elements from the last 30 days: **32,593 objects**), 24 hours from 2026-10-10 08:12 UTC, 5 km, on the same T4:

| | |
|---|---|
| Time | **27.7 s** (candidate search on the GPU 18.0 s) |
| Approaches closer than 5 km | 75,820 (2,033 closer than 1 km, 85 closer than 200 m) |
| Starlink with Starlink / with anything else / no Starlink | 44,782 / 13,682 / 17,356 |

Adding the debris and inactive objects that the active catalogue lacks brings 20,000 more approaches, most of them (17,356) between objects that are neither Starlink. The closest approach, STARLINK-30334 and BOTSAT-1 at 8 m, is the same as in the active-catalogue run. Space-Track's data is used in memory only; the repository keeps derived numbers.

### Reproducing SOCRATES

[CelesTrak's SOCRATES](https://celestrak.org/SOCRATES/) has screened active satellites against the public catalogue since 2004, with commercial software (STK/CAT). [`bench/m2_socrates.py`](bench/m2_socrates.py) takes its 100 closest conjunctions, recovers the exact element sets SOCRATES used from Space-Track's history (each object's days-from-epoch must match SOCRATES's to 0.001 day, the precision it publishes) and computes the same approaches with kessler. **All 100 were recovered and compared:**

| | Largest difference from SOCRATES | Median |
|---|---|---|
| Distance at closest approach | 4.2 m (one pair; the next largest are under 0.5 m) | 0.24 m |
| Time of closest approach | 15.5 ms | |
| Relative speed | 0.49 m/s | |

SOCRATES rounds distances to the metre, so differences below 0.5 m are its rounding. The one exception, STARLINK-34331 and STARLINK-37457 (SOCRATES 29 m, kessler 33.2 m), is being checked: the script now flags conjunctions where more than one element set fits within SOCRATES's 0.001-day precision.

SOCRATES does not publish the hard-body radius behind its "maximum probability" (Alfano's method with a 100 × 300 × 100 m covariance shape, scaled to the worst case). Solving for the radius that reproduces each value gives 2.0 to 46 m, and it follows the objects involved. Among the 15 closest: about **2 m** when neither object is a Starlink, **11 to 17 m** for a Starlink and anything else, and **24 to 35 m** for two Starlinks. So SOCRATES appears to use a size for each object, much larger for Starlink, rather than one fixed radius. (An earlier run, on only the 24 conjunctions whose element sets CelesTrak still had, saw just 5.7 to 7.0 m.) The script now groups the radii by radar cross-section class to test this.

## M3: probability of collision, checked against NASA

`kessler/pc.py` computes the 2D probability of collision (Foster, 1992; the method NASA CARA uses operationally) from two states and covariances: the combined covariance is projected onto the encounter plane and the Gaussian is integrated over the hard-body circle, in logarithms, so even probabilities of 1e-168 keep full relative precision.

It reads standard Conjunction Data Messages (CCSDS 508.0) and is compared with NASA CARA's own published values for 53 real conjunctions (for example the Hubble Space Telescope and a 1966 Diamant rocket body; Terra and a fragment of Iridium 33 at 108 m, Pc = 0.021), each with and without CARA's closest-approach adjustment ([`tests/test_pc.py`](tests/test_pc.py); the data is fetched from [nasa/CARA_Analysis_Tools](https://github.com/nasa/CARA_Analysis_Tools) at a pinned commit and not redistributed):

| Values | Max relative difference | Within 1e-8 | Within 1e-12 |
|---|---|---|---|
| 106 | 1.6e-8 | 104 | 42 |

CARA's own quadrature (64-point Gauss–Chebyshev) is accurate to about 1e-8, so the residual difference is at the level of its precision.

## M4: how often Starlink satellites maneuver, seen from public data

[`kessler/maneuvers.py`](kessler/maneuvers.py) reads one satellite's element-set history and finds the moments its orbit changed size: at each update it fits the mean semi-major axis before and after with a shared drag trend plus a step, and accepts the step if it is larger than 50 m and than 6 times the satellite's own noise. It is tested on synthetic histories with known maneuvers, and on real **debris, which cannot maneuver**: whatever it finds there is a false alarm.

[`bench/m4_maneuvers.py`](bench/m4_maneuvers.py) ran it on the six months covered by SpaceX's report to the FCC (1 December 2025 to 31 May 2026), with Space-Track's full element-set history:

| | |
|---|---|
| Starlink satellites on orbit for the whole period | 8,833 |
| Sampled at random | 500 (90,694 satellite-days) |
| Orbit changes detected | 6,504: **0.072 per satellite per day**, one every 14 days |
| Debris control: 253 fragments of COSMOS 2251, IRIDIUM 33 and FENGYUN 1C (44,747 object-days) | 92 detections: 0.0021 per object per day, **3% of the Starlink rate** |
| Scaled to the 8,833 satellites | about 115,000 orbit changes (112,000 after removing false alarms) |
| SpaceX's report: collision-avoidance maneuvers alone | **207,152** |

**The public catalogue shows only about half as many orbit changes as SpaceX reports collision-avoidance maneuvers**, even though the changes it does show also include station-keeping and every other reason to thrust. So at least roughly 45% of the reported avoidance maneuvers leave no visible step in public orbit data: either they change the orbit by less than 50 m, or the avoidance and the return to the planned orbit both happen between two public updates. Public data alone cannot count Starlink's avoidance maneuvers; it can only bound them.

The comparison is approximate. Satellites launched or retired during the period also maneuvered and are not in the sample, and the FCC figure covers the whole fleet. Next: for each detected maneuver, screen the satellite against the catalogue as it stood just before, to see which maneuvers followed a predicted close approach.

## Limits

Public orbit data (TLE/OMM) is accurate to roughly a kilometre, much less precise than what operators use internally. kessler's results will be independent estimates from public data, and it will say so wherever it reports them.
