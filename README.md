# kessler

An independent, open-source space-traffic auditor. **Live: [shakhtar-sankur.github.io/kessler](https://shakhtar-sankur.github.io/kessler/)**, updated every day.

> **Data-use notice (14 Oct 2026).** Results computed from Space-Track.org data are withheld from this
> repository and the page while approval to publish analyses of that data is requested from the 18th
> Space Defense Squadron, as Space-Track's user agreement requires. The code and methods stay public;
> the daily page now uses CelesTrak's public data only.

SpaceX reports that Starlink satellites made **207,152 collision-avoidance maneuvers** between December 2025 and May 2026. Its satellites maneuver automatically whenever the probability of a collision passes 3 in 10 million. Nobody outside SpaceX can check those numbers today.

kessler is building the tools to do it from public data:

1. **SGP4 on CPU and GPU**: one C++ propagator, compiled for both, for the whole public catalogue.
2. **All-vs-all conjunction screening on a GPU**: every tracked object against every other.
3. **Collision probability** for the close approaches, by NASA's 2D method and by Monte Carlo where it fails.
4. **Starlink maneuver detection** from public orbit history, compared with SpaceX's reports to the FCC.

Named after the Kessler syndrome: the runaway cascade of collisions that this kind of work exists to prevent.

## Status

| Milestone | State |
|---|---|
| M0: SGP4 (CPU), validated against the reference | done |
| M1: SGP4 on CUDA, full catalogue on a T4 | done |
| M2: all-vs-all screening, proven against brute force; GPU = CPU on a T4; full 32,593-object catalogue in 28 s; SOCRATES's 100 closest conjunctions reproduced (median 0.24 m) | done |
| M3: collision probability: the 2D method matches NASA CARA to 1.6e-8 on 53 real conjunctions; a two-body Monte Carlo matches CARA's on all 53 (including the 28 where the 2D method fails) and runs 15–19× faster per core than CARA's own library | done |
| M4: Starlink maneuvers vs FCC reports (results withheld pending data-use approval) | done |
| M5: daily screening on GitHub Actions (CelesTrak public data), public 3D page, archive of operator orbits | running daily |

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

Results of this run are withheld pending approval to publish analyses of Space-Track data (see the notice at the top); its speed is a property of the software: **27.7 s** for **32,593 objects** on a T4 (candidate search on the GPU 18.0 s).

### Reproducing SOCRATES

[CelesTrak's SOCRATES](https://celestrak.org/SOCRATES/) has screened active satellites against the public catalogue since 2004, with commercial software (STK/CAT). [`bench/m2_socrates.py`](bench/m2_socrates.py) takes its 100 closest conjunctions, recovers the exact element sets SOCRATES used from Space-Track's history (each object's days-from-epoch must match SOCRATES's to 0.001 day, the precision it publishes) and computes the same approaches with kessler. **All 100 were recovered and compared:**

| | Largest difference from SOCRATES | Median |
|---|---|---|
| Distance at closest approach | 4.2 m (one pair; the next largest are under 0.5 m) | 0.24 m |
| Time of closest approach | 15.5 ms | |
| Relative speed | 0.49 m/s | |

SOCRATES rounds distances to the metre, so differences below 0.5 m are its rounding. Per-conjunction results and the analysis of SOCRATES's hard-body radii are withheld pending the approval above.

## M3: probability of collision, checked against NASA

`kessler/pc.py` computes the 2D probability of collision (Foster, 1992; the method NASA CARA uses operationally) from two states and covariances: the combined covariance is projected onto the encounter plane and the Gaussian is integrated over the hard-body circle, in logarithms, so even probabilities of 1e-168 keep full relative precision.

It reads standard Conjunction Data Messages (CCSDS 508.0) and is compared with NASA CARA's own published values for 53 real conjunctions (for example the Hubble Space Telescope and a 1966 Diamant rocket body; Terra and a fragment of Iridium 33 at 108 m, Pc = 0.021), each with and without CARA's closest-approach adjustment ([`tests/test_pc.py`](tests/test_pc.py); the data is fetched from [nasa/CARA_Analysis_Tools](https://github.com/nasa/CARA_Analysis_Tools) at a pinned commit and not redistributed):

| Values | Max relative difference | Within 1e-8 | Within 1e-12 |
|---|---|---|---|
| 106 | 1.6e-8 | 104 | 42 |

CARA's own quadrature (64-point Gauss–Chebyshev) is accurate to about 1e-8, so the residual difference is at the level of its precision.

### Monte Carlo: where the 2D method is wrong

The 2D method assumes a short, straight-line encounter. CARA's own Monte Carlo (SDMC) shows how often that fails: of its 53 test conjunctions, the 2D value is off by more than a factor of 2 in **28**, too low in 25 and too high in 3, by up to 160 orders of magnitude on slow, drifting encounters.

[`src/mc.cpp`](src/mc.cpp) (called as `kessler.pc.pc_mc`) is a two-body Monte Carlo of the same kind: it draws both objects' states at TCA from their covariances, moves each along its exact Kepler orbit (checked against numerical integration to a micrometre) and counts the trials in which they pass within the hard-body radius during the encounter at TCA. Each trial is searched only in the window where it can come within that radius, from a first-order bound; trials beyond the bound's range are searched in full, and restricted and full searches give identical hits trial for trial on all 53 cases. Trials run in parallel on fixed random streams, so a result depends on the seed, not on the number of threads ([`tests/test_pc_mc.py`](tests/test_pc_mc.py)).

**Against CARA's published Monte Carlo** ([`bench/m3_sdmc.py`](bench/m3_sdmc.py), [`bench/results/m3_sdmc.json`](bench/results/m3_sdmc.json); 1.8 billion trials in 29 minutes on 4 CPU cores):

| | |
|---|---|
| Conjunctions within \|z\| < 2 / < 3 of CARA's (two-proportion test on hit counts) | **50 / 53**, **53 / 53** (largest \|z\| 2.70) |
| All 53 together | χ² = 48.0 on 53 degrees of freedom, p = 0.67: indistinguishable from CARA's |

| Case | Encounter | 2D method | CARA's Monte Carlo | kessler |
|---|---|---|---|---|
| 1: Hubble and a Delta 2 rocket body | 2.9 km/s | 6.11e-4 | 6.11e-4 | 6.23e-4 |
| 25: WorldView-2 and Fengyun-1C debris | 53.6 m/s | 4.5e-23 | 1.51e-4 | 1.53e-4 |
| 45: ICESat-2 and SpaceBEE-114 | 15.2 km/s | 3.3e-8 | 7.38e-7 | 7.7e-7 |
| 49 | 10.1 km/s | 2.4e-5 (63× too high) | 3.90e-7 | 3.6e-7 |
| 51 | 0.3 m/s | 4.5e-81 | 1.30e-6 | 1.4e-6 |
| 52: TROPICS Pathfinder and LINCS2 | 9.0 m/s | 6.5e-168 | 1.17e-5 | 9.3e-6 |

**Against CARA's own library, same inputs, one core each** ([`bench/m3_vs_nasa_lib.py`](bench/m3_vs_nasa_lib.py) builds a driver around CARA's published `libsdmc_linux.so` and runs both with the same states, covariances, radius and time window): kessler agrees within statistical error on every case run and takes **1.6–1.9 s per million trials where CARA's library takes 29–31 s (15–19×)**, on encounters from 123 m/s to 14.5 km/s; on the slowest (case 25, 54 m/s), whose trials must be searched across most of the revolution, 17.7 s against 31.2 s.

Three choices decide whether a Monte Carlo agrees, and each is easy to get wrong:

* **What the Gaussian describes.** A covariance is a linear description, and a Monte Carlo must decide in which coordinates to draw from it; on a small probability the far tail decides. kessler draws equinoctial elements with mean motion (n, a_f, a_g, χ, ψ, λ), as CARA does. Drawing Cartesian states instead tilts each sample's velocity off its orbit when the along-track uncertainty is large: case 25 comes out 17% low, and case 45 gets no hits at all in 100 million trials. Drawing the same elements with the semi-major axis in place of the mean motion gives case 45 about 20% too much.
* **How long the encounter is.** A quarter period either side of TCA (the default of CARA's standalone `Pc_SDMC`) misses slow encounters: cases 52 and 53 (9 and 11 m/s) get no hits in it. kessler searches half a period either side.
* **Which encounter is counted.** Two orbits that cross once usually cross again half a revolution later; that second close approach is a different conjunction. kessler counts only the encounter at TCA (in 25 of the 53 cases a second one lies within half a period of it).

Not done: the drag and solar-pressure cross-correlation corrections CARA can apply (its published test values do not use them), and force models beyond two-body (CARA's SDMC is two-body too).

## M4: how often Starlink satellites maneuver, seen from public data

[`kessler/maneuvers.py`](kessler/maneuvers.py) reads one satellite's element-set history and finds the moments its orbit changed size: at each update it fits the mean semi-major axis before and after with a shared drag trend plus a step, and accepts the step if it is larger than 50 m and than 6 times the satellite's own noise. It is tested on synthetic histories with known maneuvers, and on real **debris, which cannot maneuver**: whatever it finds there is a false alarm.

[`bench/m4_maneuvers.py`](bench/m4_maneuvers.py) runs it on the six months covered by SpaceX's report to the FCC (1 December 2025 to 31 May 2026) with a debris control group, and [`bench/m4_attribution.py`](bench/m4_attribution.py) tests whether the maneuvers follow predicted close approaches. Both need Space-Track's element-set history; their results are withheld pending the approval above, and anyone with their own Space-Track access can run them.

## M5: every day, in public

**Page: [shakhtar-sankur.github.io/kessler](https://shakhtar-sankur.github.io/kessler/)** — a 3D globe of the active satellites in CelesTrak's public catalogue, the day's closest approaches (select one to see both orbits around closest approach) and running statistics.

* **Daily screening** ([`.github/workflows/daily.yml`](.github/workflows/daily.yml), [`kessler/daily.py`](kessler/daily.py)): every morning (05:41 UTC) a GitHub Actions runner builds kessler, downloads CelesTrak's public catalogue of active satellites, screens every pair for the next 24 hours on its CPU, computes the worst-case probability of the 50 closest approaches, keeps the day's derived results in [`daily/`](daily) and deploys the page. Element sets are never published.
* **Archive of operator orbits** ([`.github/workflows/archive.yml`](.github/workflows/archive.yml), [`kessler/archive.py`](kessler/archive.py)): every 6 hours it stores each new version of CelesTrak's supplemental element sets, which are fitted to the operators' own ephemerides (SpaceX for Starlink) and are far more precise than the public catalogue, plus SOCRATES's 2,000 closest conjunctions. CelesTrak serves only the current versions; this history is what M4's audit needs to be repeated with operator-quality orbits. It goes to a private repository, since kessler does not redistribute source data.
* **Tests on every push** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## Limits

Public orbit data (TLE/OMM) is accurate to roughly a kilometre, much less precise than what operators use internally. kessler's results will be independent estimates from public data, and it will say so wherever it reports them.
