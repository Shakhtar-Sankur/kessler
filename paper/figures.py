"""Figures and numbers for paper/main.tex, from bench/results/*.json only.

Usage: python paper/figures.py   (writes paper/fig_agreement.pdf and paper/numbers.tex)
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import chi2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "bench", "results")


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def main():
    r = load("m3_sdmc.json")
    lib = load("m3_vs_nasa_lib.json")
    z = [x["z_vs_nasa"] for x in r]
    c2 = sum(v * v for v in z)
    ratio = [x["nasa_pc2d"] / x["nasa_sdmc"]["pc"] for x in r]

    plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.75))

    a = ax[0]
    for x in r:
        n, k = x["nasa_sdmc"], x["kessler"]
        a.errorbar(n["pc"], k["pc"] if k["hits"] else k["hi"],
                   xerr=[[n["pc"] - n["lo"]], [n["hi"] - n["pc"]]],
                   yerr=[[k["pc"] - k["lo"]] if k["hits"] else [0], [k["hi"] - k["pc"]]],
                   fmt="o", ms=2.6, lw=0.6, color="#1f3864" if abs(x["z_vs_nasa"]) < 2 else "#c0392b")
    lo, hi = 5e-8, 5e-2
    a.plot([lo, hi], [lo, hi], color="0.6", lw=0.6, zorder=0)
    a.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
          xlabel="CARA's Monte Carlo (SDMC)", ylabel="kessler's Monte Carlo")
    a.set_title("(a) 53 CARA test conjunctions, 95% intervals", fontsize=8.5)

    b = ax[1]
    floor = 1e-4
    inside = [(x["vrel_mps"], q) for x, q in zip(r, ratio) if q >= floor]
    below = [(x, q) for x, q in zip(r, ratio) if q < floor]
    b.scatter(*zip(*inside), s=8, color="#1f3864", lw=0)
    b.scatter([x["vrel_mps"] for x, _ in below], [floor * 1.6] * len(below), marker="v", s=14,
              color="#c0392b", lw=0)
    sci = lambda q: f"{q / 10 ** math.floor(math.log10(q)):.0f}e{math.floor(math.log10(q))}"
    below.sort(key=lambda t: t[0]["vrel_mps"])
    groups = []  # cases too close in velocity to label apart share one label
    for x, q in below:
        if groups and x["vrel_mps"] < 1.5 * groups[-1][-1][0]["vrel_mps"]:
            groups[-1].append((x, q))
        else:
            groups.append([(x, q)])
    for g in groups:
        label = ", ".join(str(x["case"]) for x, _ in g) + ": " + ", ".join(sci(q) for _, q in g)
        b.annotate(label, (g[0][0]["vrel_mps"], floor * 1.6), fontsize=6, rotation=90,
                   xytext=(-2, 7), textcoords="offset points", ha="center", va="bottom")
    for x, q in zip(r, ratio):
        if q > 10:
            b.annotate(str(x["case"]), (x["vrel_mps"], q), fontsize=6.5, xytext=(3, 1),
                       textcoords="offset points")
    b.axhspan(0.5, 2.0, color="0.88", zorder=0)
    b.set(xscale="log", yscale="log", xlabel="relative velocity at TCA (m/s)",
          ylabel="2D Pc / Monte Carlo Pc (CARA's)", ylim=(floor, 300))
    b.set_title("(b) 2D method vs CARA's Monte Carlo", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_agreement.pdf"))

    off = [q for q in ratio if q > 2 or q < 0.5]
    hits = sorted(x["kessler"]["hits"] for x in r)
    n = {
        "NCases": len(r),
        "WithinTwo": sum(abs(v) < 2 for v in z),
        "WithinThree": sum(abs(v) < 3 for v in z),
        "MaxAbsZ": f"{max(map(abs, z)):.2f}",
        "ChiSq": f"{c2:.1f}",
        "ChiP": f"{chi2.sf(c2, len(z)):.2f}",
        "TotalTrials": f"{sum(x['kessler']['trials'] for x in r) / 1e9:.2f}",
        "TotalMinutes": f"{sum(x['seconds'] for x in r) / 60:.0f}",
        "NasaTrials": f"{sum(x['nasa_sdmc']['trials'] for x in r) / 1e9:.1f}",
        "OffTwo": len(off),
        "OffLow": sum(q < 0.5 for q in ratio),
        "OffHigh": sum(q > 2 for q in ratio),
        "WorstOrders": f"{-math.log10(min(ratio)):.0f}",
        "FewHits": sum(h < 100 for h in hits),
        "MinHits": hits[0],
        "MedianHits": hits[len(hits) // 2],
        "MaxRelTwoD": (lambda d: f"${d / 10 ** math.floor(math.log10(d)):.1f}\\times10^{{{math.floor(math.log10(d))}}}$")(
            max(abs(x['kessler_pc2d'] / x['nasa_pc2d'] - 1) for x in r)),
        "BeyondTwo": sum(abs(v) >= 2 for v in z),
        "LibSpeedMin": f"{min(x['speedup'] for x in lib if x['vrel_mps'] > 100):.0f}",
        "LibSpeedMax": f"{max(x['speedup'] for x in lib):.0f}",
        "LibMaxAbsZ": f"{max(abs(x['z']) for x in lib):.2f}",
    }
    with open(os.path.join(HERE, "tab_speed.tex"), "w") as f:
        for x in sorted(lib, key=lambda x: -x["vrel_mps"]):
            f.write(f"{x['case']} & {x['vrel_mps']:,.0f} & {x['nasa_s']:.1f} & {x['kessler_s']:.2f} & "
                    f"{x['speedup']:.1f}$\\times$ \\\\\n")
    abl_path = os.path.join(RES, "m3_ablations.json")
    if os.path.exists(abl_path):
        abl = load("m3_ablations.json")
        names = {"cartesian": "Cartesian state", "equinoctial": "equinoctial, $a$",
                 "equinoctial-n": "equinoctial, $n$"}
        def sci(q):
            if q == 0:
                return "0"
            e = math.floor(math.log10(q))
            return f"${q / 10 ** e:.2f}\\times10^{{{e}}}$"
        rows = []
        for x in abl["sampling"]:
            k, c = x["kessler"], x["nasa_sdmc"]
            rows.append(f"{x['case']} & {names[x['space']]} & {sci(k['pc'])} & {k['hits']:,} / {k['trials'] / 1e6:.0f}M"
                        f" & {sci(c['pc'])} & ${k['z_vs_nasa']:+.1f}$ \\\\")
        tail_path = os.path.join(RES, "m3_tail.json")
        if os.path.exists(tail_path):
            rows.append("\\midrule")
            for x in load("m3_tail.json"):
                c = x["nasa_sdmc"]
                rows.append(f"{x['case']} & {names[x['space']]} & {sci(x['pc'])} & {x['hits']:,} / {x['trials'] / 1e6:,.0f}M"
                            f" & {sci(c['pc'])} & ${x['z_vs_nasa']:+.1f}$ \\\\")
                n["TailPc" + ("N" if x["space"] == "equinoctial-n" else "A")] = sci(x["pc"])
                n["TailZ" + ("N" if x["space"] == "equinoctial-n" else "A")] = f"{x['z_vs_nasa']:+.1f}"
                n["TailHits" + ("N" if x["space"] == "equinoctial-n" else "A")] = f"{x['hits']:,}"
        with open(os.path.join(HERE, "tab_sampling.tex"), "w") as f:
            f.write("\n".join(rows) + "\n")
        rows = []
        for x in abl["window"]:
            k, c = x["kessler"], x["nasa_sdmc"]
            what = "TCA only" if x["counted"].startswith("encounter") else "every approach"
            frac = {0.25: "$\\pm P/4$", 0.5: "$\\pm P/2$", 1.0: "$\\pm P$"}[x["half_window_periods"]]
            rows.append(f"{x['case']} & {frac} & {what} & {sci(k['pc'])} & {k['hits']:,} & {sci(c['pc'])} & ${k['z_vs_nasa']:+.1f}$ \\\\")
        with open(os.path.join(HERE, "tab_window.tex"), "w") as f:
            f.write("\n".join(rows) + "\n")
        n["SecondWindows"] = abl["second_encounter"].get("cases_with_more_than_one_window", "?")
    wc_path = os.path.join(RES, "m3_windows_check.json")
    if os.path.exists(wc_path):
        wc = load("m3_windows_check.json")
        n["WcCases"] = len(wc)
        n["WcSame"] = sum(x["restricted"] == x["full"] for x in wc)
        n["WcTrials"] = f"{wc[0]['trials']:,}"
        n["WcHits"] = f"{sum(x['full'] for x in wc):,}"
    with open(os.path.join(HERE, "numbers.tex"), "w") as f:
        f.write("% generated by paper/figures.py from bench/results; do not edit\n")
        for k, val in n.items():
            f.write(f"\\newcommand{{\\n{k}}}{{{val}}}\n")
    print(n)


if __name__ == "__main__":
    main()
