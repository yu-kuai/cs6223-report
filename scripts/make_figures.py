"""Draw all paper figures into ../figures (PDF for LaTeX, PNG for preview).
usage: PYTHONPATH=<matplotlib dir> python scripts/make_figures.py [project_dir]
Numbers for the bar/audit charts come from results/analysis/compare_v2*.txt and results/audit/claude_labels_summary.md
of the project; the over-rounds figure is computed from the per-test logs (events.jsonl + judge_ext_deepseek.jsonl)."""
import glob, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")
PROJ = sys.argv[1] if len(sys.argv) > 1 else "/mnt/scratch/k/kuaiyu/cs6223/project"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e3e3e3", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
C = {"HW": "#8C96A6", "SJ": "#D9A441", "OJ": "#6BAED6", "BO": "#E6855E", "BR": "#2F5DA8"}
FAIR, TRICK, BROKEN = "#2F5DA8", "#E6A23C", "#C8471A"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ 1. text: distinct questions
def fig_text():
    models = ["Qwen3-1.7B", "Qwen3-8B", "Qwen3-32B"]
    rows = [  # name, color, mean, std, strict mean
        ("Static", C["HW"], [99, 49, 33], [4, 4, 6], [98, 49, 33]),
        ("Self-Judge", C["SJ"], [109, 40, 26], [59, 20, 3], [108, 40, 26]),
        ("External-Judge", C["OJ"], [55, 60, 28], [13, 26, 7], [55, 60, 28]),
        ("Greedy", C["BO"], [170, 135, 80], [28, 129, 84], [169, 57, 31]),
        ("Guarded", C["BR"], [100, 59, 39], [20, 40, 15], [100, 59, 39]),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    x = np.arange(3); w = 0.16
    for i, (n, col, m, s, st) in enumerate(rows):
        xs = x + (i - 2) * w
        ax.bar(xs, st, w * 0.92, color=col, label=n, zorder=2)
        extra = np.array(m) - np.array(st)
        if extra.max() > 2:
            ax.bar(xs, extra, w * 0.92, bottom=st, color="white", edgecolor=col, hatch="////", linewidth=0.8, zorder=2,
                   label="Greedy: tricks (double negation, fragmented question)")
        ax.errorbar(xs, m, yerr=s, fmt="none", ecolor="#333333", elinewidth=0.7, capsize=1.8, zorder=3)
    ax.set_xticks(x, models)
    ax.set_ylabel("distinct questions with\na real failure")
    ax.set_ylim(0, 280)
    h, l = ax.get_legend_handles_labels()
    order = [0, 1, 2, 3, 5, 4]   # put the hatched "tricks" entry last
    ax.legend([h[i] for i in order], [l[i] for i in order], ncol=3, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.0)
    save(fig, "fig_text_distinct")


# ------------------------------------------------------------------ 2. images
def fig_images():
    models = ["Qwen3-VL-2B", "Qwen3-VL-8B", "Qwen3-VL-32B"]
    fig, ax = plt.subplots(figsize=(5.5, 2.3))
    x = np.arange(3); w = 0.22
    hw, hws = [40, 29, 16], [9, 3, 2]
    sj, sjs = [27, 46, 12], [9, 8, 11]
    br, brs = [39, 101, 50], [28, 37, 28]
    corr = [20, 13, 37]
    ax.bar(x - w, hw, w * 0.92, color=C["HW"], label="Static", zorder=2)
    ax.bar(x, sj, w * 0.92, color=C["SJ"], label="Self-Judge", zorder=2)
    ax.bar(x + w, br, w * 0.92, color="white", edgecolor=C["BR"], hatch="////", linewidth=0.8, zorder=2,
           label="Guarded, as graded by DeepSeek")
    ax.bar(x + w, corr, w * 0.92, color=C["BR"], zorder=3, label="Guarded, corrected by hand audit")
    for xs, m, s in [(x - w, hw, hws), (x, sj, sjs), (x + w, br, brs)]:
        ax.errorbar(xs, m, yerr=s, fmt="none", ecolor="#333333", elinewidth=0.7, capsize=1.8, zorder=4)
    for i in range(3):
        ax.text(x[i] + w, corr[i] + 2, f"~{corr[i]}", ha="center", va="bottom", fontsize=7.5, color=C["BR"], zorder=5,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.set_xticks(x, models)
    ax.set_ylabel("distinct questions\nwith a failure")
    ax.set_ylim(0, 160)
    ax.legend(ncol=2, frameon=False, loc="upper left", handlelength=1.4)
    save(fig, "fig_image_distinct")


# ------------------------------------------------------------------ 3. progress over actual rounds (no rescaling)
def load_run(d):
    """per round: tests that reached the target, real (DeepSeek VALID) and broken failures, and the set of seeds
    with a real failure; rounds are the loop's own rounds (48 tests each)."""
    ev = [json.loads(l) for l in open(os.path.join(d, "events.jsonl"))]
    ext = {}
    for l in open(os.path.join(d, "judge_ext_deepseek.jsonl")):
        r = json.loads(l); ext[r["line"]] = r["ext"]
    R = max(e["round"] for e in ev) + 1
    n, tp, fp = np.zeros(R), np.zeros(R), np.zeros(R)
    seen, cum = set(), np.zeros(R)
    for i, e in enumerate(ev):
        r = e["round"]
        if e["status"] not in ("pass", "violation", "unparsed"):
            continue
        n[r] += 1
        if e["status"] == "violation":
            if ext.get(i) == "VALID":
                tp[r] += 1; seen.add(e["seed_id"])
            else:
                fp[r] += 1
        cum[r] = len(seen)
    cum = np.maximum.accumulate(cum)
    return n, tp, fp, cum


PANELS = [("results/fuzz", "qwen3-1.7b_self", "Text 1.7B"), ("results/fuzz", "qwen3-8b_self", "Text 8B"),
          ("results/fuzz", "qwen3-32b_self", "Text 32B"), ("results/fuzz_vqa", "qwen3-vl-2b_self", "Image 2B"),
          ("results/fuzz_vqa", "qwen3-vl-8b_self", "Image 8B"), ("results/fuzz_vqa", "qwen3-vl-32b_self", "Image 32B")]
SETUPS = [("Static", "fixed*", C["HW"]), ("Guarded", "full_v4*", C["BR"])]


def runs(base, m, pat):
    for d in sorted(glob.glob(os.path.join(PROJ, base, m, pat))):
        if "smoke" in d or "cancel" in d or not os.path.exists(os.path.join(d, "judge_ext_deepseek.jsonl")):
            continue
        yield d


def fig_rounds():
    """cumulative distinct questions with a real failure vs. round number, one line per run"""
    fig, axs = plt.subplots(2, 3, figsize=(5.6, 3.7))
    table = []
    for ax, (base, m, title) in zip(axs.flat, PANELS):
        for setup, pat, col in SETUPS:
            for k, d in enumerate(runs(base, m, pat)):
                n, tp, fp, cum = load_run(d)
                ax.plot(np.arange(1, len(cum) + 1), cum, color=col, lw=1.2, alpha=0.9,
                        label=setup if k == 0 else None)
                ax.plot(len(cum), cum[-1], "o", color=col, ms=2.5)
                table.append({"panel": title, "setup": setup, "run": os.path.basename(d), "rounds": len(cum),
                              "distinct_real": int(cum[-1]),
                              "real_per_100_by_round": [round(100 * t / max(x, 1), 1) for t, x in zip(tp, n)],
                              "broken_per_100_by_round": [round(100 * f / max(x, 1), 1) for f, x in zip(fp, n)]})
        ax.set_title(title)
        ax.set_ylim(bottom=0); ax.set_xlim(left=0)
    for ax in axs[1]:
        ax.set_xlabel("round (48 tests each)")
    for ax in axs[:, 0]:
        ax.set_ylabel("distinct questions\nwith a real failure")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_over_rounds")
    json.dump(table, open(os.path.join(OUT, "over_rounds_data.json"), "w"))


def fig_rates(win=10):
    """real and broken failures per 100 tests in consecutive blocks of `win` rounds (actual round numbers),
    pooled over the 3 runs that are still running in that block"""
    fig, axs = plt.subplots(2, 3, figsize=(5.6, 3.7))
    for ax, (base, m, title) in zip(axs.flat, PANELS):
        for setup, pat, col in SETUPS:
            acc = {}
            for d in runs(base, m, pat):
                n, tp, fp, _ = load_run(d)
                for r in range(len(n)):
                    b = r // win
                    a = acc.setdefault(b, [0, 0, 0]); a[0] += n[r]; a[1] += tp[r]; a[2] += fp[r]
            bs = sorted(b for b in acc if acc[b][0] >= 100)   # skip blocks with too few tests
            x = [(b + 0.5) * win for b in bs]
            ax.plot(x, [100 * acc[b][1] / acc[b][0] for b in bs], "-o", color=col, ms=2.5, lw=1.3, label=f"{setup}: real")
            ax.plot(x, [100 * acc[b][2] / acc[b][0] for b in bs], "--", color=col, lw=1.0, label=f"{setup}: broken")
        ax.set_title(title)
        ax.set_ylim(bottom=0); ax.set_xlim(left=0)
    for ax in axs[1]:
        ax.set_xlabel("round (48 tests each)")
    for ax in axs[:, 0]:
        ax.set_ylabel("per 100 tests")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02), columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_rates_over_rounds")


# ------------------------------------------------------------------ 4. judges
def fig_judges():
    rows = [  # label, fair, trick, broken (counts of failures the judge accepted)
        ("Tested model judges itself\n(Self-Judge era, n=102)", 58, 6, 38),
        ("Qwen3-32B judges\n(Self-Judge era, n=78)", 52, 9, 17),
        ("DeepSeek grader\n(Self-Judge era, n=25)", 23, 2, 0),
        ("GLM in-loop judge\n(Greedy, n=54)", 22, 32, 0),
        ("DeepSeek grader\n(Greedy, n=45)", 15, 30, 0),
        ("DeepSeek grader, images\n(Static, n=20)", 20, 0, 0),
        ("DeepSeek grader, images\n(Guarded, n=20)", 9, 3, 8),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, a, b, c) in zip(y, rows):
        tot = a + b + c
        left = 0
        for v, col in [(a, FAIR), (b, TRICK), (c, BROKEN)]:
            p = 100 * v / tot
            ax.barh(yi, p, left=left, color=col, height=0.66, zorder=2)
            if p >= 9:
                ax.text(left + p / 2, yi, f"{p:.0f}%", ha="center", va="center", color="white", fontsize=7.5,
                        fontweight="bold", zorder=3)
            left += p
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the failures the judge accepted as real (%)")
    ax.grid(axis="x", color="#e3e3e3"); ax.grid(axis="y", visible=False)
    for v, col, n in [(0, FAIR, "fair"), (0, TRICK, "trick"), (0, BROKEN, "broken")]:
        ax.barh(-5, 0, color=col, label=n)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig_judges")


# ------------------------------------------------------------------ 5. quantity vs quality (text)
def fig_tradeoff():
    pts = {  # config: [(real share %, distinct questions strict) for 1.7B, 8B, 32B]
        "Static": [(76, 98), (80, 49), (79, 33)],
        "Self-Judge": [(56, 108), (15, 40), (30, 26)],
        "External-Judge": [(80, 55), (48, 60), (44, 28)],
        "Greedy": [(80, 169), (25, 57), (28, 31)],
        "Guarded": [(89, 100), (42, 59), (62, 39)],
    }
    col = {"Static": C["HW"], "Self-Judge": C["SJ"], "External-Judge": C["OJ"], "Greedy": C["BO"], "Guarded": C["BR"]}
    mk = ["o", "s", "^"]
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    for n, ps in pts.items():
        for (r, d), m in zip(ps, mk):
            ax.scatter(r, d, color=col[n], marker=m, s=34, zorder=3, edgecolor="white", linewidth=0.5)
    for n in pts:
        ax.scatter([], [], color=col[n], marker="o", s=30, label=n)
    for m, lab in zip(mk, ["1.7B", "8B", "32B"]):
        ax.scatter([], [], color="#555555", marker=m, s=30, label=lab)
    ax.set_xlabel("share of answer changes that are real (%), strict")
    ax.set_ylabel("distinct questions with\na real failure (strict)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 190)
    ax.grid(axis="x", color="#e3e3e3")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=7.5, handletextpad=0.3, columnspacing=0.8)
    save(fig, "fig_tradeoff")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_text(); fig_images(); fig_rounds(); fig_rates(); fig_judges(); fig_tradeoff()
