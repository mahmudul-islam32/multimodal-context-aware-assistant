"""Study analysis (EVALUATION.md analysis plan).

    python -m eval.analysis.analyze_study --tasks results/study/export/tasks.csv --questionnaires q.csv
    python -m eval.analysis.analyze_study --simulate 12     # pipeline check on SIMULATED data

Paired comparisons A vs B per participant: descriptive stats (median, IQR),
normality check of differences (Shapiro-Wilk), paired t-test or Wilcoxon
signed-rank, effect size (Cohen's d_z / rank-biserial r) with bootstrap 95% CI,
and an order-effect check (difference by AB vs BA order).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = {"task_time_s": "lower", "errors": "lower", "sus": "higher", "tlx": "lower", "trust": "higher"}


def simulate(n: int, seed: int = 1) -> pd.DataFrame:
    """SIMULATED participants for testing the analysis pipeline only.
    Effects are invented; never report these numbers as results."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, n + 1):
        pid = f"P{i:02d}"
        skill = rng.normal(0, 1)
        order = "AB" if i % 2 else "BA"
        for cond in "AB":
            learn = -15 if (order[1] == cond) else 0
            eff = cond == "A"
            rows.append({
                "participant": pid, "condition": cond, "order": order,
                "task_time_s": max(120, 420 - 35 * skill - (45 if eff else 0) + learn + rng.normal(0, 40)),
                "errors": max(0, int(rng.poisson(1.0 if eff else 1.7))),
                "sus": float(np.clip(rng.normal(78 if eff else 68, 9), 0, 100)),
                "tlx": float(np.clip(rng.normal(38 if eff else 47, 10), 0, 100)),
                "trust": float(np.clip(rng.normal(5.2 if eff else 4.7, 0.7), 1, 7)),
            })
    df = pd.DataFrame(rows)
    df.attrs["simulated"] = True
    return df


def _boot_ci(d: np.ndarray, fn, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = [fn(rng.choice(d, len(d), replace=True)) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def dz(d: np.ndarray) -> float:
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0


def rank_biserial(d: np.ndarray) -> float:
    d = d[d != 0]
    if not len(d):
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def compare(df: pd.DataFrame, metric: str) -> dict:
    w = df.pivot(index="participant", columns="condition", values=metric).dropna()
    d = (w["A"] - w["B"]).to_numpy(float)
    normal = len(d) >= 3 and stats.shapiro(d).pvalue > 0.05 if np.ptp(d) > 0 else False
    if normal:
        test, p = "paired t", float(stats.ttest_rel(w["A"], w["B"]).pvalue)
        es_name, es = "d_z", dz(d)
        ci = _boot_ci(d, dz)
    else:
        test = "Wilcoxon"
        p = float(stats.wilcoxon(w["A"], w["B"], zero_method="zsplit").pvalue) if np.any(d != 0) else 1.0
        es_name, es = "r_rb", rank_biserial(d)
        ci = _boot_ci(d, rank_biserial)
    q = lambda s: [float(np.percentile(s, 25)), float(np.percentile(s, 75))]  # noqa: E731
    out = {
        "n": int(len(d)), "median_A": float(w["A"].median()), "median_B": float(w["B"].median()),
        "iqr_A": q(w["A"]), "iqr_B": q(w["B"]), "test": test, "p": p, "effect": es_name,
        "effect_size": es, "effect_ci95": ci, "direction_expected": METRICS.get(metric, "?"),
    }
    if "order" in df:
        order = df.drop_duplicates("participant").set_index("participant")["order"]
        dd = pd.Series(d, index=w.index)
        groups = [dd[order.reindex(dd.index) == o].to_numpy() for o in ("AB", "BA")]
        if all(len(g) >= 2 for g in groups):
            out["order_effect_p"] = float(stats.mannwhitneyu(*groups).pvalue)
    return out


def analyze(df: pd.DataFrame) -> dict:
    return {m: compare(df, m) for m in METRICS if m in df}


def plot(df: pd.DataFrame, res: dict, path: Path, simulated: bool) -> None:
    import matplotlib.pyplot as plt

    ms = [m for m in METRICS if m in df]
    fig, axes = plt.subplots(1, len(ms), figsize=(3.1 * len(ms), 3.6))
    labels = {"task_time_s": "Task time (s)", "errors": "Errors", "sus": "SUS (0-100)", "tlx": "NASA-TLX raw", "trust": "Trust (1-7)"}
    for ax, m in zip(axes, ms, strict=True):
        w = df.pivot(index="participant", columns="condition", values=m)
        for _, r in w.iterrows():
            ax.plot([0, 1], [r["A"], r["B"]], color="#94a3b8", lw=1, alpha=.7, zorder=1)
        ax.scatter(np.zeros(len(w)), w["A"], color="#2563eb", zorder=2, s=22, label="A context-aware")
        ax.scatter(np.ones(len(w)), w["B"], color="#64748b", zorder=2, s=22, label="B plain chatbot")
        ax.boxplot([w["A"], w["B"]], positions=[0, 1], widths=.35, showfliers=False, medianprops={"color": "#0f172a"})
        ax.set_xticks([0, 1], ["A", "B"])
        r = res[m]
        ax.set_title(f"{labels[m]}\n{r['effect']}={r['effect_size']:.2f}, p={r['p']:.3f}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, loc="upper right", frameon=False)
    if simulated:
        fig.suptitle("SIMULATED DATA - analysis pipeline check, not study results", color="#b91c1c", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks")
    ap.add_argument("--questionnaires", help="CSV with participant, condition, sus, tlx, trust")
    ap.add_argument("--simulate", type=int)
    ap.add_argument("--out", default="results/study")
    a = ap.parse_args(argv)
    if a.simulate:
        df, sim = simulate(a.simulate), True
    else:
        df = pd.read_csv(a.tasks)
        if a.questionnaires:
            df = df.merge(pd.read_csv(a.questionnaires), on=["participant", "condition"], how="left")
        sim = False
    res = analyze(df)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "simulated_" if sim else ""
    (out / f"{tag}analysis.json").write_text(json.dumps({"simulated": sim, "results": res}, indent=2))
    plot(df, res, out / f"{tag}paired_comparison.png", sim)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
