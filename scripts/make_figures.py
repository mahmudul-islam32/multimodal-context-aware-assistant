"""Generate README figures and animations into docs/assets/.

    python -m mcaa.evaluation          # writes results/metrics.json first
    python scripts/make_figures.py

Colours: validated reference palette (categorical slots in fixed order,
single-hue sequential ramp for magnitude). Every series is direct-labelled
or in a legend so identity never relies on colour alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from mcaa.config import load_context_config
from mcaa.context import ContextEngine
from mcaa.har.models import FeatureModel
from mcaa.sensors import detect_drops, resample
from mcaa.sensors.preprocess import preprocess_block, separate_gravity, sliding_windows
from mcaa.sensors.synthetic import (
    ACTIVITY_TO_TASK,
    SubjectProfile,
    generate_segment,
    make_dataset,
    scripted_session,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
STATE_COL = {"IDLE": S1, "HANDS_BUSY": S2, "MOVING": S3, "NOISY": S4}
SEQ = LinearSegmentedColormap.from_list("seq", ["#f3f7fd", "#bcd5f3", "#6aa3e6", "#2a78d6", "#174a8c"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
    "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- 1 signal pipeline
def fig_signal():
    prof = SubjectProfile(3, seed=5)
    seg = np.vstack([generate_segment("STANDING", 3, prof), generate_segment("WALKING", 4, prof)])
    t50 = np.arange(len(seg)) / 50
    # simulate a jittery 20 Hz phone stream with a dropout
    rng = np.random.default_rng(0)
    t20 = np.arange(0, t50[-1], 1 / 20) + rng.normal(0, 0.004, int(np.ceil(t50[-1] * 20)))
    t20 = np.sort(t20[(t20 > 0) & (t20 < t50[-1])])
    t20 = t20[(t20 < 4.2) | (t20 > 4.65)]
    x20 = np.column_stack([np.interp(t20, t50, seg[:, c]) for c in range(6)]) + rng.normal(0, 0.03, (len(t20), 6))
    tr, xr = resample(t20, x20, 50)
    xf = preprocess_block(xr)
    body, grav = separate_gravity(xf[:, :3])
    drops = detect_drops(t20, 20)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True)
    ax = axes[0]
    ax.plot(t20, x20[:, 0], "o", ms=3, color=MUTED, label="raw phone samples (≈20 Hz, jitter)")
    for a, b, n in drops:
        ax.axvspan(a, b, color=S4, alpha=0.25, lw=0)
        ax.text((a + b) / 2, ax.get_ylim()[1] if False else 1.45, f"drop: {n} samples", ha="center", fontsize=8, color=INK2)
    ax.set_title("1  Raw stream: irregular timestamps, a dropout is detected and reported")
    ax.set_ylabel("acc x (g)")
    ax.set_ylim(0.5, 1.55)
    ax.legend(loc="lower left", fontsize=8)
    ax = axes[1]
    ax.plot(tr, xr[:, 0], color=MUTED, lw=1, label="resampled to 50 Hz")
    ax.plot(tr, xf[:, 0], color=S1, label="low-pass 20 Hz (Butterworth, zero phase)")
    ax.set_ylabel("acc x (g)")
    ax.set_title("2  Resampled onto a uniform 50 Hz grid and filtered")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(0.5, 1.55)
    ax = axes[2]
    ax.plot(tr, grav[:, 0], color=S2, label="gravity (< 0.3 Hz)")
    ax.plot(tr, body[:, 0], color=S3, label="body acceleration")
    for i, s in enumerate(range(0, len(tr) - 128 + 1, 64)):
        y = -0.42 - 0.07 * (i % 2)
        ax.plot([tr[s] + 0.05, tr[s + 127] - 0.05], [y, y], color=INK2, lw=2, solid_capstyle="round")
    ax.text(tr[0], -0.62, "windows: 128 samples (2.56 s), 50 % overlap → HAR model", fontsize=8, color=INK2)
    ax.set_ylim(-0.7, 1.6)
    ax.set_ylabel("g")
    ax.set_xlabel("time (s)")
    ax.set_title("3  Gravity / body split and sliding windows")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "signal_pipeline.png")


# ---------------------------------------------------------------- 2 HAR
def fig_har(m):
    h = m["har"]
    rf = h["rf"]
    labels = [lab.replace("WALKING_", "WALK_").replace("_", " ").lower() for lab in rf["labels"]]
    cm = np.array(rf["confusion"], float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    a1.grid(False)
    im = a1.imshow(cmn, cmap=SEQ, vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cmn[i, j] >= 0.01:
                a1.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.55 else INK)
    a1.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    a1.set_yticks(range(len(labels)), labels)
    a1.set_xlabel("predicted")
    a1.set_ylabel("true")
    a1.set_title(f"Random forest, leave-subjects-out (macro-F1 {rf['macro_f1']:.3f})")
    fig.colorbar(im, ax=a1, fraction=0.046, label="row-normalised share")
    # per-class F1, two series
    cls = rf["labels"]
    y = np.arange(len(cls))
    has_cnn = isinstance(h.get("cnn"), dict)
    hgt = 0.38 if has_cnn else 0.6
    a2.barh(y - (hgt / 2 if has_cnn else 0), [rf["per_class_f1"][c] for c in cls], height=hgt - 0.04, color=S1, label=f"features + RF ({rf['macro_f1']:.3f})")
    if has_cnn:
        cn = h["cnn"]
        a2.barh(y + hgt / 2, [cn["per_class_f1"][c] for c in cls], height=hgt - 0.04, color=S2, label=f"1D-CNN ({cn['macro_f1']:.3f})")
    a2.set_yticks(y, labels)
    a2.invert_yaxis()
    a2.set_xlim(0.5, 1.0)
    a2.set_xlabel("F1 per class")
    a2.set_title("Per-class F1 (legend: macro-F1)")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
    a2.grid(axis="y", visible=False)
    fig.suptitle(f"Activity recognition on SYNTHETIC IMU data ({h['n_subjects']} simulated subjects, {h['n_windows']} windows)", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    save(fig, "har_results.png")


# ---------------------------------------------------------------- 3 context timeline (+ GIF)
def context_run():
    script = [("STANDING", 20), ("WALKING", 20), ("WORKING", 40), ("WALKING", 15), ("STANDING", 20)]
    t, x, labels = scripted_session(script)
    X, y, _ = make_dataset(n_subjects=6, seconds_per_activity=25)
    model = FeatureModel(n_estimators=120).fit(X, y)
    clock = {"t": 0.0}
    eng = ContextEngine(load_context_config(), clock=lambda: clock["t"])
    eng.touch()
    W = sliding_windows(preprocess_block(x))
    rows = []
    for i, w in enumerate(W):
        clock["t"] = (i * 64 + 128) / 50
        pr = model.predict_proba(w[None])[0]
        proba = dict(zip(model.classes_, pr, strict=True))
        agg = {"idle": 0, "moving": 0, "hands_busy": 0}
        for c, p in proba.items():
            agg[ACTIVITY_TO_TASK[c]] += p
        raw = {"idle": "IDLE", "moving": "MOVING", "hands_busy": "HANDS_BUSY"}[max(agg, key=agg.get)]
        st = eng.update_activity(proba)
        truth = {"idle": "IDLE", "moving": "MOVING", "hands_busy": "HANDS_BUSY"}[ACTIVITY_TO_TASK[labels[i * 64 + 127]]]
        rows.append({"t": clock["t"], **agg, "raw": raw, "state": st.state, "truth": truth})
    return t, x, rows, script


def _bands(ax, ts, states, y, h):
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            ax.broken_barh([(ts[start] - 0.64, ts[i - 1] - ts[start] + 1.28)], (y, h), color=STATE_COL[states[start]], edgecolor=SURFACE, lw=1)
            start = i


def draw_context(fig, axes, t, x, rows, upto=None):
    ts = np.array([r["t"] for r in rows])
    n = len(rows) if upto is None else upto
    a0, a1, a2 = axes
    for a in axes:
        a.cla()
    mag = np.linalg.norm(x[:, 3:], axis=1)
    k = int(ts[n - 1] * 50) if n else 0
    a0.plot(t[:k], mag[:k], color=MUTED, lw=0.8)
    a0.set_ylabel("|gyro| (rad/s)")
    a0.set_title("Wrist IMU (simulated session: stand → walk → manual work → walk → stand)")
    a0.set_xlim(0, t[-1])
    a0.set_ylim(0, max(2.0, mag.max() * 1.05))
    for key, col, lab in (("idle", S1, "p(idle)"), ("moving", S3, "p(moving)"), ("hands_busy", S2, "p(hands busy)")):
        a1.plot(ts[:n], [r[key] for r in rows[:n]], color=col, label=lab)
    a1.set_ylim(-0.03, 1.05)
    a1.set_xlim(0, t[-1])
    a1.set_ylabel("probability")
    a1.set_title("HAR model output, aggregated to task-relevant groups")
    a1.legend(loc="center", bbox_to_anchor=(0.5, 0.5), ncol=3, fontsize=8, facecolor=SURFACE, frameon=True, edgecolor=SURFACE)
    a2.grid(False)
    for yy, key in ((2, "truth"), (1, "raw"), (0, "state")):
        _bands(a2, ts[:n], [r[key] for r in rows[:n]], yy + 0.1, 0.8)
    a2.set_yticks([0.5, 1.5, 2.5], ["context engine\n(hysteresis)", "raw classifier\n(argmax)", "ground truth"])
    a2.set_ylim(0, 3)
    a2.set_xlim(0, t[-1])
    a2.set_xlabel("time (s)")
    a2.set_title("Context state: one explicit, debounced state drives the output policy")
    a2.legend(handles=[Patch(color=c, label=s) for s, c in STATE_COL.items() if s != "NOISY"], loc="upper center",
              bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=9)


def fig_context(t, x, rows):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=False, gridspec_kw={"height_ratios": [1, 1.2, 1.1]})
    draw_context(fig, axes, t, x, rows)
    fig.tight_layout()
    save(fig, "context_timeline.png")


def gif_context(t, x, rows):
    frames = []
    fig, axes = plt.subplots(3, 1, figsize=(8, 5.8), gridspec_kw={"height_ratios": [1, 1.2, 1.1]})
    for n in list(range(2, len(rows), 2)) + [len(rows)] * 6:
        draw_context(fig, axes, t, x, rows, n)
        st = rows[n - 1]["state"]
        fig.suptitle(f"t = {rows[n - 1]['t']:5.1f} s    context: {st}", x=0.02, ha="left", fontsize=12, weight="bold", color=INK)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(OUT / "context_engine.gif", frames, duration=0.12, loop=0)
    print("wrote", OUT / "context_engine.gif")


# ---------------------------------------------------------------- 4 RAG + safety
def fig_rag(m):
    r, _s = m["rag"], m["security"]
    metrics = [("recall_at_1", "Recall@1"), ("recall_at_4", "Recall@4"), ("answer_accuracy", "Answer accuracy"),
               ("faithfulness", "Faithfulness"), ("oos_refusal_rate", "Out-of-scope refused")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    xs = np.arange(len(metrics))
    for k, (scn, col) in enumerate((("technician", S1), ("care_worker", S2))):
        vals = [r[scn][key] for key, _ in metrics]
        bars = a1.bar(xs + (k - 0.5) * 0.38, vals, width=0.36, color=col, label=f"{scn.replace('_', ' ')} (n={r[scn]['n_in_scope']} in / {r[scn]['n_out_of_scope']} out)")
        for b, v in zip(bars, vals, strict=True):
            a1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
    a1.set_xticks(xs, [lab.replace(" ", "\n", 1) if len(lab) > 10 else lab for _, lab in metrics])
    a1.set_ylim(0, 1.12)
    a1.set_title("Gold question set (higher is better)")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    a1.grid(axis="x", visible=False)
    # refusal-gate margin: retrieval confidence of in-scope vs out-of-scope questions
    import yaml

    from mcaa.config import load_scenario
    from mcaa.rag import DocumentStore, RAGPipeline

    gold = yaml.safe_load((ROOT / "eval" / "gold_questions.yaml").read_text())
    rng = np.random.default_rng(0)
    for row, scn in enumerate(("technician", "care_worker")):
        sc = load_scenario(scn)
        st = DocumentStore()
        st.ingest_dir(sc.docs_path())
        p = RAGPipeline(st, sc.escalation)
        for kind, qs, col, mk in (("in scope", [i["q"] for i in gold[scn]["in_scope"]], S1, "o"),
                                  ("out of scope", gold[scn]["out_of_scope"], S2, "D")):
            hits = [p.retriever.search(q, k=4) for q in qs]
            cov = [h[0].coverage if h else 0.0 for h in hits]
            yy = row + (0.12 if kind == "in scope" else -0.12) + rng.uniform(-0.05, 0.05, len(cov))
            a2.scatter(cov, yy, s=36, color=col, marker=mk, edgecolor=SURFACE, linewidth=1, zorder=3,
                       label=kind if row == 0 else None)
    a2.axvline(0.5, color=INK2, lw=1, ls="--")
    a2.text(0.51, 1.42, "refuse below 0.5", fontsize=8, color=INK2)
    a2.set_yticks([0, 1], ["technician", "care worker"])
    a2.set_ylim(-0.5, 1.55)
    a2.set_xlim(-0.03, 1.03)
    a2.set_xlabel("query-term coverage of best chunk")
    a2.set_title("Refusal gate: in-scope vs out-of-scope questions")
    a2.legend(loc="lower right", fontsize=8, facecolor=SURFACE, frameon=True, edgecolor=SURFACE)
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "rag_safety.png")


if __name__ == "__main__":
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    fig_signal()
    fig_har(metrics)
    fig_rag(metrics)
    t, x, rows = context_run()[:3]
    fig_context(t, x, rows)
    gif_context(t, x, rows)
