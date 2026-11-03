"""F-15: offline evaluation harness. One command, one report.

    mcaa-eval                 # full run -> results/metrics.json + results/REPORT.md
    mcaa-eval --quick         # CI-sized run
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import yaml

from . import __version__
from .config import ROOT, load_context_config, load_policy_config, load_scenario
from .context import ContextEngine
from .har.models import FeatureModel, evaluate_loso, single_window_latency_ms
from .policy import OutputPolicy
from .rag import DocumentStore, RAGPipeline, redact
from .rag.guardrails import InjectionDetector
from .sensors import detect_drops, resample
from .sensors.preprocess import preprocess_block, sliding_windows
from .sensors.synthetic import ACTIVITIES, ACTIVITY_TO_TASK, make_dataset, scripted_session
from .speech.wer import corpus_wer

EVAL = ROOT / "eval"


def _pipeline(scn: str) -> RAGPipeline:
    sc = load_scenario(scn)
    st = DocumentStore()
    st.ingest_dir(sc.docs_path())
    return RAGPipeline(st, sc.escalation)


def eval_rag(gold: dict) -> dict:
    out = {}
    lat = []
    for scn, items in gold.items():
        p = _pipeline(scn)
        rec1 = rec4 = correct = refused_in = 0
        faith = []
        rows = []
        for it in items["in_scope"]:
            hits = p.retriever.search(it["q"], k=4)
            ok = [h.chunk.doc_id == it["doc"] and h.chunk.section.startswith(it["section"]) for h in hits]
            rec1 += bool(ok[:1] and ok[0])
            rec4 += any(ok)
            a = p.ask(it["q"])
            lat.append(a.latency_ms)
            refused_in += a.refused
            hit_fact = (not a.refused) and it["fact"].lower() in a.answer.lower()
            correct += hit_fact
            if not a.refused:
                faith.append(a.faithfulness)
            rows.append({"q": it["q"], "refused": a.refused, "correct": hit_fact})
        n = len(items["in_scope"])
        oos = [p.ask(q) for q in items["out_of_scope"]]
        out[scn] = {
            "n_in_scope": n,
            "n_out_of_scope": len(oos),
            "recall_at_1": rec1 / n,
            "recall_at_4": rec4 / n,
            "answer_accuracy": correct / n,
            "faithfulness": float(np.mean(faith)) if faith else 0.0,
            "false_refusal_rate": refused_in / n,
            "oos_refusal_rate": sum(a.refused for a in oos) / len(oos),
            "failures": [r for r in rows if not r["correct"]],
        }
    out["latency_ms_p50"] = float(np.percentile(lat, 50))
    out["latency_ms_p95"] = float(np.percentile(lat, 95))
    return out


def eval_security(red: dict) -> dict:
    p = _pipeline("technician")
    det = InjectionDetector()
    direct = [p.ask(q) for q in red["direct"]]
    succeeded = [a.query for a in direct if not (a.blocked or a.refused)]
    doc_res = []
    for d in red["document_attacks"]:
        pp = _pipeline(d["scenario"])
        quarantined = d["doc_id"] not in pp.store.documents
        ans = pp.ask(d["probe"])
        doc_res.append({"doc_id": d["doc_id"], "quarantined": quarantined, "leaked": d["forbidden"].lower() in ans.answer.lower()})
    fp = [q for q in red["benign_controls"] if det.scan(q).flagged]
    tp = fn = fpos = 0
    for item in red["pii"]:
        _, found = redact(item["text"])
        exp = set(item["expect"])
        tp += len(exp & set(found))
        fn += len(exp - set(found))
        fpos += len(set(found) - exp)
    return {
        "direct_attacks": len(direct),
        "attack_success_rate": len(succeeded) / len(direct),
        "succeeded": succeeded,
        "blocked_by_input_guard": sum(a.blocked for a in direct),
        "document_attacks": doc_res,
        "benign_false_positive_rate": len(fp) / len(red["benign_controls"]),
        "pii_recall": tp / max(1, tp + fn),
        "pii_precision": tp / max(1, tp + fpos),
    }


def eval_har(quick: bool, seed: int = 42) -> dict:
    n_sub, sec = (6, 20) if quick else (12, 45)
    X, y, g = make_dataset(n_subjects=n_sub, seconds_per_activity=sec, seed=seed)
    res = {"dataset": "synthetic (see mcaa.sensors.synthetic)", "n_windows": len(y), "n_subjects": n_sub}
    rf = evaluate_loso(lambda: FeatureModel(n_estimators=100 if quick else 200, seed=seed), X, y, g, n_splits=3 if quick else 4)
    m = FeatureModel(n_estimators=100, seed=seed).fit(X, y)
    res["rf"] = {**rf.to_dict(), "single_window_latency_ms": single_window_latency_ms(m, X[0])}
    try:
        from .har.cnn import CNNModel

        cnn = evaluate_loso(lambda: CNNModel(epochs=6 if quick else 15, seed=seed), X, y, g, n_splits=3 if quick else 4)
        cm = CNNModel(epochs=4, seed=seed).fit(X, y)
        res["cnn"] = {**cnn.to_dict(), "single_window_latency_ms": single_window_latency_ms(cm, X[0])}
    except ImportError:
        res["cnn"] = "skipped (torch not installed)"
    # calibration: model without WORKING, then calibrate with 30 s of the target user
    base = [a for a in ACTIVITIES if a != "WORKING"]
    Xb, yb, _ = make_dataset(n_subjects=n_sub, seconds_per_activity=sec, activities=tuple(base), seed=seed)
    Xt, yt, _ = make_dataset(n_subjects=1, seconds_per_activity=30, seed=seed + 100)
    Xc, yc, _ = make_dataset(n_subjects=1, seconds_per_activity=30, seed=seed + 100)  # calibration recording
    Xt, yt = Xt[1::2], yt[1::2]
    Xc, yc = Xc[::2], yc[::2]
    fm = FeatureModel(n_estimators=100, seed=seed).fit(Xb, yb)
    task_true = np.array([ACTIVITY_TO_TASK[a] for a in yt])
    before = np.mean(np.array([ACTIVITY_TO_TASK[a] for a in fm.predict(Xt)]) == task_true)
    fm.calibrate(Xc, yc)
    after = np.mean(np.array([ACTIVITY_TO_TASK[a] for a in fm.predict(Xt)]) == task_true)
    res["calibration"] = {"task_state_accuracy_before": float(before), "task_state_accuracy_after": float(after), "calibration_seconds": 30 * len(ACTIVITIES)}
    return res


def eval_sensors() -> dict:
    fs_src = 20.0
    t = np.arange(0, 20, 1 / fs_src)
    x = np.sin(2 * np.pi * 1.5 * t)[:, None]
    tr, xr = resample(t, x, 50.0)
    err = float(np.abs(xr[:, 0] - np.sin(2 * np.pi * 1.5 * tr)).max())
    t50 = np.arange(0, 10, 0.02)
    t_drop = np.delete(t50, np.r_[100:110, 300:303])
    drops = detect_drops(t_drop, 50.0)
    return {"resample_20_to_50hz_max_abs_err": err, "drops_detected": [d[2] for d in drops], "drops_expected": [10, 3]}


def eval_context() -> dict:
    """Run a scripted session through HAR + context engine; measure flicker and accuracy."""
    script = [("STANDING", 20), ("WALKING", 20), ("WORKING", 40), ("WALKING", 15), ("STANDING", 20)]
    t, x, labels = scripted_session(script)
    model = FeatureModel(n_estimators=100).fit(*make_dataset(n_subjects=6, seconds_per_activity=25)[:2])
    clock = {"t": 0.0}
    eng = ContextEngine(load_context_config(), clock=lambda: clock["t"])
    eng.touch()
    xf = preprocess_block(x)
    W = sliding_windows(xf)
    states, truth = [], []
    want = {"idle": "IDLE", "moving": "MOVING", "hands_busy": "HANDS_BUSY"}
    raw_changes, prev_raw = 0, None
    for i, w in enumerate(W):
        clock["t"] = (i * 64 + 128) / 50.0
        pr = model.predict_proba(w[None])[0]
        raw = max(zip(pr, model.classes_, strict=True))[1]
        raw_changes += prev_raw is not None and ACTIVITY_TO_TASK[raw] != ACTIVITY_TO_TASK[prev_raw]
        prev_raw = raw
        st = eng.update_activity(dict(zip(model.classes_, pr, strict=True)))
        states.append(st.state)
        truth.append(want[ACTIVITY_TO_TASK[labels[min(i * 64 + 127, len(labels) - 1)]]])
    changes = sum(a != b for a, b in zip(states, states[1:], strict=False))
    dwell = [h.since for h in eng.history]
    gaps = np.diff(dwell) if len(dwell) > 1 else np.array([np.inf])
    return {
        "windows": len(states),
        "true_segments": len(script),
        "state_changes": int(changes),
        "raw_classifier_changes": int(raw_changes),
        "min_seconds_between_changes": float(gaps.min()) if len(gaps) else None,
        "agreement_with_truth": float(np.mean(np.array(states) == np.array(truth))),
        "timeline": {"t": [(i * 64 + 128) / 50.0 for i in range(len(states))], "state": states, "truth": truth},
    }


def eval_policy(gold: dict) -> dict:
    pol = OutputPolicy(load_policy_config())
    p = _pipeline("technician")
    limits = {s: r.max_voice_words for s, r in pol.cfg.states.items()}
    viol = 0
    n = 0
    for it in gold["technician"]["in_scope"]:
        a = p.ask(it["q"])
        for s in limits:
            plan = pol.plan(s, a.answer, a.short_answer, a.citations, a.refused)
            words = len(plan.voice_text.split())
            n += 1
            # citation tag ("From the ..., section ...") may add up to 8 words
            viol += words > limits[s] + 8 or (s == "NOISY" and plan.speak)
    return {"plans_checked": n, "limit_violations": viol, "policy_version": pol.cfg.version}


def eval_asr() -> dict:
    """WER on recorded test clips (eval/asr/*.wav + transcripts.yaml) if a backend is installed."""
    meta = EVAL / "asr" / "transcripts.yaml"
    if not meta.exists():
        return {"status": "skipped: no recorded test set in eval/asr/ (record clips in the target setting)"}
    from .speech.asr import UnavailableASR, decode_wav, get_asr

    asr = get_asr()
    if isinstance(asr, UnavailableASR):
        return {"status": "skipped: no ASR backend installed (pip install '.[speech]')"}
    items = yaml.safe_load(meta.read_text())
    pairs, lat = [], []
    for it in items:
        audio, sr = decode_wav((EVAL / "asr" / it["file"]).read_bytes())
        r = asr.transcribe(audio, sr)
        pairs.append((it["text"], r.text))
        lat.append(r.latency_ms)
    return {"status": "ok", "backend": asr.name, "wer": corpus_wer(pairs), "latency_ms_p50": float(np.median(lat)), "n": len(pairs)}


def to_markdown(m: dict) -> str:
    h, r, s, c = m["har"], m["rag"], m["security"], m["context"]
    lines = [
        f"# Evaluation report (mcaa {m['meta']['version']})",
        "",
        f"Generated {m['meta']['date']} on {m['meta']['machine']}. Quick mode: {m['meta']['quick']}. Seed: 42.",
        "",
        "> HAR numbers are on the **synthetic** IMU generator, not on UCI HAR / WISDM. Run",
        "> `python -m mcaa.data.download --dataset uci-har` and `python -m mcaa.har.train --dataset uci-har` for real-data results.",
        "",
        "## Activity recognition (leave-subjects-out)",
        "",
        "| Model | Macro-F1 | Fold std | Latency / window (ms) |",
        "|---|---|---|---|",
        f"| Features + random forest | {h['rf']['macro_f1']:.3f} | {h['rf']['macro_f1_fold_std']:.3f} | {h['rf']['single_window_latency_ms']:.1f} |",
    ]
    if isinstance(h["cnn"], dict):
        lines.append(f"| 1D-CNN (PyTorch) | {h['cnn']['macro_f1']:.3f} | {h['cnn']['macro_f1_fold_std']:.3f} | {h['cnn']['single_window_latency_ms']:.1f} |")
    cal = h["calibration"]
    lines += [
        "",
        f"Calibration (new class WORKING learned from a short recording): task-state accuracy {cal['task_state_accuracy_before']:.2f} → {cal['task_state_accuracy_after']:.2f}.",
        "",
        "## Verified-docs RAG",
        "",
        "| Scenario | Recall@1 | Recall@4 | Answer accuracy | Faithfulness | False refusals | Out-of-scope refused |",
        "|---|---|---|---|---|---|---|",
    ]
    for scn in ("technician", "care_worker"):
        x = r[scn]
        lines.append(f"| {scn} | {x['recall_at_1']:.2f} | {x['recall_at_4']:.2f} | {x['answer_accuracy']:.2f} | {x['faithfulness']:.2f} | {x['false_refusal_rate']:.2f} | {x['oos_refusal_rate']:.2f} |")
    lines += [
        "",
        f"Answer latency (extractive generator): p50 {r['latency_ms_p50']:.1f} ms, p95 {r['latency_ms_p95']:.1f} ms.",
        "",
        "## Security and privacy",
        "",
        f"- Direct injection attacks: {s['direct_attacks']}, attack success rate **{s['attack_success_rate']:.2f}**",
        "- Document attacks: " + ", ".join(f"{d['doc_id']} quarantined={d['quarantined']} leaked={d['leaked']}" for d in s["document_attacks"]),
        f"- Benign-query false-positive rate: {s['benign_false_positive_rate']:.2f}",
        f"- PII redaction recall {s['pii_recall']:.2f}, precision {s['pii_precision']:.2f}",
        "",
        "## Context engine (scripted session)",
        "",
        f"- {c['true_segments']} true segments; raw classifier task-state flips: {c['raw_classifier_changes']}; engine state changes: {c['state_changes']}",
        f"- Minimum time between changes: {c['min_seconds_between_changes']:.1f} s; agreement with ground truth: {c['agreement_with_truth']:.2f}",
        "",
        "## Output policy",
        "",
        f"- {m['policy']['plans_checked']} plans checked, word-limit violations: {m['policy']['limit_violations']} ({m['policy']['policy_version']})",
        "",
        "## Sensor pipeline",
        "",
        f"- Resampling 20 → 50 Hz max abs error: {m['sensors']['resample_20_to_50hz_max_abs_err']:.4f}",
        f"- Dropped samples detected: {m['sensors']['drops_detected']} (expected {m['sensors']['drops_expected']})",
        "",
        "## Speech",
        "",
        f"- ASR: {m['asr'].get('status')}" + (f", WER {m['asr']['wer']:.3f}" if m["asr"].get("status") == "ok" else ""),
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    t0 = time.time()
    gold = yaml.safe_load((EVAL / "gold_questions.yaml").read_text())
    red = yaml.safe_load((EVAL / "redteam.yaml").read_text())
    m = {
        "meta": {"version": __version__, "date": time.strftime("%Y-%m-%d"), "machine": f"{platform.processor() or platform.machine()} / {platform.system()}", "quick": a.quick},
        "sensors": eval_sensors(),
        "har": eval_har(a.quick),
        "context": eval_context(),
        "rag": eval_rag(gold),
        "security": eval_security(red),
        "policy": eval_policy(gold),
        "asr": eval_asr(),
    }
    m["meta"]["runtime_s"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, indent=2, default=float))
    (out / "REPORT.md").write_text(to_markdown(m))
    print(to_markdown(m))
    return m


if __name__ == "__main__":
    main()
