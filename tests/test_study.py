import pandas as pd
import pytest

from mcaa.study import EventLogger, schedule_for
from mcaa.study.conditions import validate_pid


def test_counterbalanced_schedule():
    seqs = [tuple((b["condition"], b["task_set"]) for b in schedule_for(f"P{i:02d}")) for i in range(1, 5)]
    assert len(set(seqs)) == 4
    firsts = [s[0][0] for s in seqs]
    assert firsts.count("A") == firsts.count("B") == 2
    with pytest.raises(ValueError):
        validate_pid("Anna Schmidt")


def test_logging_export_and_deletion(tmp_path):
    lg = EventLogger(tmp_path / "s.db")
    lg.start_session("P01", "A", "A")
    lg.log("task_start")
    lg.log("query", query="my name is Mr Hansen, how do I lock out?")
    lg.log("answer", refused=True)
    lg.log("task_end")
    lg.end_session()
    q = lg.events("query").payload.iloc[0]
    assert "Hansen" not in q
    paths = lg.export(tmp_path / "out", "csv")
    tasks = pd.read_csv(paths[1])
    assert tasks.refusals.iloc[0] == 1 and tasks.queries.iloc[0] == 1
    pytest.importorskip("pyarrow")
    lg.export(tmp_path / "pq", "parquet")
    assert lg.delete_participant("P01") == 4
    assert lg.events(participant="P01").empty


def test_analysis_runs_on_simulated_data(tmp_path):
    from eval.analysis.analyze_study import analyze, simulate

    res = analyze(simulate(10))
    assert set(res) == {"task_time_s", "errors", "sus", "tlx", "trust"}
    assert res["task_time_s"]["test"] in ("paired t", "Wilcoxon")
