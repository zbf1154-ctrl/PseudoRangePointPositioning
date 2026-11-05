import csv
import json

import numpy as np

from spp.export import format_summary, save_outputs, write_solutions_csv
from spp.pipeline import RunResult


def test_statistics_keys(short_result):
    stats = short_result.statistics()
    expected = {
        "epochs",
        "mean_3d",
        "median_3d",
        "p95_3d",
        "rms_e",
        "rms_n",
        "rms_u",
        "mean_residual",
        "mean_pdop",
    }
    assert expected.issubset(stats)
    assert stats["epochs"] == 11.0
    assert np.isfinite(stats["mean_3d"])


def test_to_records(short_result):
    records = short_result.to_records()
    assert len(records) == 11
    first = records[0]
    assert {"time", "x", "y", "z", "east", "north", "up", "error_3d"}.issubset(first)
    assert first["time"] == "2020-01-03 00:00:00"


def test_save_outputs(short_result, tmp_path):
    outputs = save_outputs(short_result, str(tmp_path), prefix="demo")
    assert set(outputs) == {"solutions", "summary"}
    assert outputs["solutions"].is_file()
    assert outputs["summary"].is_file()

    with outputs["solutions"].open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 11
    assert rows[0]["mode"] == "dual"

    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["station"] == "ABMF"
    assert summary["statistics"]["epochs"] == 11.0


def test_write_solutions_csv(short_result, tmp_path):
    path = write_solutions_csv(tmp_path / "only.csv", short_result)
    assert path.is_file()
    assert path.read_text(encoding="utf-8-sig").startswith("time,x,y,z")


def test_format_summary(short_result):
    text = format_summary(short_result)
    assert "ABMF" in text
    assert "3D 误差" in text
    assert "dual=11" in text


def test_run_result_defaults():
    empty = RunResult()
    assert empty.solved_epochs == 0
    assert empty.statistics() == {}
    assert empty.to_records() == []
