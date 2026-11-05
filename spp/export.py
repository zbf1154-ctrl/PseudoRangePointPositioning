"""结果导出（CSV / JSON）。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional

from .pipeline import RunResult

SOLUTION_FIELDS = [
    "time",
    "x",
    "y",
    "z",
    "latitude",
    "longitude",
    "height",
    "east",
    "north",
    "up",
    "dx",
    "dy",
    "dz",
    "error_3d",
    "clock_bias_m",
    "clock_bias_s",
    "residual_rms",
    "pdop",
    "satellites",
    "iterations",
    "mode",
]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_solutions_csv(
    path: str,
    result: RunResult,
    use_truth: bool = True,
) -> Path:
    """将逐历元解算结果写入 CSV。"""
    destination = Path(path)
    _ensure_parent(destination)
    records = result.to_records(use_truth=use_truth)
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOLUTION_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in SOLUTION_FIELDS})
    return destination


def write_summary_json(
    path: str,
    result: RunResult,
    use_truth: bool = True,
) -> Path:
    """将配置、统计量与失败历元写入 JSON。"""
    destination = Path(path)
    _ensure_parent(destination)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(result.summary(use_truth=use_truth), handle, ensure_ascii=False, indent=2)
    return destination


def write_failures_csv(path: str, result: RunResult) -> Path:
    """将无法解算的历元写入 CSV。"""
    destination = Path(path)
    _ensure_parent(destination)
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "reason", "satellites"])
        for failure in result.failures:
            writer.writerow([failure.time.isoformat(sep=" "), failure.reason, failure.satellites])
    return destination


def save_outputs(
    result: RunResult,
    output_dir: str,
    prefix: str = "spp",
    use_truth: bool = True,
    write_failures: bool = True,
) -> Dict[str, Path]:
    """保存 CSV/JSON 结果，返回文件路径字典。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "solutions": write_solutions_csv(directory / f"{prefix}_solutions.csv", result, use_truth),
        "summary": write_summary_json(directory / f"{prefix}_summary.json", result, use_truth),
    }
    if write_failures and result.failures:
        outputs["failures"] = write_failures_csv(
            directory / f"{prefix}_failures.csv", result
        )
    return outputs


def format_summary(result: RunResult, use_truth: bool = True) -> str:
    """返回适合终端/GUI 展示的文本摘要。"""
    stats = result.statistics(use_truth=use_truth)
    if not stats:
        return "没有可用的解算结果。"
    modes = ", ".join(f"{key}={value}" for key, value in sorted(result.mode_counts().items()))
    reference = "IGS 参考坐标" if use_truth and result.dataset and result.dataset.truth_position is not None else "先验坐标"
    lines = [
        f"测站            : {result.dataset.station if result.dataset else '—'}",
        f"时间窗          : {result.window.describe() if result.window else '—'}",
        f"有效/总历元     : {int(stats['epochs'])} / {int(stats['total_epochs'])}"
        f"（失败 {int(stats['rejected_epochs'])}）",
        f"解算模式        : {modes or '—'}",
        f"参考坐标        : {reference}",
        f"3D 误差 均值    : {stats['mean_3d']:.2f} m",
        f"3D 误差 中位数  : {stats['median_3d']:.2f} m",
        f"3D 误差 P95     : {stats['p95_3d']:.2f} m",
        f"3D 误差 最大    : {stats['max_3d']:.2f} m",
        f"ENU 均值 (m)    : E {stats['mean_e']:+.2f}  N {stats['mean_n']:+.2f}  U {stats['mean_u']:+.2f}",
        f"ENU RMS  (m)    : E {stats['rms_e']:.2f}  N {stats['rms_n']:.2f}  U {stats['rms_u']:.2f}",
        f"观测残差 RMS    : {stats['mean_residual']:.2f} m",
        f"平均 PDOP       : {stats['mean_pdop']:.2f}",
        f"平均卫星数      : {stats['mean_satellites']:.1f}",
        f"耗时            : {stats['elapsed_seconds']:.2f} s",
    ]
    return "\n".join(lines)
