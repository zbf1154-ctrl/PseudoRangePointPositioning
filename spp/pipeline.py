"""数据装载与全流程编排。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .config import (
    ABMF_APPROX_POSITION,
    SolverConfig,
    TimeWindow,
)
from .coords import ecef_offset_to_enu, ecef_to_geodetic
from .models import Dataset, EpochFailure, Solution
from .rinex_clk import read_rinex_clk
from .rinex_obs import read_rinex_obs
from .solver import SppSolver
from .sp3 import build_orbit, read_sp3

Logger = Callable[[str], None]


class PipelineError(RuntimeError):
    """流程执行失败。"""


def _noop(_message: str) -> None:
    return None


@dataclass
class RunResult:
    """一次完整解算的结果。"""

    solutions: List[Solution] = field(default_factory=list)
    failures: List[EpochFailure] = field(default_factory=list)
    dataset: Optional[Dataset] = None
    elapsed_seconds: float = 0.0
    config: SolverConfig = field(default_factory=SolverConfig)
    window: Optional[TimeWindow] = None

    @property
    def solved_epochs(self) -> int:
        return len(self.solutions)

    @property
    def total_epochs(self) -> int:
        return self.dataset.epoch_count if self.dataset is not None else 0

    @property
    def rejected_epochs(self) -> int:
        return len(self.failures)

    def offsets_ecef(self, use_truth: bool = True) -> np.ndarray:
        if not self.solutions or self.dataset is None:
            return np.zeros((0, 3))
        reference = self.dataset.truth_position if (
            use_truth and self.dataset.truth_position is not None
        ) else self.dataset.reference_position
        return np.array(
            [solution.position - np.asarray(reference, dtype=float) for solution in self.solutions]
        )

    def offsets_enu(self, use_truth: bool = True) -> np.ndarray:
        if self.dataset is None:
            return np.zeros((0, 3))
        reference = self.dataset.truth_position if (
            use_truth and self.dataset.truth_position is not None
        ) else self.dataset.reference_position
        latitude, longitude, _ = ecef_to_geodetic(reference)
        offsets = self.offsets_ecef(use_truth=use_truth)
        if offsets.size == 0:
            return np.zeros((0, 3))
        return np.array(
            [ecef_offset_to_enu(offset, latitude, longitude) for offset in offsets]
        )

    def statistics(self, use_truth: bool = True) -> Dict[str, float]:
        """返回精度统计量（单位：米）。"""
        if not self.solutions:
            return {}
        offsets = self.offsets_ecef(use_truth=use_truth)
        distances = np.linalg.norm(offsets, axis=1)
        enu = self.offsets_enu(use_truth=use_truth)
        residuals = np.array([solution.residual_rms for solution in self.solutions], dtype=float)
        pdops = np.array([solution.pdop for solution in self.solutions], dtype=float)
        satellites = np.array([solution.satellites for solution in self.solutions], dtype=float)
        result = {
            "epochs": float(len(self.solutions)),
            "total_epochs": float(self.total_epochs),
            "rejected_epochs": float(self.rejected_epochs),
            "mean_3d": float(np.mean(distances)),
            "median_3d": float(np.median(distances)),
            "p95_3d": float(np.percentile(distances, 95)),
            "max_3d": float(np.max(distances)),
            "rms_e": float(np.sqrt(np.mean(enu[:, 0] ** 2))) if enu.size else float("nan"),
            "rms_n": float(np.sqrt(np.mean(enu[:, 1] ** 2))) if enu.size else float("nan"),
            "rms_u": float(np.sqrt(np.mean(enu[:, 2] ** 2))) if enu.size else float("nan"),
            "mean_e": float(np.mean(enu[:, 0])) if enu.size else float("nan"),
            "mean_n": float(np.mean(enu[:, 1])) if enu.size else float("nan"),
            "mean_u": float(np.mean(enu[:, 2])) if enu.size else float("nan"),
            "mean_residual": float(np.mean(residuals)),
            "mean_pdop": float(np.mean(pdops)),
            "mean_satellites": float(np.mean(satellites)),
            "elapsed_seconds": float(self.elapsed_seconds),
        }
        return result

    def mode_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for solution in self.solutions:
            counts[solution.mode] = counts.get(solution.mode, 0) + 1
        return counts

    def to_records(self, use_truth: bool = True) -> List[dict]:
        if self.dataset is None:
            return []
        reference = self.dataset.truth_position if (
            use_truth and self.dataset.truth_position is not None
        ) else self.dataset.reference_position
        latitude, longitude, _ = ecef_to_geodetic(reference)
        records: List[dict] = []
        for solution in self.solutions:
            record = solution.to_dict(reference)
            enu = ecef_offset_to_enu(solution.position - reference, latitude, longitude)
            record.update(
                {
                    "east": float(enu[0]),
                    "north": float(enu[1]),
                    "up": float(enu[2]),
                }
            )
            records.append(record)
        return records

    def summary(self, use_truth: bool = True) -> dict:
        dataset = self.dataset
        return {
            "station": dataset.station if dataset else "",
            "window": self.window.describe() if self.window else "",
            "sources": dict(dataset.source_files) if dataset else {},
            "config": self.config.as_dict(),
            "statistics": self.statistics(use_truth=use_truth),
            "modes": self.mode_counts(),
            "failures": [
                {"time": failure.time.isoformat(sep=" "), "reason": failure.reason,
                 "satellites": failure.satellites}
                for failure in self.failures
            ],
        }


def load_dataset(
    sp3_path: str,
    clk_path: str,
    obs_path: str,
    window: Optional[TimeWindow] = None,
    logger: Optional[Logger] = None,
    station: Optional[str] = None,
) -> Dataset:
    """装载三类产品并完成一致性检查。

    SP3 与钟差文件按全时段装载，以满足信号发射时刻的插值需求；
    观测文件按时间窗过滤。
    """
    log = logger or _noop
    for label, path in (("SP3", sp3_path), ("钟差", clk_path), ("观测", obs_path)):
        if not Path(path).is_file():
            raise PipelineError(f"{label}文件不存在：{path}")

    log("解析观测文件 …")
    observation_file = read_rinex_obs(obs_path, window)
    log(
        f"  历元 {len(observation_file.epochs)} 个，"
        f"观测类型 {len(observation_file.observation_types)} 类"
    )

    log("解析钟差文件 …")
    clock_file = read_rinex_clk(clk_path, None)
    log(
        f"  历元 {len(clock_file.epochs)} 个，"
        f"卫星钟差 {len(clock_file.epochs[0].satellites)} 颗"
    )

    log("解析精密轨道文件 …")
    sp3_file = read_sp3(sp3_path, None)
    orbit = build_orbit(sp3_file)
    log(
        f"  历元 {len(orbit.times)} 个，卫星 {len(orbit.satellites)} 颗，"
        f"采样 {orbit.interval_seconds:.0f} s"
    )

    from .models import SatelliteClockSeries

    clocks = SatelliteClockSeries(clock_file.epochs)

    station_name = (
        station
        or observation_file.marker_name
        or next(iter(clock_file.station_positions), "")
    )
    reference = observation_file.approx_position
    if reference is None:
        reference = clock_file.station_position(station_name)
    if reference is None:
        reference = np.array(ABMF_APPROX_POSITION, dtype=float)
        log("  观测文件缺少近似坐标，使用内置先验坐标")

    truth = clock_file.station_position(station_name)
    if truth is None:
        truth = np.array(ABMF_APPROX_POSITION, dtype=float)

    coefficients = None
    if (
        observation_file.ionosphere_alpha is not None
        and observation_file.ionosphere_beta is not None
    ):
        coefficients = (
            observation_file.ionosphere_alpha,
            observation_file.ionosphere_beta,
        )

    dataset = Dataset(
        observations=observation_file.epochs,
        clocks=clocks,
        orbit=orbit,
        reference_position=np.asarray(reference, dtype=float),
        station=station_name,
        ionosphere_coefficients=coefficients,
        source_files={"sp3": str(sp3_path), "clk": str(clk_path), "obs": str(obs_path)},
        truth_position=np.asarray(truth, dtype=float),
        window=window,
    )

    _check_coverage(dataset, log)
    return dataset


def _check_coverage(dataset: Dataset, log: Logger) -> None:
    if not dataset.observations:
        return
    first = dataset.observations[0].time
    last = dataset.observations[-1].time
    if dataset.orbit.start > first or dataset.orbit.end < last + timedelta(minutes=5):
        log("警告：SP3 轨道时间范围未完全覆盖观测时段")
    if dataset.clocks.start > first or dataset.clocks.end < last:
        log("警告：钟差时间范围未完全覆盖观测时段")


def run(
    sp3_path: str,
    clk_path: str,
    obs_path: str,
    window: Optional[TimeWindow] = None,
    config: Optional[SolverConfig] = None,
    logger: Optional[Logger] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    station: Optional[str] = None,
) -> RunResult:
    """执行「装载 → 解算」全流程。"""
    log = logger or _noop
    started = time.perf_counter()
    dataset = load_dataset(sp3_path, clk_path, obs_path, window, log, station=station)
    solver = SppSolver(dataset, config)
    log(f"开始解算，共 {len(dataset.observations)} 个历元 …")
    solutions = solver.solve(progress=progress, should_stop=should_stop)
    elapsed = time.perf_counter() - started

    if not solutions:
        reason = solver.failures[0].reason if solver.failures else "未知原因"
        raise PipelineError(
            f"所有历元均无法完成解算（{reason}），请检查数据完整性与时间窗设置"
        )

    log(
        f"解算完成：{len(solutions)}/{len(dataset.observations)} 个历元有效，"
        f"耗时 {elapsed:.2f} s"
    )
    return RunResult(
        solutions=solutions,
        failures=solver.failures,
        dataset=dataset,
        elapsed_seconds=elapsed,
        config=config or SolverConfig(),
        window=window,
    )
