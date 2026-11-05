"""逐历元伪距单点定位解算。

观测方程
--------
``P = ρ + c·δt_r - c·δt^s + I + T``

其中 ``ρ`` 为接收机与卫星的几何距离，``δt_r`` 为接收机钟差，``δt^s`` 为卫星钟差，
``I``、``T`` 分别为电离层与对流层延迟。线性化后以 Gauss-Newton 迭代求解，
并采用高度角定权与 Huber 型残差剔除提高抗差能力。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import atmosphere
from .config import (
    EARTH_ROTATION_RATE,
    SPEED_OF_LIGHT,
    SolverConfig,
)
from .coords import azimuth_elevation, ecef_to_geodetic
from .interpolate import OrbitInterpolator
from .models import Dataset, EpochFailure, ObservationEpoch, Solution

_INITIAL_TRANSIT_TIME = 0.075  # 初始信号传播时间，s


@dataclass(frozen=True)
class Measurement:
    """单颗卫星的有效伪距观测。"""

    sv: str
    pseudorange: float
    clock_bias_s: float
    mode: str


@dataclass(frozen=True)
class Geometry:
    """卫星几何信息。"""

    satellite: np.ndarray
    transit_time: float
    range: float


class SppSolver:
    """伪距单点定位求解器。"""

    def __init__(
        self,
        dataset: Dataset,
        config: Optional[SolverConfig] = None,
        interpolator: Optional[OrbitInterpolator] = None,
    ) -> None:
        self.dataset = dataset
        self.config = config or SolverConfig()
        self.interpolator = interpolator or OrbitInterpolator(dataset.orbit)
        self.failures: List[EpochFailure] = []
        self.last_error: str = ""

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def solve(
        self,
        progress: Optional[Callable[[int, int], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> List[Solution]:
        """解算全部历元，返回有效结果列表；失败历元记录在 ``failures``。"""
        results: List[Solution] = []
        self.failures = []
        previous: Optional[np.ndarray] = None
        total = len(self.dataset.observations)

        for index, epoch in enumerate(self.dataset.observations):
            if should_stop is not None and should_stop():
                self.failures.append(EpochFailure(epoch.time, "用户中止", epoch.satellite_count))
                break

            initial = previous if previous is not None else self.dataset.reference_position
            self.last_error = ""
            try:
                solution = self.solve_epoch(epoch, initial)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                solution = None
                self.last_error = f"解算异常: {error}"

            if solution is not None:
                results.append(solution)
                previous = solution.position
            else:
                self.failures.append(
                    EpochFailure(
                        time=epoch.time,
                        reason=self.last_error or "历元不可解",
                        satellites=epoch.satellite_count,
                    )
                )
                previous = None

            if progress is not None:
                progress(index + 1, total)

        return results

    def solve_epoch(
        self,
        epoch: ObservationEpoch,
        initial: Optional[np.ndarray] = None,
    ) -> Optional[Solution]:
        """解算单个历元，无法解算时返回 ``None``。"""
        mode = self._choose_mode(epoch)
        if mode is None:
            self.last_error = "可用卫星数不足"
            return None

        measurements = self._measurements(epoch, mode)
        if len(measurements) < self.config.min_satellites:
            self.last_error = f"{mode} 模式有效观测不足（{len(measurements)}）"
            return None

        solution = self._solve_mode(epoch, measurements, mode, initial)
        if solution is None and self.config.frequency_mode == "auto" and mode == "dual":
            # 双频失败时自动退化为单频
            single = self._measurements(epoch, "single")
            if len(single) >= self.config.min_satellites:
                solution = self._solve_mode(epoch, single, "single", initial)
        return solution

    # ------------------------------------------------------------------
    # 模式与观测值选择
    # ------------------------------------------------------------------
    def _choose_mode(self, epoch: ObservationEpoch) -> Optional[str]:
        config = self.config
        single_count = sum(
            1 for sv in epoch.satellites() if epoch.pseudorange(sv) is not None
        )
        dual_count = sum(
            1 for sv in epoch.satellites() if epoch.ionosphere_free(sv) is not None
        )

        if config.frequency_mode == "single":
            return "single" if single_count >= config.min_satellites else None
        if config.frequency_mode == "dual":
            return "dual" if dual_count >= config.min_satellites else None

        if dual_count >= max(config.min_dual_satellites, config.min_satellites):
            return "dual"
        if single_count >= config.min_satellites:
            return "single"
        if dual_count >= config.min_satellites:
            return "dual"
        return None

    def _measurements(self, epoch: ObservationEpoch, mode: str) -> List[Measurement]:
        items: List[Measurement] = []
        for sv in epoch.satellites():
            if not self.interpolator.has_satellite(sv):
                continue
            pseudorange = (
                epoch.ionosphere_free(sv) if mode == "dual" else epoch.pseudorange(sv)
            )
            if pseudorange is None:
                continue
            clock = self.dataset.clocks.bias(sv, epoch.time)
            if clock is None:
                continue
            items.append(Measurement(sv, float(pseudorange), float(clock), mode))
        return items

    # ------------------------------------------------------------------
    # 几何
    # ------------------------------------------------------------------
    def _geometry(self, sv: str, time: datetime, receiver: np.ndarray) -> Optional[Geometry]:
        """返回自转改正后的卫星位置、传播时间与几何距离。"""
        transit_time = _INITIAL_TRANSIT_TIME
        satellite: Optional[np.ndarray] = None
        distance = 0.0

        for _ in range(8):
            raw = self.interpolator.position(sv, time - timedelta(seconds=transit_time))
            if raw is None:
                return None
            satellite = _rotate_z(raw, EARTH_ROTATION_RATE * transit_time)
            distance = float(np.linalg.norm(satellite - receiver))
            if not np.isfinite(distance) or distance < 1.0e6:
                return None
            updated = distance / SPEED_OF_LIGHT
            if abs(updated - transit_time) < 1.0e-12:
                transit_time = updated
                break
            transit_time = updated

        if satellite is None:
            return None
        return Geometry(satellite=satellite, transit_time=transit_time, range=distance)

    # ------------------------------------------------------------------
    # Gauss-Newton
    # ------------------------------------------------------------------
    def _solve_mode(
        self,
        epoch: ObservationEpoch,
        measurements: Sequence[Measurement],
        mode: str,
        initial: Optional[np.ndarray],
    ) -> Optional[Solution]:
        config = self.config
        position = np.asarray(
            initial if initial is not None else self.dataset.reference_position, dtype=float
        ).copy()
        clock_bias = 0.0
        remaining = list(measurements)
        evaluations: Dict[str, Tuple[float, float, float, float]] = {}
        design: Optional[np.ndarray] = None
        iterations = 0

        for _round in range(1 + config.max_residual_rounds):
            solved = self._gauss_newton(epoch, remaining, position, clock_bias, mode)
            if solved is None:
                return None
            position, clock_bias, evaluations, design, iterations = solved

            if not config.robust or len(remaining) <= config.min_satellites:
                break

            outliers = {
                sv
                for sv, (residual, sigma, _elevation, _azimuth) in evaluations.items()
                if sigma > 0.0 and abs(residual) > config.robust_k * sigma
            }
            if not outliers:
                break
            candidate = [item for item in remaining if item.sv not in outliers]
            if len(candidate) < config.min_satellites:
                break
            remaining = candidate

        if not evaluations:
            self.last_error = "无有效残差"
            return None

        residuals = np.array([value[0] for value in evaluations.values()], dtype=float)
        residual_rms = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else float("nan")
        pdop = self._pdop(design)

        if not np.isfinite(pdop) or pdop > config.max_pdop:
            self.last_error = f"PDOP 超限（{pdop:.2f}）"
            return None

        latitude, longitude, height = ecef_to_geodetic(position)
        return Solution(
            time=epoch.time,
            position=position,
            clock_bias_m=float(clock_bias),
            residual_rms=residual_rms,
            pdop=float(pdop),
            satellites=len(evaluations),
            iterations=int(iterations),
            mode=mode,
            latitude=latitude,
            longitude=longitude,
            height=height,
            residuals={sv: float(value[0]) for sv, value in evaluations.items()},
            elevations={sv: float(value[2]) for sv, value in evaluations.items()},
            azimuths={sv: float(value[3]) for sv, value in evaluations.items()},
        )

    def _gauss_newton(
        self,
        epoch: ObservationEpoch,
        measurements: Sequence[Measurement],
        position: np.ndarray,
        clock_bias: float,
        mode: str,
    ) -> Optional[
        Tuple[np.ndarray, float, Dict[str, Tuple[float, float, float, float]], Optional[np.ndarray], int]
    ]:
        config = self.config
        x = position.copy()
        bias = float(clock_bias)
        evaluations: Dict[str, Tuple[float, float, float, float]] = {}
        design: Optional[np.ndarray] = None
        iterations = 0

        for iterations in range(1, config.max_iterations + 1):
            latitude, longitude, height = ecef_to_geodetic(x)
            up, east, north = _local_frame(latitude, longitude)
            rows: List[List[float]] = []
            rhs: List[float] = []
            scales: List[float] = []

            for measurement in measurements:
                geometry = self._geometry(measurement.sv, epoch.time, x)
                if geometry is None:
                    continue
                unit = (geometry.satellite - x) / geometry.range
                elevation = math.degrees(
                    math.asin(max(-1.0, min(1.0, float(np.dot(unit, up)))))
                )
                if elevation < config.elevation_cutoff_deg:
                    continue
                azimuth = math.degrees(
                    math.atan2(float(np.dot(unit, east)), float(np.dot(unit, north)))
                ) % 360.0

                corrected = self._corrected_pseudorange(
                    measurement,
                    epoch.time,
                    geometry.transit_time,
                    latitude,
                    longitude,
                    height,
                    elevation,
                    azimuth,
                    mode,
                )
                if corrected is None:
                    continue
                sigma = self._sigma(elevation, mode)
                rows.append([-unit[0], -unit[1], -unit[2], 1.0])
                rhs.append(corrected - geometry.range - bias)
                scales.append(1.0 / sigma)

            if len(rows) < config.min_satellites:
                self.last_error = f"{mode} 模式几何有效卫星不足（{len(rows)}）"
                return None

            matrix = np.asarray(rows, dtype=float)
            vector = np.asarray(rhs, dtype=float)
            scale = np.asarray(scales, dtype=float)
            try:
                solution, *_ = np.linalg.lstsq(
                    matrix * scale[:, None], vector * scale, rcond=None
                )
            except np.linalg.LinAlgError:
                self.last_error = "法方程奇异"
                return None
            if not np.all(np.isfinite(solution)):
                self.last_error = "解算结果非有限值"
                return None

            increment = float(np.linalg.norm(solution[:3]))
            if increment > 1.0e7:  # 异常步长阻尼
                factor = 1.0e7 / increment
                solution = solution * factor
                increment = 1.0e7

            x = x + solution[:3]
            bias += float(solution[3])
            design = matrix

            if increment < config.convergence_threshold_m:
                break

        evaluations = self._residuals(epoch, measurements, x, bias, mode)
        if not evaluations:
            self.last_error = "无有效残差"
            return None
        return x, bias, evaluations, design, iterations

    def _corrected_pseudorange(
        self,
        measurement: Measurement,
        time: datetime,
        transit_time: float,
        latitude: float,
        longitude: float,
        height: float,
        elevation: float,
        azimuth: float,
        mode: str,
    ) -> Optional[float]:
        config = self.config
        clock = self.dataset.clocks.bias(
            measurement.sv, time - timedelta(seconds=transit_time)
        )
        if clock is None:
            clock = measurement.clock_bias_s
        corrected = measurement.pseudorange + SPEED_OF_LIGHT * clock
        if config.use_troposphere:
            corrected -= atmosphere.troposphere_delay(latitude, height, elevation)
        if mode == "single" and config.use_ionosphere:
            corrected -= atmosphere.ionosphere_delay(
                self.dataset.ionosphere_coefficients,
                latitude,
                longitude,
                elevation,
                azimuth,
                _seconds_of_day(time),
            )
        return corrected

    def _residuals(
        self,
        epoch: ObservationEpoch,
        measurements: Sequence[Measurement],
        position: np.ndarray,
        clock_bias: float,
        mode: str,
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """计算验后残差：(残差, 观测噪声, 高度角, 方位角)。"""
        config = self.config
        latitude, longitude, height = ecef_to_geodetic(position)
        up, east, north = _local_frame(latitude, longitude)
        result: Dict[str, Tuple[float, float, float, float]] = {}

        for measurement in measurements:
            geometry = self._geometry(measurement.sv, epoch.time, position)
            if geometry is None:
                continue
            unit = (geometry.satellite - position) / geometry.range
            elevation = math.degrees(
                math.asin(max(-1.0, min(1.0, float(np.dot(unit, up)))))
            )
            if elevation < config.elevation_cutoff_deg:
                continue
            azimuth = math.degrees(
                math.atan2(float(np.dot(unit, east)), float(np.dot(unit, north)))
            ) % 360.0

            corrected = self._corrected_pseudorange(
                measurement,
                epoch.time,
                geometry.transit_time,
                latitude,
                longitude,
                height,
                elevation,
                azimuth,
                mode,
            )
            if corrected is None:
                continue
            sigma = self._sigma(elevation, mode)
            result[measurement.sv] = (
                corrected - (geometry.range + clock_bias),
                sigma,
                elevation,
                azimuth,
            )
        return result

    def _sigma(self, elevation_deg: float, mode: str) -> float:
        config = self.config
        sigma = config.sigma_zenith_m + config.sigma_elevation_m / max(
            math.sin(math.radians(max(elevation_deg, 1.0))), 1.0e-3
        )
        if mode == "dual":
            sigma *= config.dual_sigma_scale
        return sigma

    # ------------------------------------------------------------------
    # 残差与统计
    # ------------------------------------------------------------------
    @staticmethod
    def _pdop(design: Optional[np.ndarray]) -> float:
        if design is None or design.shape[0] < 4:
            return float("inf")
        normal = design.T @ design
        try:
            covariance = np.linalg.inv(normal)
        except np.linalg.LinAlgError:
            return float("inf")
        trace = covariance[0, 0] + covariance[1, 1] + covariance[2, 2]
        if trace <= 0.0 or not np.isfinite(trace):
            return float("inf")
        return math.sqrt(trace)


def _local_frame(
    latitude_deg: float, longitude_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回测站处的 (天顶, 东, 北) 单位矢量。"""
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    return up, east, north


def _rotate_z(position: np.ndarray, theta: float) -> np.ndarray:
    """绕 Z 轴旋转（地球自转改正）。"""
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)
    return np.array(
        [
            cos_theta * position[0] + sin_theta * position[1],
            -sin_theta * position[0] + cos_theta * position[1],
            position[2],
        ]
    )


def _seconds_of_day(time: datetime) -> float:
    return time.hour * 3600.0 + time.minute * 60.0 + time.second + time.microsecond * 1e-6
