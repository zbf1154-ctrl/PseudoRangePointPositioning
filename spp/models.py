"""数据模型定义。

所有位置量的单位均为米，时间均为 GPS 时（``datetime``）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import (
    IF_COEF_L1,
    IF_COEF_L2,
    PSEUDORANGE_MAX,
    PSEUDORANGE_MIN,
    PSEUDORANGE_PRIORITY,
    SPEED_OF_LIGHT,
)


def _valid_pseudorange(value: Optional[float]) -> bool:
    return (
        value is not None
        and np.isfinite(value)
        and PSEUDORANGE_MIN <= float(value) <= PSEUDORANGE_MAX
    )


@dataclass
class ObservationEpoch:
    """一个观测历元。

    ``values`` 的结构为 ``{卫星: {观测类型: 观测值}}``，观测值单位统一为米。
    """

    time: datetime
    values: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def satellites(self) -> List[str]:
        return sorted(self.values)

    def has_satellite(self, sv: str) -> bool:
        return sv in self.values

    def observation(self, sv: str, names: Sequence[str]) -> Optional[float]:
        """按给定顺序返回第一个有效观测值（米）。"""
        observations = self.values.get(sv)
        if not observations:
            return None
        for name in names:
            value = observations.get(name)
            if _valid_pseudorange(value):
                return float(value)
        return None

    def pseudorange(self, sv: str) -> Optional[float]:
        """按优先级选取单频伪距（米）。"""
        return self.observation(sv, PSEUDORANGE_PRIORITY)

    def ionosphere_free(self, sv: str) -> Optional[float]:
        """无电离层组合伪距（米）。缺少 P1/P2 时返回 ``None``。"""
        observations = self.values.get(sv)
        if not observations:
            return None
        p1 = observations.get("P1")
        p2 = observations.get("P2")
        if not (_valid_pseudorange(p1) and _valid_pseudorange(p2)):
            return None
        return IF_COEF_L1 * float(p1) + IF_COEF_L2 * float(p2)

    @property
    def satellite_count(self) -> int:
        return len(self.values)


@dataclass
class ClockEpoch:
    """钟差文件中的一个历元。钟差单位：秒。"""

    time: datetime
    satellites: Dict[str, float] = field(default_factory=dict)
    stations: Dict[str, float] = field(default_factory=dict)

    @property
    def satellite_count(self) -> int:
        return len(self.satellites)


class SatelliteClockSeries:
    """卫星/测站钟差序列，支持对任意时刻做线性插值。"""

    def __init__(
        self,
        epochs: Sequence[ClockEpoch],
        base: Optional[datetime] = None,
        extrapolation_limit_seconds: float = 60.0,
    ) -> None:
        if not epochs:
            raise ValueError("钟差序列为空")
        epochs = sorted(epochs, key=lambda item: item.time)
        self.base = base or epochs[0].time
        self._times = np.array(
            [(epoch.time - self.base).total_seconds() for epoch in epochs], dtype=float
        )
        # 信号发射时刻可能略早于首个钟差历元（如 00:00:00 历元），允许有限外推。
        self._extrapolation_limit = float(extrapolation_limit_seconds)
        self._satellites: Dict[str, np.ndarray] = {}
        self._stations: Dict[str, np.ndarray] = {}

        names = {sv for epoch in epochs for sv in epoch.satellites}
        for sv in names:
            self._satellites[sv] = np.array(
                [epoch.satellites.get(sv, np.nan) for epoch in epochs], dtype=float
            )
        station_names = {name for epoch in epochs for name in epoch.stations}
        for name in station_names:
            self._stations[name] = np.array(
                [epoch.stations.get(name, np.nan) for epoch in epochs], dtype=float
            )

    @property
    def times(self) -> np.ndarray:
        return self._times.copy()

    @property
    def satellites(self) -> List[str]:
        return sorted(self._satellites)

    @property
    def stations(self) -> List[str]:
        return sorted(self._stations)

    @property
    def start(self) -> datetime:
        return self.base

    @property
    def end(self) -> datetime:
        from datetime import timedelta

        return self.base + timedelta(seconds=float(self._times[-1]))

    def station_bias(self, name: str) -> Optional[float]:
        """返回测站钟差的中位数（秒），用于质量检查。"""
        values = self._stations.get(name)
        if values is None:
            return None
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        return float(np.median(finite))

    def bias(self, sv: str, time: datetime) -> Optional[float]:
        """返回 ``time`` 时刻的卫星钟差（秒）。越界或无效返回 ``None``。"""
        values = self._satellites.get(sv)
        if values is None:
            return None
        offset = (time - self.base).total_seconds()
        if offset < self._times[0] - self._extrapolation_limit:
            return None
        if offset > self._times[-1] + self._extrapolation_limit:
            return None
        finite = np.isfinite(values)
        if finite.sum() < 2:
            return None
        interpolated = float(np.interp(offset, self._times[finite], values[finite]))
        if not np.isfinite(interpolated):
            return None
        return interpolated

    def coverage(self, sv: str) -> Optional[tuple]:
        """返回卫星钟差的有效时间范围（秒，相对 ``base``）。"""
        values = self._satellites.get(sv)
        if values is None:
            return None
        finite = np.where(np.isfinite(values))[0]
        if finite.size < 2:
            return None
        return float(self._times[finite[0]]), float(self._times[finite[-1]])


@dataclass
class Sp3Orbit:
    """SP3 精密轨道数据（位置单位：米）。"""

    times: List[datetime]
    satellites: List[str]
    positions: Dict[str, np.ndarray]          # 卫星 -> (n, 3)
    interval_seconds: float = 900.0
    coordinate_system: str = "UNKNOWN"

    def __post_init__(self) -> None:
        if not self.times:
            raise ValueError("SP3 轨道为空")
        self._base = self.times[0]
        self._seconds = np.array(
            [(time - self._base).total_seconds() for time in self.times], dtype=float
        )

    @property
    def base(self) -> datetime:
        return self._base

    @property
    def seconds_array(self) -> np.ndarray:
        """各历元相对首历元的秒数。"""
        return self._seconds

    @property
    def start(self) -> datetime:
        return self.times[0]

    @property
    def end(self) -> datetime:
        return self.times[-1]

    def seconds(self, time: datetime) -> float:
        return (time - self._base).total_seconds()

    def position(self, sv: str, time: datetime) -> Optional[np.ndarray]:
        """返回卫星在 ``time`` 时刻的位置（米），越界返回 ``None``。

        说明：本方法是简单的逐分量线性插值，仅作为兼容接口；
        解算器使用 :class:`spp.interpolate.OrbitInterpolator` 的高阶插值。
        """
        values = self.positions.get(sv)
        if values is None:
            return None
        offset = self.seconds(time)
        if offset < self._seconds[0] or offset > self._seconds[-1]:
            return None
        finite = np.all(np.isfinite(values), axis=1)
        if finite.sum() < 2:
            return None
        index = np.where(finite)[0]
        seconds = self._seconds[index]
        result = np.array(
            [np.interp(offset, seconds, values[index, axis]) for axis in range(3)]
        )
        return result


@dataclass
class Solution:
    """单个历元的定位结果。"""

    time: datetime
    position: np.ndarray                      # ECEF，m
    clock_bias_m: float                       # 接收机钟差等效距离，m
    residual_rms: float                       # 验后残差 RMS，m
    pdop: float
    satellites: int
    iterations: int
    mode: str                                 # "single" 或 "dual"
    latitude: float = 0.0
    longitude: float = 0.0
    height: float = 0.0
    residuals: Dict[str, float] = field(default_factory=dict)
    elevations: Dict[str, float] = field(default_factory=dict)
    azimuths: Dict[str, float] = field(default_factory=dict)

    @property
    def clock_bias_s(self) -> float:
        return self.clock_bias_m / SPEED_OF_LIGHT

    def offset_ecef(self, reference: np.ndarray) -> np.ndarray:
        return self.position - np.asarray(reference, dtype=float)

    def to_dict(self, reference: Optional[np.ndarray] = None) -> dict:
        record = {
            "time": self.time.isoformat(sep=" "),
            "x": float(self.position[0]),
            "y": float(self.position[1]),
            "z": float(self.position[2]),
            "latitude": float(self.latitude),
            "longitude": float(self.longitude),
            "height": float(self.height),
            "clock_bias_m": float(self.clock_bias_m),
            "clock_bias_s": float(self.clock_bias_s),
            "residual_rms": float(self.residual_rms),
            "pdop": float(self.pdop),
            "satellites": int(self.satellites),
            "iterations": int(self.iterations),
            "mode": self.mode,
        }
        if reference is not None:
            offset = self.offset_ecef(reference)
            record.update(
                {
                    "dx": float(offset[0]),
                    "dy": float(offset[1]),
                    "dz": float(offset[2]),
                    "error_3d": float(np.linalg.norm(offset)),
                }
            )
        return record


@dataclass
class EpochFailure:
    """无法完成解算的历元及原因。"""

    time: datetime
    reason: str
    satellites: int = 0
    details: str = ""


@dataclass
class Dataset:
    """一次解算所需的全部输入数据。"""

    observations: List[ObservationEpoch]
    clocks: SatelliteClockSeries
    orbit: Sp3Orbit
    reference_position: np.ndarray
    station: str = ""
    ionosphere_coefficients: Optional[tuple] = None
    source_files: Dict[str, str] = field(default_factory=dict)
    truth_position: Optional[np.ndarray] = None
    window: Optional[object] = None

    @property
    def epoch_count(self) -> int:
        return len(self.observations)

    @property
    def satellite_count(self) -> int:
        return len(self.orbit.satellites)

    @property
    def has_truth(self) -> bool:
        return self.truth_position is not None
