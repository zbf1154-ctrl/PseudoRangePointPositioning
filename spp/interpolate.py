"""轨道插值。

采用重心形式的拉格朗日插值，相比直接展开多项式具有更好的数值稳定性。
每个查询时刻取 ``order + 1`` 个连续节点，节点数量不足以覆盖时返回 ``None``。
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .models import Sp3Orbit


def lagrange_interpolate(
    nodes: Sequence[float],
    values: Sequence[float],
    query: float,
) -> float:
    """一维拉格朗日插值（重心形式）。

    Parameters
    ----------
    nodes:
        已知节点横坐标，必须严格递增。
    values:
        节点对应的纵坐标。
    query:
        待插值横坐标。
    """
    x = np.asarray(nodes, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.size == 0 or x.size != y.size:
        raise ValueError("节点与观测值数量不一致")
    if x.size == 1:
        return float(y[0])

    offsets = x - query
    exact = np.where(offsets == 0.0)[0]
    if exact.size:
        return float(y[exact[0]])

    weights = np.ones(x.size, dtype=float)
    for i in range(x.size):
        product = 1.0
        for j in range(x.size):
            if i != j:
                product *= x[i] - x[j]
        weights[i] = 1.0 / product

    normalised = weights / offsets
    total = normalised.sum()
    if total == 0.0 or not np.isfinite(total):
        raise ValueError("插值权重退化")
    return float(normalised @ y / total)


class OrbitInterpolator:
    """SP3 轨道插值器。

    Parameters
    ----------
    orbit:
        轨道数据，位置单位米。
    order:
        插值阶数，默认 9 阶（10 个节点）。对 900 s 采样的 IGS 产品，
        9 阶插值的位置误差在厘米量级。
    extrapolation_limit_seconds:
        允许在轨道时间范围外轻微外推的最大秒数。
    """

    def __init__(
        self,
        orbit: Sp3Orbit,
        order: int = 9,
        extrapolation_limit_seconds: Optional[float] = None,
    ) -> None:
        self.orbit = orbit
        self.order = max(1, int(order))
        self._series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        seconds = orbit.seconds_array
        for sv, positions in orbit.positions.items():
            valid = np.all(np.isfinite(positions), axis=1)
            if valid.sum() >= 2:
                self._series[sv] = (seconds[valid], positions[valid])
        interval = max(orbit.interval_seconds, 1.0)
        if extrapolation_limit_seconds is None:
            extrapolation_limit_seconds = min(interval * 0.5, 120.0)
        self._extrapolation_limit = float(extrapolation_limit_seconds)

    @property
    def satellites(self):
        return sorted(self._series)

    @property
    def order_(self) -> int:
        return self.order

    def has_satellite(self, sv: str) -> bool:
        return sv in self._series

    def position_at_seconds(self, sv: str, query: float) -> Optional[np.ndarray]:
        """按相对首历元的秒数查询卫星位置（米）。"""
        series = self._series.get(sv)
        if series is None:
            return None
        seconds, positions = series
        if query < seconds[0] - self._extrapolation_limit:
            return None
        if query > seconds[-1] + self._extrapolation_limit:
            return None

        count = min(self.order + 1, seconds.size)
        insert = int(np.searchsorted(seconds, query))
        start = min(max(insert - count // 2, 0), seconds.size - count)
        nodes = seconds[start:start + count]
        values = positions[start:start + count]

        offsets = nodes - query
        exact = np.where(offsets == 0.0)[0]
        if exact.size:
            return values[exact[0]].copy()

        weights = np.ones(count, dtype=float)
        for i in range(count):
            product = 1.0
            for j in range(count):
                if i != j:
                    product *= nodes[i] - nodes[j]
            weights[i] = 1.0 / product

        normalised = weights / offsets
        total = normalised.sum()
        if total == 0.0 or not np.isfinite(total):
            return None
        return (normalised @ values) / total

    def position(self, sv: str, time: datetime) -> Optional[np.ndarray]:
        """返回卫星在 ``time`` 时刻的三维位置（米）。"""
        return self.position_at_seconds(sv, (time - self.orbit.base).total_seconds())


def interpolate_orbit_to_times(
    orbit: Sp3Orbit,
    times: Sequence[datetime],
    order: int = 9,
) -> Dict[str, np.ndarray]:
    """将轨道插值到给定时间轴。

    Returns
    -------
    ``{卫星: (len(times), 3)}``；无法插值的位置为 ``NaN``。
    """
    interpolator = OrbitInterpolator(orbit, order=order)
    output: Dict[str, np.ndarray] = {}
    for sv in interpolator.satellites:
        values = np.full((len(times), 3), np.nan)
        for row, time in enumerate(times):
            position = interpolator.position(sv, time)
            if position is not None:
                values[row] = position
        output[sv] = values
    return output
