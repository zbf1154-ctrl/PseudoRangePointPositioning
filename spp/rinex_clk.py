"""RINEX 3.0 钟差文件解析。

时间系统偏移记录（如 ``AS GPST``、``AS GLPS``）与卫星钟差记录共用 ``AS`` 类型，
本模块通过标识符模式加以区分，仅保留卫星与测站钟差。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from .config import TimeWindow
from .models import ClockEpoch

_SATELLITE_RE = re.compile(r"^[A-Z]\d{2}$")
_TIME_SYSTEM_IDS = {
    "GPST",
    "GLPS",
    "GLSV",
    "GALT",
    "BDST",
    "QZST",
    "IRNT",
    "UTC",
    "TAI",
    "GPS",
    "GLO",
    "GAL",
    "BDS",
    "QZS",
}


class RinexClkError(ValueError):
    """钟差文件解析失败。"""


@dataclass
class RinexClkFile:
    """RINEX 3.0 钟差文件内容。"""

    path: str = ""
    version: str = ""
    clock_types: List[str] = field(default_factory=list)
    time_system: str = ""
    station_positions: Dict[str, np.ndarray] = field(default_factory=dict)
    epochs: List[ClockEpoch] = field(default_factory=list)
    header_line_count: int = 0

    @property
    def epoch_count(self) -> int:
        return len(self.epochs)

    def station_position(self, name: str) -> Optional[np.ndarray]:
        return self.station_positions.get(name)


def _parse_station_position(line: str) -> Optional[tuple]:
    parts = line[:60].split()
    if len(parts) < 5:
        return None
    name = parts[0]
    try:
        coordinates = [float(parts[-3]), float(parts[-2]), float(parts[-1])]
    except ValueError:
        return None
    if max(abs(value) for value in coordinates) > 1.0e7:  # 文件中以毫米记录
        coordinates = [value / 1000.0 for value in coordinates]
    return name, np.array(coordinates, dtype=float)


def read_rinex_clk(path: str, window: Optional[TimeWindow] = None) -> RinexClkFile:
    """读取 RINEX 3.0 钟差文件。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    result = RinexClkFile(path=path)
    index = 0
    expected_types = 0

    while index < len(lines):
        line = lines[index]
        if "END OF HEADER" in line:
            index += 1
            break
        label = line[60:].strip()
        if label == "RINEX VERSION / TYPE":
            result.version = line[:9].strip()
        elif label == "# / TYPES OF DATA":
            if expected_types == 0:
                try:
                    expected_types = int(line[:6])
                except ValueError:
                    expected_types = 0
            result.clock_types.extend(line[6:60].split())
            while expected_types and len(result.clock_types) < expected_types:
                index += 1
                if index >= len(lines):
                    break
                result.clock_types.extend(lines[index][6:60].split())
            if expected_types:
                result.clock_types = result.clock_types[:expected_types]
        elif label == "TIME SYSTEM ID":
            result.time_system = line[:60].strip()
        elif label == "SOLN STA NAME / NUM":
            parsed = _parse_station_position(line)
            if parsed is not None:
                result.station_positions.setdefault(parsed[0], parsed[1])
        index += 1
    result.header_line_count = index

    clock_types = set(result.clock_types) or {"AS", "AR"}
    epochs: Dict[datetime, ClockEpoch] = {}

    while index < len(lines):
        raw = lines[index]
        index += 1
        parts = raw.split()
        if len(parts) < 9:
            continue
        if parts[0] not in clock_types:
            continue
        name = parts[1]
        try:
            year, month, day, hour, minute = (int(parts[k]) for k in range(2, 7))
            second = float(parts[7])
            n_coefficients = int(parts[8])
        except (ValueError, IndexError):
            continue
        try:
            epoch_time = datetime(
                year,
                month,
                day,
                hour,
                minute,
                int(second),
                int(round((second % 1) * 1e6)) % 1_000_000,
            )
        except ValueError:
            continue

        coefficients = []
        for token in parts[9:9 + n_coefficients]:
            try:
                coefficients.append(float(token))
            except ValueError:
                coefficients.append(math.nan)
        bias = coefficients[0] if coefficients else math.nan
        if not np.isfinite(bias):
            continue

        epoch = epochs.get(epoch_time)
        if epoch is None:
            epoch = ClockEpoch(time=epoch_time)
            epochs[epoch_time] = epoch

        if _SATELLITE_RE.match(name):
            epoch.satellites[name] = bias
        elif name in _TIME_SYSTEM_IDS:
            continue
        elif 3 <= len(name) <= 4:
            epoch.stations[name] = bias

    result.epochs = [epochs[key] for key in sorted(epochs)]
    if window is not None:
        result.epochs = [epoch for epoch in result.epochs if window.contains(epoch.time)]
    if not result.epochs:
        raise RinexClkError(f"{path} 未解析到有效钟差历元")
    return result
