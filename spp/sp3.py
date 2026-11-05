"""SP3 精密轨道文件解析。

支持 SP3-a/b/c/d 格式；坐标单位由 km 统一转换为 m，缺失位置（文件中的 0.0）
记为 ``NaN``。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import TimeWindow
from .models import Sp3Orbit

_SATELLITE_RE = re.compile(r"[A-Z]\d{2}")
_EPOCH_RE = re.compile(
    r"^\*\s+(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d*)?)"
)
_POSITION_RE = re.compile(
    r"^P([A-Z]\d{2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
)


class Sp3Error(ValueError):
    """SP3 文件解析失败。"""


@dataclass
class Sp3File:
    """SP3 文件解析结果。"""

    path: str = ""
    version: str = ""
    coordinate_system: str = ""
    orbit_type: str = ""
    interval_seconds: float = 900.0
    epochs: List[Tuple[datetime, Dict[str, np.ndarray]]] = field(default_factory=list)
    satellites: List[str] = field(default_factory=list)
    gps_week: int = 0
    mjd: int = 0
    header_line_count: int = 0

    @property
    def epoch_count(self) -> int:
        return len(self.epochs)


def read_sp3(path: str, window: Optional[TimeWindow] = None) -> Sp3File:
    """读取 SP3 精密轨道文件，位置单位为米。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    if not lines or not lines[0].startswith("#"):
        raise Sp3Error(f"{path} 不是有效的 SP3 文件")

    result = Sp3File(path=path, version=lines[0][:3].strip())
    satellites: List[str] = []
    index = 1

    while index < len(lines):
        line = lines[index]
        if line.startswith("*"):
            break
        if line.startswith("##"):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    result.gps_week = int(parts[1])
                    result.interval_seconds = float(parts[3])
                    result.mjd = int(parts[4]) if len(parts) > 4 else 0
                except (ValueError, IndexError):
                    pass
        elif line.startswith("+"):
            # 排除 "++" 精度行
            if not line.startswith("++"):
                satellites.extend(_SATELLITE_RE.findall(line[1:]))
        elif line.startswith("%c"):
            # 首行 ``%c G  cc GPS ...`` 给出坐标系，后续 ``%c cc cc ...`` 为说明行
            tokens = line.split()
            if len(tokens) >= 2 and not result.coordinate_system and tokens[1] != "cc":
                result.coordinate_system = tokens[1]
        index += 1
    result.header_line_count = index
    result.satellites = sorted(set(satellites))

    current_time: Optional[datetime] = None
    current: Dict[str, np.ndarray] = {}

    while index < len(lines):
        line = lines[index]
        index += 1
        if line.startswith("*"):
            epoch_match = _EPOCH_RE.match(line)
            if current_time is not None and current:
                if window is None or window.contains(current_time):
                    result.epochs.append((current_time, current))
            current = {}
            if not epoch_match:
                current_time = None
                continue
            year, month, day, hour, minute = (int(epoch_match.group(k)) for k in range(1, 6))
            second = float(epoch_match.group(6))
            try:
                current_time = datetime(
                    year,
                    month,
                    day,
                    hour,
                    minute,
                    int(second),
                    int(round((second % 1) * 1e6)) % 1_000_000,
                )
            except ValueError:
                current_time = None
        elif line.startswith("P"):
            position_match = _POSITION_RE.match(line)
            if not position_match:
                continue
            values = [float(position_match.group(k)) for k in range(2, 5)]
            if all(value == 0.0 for value in values):
                continue
            if any(abs(value) > 1.0e5 for value in values):
                # 非法/缺失标记
                continue
            current[position_match.group(1)] = np.array(
                [value * 1000.0 for value in values], dtype=float
            )

    if current_time is not None and current and (window is None or window.contains(current_time)):
        result.epochs.append((current_time, current))

    if not result.epochs:
        raise Sp3Error(f"{path} 未解析到有效轨道历元")
    return result


def build_orbit(sp3: Sp3File) -> Sp3Orbit:
    """将解析结果转换为按卫星组织的轨道对象。"""
    times = [epoch_time for epoch_time, _ in sp3.epochs]
    satellites = sorted({sv for _, positions in sp3.epochs for sv in positions})
    count = len(times)
    arrays = {sv: np.full((count, 3), math.nan) for sv in satellites}
    for row, (_, positions) in enumerate(sp3.epochs):
        for sv, position in positions.items():
            arrays[sv][row] = position
    return Sp3Orbit(
        times=times,
        satellites=satellites,
        positions=arrays,
        interval_seconds=sp3.interval_seconds,
        coordinate_system=sp3.coordinate_system or "UNKNOWN",
    )
