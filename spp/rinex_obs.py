"""RINEX 2.11 观测文件解析。

实现要点
--------
1. 观测值按固定 16 列字段解析，每行 5 个观测值，缺失字段补 ``NaN``；
   空行同样补齐 5 个 ``NaN``，保证观测类型索引不会因缺失数据而错位。
2. 卫星列表允许跨行续排，续行数量由卫星数动态确定，不依赖 12 颗/行的硬编码。
3. 支持事件标志（epoch flag ≥ 2）的记录跳过。
4. 数值允许 ``D`` 指数形式。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import TimeWindow
from .models import ObservationEpoch

OBSERVATIONS_PER_LINE = 5
FIELD_WIDTH = 16
VALUE_WIDTH = 14
SATELLITE_LIST_WIDTH = 36  # 每行 12 个卫星，每个 3 字符

_EPOCH_RE = re.compile(
    r"^\s*(\d{2,4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d*)?)"
    r"\s+(\d{1,2})\s*(\d{1,3})"
)
_FIELDS = tuple((j * FIELD_WIDTH, j * FIELD_WIDTH + VALUE_WIDTH) for j in range(OBSERVATIONS_PER_LINE))


class RinexObsError(ValueError):
    """观测文件解析失败。"""


@dataclass
class RinexObsFile:
    """RINEX 2.x 观测文件内容。"""

    path: str
    version: str = ""
    marker_name: str = ""
    marker_number: str = ""
    approx_position: Optional[np.ndarray] = None
    antenna_delta: np.ndarray = field(default_factory=lambda: np.zeros(3))
    observation_types: List[str] = field(default_factory=list)
    interval: float = 0.0
    first_observation: Optional[datetime] = None
    ionosphere_alpha: Optional[np.ndarray] = None
    ionosphere_beta: Optional[np.ndarray] = None
    epochs: List[ObservationEpoch] = field(default_factory=list)
    header_line_count: int = 0

    @property
    def epoch_count(self) -> int:
        return len(self.epochs)


def _to_float(text: str) -> float:
    text = text.strip()
    if not text:
        return math.nan
    try:
        return float(text.replace("D", "E").replace("d", "e"))
    except ValueError:
        return math.nan


def _parse_header(lines: List[str]) -> Tuple[RinexObsFile, int]:
    header = RinexObsFile(path="")
    index = 0
    expected_types = 0
    while index < len(lines):
        line = lines[index]
        if "END OF HEADER" in line:
            index += 1
            break
        label = line[60:].strip()
        if label == "RINEX VERSION / TYPE":
            header.version = line[:9].strip()
        elif label == "MARKER NAME":
            header.marker_name = line[:60].strip()
        elif label == "MARKER NUMBER":
            header.marker_number = line[:60].strip()
        elif label == "APPROX POSITION XYZ":
            values = [_to_float(v) for v in line[:60].split()]
            if len(values) >= 3 and all(np.isfinite(values[:3])):
                header.approx_position = np.array(values[:3], dtype=float)
        elif label == "ANTENNA: DELTA H/E/N":
            values = [_to_float(v) for v in line[:60].split()]
            if len(values) >= 3 and all(np.isfinite(values[:3])):
                header.antenna_delta = np.array(values[:3], dtype=float)
        elif label == "# / TYPES OF OBSERV":
            if expected_types == 0:
                try:
                    expected_types = int(line[:6])
                except ValueError:
                    expected_types = 0
            header.observation_types.extend(line[6:60].split())
            while expected_types and len(header.observation_types) < expected_types:
                index += 1
                if index >= len(lines):
                    break
                header.observation_types.extend(lines[index][6:60].split())
            if expected_types:
                header.observation_types = header.observation_types[:expected_types]
        elif label == "INTERVAL":
            header.interval = _to_float(line[:10])
        elif label == "TIME OF FIRST OBS":
            parts = line[:43].split()
            if len(parts) >= 6:
                year = int(parts[0])
                if year < 100:
                    year += 2000 if year < 80 else 1900
                second = float(parts[5])
                try:
                    header.first_observation = datetime(
                        year,
                        int(parts[1]),
                        int(parts[2]),
                        int(parts[3]),
                        int(parts[4]),
                        int(second),
                        int(round((second % 1) * 1e6)) % 1_000_000,
                    )
                except ValueError:
                    header.first_observation = None
        elif label == "ION ALPHA":
            values = [_to_float(v) for v in line[2:60].split()]
            if len(values) >= 4:
                header.ionosphere_alpha = np.array(values[:4], dtype=float)
        elif label == "ION BETA":
            values = [_to_float(v) for v in line[2:60].split()]
            if len(values) >= 4:
                header.ionosphere_beta = np.array(values[:4], dtype=float)
        index += 1
    header.header_line_count = index
    return header, index


def read_rinex_obs(path: str, window: Optional[TimeWindow] = None) -> RinexObsFile:
    """读取 RINEX 2.x 观测文件。

    Parameters
    ----------
    path:
        文件路径。
    window:
        仅保留该时间窗内的历元；``None`` 表示不做过滤。
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    header, index = _parse_header(lines)
    header.path = path

    try:
        major = int(float(header.version)) if header.version else 2
    except ValueError:
        major = 2
    if major >= 3:
        raise RinexObsError(
            f"{path} 为 RINEX {header.version} 格式；当前实现仅支持 RINEX 2.x 观测文件"
        )
    if not header.observation_types:
        raise RinexObsError(f"{path} 未找到观测类型定义")

    types = header.observation_types
    n_types = len(types)
    lines_per_satellite = max(1, math.ceil(n_types / OBSERVATIONS_PER_LINE))

    while index < len(lines):
        line = lines[index]
        match = _EPOCH_RE.match(line)
        if not match:
            index += 1
            continue

        year = int(match.group(1))
        if year < 100:
            year += 2000 if year < 80 else 1900
        month, day, hour, minute = (int(match.group(k)) for k in range(2, 6))
        second = float(match.group(6))
        flag = int(match.group(7))
        n_satellites = int(match.group(8))

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
            index += 1
            continue

        index += 1

        if flag not in (0, 1):
            # 事件记录：跳过 n_satellites 行特殊记录
            index += n_satellites
            continue

        if n_satellites <= 0:
            continue

        # 卫星列表可能跨行，按需继续读取
        satellite_text = line[32:32 + SATELLITE_LIST_WIDTH] if len(line) > 32 else ""
        while len(satellite_text.strip().replace(" ", "")) < 3 * n_satellites and index < len(lines):
            candidate = lines[index]
            if _EPOCH_RE.match(candidate):
                break
            satellite_text += candidate[32:32 + SATELLITE_LIST_WIDTH] if len(candidate) > 32 else ""
            index += 1

        satellites = [
            satellite_text[start:start + 3].strip()
            for start in range(0, 3 * n_satellites, 3)
        ]
        satellites = [sv for sv in satellites if sv]
        if len(satellites) < n_satellites:
            # 列表不完整时仍按已解析卫星读取，避免整体失败
            n_satellites = len(satellites)

        keep = window is None or window.contains(epoch_time)
        epoch = ObservationEpoch(time=epoch_time) if keep else None

        for satellite in satellites:
            values: List[float] = []
            for _ in range(lines_per_satellite):
                if index >= len(lines):
                    break
                raw = lines[index]
                if _EPOCH_RE.match(raw):
                    break
                index += 1
                if not raw.strip():
                    values.extend([math.nan] * OBSERVATIONS_PER_LINE)
                    continue
                for start, end in _FIELDS:
                    values.append(_to_float(raw[start:end]) if start < len(raw) else math.nan)
            if len(values) < n_types:
                values.extend([math.nan] * (n_types - len(values)))

            if epoch is None:
                continue
            record: Dict[str, float] = {}
            for position, obs_type in enumerate(types):
                value = values[position]
                if np.isfinite(value) and value > 0.0:
                    record[obs_type] = value
            if record:
                epoch.values[satellite] = record

        if epoch is not None and epoch.values:
            header.epochs.append(epoch)

    if not header.epochs:
        raise RinexObsError(f"{path} 未解析到有效观测历元")
    return header
