"""坐标转换工具（WGS-84 / ECEF / ENU）。"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .config import WGS84_A, WGS84_E2


def ecef_to_geodetic(position: np.ndarray) -> Tuple[float, float, float]:
    """ECEF 直角坐标转大地坐标（WGS-84）。

    Parameters
    ----------
    position:
        ECEF 坐标，单位 m。

    Returns
    -------
    (lat_deg, lon_deg, height_m)
    """
    x, y, z = (float(value) for value in position)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)

    if p < 1.0e-9:  # 位于极轴
        lat = math.copysign(math.pi / 2.0, z)
        height = abs(z) - WGS84_A * math.sqrt(1.0 - WGS84_E2)
        return math.degrees(lat), math.degrees(lon), height

    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    height = 0.0
    for _ in range(12):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        height = p / math.cos(lat) - n
        updated = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + height)))
        if abs(updated - lat) < 1.0e-14:
            lat = updated
            break
        lat = updated

    return math.degrees(lat), math.degrees(lon), height


def geodetic_to_ecef(lat_deg: float, lon_deg: float, height_m: float) -> np.ndarray:
    """大地坐标转 ECEF 直角坐标（WGS-84）。"""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + height_m) * math.cos(lat) * math.cos(lon)
    y = (n + height_m) * math.cos(lat) * math.sin(lon)
    z = (n * (1.0 - WGS84_E2) + height_m) * math.sin(lat)
    return np.array([x, y, z], dtype=float)


def enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    """返回 ECEF 到 ENU 的旋转矩阵（3×3）。

    行向量依次为东、北、天顶方向。
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )


def ecef_offset_to_enu(offset: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    """将 ECEF 坐标差转换为 ENU 分量。"""
    return enu_matrix(lat_deg, lon_deg) @ np.asarray(offset, dtype=float)


def enu_offset_to_ecef(offset: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    """将 ENU 分量转换为 ECEF 坐标差。"""
    return enu_matrix(lat_deg, lon_deg).T @ np.asarray(offset, dtype=float)


def azimuth_elevation(
    receiver: np.ndarray,
    satellite: np.ndarray,
    lat_deg: float,
    lon_deg: float,
) -> Tuple[float, float, float]:
    """计算卫星相对测站的方位角、高度角与距离。

    Returns
    -------
    (azimuth_deg, elevation_deg, range_m)
    """
    receiver = np.asarray(receiver, dtype=float)
    satellite = np.asarray(satellite, dtype=float)
    delta = satellite - receiver
    distance = float(np.linalg.norm(delta))
    if distance == 0.0:
        return 0.0, 0.0, 0.0

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])

    unit = delta / distance
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(unit, up))))))
    azimuth = math.degrees(math.atan2(float(np.dot(unit, east)), float(np.dot(unit, north)))) % 360.0
    return azimuth, elevation, distance
