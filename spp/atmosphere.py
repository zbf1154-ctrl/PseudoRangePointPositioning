"""大气延迟改正模型。

对流层采用 Saastamoinen 模型（标准大气 + 湿度 0.7），映射函数采用 Niell 湿项近似；
电离层采用 Klobuchar 模型（需要广播星历给出的 α、β 系数）。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from .config import SPEED_OF_LIGHT


def standard_atmosphere(height_m: float) -> Tuple[float, float]:
    """返回标准大气在给定大地高处的气压（hPa）与温度（K）。"""
    height_km = height_m / 1000.0
    base = 1.0 - 2.2557e-5 * height_m
    if base <= 0.0:
        return 0.0, 0.0
    pressure = 1013.25 * base ** 5.2568
    temperature = 288.15 - 6.5 * height_km
    return pressure, temperature


def saastamoinen_zenith(
    latitude_deg: float,
    height_m: float,
    relative_humidity: float = 0.7,
) -> Tuple[float, float]:
    """Saastamoinen 天顶延迟。

    Returns
    -------
    (zenith_hydrostatic_m, zenith_wet_m)
    """
    pressure, temperature = standard_atmosphere(height_m)
    if pressure <= 0.0 or temperature <= 0.0:
        return 0.0, 0.0

    # 水汽压公式中的温度必须为摄氏度；此前使用开尔文会导致约 500 m 的量级错误。
    temperature_c = temperature - 273.15
    water_vapour = relative_humidity * 6.108 * math.exp(
        17.15 * temperature_c / (234.9 + temperature_c)
    )

    lat = math.radians(latitude_deg)
    height_km = height_m / 1000.0
    zenith_hydrostatic = (
        0.0022768
        * pressure
        / (1.0 - 0.00266 * math.cos(2.0 * lat) - 0.00028 * height_km)
    )
    zenith_wet = 0.002277 * (1255.0 / temperature + 0.05) * water_vapour
    return float(zenith_hydrostatic), float(zenith_wet)


def troposphere_delay(
    latitude_deg: float,
    height_m: float,
    elevation_deg: float,
    relative_humidity: float = 0.7,
) -> float:
    """Saastamoinen 对流层延迟及其投影。

    Parameters
    ----------
    latitude_deg: 测站大地纬度
    height_m: 测站大地高
    elevation_deg: 卫星高度角
    relative_humidity: 相对湿度（0–1）

    Returns
    -------
    对流层延迟，单位米；高度角过低时返回 0。
    """
    if elevation_deg <= 3.0:
        return 0.0

    zenith_hydrostatic, zenith_wet = saastamoinen_zenith(
        latitude_deg, height_m, relative_humidity
    )
    if zenith_hydrostatic == 0.0 and zenith_wet == 0.0:
        return 0.0

    elevation = math.radians(elevation_deg)
    tangent = math.tan(elevation)
    mapping = 1.0 / (math.sin(elevation) + 0.00143 / (tangent + 0.0445))
    return (zenith_hydrostatic + zenith_wet) * mapping


def klobuchar_delay(
    alpha: Sequence[float],
    beta: Sequence[float],
    latitude_deg: float,
    longitude_deg: float,
    elevation_deg: float,
    azimuth_deg: float,
    gps_seconds: float,
) -> float:
    """Klobuchar 电离层延迟（单频）。

    输入角度均为度，``gps_seconds`` 为当日 GPS 秒。返回单位为米。
    """
    if len(alpha) < 4 or len(beta) < 4:
        return 0.0

    semicircle = 180.0
    elevation = elevation_deg / semicircle
    azimuth = azimuth_deg / semicircle
    phi_u = latitude_deg / semicircle
    lam_u = longitude_deg / semicircle

    if elevation + 0.11 <= 0.0:
        return 0.0

    psi = 0.0137 / (elevation + 0.11) - 0.022
    phi_i = phi_u + psi * math.cos(azimuth)
    phi_i = max(-0.416, min(0.416, phi_i))
    lam_i = lam_u + psi * math.sin(azimuth) / math.cos(phi_i * math.pi)
    phi_m = phi_i + 0.064 * math.cos((lam_i - 1.617) * math.pi)

    local_time = 4.32e4 * lam_i + gps_seconds
    local_time = math.fmod(local_time, 86400.0)
    if local_time < 0.0:
        local_time += 86400.0

    amplitude = sum(alpha[n] * phi_m ** n for n in range(4))
    period = sum(beta[n] * phi_m ** n for n in range(4))
    amplitude = max(amplitude, 0.0)
    period = max(period, 72000.0)

    factor = 1.0 + 16.0 * (0.53 - elevation) ** 3
    phase = 2.0 * math.pi * (local_time - 50400.0) / period

    if abs(phase) < 1.57:
        delay_seconds = factor * (
            5.0e-9 + amplitude * (1.0 - phase ** 2 / 2.0 + phase ** 4 / 24.0)
        )
    else:
        delay_seconds = factor * 5.0e-9

    return delay_seconds * SPEED_OF_LIGHT


def ionosphere_delay(
    coefficients: Optional[tuple],
    latitude_deg: float,
    longitude_deg: float,
    elevation_deg: float,
    azimuth_deg: float,
    gps_seconds: float,
) -> float:
    """按可用系数计算电离层延迟；系数缺失时返回 0。"""
    if coefficients is None:
        return 0.0
    alpha, beta = coefficients
    if alpha is None or beta is None:
        return 0.0
    return klobuchar_delay(
        np.asarray(alpha, dtype=float),
        np.asarray(beta, dtype=float),
        latitude_deg,
        longitude_deg,
        elevation_deg,
        azimuth_deg,
        gps_seconds,
    )
