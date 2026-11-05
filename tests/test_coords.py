import math

import numpy as np
import pytest

from spp.coords import (
    azimuth_elevation,
    ecef_offset_to_enu,
    ecef_to_geodetic,
    enu_matrix,
    geodetic_to_ecef,
)


@pytest.mark.parametrize(
    "lat, lon, height",
    [
        (0.0, 0.0, 0.0),
        (16.2623, -61.5275, -25.5),
        (45.0, 90.0, 1200.0),
        (-33.0, 151.0, 50.0),
        (80.0, -120.0, 3000.0),
    ],
)
def test_geodetic_roundtrip(lat, lon, height):
    ecef = geodetic_to_ecef(lat, lon, height)
    lat2, lon2, h2 = ecef_to_geodetic(ecef)
    assert lat2 == pytest.approx(lat, abs=1e-7)
    assert lon2 == pytest.approx(lon, abs=1e-7)
    assert h2 == pytest.approx(height, abs=1e-4)


def test_enu_matrix_is_orthonormal():
    matrix = enu_matrix(16.2623, -61.5275)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_ecef_offset_to_enu_known_direction():
    lat, lon = 0.0, 0.0
    # 在赤道/本初子午线处，+X 为天顶方向，+Z 为北向
    enu_up = ecef_offset_to_enu(np.array([10.0, 0.0, 0.0]), lat, lon)
    np.testing.assert_allclose(enu_up, [0.0, 0.0, 10.0], atol=1e-12)
    enu_north = ecef_offset_to_enu(np.array([0.0, 0.0, 10.0]), lat, lon)
    np.testing.assert_allclose(enu_north, [0.0, 10.0, 0.0], atol=1e-12)


def test_azimuth_elevation_zenith():
    receiver = geodetic_to_ecef(0.0, 0.0, 0.0)
    satellite = receiver + np.array([20_000_000.0, 0.0, 0.0])
    azimuth, elevation, distance = azimuth_elevation(receiver, satellite, 0.0, 0.0)
    assert elevation == pytest.approx(90.0, abs=1e-6)
    assert distance == pytest.approx(20_000_000.0)
    assert math.isfinite(azimuth)
