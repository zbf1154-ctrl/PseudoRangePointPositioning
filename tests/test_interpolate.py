from datetime import datetime, timedelta

import numpy as np
import pytest

from spp.interpolate import OrbitInterpolator, lagrange_interpolate
from spp.models import Sp3Orbit


def test_lagrange_interpolate_quadratic():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = x ** 2
    assert lagrange_interpolate(x, y, 1.5) == pytest.approx(2.25)
    assert lagrange_interpolate(x, y, 2.0) == pytest.approx(4.0)


def test_lagrange_interpolate_rejects_mismatch():
    with pytest.raises(ValueError):
        lagrange_interpolate([0.0, 1.0], [0.0], 0.5)


def _synthetic_orbit(count=20, interval=900.0):
    base = datetime(2020, 1, 3, 0, 0, 0)
    times = [base + timedelta(seconds=interval * i) for i in range(count)]
    seconds = np.array([interval * i for i in range(count)], dtype=float)
    # 二次多项式轨迹，3 阶以上插值应精确
    positions = np.column_stack(
        [
            1.0e7 + 100.0 * seconds + 0.001 * seconds ** 2,
            -2.0e7 + 200.0 * seconds - 0.002 * seconds ** 2,
            3.0e6 + 300.0 * seconds + 0.003 * seconds ** 2,
        ]
    )
    return Sp3Orbit(
        times=times,
        satellites=["G01"],
        positions={"G01": positions},
        interval_seconds=interval,
    )


def test_orbit_interpolator_exact_node():
    orbit = _synthetic_orbit()
    interpolator = OrbitInterpolator(orbit, order=5)
    expected = orbit.positions["G01"][3]
    actual = interpolator.position("G01", orbit.times[3])
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_orbit_interpolator_quadratic_reproduction():
    orbit = _synthetic_orbit()
    interpolator = OrbitInterpolator(orbit, order=5)
    query = orbit.times[6] + timedelta(seconds=123.0)
    t = 6 * 900.0 + 123.0
    expected = np.array(
        [
            1.0e7 + 100.0 * t + 0.001 * t ** 2,
            -2.0e7 + 200.0 * t - 0.002 * t ** 2,
            3.0e6 + 300.0 * t + 0.003 * t ** 2,
        ]
    )
    np.testing.assert_allclose(interpolator.position("G01", query), expected, atol=1e-3)


def test_orbit_interpolator_out_of_range():
    orbit = _synthetic_orbit()
    interpolator = OrbitInterpolator(orbit, order=5, extrapolation_limit_seconds=10.0)
    before = orbit.times[0] - timedelta(seconds=60)
    after = orbit.times[-1] + timedelta(seconds=60)
    assert interpolator.position("G01", before) is None
    assert interpolator.position("G01", after) is None
    assert interpolator.position("UNKNOWN", orbit.times[0]) is None
