import math

import pytest

from spp.atmosphere import (
    ionosphere_delay,
    klobuchar_delay,
    saastamoinen_zenith,
    standard_atmosphere,
    troposphere_delay,
)


def test_standard_atmosphere_sea_level():
    pressure, temperature = standard_atmosphere(0.0)
    assert pressure == pytest.approx(1013.25, abs=0.1)
    assert temperature == pytest.approx(288.15, abs=0.01)


def test_zenith_delay_is_physical():
    hydrostatic, wet = saastamoinen_zenith(16.26, -25.5)
    assert 2.0 < hydrostatic < 2.6
    assert 0.0 <= wet < 0.6


def test_troposphere_mapping_increases_toward_horizon():
    zenith = troposphere_delay(16.26, -25.5, 90.0)
    low = troposphere_delay(16.26, -25.5, 5.0)
    assert 2.0 < zenith < 2.8
    assert low > 4.0 * zenith
    assert troposphere_delay(16.26, -25.5, 2.0) == 0.0


def test_klobuchar_delay_is_bounded():
    alpha = (1.0e-8, 0.0, -1.0e-7, 0.0)
    beta = (1.0e5, 0.0, -1.0e5, 0.0)
    delay = klobuchar_delay(
        alpha, beta, 16.26, -61.53, 30.0, 120.0, 0.0
    )
    assert math.isfinite(delay)
    assert 0.0 <= delay < 50.0


def test_ionosphere_without_coefficients_is_zero():
    assert ionosphere_delay(None, 16.26, -61.53, 30.0, 120.0, 0.0) == 0.0
