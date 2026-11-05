from datetime import datetime

import numpy as np
import pytest

from spp.config import TimeWindow
from spp.rinex_clk import read_rinex_clk
from spp.rinex_obs import read_rinex_obs
from spp.sp3 import build_orbit, read_sp3


def test_read_rinex_obs(data_files, short_window):
    _, _, obs = data_files
    parsed = read_rinex_obs(str(obs), short_window)
    assert parsed.marker_name == "ABMF"
    assert parsed.version.startswith("2.11")
    assert len(parsed.observation_types) == 26
    assert parsed.epoch_count == 11  # 00:00 ~ 00:05，30 s 采样
    first = parsed.epochs[0]
    assert first.time == datetime(2020, 1, 3, 0, 0, 0)
    assert len(first.satellites()) == 34
    assert first.values["G16"]["C1"] == pytest.approx(23460886.408)
    assert first.values["G16"]["P1"] == pytest.approx(23460885.393)
    assert first.pseudorange("G16") == pytest.approx(23460886.408)


def test_read_rinex_clk(data_files):
    _, clk, _ = data_files
    parsed = read_rinex_clk(str(clk))
    assert parsed.epoch_count == 2880
    assert parsed.version.startswith("3.0")
    assert "G01" in parsed.epochs[0].satellites
    assert parsed.epochs[0].satellites["G01"] == pytest.approx(-2.499858088370e-04)
    position = parsed.station_position("ABMF")
    assert position is not None
    np.testing.assert_allclose(
        position, [2919785.787, -5383744.956, 1774604.860], atol=1e-6
    )


def test_read_sp3(data_files):
    sp3_path, _, _ = data_files
    parsed = read_sp3(str(sp3_path))
    assert parsed.epoch_count == 96
    assert len(parsed.satellites) == 32
    assert parsed.interval_seconds == pytest.approx(900.0)
    assert parsed.coordinate_system == "G"
    first_time, positions = parsed.epochs[0]
    assert first_time == datetime(2020, 1, 3, 0, 0, 0)
    np.testing.assert_allclose(
        positions["G01"],
        [10653452.144, -20074341.758, -13675038.579],
        atol=1e-3,
    )
    orbit = build_orbit(parsed)
    assert len(orbit.times) == 96
    assert len(orbit.satellites) == 32
