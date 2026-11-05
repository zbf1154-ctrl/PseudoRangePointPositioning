from datetime import datetime

import pytest

from spp.config import SolverConfig, TimeWindow


def test_time_window_contains_boundaries():
    window = TimeWindow(datetime(2020, 1, 3, 0, 0), datetime(2020, 1, 3, 1, 0))
    assert window.contains(datetime(2020, 1, 3, 0, 0))
    assert window.contains(datetime(2020, 1, 3, 1, 0))
    assert not window.contains(datetime(2020, 1, 3, 1, 0, 1))


def test_time_window_rejects_reversed_range():
    with pytest.raises(ValueError):
        TimeWindow(datetime(2020, 1, 3, 1, 0), datetime(2020, 1, 3, 0, 0))


def test_solver_config_validation():
    with pytest.raises(ValueError):
        SolverConfig(frequency_mode="bad")
    with pytest.raises(ValueError):
        SolverConfig(min_satellites=3)
    with pytest.raises(ValueError):
        SolverConfig(robust_k=0)


def test_solver_config_roundtrip():
    config = SolverConfig(frequency_mode="single", use_troposphere=False)
    restored = SolverConfig.from_dict(config.as_dict())
    assert restored == config
