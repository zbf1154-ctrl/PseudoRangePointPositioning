"""pytest 公共夹具。"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spp.config import TimeWindow
from spp.pipeline import load_dataset, run

DATA_DIR = PROJECT_ROOT / "data"
SP3 = DATA_DIR / "igs20865.sp3"
CLK = DATA_DIR / "igs20865.clk_30s"
OBS = DATA_DIR / "abmf0030.20o"

START = datetime(2020, 1, 3, 0, 0, 0)
SHORT_END = START + timedelta(minutes=5)


@pytest.fixture(scope="session")
def data_files():
    if not (SP3.is_file() and CLK.is_file() and OBS.is_file()):
        pytest.skip("示例数据不完整")
    return SP3, CLK, OBS


@pytest.fixture(scope="session")
def short_window():
    return TimeWindow(START, SHORT_END)


@pytest.fixture(scope="session")
def dataset(data_files, short_window):
    sp3, clk, obs = data_files
    return load_dataset(str(sp3), str(clk), str(obs), short_window)


@pytest.fixture(scope="session")
def short_result(data_files, short_window):
    sp3, clk, obs = data_files
    return run(str(sp3), str(clk), str(obs), short_window)
