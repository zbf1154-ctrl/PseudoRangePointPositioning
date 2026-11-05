"""伪距单点定位（SPP）工具包。

模块划分
--------
config      常量、路径与解算配置
models      数据模型（观测、钟差、轨道、解算结果）
rinex_obs   RINEX 2.x 观测文件解析
rinex_clk   RINEX 3.0 钟差文件解析
sp3         SP3 精密轨道文件解析
interpolate 拉格朗日轨道插值
atmosphere  对流层与电离层模型
coords      ECEF / 大地坐标 / ENU 转换
solver      逐历元抗差最小二乘解算
pipeline    数据装载与全流程编排
export      CSV / JSON 结果导出
plotting    结果可视化
cli         命令行入口
gui         Tkinter 图形界面
"""

from __future__ import annotations

__version__ = "2.1.0"

from .config import SolverConfig, TimeWindow, default_paths, default_solver_config
from .models import Dataset, EpochFailure, ObservationEpoch, SatelliteClockSeries, Solution, Sp3Orbit
from .pipeline import PipelineError, RunResult, load_dataset, run
from .solver import SppSolver

__all__ = [
    "__version__",
    "Dataset",
    "EpochFailure",
    "ObservationEpoch",
    "PipelineError",
    "RunResult",
    "SatelliteClockSeries",
    "Solution",
    "SolverConfig",
    "Sp3Orbit",
    "SppSolver",
    "TimeWindow",
    "default_paths",
    "default_solver_config",
    "load_dataset",
    "run",
]
