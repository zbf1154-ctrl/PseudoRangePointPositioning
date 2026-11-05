"""全局常量、路径与求解配置。

本模块只依赖标准库，便于被其余模块安全导入。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# --------------------------------------------------------------------------
# 项目路径
# --------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_SP3_FILE = DATA_DIR / "igs20865.sp3"
DEFAULT_CLK_FILE = DATA_DIR / "igs20865.clk_30s"
DEFAULT_OBS_FILE = DATA_DIR / "abmf0030.20o"

# --------------------------------------------------------------------------
# 物理常量（IERS / WGS-84）
# --------------------------------------------------------------------------
SPEED_OF_LIGHT = 299_792_458.0          # 真空光速，m/s
EARTH_ROTATION_RATE = 7.2921151467e-5   # 地球自转角速度，rad/s
EARTH_GRAVITY = 3.986005e14             # 地球引力常数，m³/s²

# --------------------------------------------------------------------------
# GNSS 频率
# --------------------------------------------------------------------------
GPS_FREQ_L1 = 1575.42e6
GPS_FREQ_L2 = 1227.60e6
GPS_FREQ_L5 = 1176.45e6

# 无电离层组合系数（L1/L2）
GAMMA_L1_L2 = (GPS_FREQ_L1 / GPS_FREQ_L2) ** 2
IF_COEF_L1 = GAMMA_L1_L2 / (GAMMA_L1_L2 - 1.0)
IF_COEF_L2 = -1.0 / (GAMMA_L1_L2 - 1.0)

# --------------------------------------------------------------------------
# WGS-84 椭球
# --------------------------------------------------------------------------
WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# --------------------------------------------------------------------------
# 默认处理范围
# --------------------------------------------------------------------------
DEFAULT_START = datetime(2020, 1, 3, 0, 0, 0)
DEFAULT_END = datetime(2020, 1, 3, 1, 0, 0)
DEFAULT_STATION = "ABMF"

# ABMF 测站近似坐标（观测文件头，仅作迭代初值）
ABMF_APPROX_POSITION = (2_919_785.712, -5_383_745.067, 1_774_604.692)

# ABMF 测站参考坐标（IGS 钟差文件头 SOLN STA，用于精度评估）
ABMF_REFERENCE_POSITION = (2_919_785.787, -5_383_744.956, 1_774_604.860)

# 伪距观测量的选取优先级：仅使用 L1 频点的测距码，避免不同频点混用引入偏差
PSEUDORANGE_PRIORITY = ("C1", "P1")
DUAL_PSEUDORANGE_PRIORITY = ("P1", "P2")

# 合理伪距区间，用于剔除解析异常值（m）
PSEUDORANGE_MIN = 1.0e7
PSEUDORANGE_MAX = 5.0e7

# 频率模式
FREQUENCY_MODES = ("auto", "single", "dual")


@dataclass(frozen=True)
class SolverConfig:
    """单点定位解算参数。

    Attributes
    ----------
    min_satellites:
        单个历元参与解算的最少卫星数，低于该值则判定该历元无效。
    max_iterations:
        Gauss-Newton 最大迭代次数。
    convergence_threshold_m:
        位置增量收敛阈值，单位 m。
    elevation_cutoff_deg:
        高度角截止角，单位度。
    use_troposphere:
        是否施加 Saastamoinen 对流层延迟改正。
    use_ionosphere:
        是否施加 Klobuchar 电离层延迟改正（需观测文件头包含 α、β 系数）。
    frequency_mode:
        ``auto`` 在有足够双频伪距时采用无电离层组合，否则退回单频；
        ``single`` 强制单频；``dual`` 强制双频（不足最少卫星数则历元无效）。
    max_pdop:
        PDOP 上限，超过该值的历元判定为几何结构不可用。
    sigma_zenith_m / sigma_elevation_m:
        伪距观测噪声模型 σ = a + b / sin(el)，单位 m。
    robust:
        是否启用抗差估计（Huber 型残差剔除）。
    robust_k:
        残差剔除阈值倍数（以标准化残差计）。
    max_residual_rounds:
        抗差剔除的最大轮数。
    min_dual_satellites:
        ``auto`` 模式下切换双频所需的最少双频卫星数。
    dual_sigma_scale:
        双频无电离层组合的噪声放大因子。
    """

    min_satellites: int = 4
    max_iterations: int = 20
    convergence_threshold_m: float = 1.0e-4
    elevation_cutoff_deg: float = 5.0
    use_troposphere: bool = True
    use_ionosphere: bool = False
    frequency_mode: str = "auto"
    max_pdop: float = 30.0
    sigma_zenith_m: float = 0.3
    sigma_elevation_m: float = 0.9
    robust: bool = True
    robust_k: float = 3.0
    max_residual_rounds: int = 2
    min_dual_satellites: int = 5
    dual_sigma_scale: float = 3.0

    def __post_init__(self) -> None:
        if self.frequency_mode not in FREQUENCY_MODES:
            raise ValueError(
                f"frequency_mode 必须为 {FREQUENCY_MODES} 之一，当前为 {self.frequency_mode!r}"
            )
        if self.min_satellites < 4:
            raise ValueError("min_satellites 不能小于 4")
        if self.max_iterations < 1:
            raise ValueError("max_iterations 必须为正整数")
        if self.elevation_cutoff_deg < 0.0:
            raise ValueError("elevation_cutoff_deg 不能为负")
        if self.robust_k <= 0.0:
            raise ValueError("robust_k 必须为正数")

    @classmethod
    def from_dict(cls, values: dict) -> "SolverConfig":
        """从字典构造配置，忽略未知字段。"""
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})

    def as_dict(self) -> dict:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    def describe(self) -> str:
        return (
            f"mode={self.frequency_mode}, tropo={self.use_troposphere}, "
            f"iono={self.use_ionosphere}, robust={self.robust}(k={self.robust_k}), "
            f"cutoff={self.elevation_cutoff_deg}°, min_sats={self.min_satellites}, "
            f"max_pdop={self.max_pdop}"
        )


@dataclass(frozen=True)
class TimeWindow:
    """处理时间窗（GPS 时）。``None`` 表示该侧不设限制。"""

    start: Optional[datetime] = DEFAULT_START
    end: Optional[datetime] = DEFAULT_END

    def contains(self, time: datetime) -> bool:
        if self.start is not None and time < self.start:
            return False
        if self.end is not None and time > self.end:
            return False
        return True

    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("时间窗起点不能晚于终点")

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.start is None or self.end is None:
            return None
        return (self.end - self.start).total_seconds()

    def describe(self) -> str:
        start = self.start.isoformat(sep=" ") if self.start else "—"
        end = self.end.isoformat(sep=" ") if self.end else "—"
        return f"[{start} → {end}]"


def default_solver_config() -> SolverConfig:
    """返回项目默认配置（便于 GUI/CLI 共用）。"""
    return SolverConfig()


def default_paths() -> Tuple[Path, Path, Path]:
    """返回默认的 (SP3, CLK, OBS) 路径。"""
    return DEFAULT_SP3_FILE, DEFAULT_CLK_FILE, DEFAULT_OBS_FILE
