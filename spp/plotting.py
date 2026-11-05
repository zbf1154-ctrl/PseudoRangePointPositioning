"""结果可视化。

模块只构造 ``matplotlib.figure.Figure``，不主动切换后端；CLI 使用 Agg，
GUI 使用 TkAgg，从而避免相互干扰。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.dates as mdates
from matplotlib.figure import Figure

from .pipeline import RunResult

_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
]


def configure_fonts() -> None:
    matplotlib.rcParams["font.sans-serif"] = _FONT_CANDIDATES
    matplotlib.rcParams["axes.unicode_minus"] = False


def _prepare(result: RunResult) -> Optional[tuple]:
    if not result.solutions:
        return None
    times = [solution.time for solution in result.solutions]
    enu = result.offsets_enu(use_truth=True)
    return times, enu


def _apply_time_axis(axis, times) -> None:
    duration = (times[-1] - times[0]).total_seconds() if len(times) > 1 else 0.0
    axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M:%S" if duration < 600 else "%H:%M")
    )
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))


def create_enu_figure(result: RunResult, figsize: tuple = (10, 7)) -> Optional[Figure]:
    """绘制 E/N/U 误差时间序列。"""
    prepared = _prepare(result)
    if prepared is None:
        return None
    times, enu = prepared
    configure_fonts()
    figure = Figure(figsize=figsize, dpi=110, constrained_layout=True)
    axes = figure.subplots(3, 1, sharex=True)
    labels = ("East", "North", "Up")
    colors = ("#1f77b4", "#2ca02c", "#d62728")
    for index, axis in enumerate(axes):
        axis.plot(times, enu[:, index], color=colors[index], linewidth=1.2)
        axis.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
        axis.set_ylabel(f"{labels[index]} error [m]")
        axis.grid(True, alpha=0.3)
        mean = float(enu[:, index].mean())
        axis.axhline(mean, color=colors[index], linewidth=0.8, linestyle=":", alpha=0.8)
        axis.text(
            0.01,
            0.92,
            f"mean={mean:+.2f} m, rms={math.sqrt(float((enu[:, index] ** 2).mean())):.2f} m",
            transform=axis.transAxes,
            fontsize=8,
            va="top",
        )
    _apply_time_axis(axes[-1], times)
    axes[-1].set_xlabel("GPS time")
    figure.suptitle("SPP position error (ENU, relative to IGS reference)", fontsize=12)
    return figure


def create_quality_figure(result: RunResult, figsize: tuple = (10, 7)) -> Optional[Figure]:
    """绘制残差、PDOP 与卫星数时间序列。"""
    if not result.solutions:
        return None
    configure_fonts()
    times = [solution.time for solution in result.solutions]
    residuals = [solution.residual_rms for solution in result.solutions]
    pdops = [solution.pdop for solution in result.solutions]
    satellites = [solution.satellites for solution in result.solutions]

    figure = Figure(figsize=figsize, dpi=110, constrained_layout=True)
    axes = figure.subplots(3, 1, sharex=True)
    axes[0].plot(times, residuals, color="#9467bd", linewidth=1.2)
    axes[0].set_ylabel("Residual RMS [m]")
    axes[1].plot(times, pdops, color="#ff7f0e", linewidth=1.2)
    axes[1].set_ylabel("PDOP")
    axes[2].plot(times, satellites, color="#17becf", linewidth=1.2, marker=".", markersize=3)
    axes[2].set_ylabel("Satellites")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    _apply_time_axis(axes[-1], times)
    axes[-1].set_xlabel("GPS time")
    figure.suptitle("SPP solution quality", fontsize=12)
    return figure


def create_skyplot_figure(
    result: RunResult, figsize: tuple = (7, 7), solution_index: int = -1
) -> Optional[Figure]:
    """绘制单个历元的卫星天空图（方位角/高度角）。"""
    if not result.solutions:
        return None
    solution = result.solutions[solution_index]
    if not solution.azimuths:
        return None
    configure_fonts()
    figure = Figure(figsize=figsize, dpi=110, constrained_layout=True)
    axis = figure.add_subplot(111, projection="polar")
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_ylim(0, 90)
    axis.set_yticks([0, 30, 60])
    axis.set_yticklabels(["90°", "60°", "30°"], fontsize=8)
    axis.set_rlabel_position(135)
    axis.set_title(
        f"Skyplot at {solution.time.isoformat(sep=' ')}\n"
        f"PDOP={solution.pdop:.2f}, satellites={solution.satellites}",
        fontsize=11,
    )
    for sv, azimuth in solution.azimuths.items():
        elevation = solution.elevations.get(sv, 0.0)
        theta = math.radians(azimuth)
        radius = 90.0 - elevation
        axis.scatter([theta], [radius], s=28, color="#1f77b4")
        axis.text(theta, radius, sv, fontsize=7, ha="center", va="bottom")
    return figure


def save_figure(figure: Figure, path: str, dpi: int = 150) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi)
    return destination


def save_default_figures(result: RunResult, output_dir: str, prefix: str = "spp") -> dict:
    """保存默认的误差图、质量图与天空图。"""
    directory = Path(output_dir)
    paths = {}
    for name, factory in (
        ("enu_error", create_enu_figure),
        ("quality", create_quality_figure),
        ("skyplot", create_skyplot_figure),
    ):
        figure = factory(result)
        if figure is not None:
            paths[name] = save_figure(figure, directory / f"{prefix}_{name}.png")
    return paths
