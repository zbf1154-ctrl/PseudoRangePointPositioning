"""命令行入口。

示例
----
``python -m spp run``
``python -m spp run --start "2020-01-03 00:00:00" --end "2020-01-03 00:10:00" --no-plot``
``python -m spp gui``
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from . import __version__
from .config import (
    DATA_DIR,
    DEFAULT_CLK_FILE,
    DEFAULT_END,
    DEFAULT_OBS_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SP3_FILE,
    DEFAULT_START,
    SolverConfig,
    TimeWindow,
    default_paths,
)
from .export import format_summary, save_outputs
from .pipeline import PipelineError, run


def _parse_datetime(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"无法解析时间 {value!r}，请使用 'YYYY-MM-DD HH:MM:SS' 或 ISO 格式"
        ) from error
    return parsed.replace(tzinfo=None)


def _build_config(args: argparse.Namespace) -> SolverConfig:
    return SolverConfig(
        frequency_mode=args.mode,
        use_troposphere=not args.no_tropo,
        use_ionosphere=args.iono,
        robust=not args.no_robust,
        robust_k=args.robust_k,
        elevation_cutoff_deg=args.cutoff,
        min_satellites=args.min_sats,
        max_pdop=args.max_pdop,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spp",
        description="伪距单点定位（SPP）工具：解析 RINEX/SP3/CLK，逐历元解算并导出结果。",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="执行完整解算流程")
    run_parser.add_argument("--sp3", default=str(DEFAULT_SP3_FILE), help="SP3 精密轨道文件")
    run_parser.add_argument("--clk", default=str(DEFAULT_CLK_FILE), help="RINEX 钟差文件")
    run_parser.add_argument("--obs", default=str(DEFAULT_OBS_FILE), help="RINEX 2.x 观测文件")
    run_parser.add_argument("--start", type=_parse_datetime, default=DEFAULT_START, help="时间窗起点")
    run_parser.add_argument("--end", type=_parse_datetime, default=DEFAULT_END, help="时间窗终点")
    run_parser.add_argument("--station", default=None, help="测站名（默认取观测文件 MARKER NAME）")
    run_parser.add_argument(
        "--mode",
        choices=("auto", "single", "dual"),
        default="auto",
        help="频率模式：auto/single/dual",
    )
    run_parser.add_argument("--cutoff", type=float, default=5.0, help="高度角截止角（度）")
    run_parser.add_argument("--min-sats", type=int, default=4, help="最少卫星数")
    run_parser.add_argument("--max-pdop", type=float, default=30.0, help="PDOP 上限")
    run_parser.add_argument("--no-tropo", action="store_true", help="关闭对流层改正")
    run_parser.add_argument("--iono", action="store_true", help="启用 Klobuchar 电离层改正")
    run_parser.add_argument("--no-robust", action="store_true", help="关闭抗差残差剔除")
    run_parser.add_argument("--robust-k", type=float, default=3.0, help="抗差阈值倍数")
    run_parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    run_parser.add_argument("--prefix", default="spp", help="输出文件前缀")
    run_parser.add_argument("--no-export", action="store_true", help="不写 CSV/JSON")
    run_parser.add_argument("--no-plot", action="store_true", help="不生成图片")
    run_parser.add_argument("--quiet", action="store_true", help="不打印解析过程")

    subparsers.add_parser("gui", help="启动 Tkinter 图形界面")
    subparsers.add_parser("info", help="显示版本、默认数据路径与数据规模")
    return parser


def _logger(quiet: bool):
    if quiet:
        return lambda _message: None

    def log(message: str) -> None:
        print(message, flush=True)

    return log


def _progress_printer():
    last_percent = -1

    def progress(current: int, total: int) -> None:
        nonlocal last_percent
        if total <= 0:
            return
        percent = int(100.0 * current / total)
        if current == total or percent >= last_percent + 10:
            last_percent = percent
            print(f"  解算进度：{current}/{total} ({percent:3d}%)", flush=True)

    return progress


def run_command(args: argparse.Namespace) -> int:
    try:
        config = _build_config(args)
        window = TimeWindow(args.start, args.end)
    except ValueError as error:
        print(f"参数错误：{error}", file=sys.stderr)
        return 2

    log = _logger(args.quiet)
    progress = None if args.quiet else _progress_printer()

    try:
        result = run(
            sp3_path=args.sp3,
            clk_path=args.clk,
            obs_path=args.obs,
            window=window,
            config=config,
            logger=log,
            progress=progress,
            station=args.station,
        )
    except (PipelineError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1

    print()
    print(format_summary(result, use_truth=True))

    outputs = {}
    if not args.no_export:
        outputs.update(save_outputs(result, args.out, prefix=args.prefix))
    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")  # 无显示环境下也能生成图片
        from .plotting import save_default_figures

        outputs.update(save_default_figures(result, args.out, prefix=args.prefix))

    if outputs:
        print("\n输出文件：")
        for name, path in outputs.items():
            print(f"  {name:10s} {path}")
    if result.failures:
        print(f"\n有 {len(result.failures)} 个历元未解算，详见失败记录。")
    return 0


def info_command(_args: argparse.Namespace) -> int:
    sp3, clk, obs = default_paths()
    print(f"版本       : {__version__}")
    print(f"项目数据   : {DATA_DIR}")
    for label, path in (("SP3", sp3), ("CLK", clk), ("OBS", obs)):
        if Path(path).is_file():
            size_mb = Path(path).stat().st_size / 1024 / 1024
            print(f"{label:10s}: {path}  ({size_mb:.1f} MB)")
        else:
            print(f"{label:10s}: {path}  (缺失)")
    print("默认窗口   :", TimeWindow().describe())
    print("默认配置   :", SolverConfig().describe())
    return 0


def gui_command(_args: argparse.Namespace) -> int:
    from .gui import main as gui_main

    return gui_main()


def main(argv: Optional[List[str]] = None) -> int:
    # 保证 Windows 控制台输出中文时不因默认 GBK 编码而抛异常。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    if args.command == "run":
        return run_command(args)
    if args.command == "gui":
        return gui_command(args)
    if args.command == "info":
        return info_command(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
