"""Tkinter 图形界面。

界面只负责收集参数、展示日志与结果；解算在后台线程中执行，
通过 ``queue.Queue`` 把日志、进度与结果安全地传回主线程。
"""

from __future__ import annotations

import queue
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .config import (
    DEFAULT_CLK_FILE,
    DEFAULT_END,
    DEFAULT_OBS_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SP3_FILE,
    DEFAULT_START,
    DEFAULT_STATION,
    SolverConfig,
    TimeWindow,
)
from .export import format_summary, write_solutions_csv, write_summary_json
from .pipeline import RunResult, run


def _format_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _parse_time(value: str) -> datetime:
    text = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(text)


class SppApplication:
    """主窗口。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("伪距单点定位 SPP · ABMF")
        self.root.geometry("1120x820")
        self.root.minsize(960, 700)

        self.result: Optional[RunResult] = None
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._figure_canvases: dict = {}

        self.sp3_var = tk.StringVar(value=str(DEFAULT_SP3_FILE))
        self.clk_var = tk.StringVar(value=str(DEFAULT_CLK_FILE))
        self.obs_var = tk.StringVar(value=str(DEFAULT_OBS_FILE))
        self.start_var = tk.StringVar(value=_format_time(DEFAULT_START))
        self.end_var = tk.StringVar(value=_format_time(DEFAULT_END))
        self.station_var = tk.StringVar(value=DEFAULT_STATION)
        self.mode_var = tk.StringVar(value="auto")
        self.cutoff_var = tk.DoubleVar(value=5.0)
        self.min_sats_var = tk.IntVar(value=4)
        self.max_pdop_var = tk.DoubleVar(value=30.0)
        self.tropo_var = tk.BooleanVar(value=True)
        self.iono_var = tk.BooleanVar(value=False)
        self.robust_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="就绪")

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(120, self._poll_queue)

    # ------------------------------------------------------------------
    # 布局
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)

        files = ttk.LabelFrame(outer, text="输入数据", padding=8)
        files.grid(row=0, column=0, sticky="ew")
        files.columnconfigure(1, weight=1)
        for row, (label, variable) in enumerate(
            (("SP3 轨道", self.sp3_var), ("CLK 钟差", self.clk_var), ("OBS 观测", self.obs_var))
        ):
            ttk.Label(files, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            ttk.Entry(files, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=3)
            ttk.Button(files, text="浏览…", command=lambda var=variable: self._browse(var)).grid(
                row=row, column=2, padx=(8, 0), pady=3
            )

        settings = ttk.LabelFrame(outer, text="处理设置", padding=8)
        settings.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        for column in (1, 3, 5):
            settings.columnconfigure(column, weight=1)

        ttk.Label(settings, text="起始时间").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.start_var, width=21).grid(row=0, column=1, sticky="w", padx=(4, 12))
        ttk.Label(settings, text="结束时间").grid(row=0, column=2, sticky="w")
        ttk.Entry(settings, textvariable=self.end_var, width=21).grid(row=0, column=3, sticky="w", padx=(4, 12))
        ttk.Label(settings, text="测站").grid(row=0, column=4, sticky="w")
        ttk.Entry(settings, textvariable=self.station_var, width=10).grid(row=0, column=5, sticky="w", padx=4)

        ttk.Label(settings, text="频率模式").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(
            settings,
            textvariable=self.mode_var,
            values=("auto", "single", "dual"),
            state="readonly",
            width=10,
        ).grid(row=1, column=1, sticky="w", padx=(4, 12), pady=(6, 0))
        ttk.Label(settings, text="截止高度角 (°)").grid(row=1, column=2, sticky="w", pady=(6, 0))
        ttk.Spinbox(settings, from_=0, to=45, increment=1, textvariable=self.cutoff_var, width=8).grid(
            row=1, column=3, sticky="w", padx=(4, 12), pady=(6, 0)
        )
        ttk.Label(settings, text="最少卫星数").grid(row=1, column=4, sticky="w", pady=(6, 0))
        ttk.Spinbox(settings, from_=4, to=12, increment=1, textvariable=self.min_sats_var, width=8).grid(
            row=1, column=5, sticky="w", padx=4, pady=(6, 0)
        )

        ttk.Label(settings, text="PDOP 上限").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(
            settings, from_=1, to=100, increment=1, textvariable=self.max_pdop_var, width=8
        ).grid(row=2, column=1, sticky="w", padx=(4, 12), pady=(6, 0))
        checks = ttk.Frame(settings)
        checks.grid(row=2, column=2, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Checkbutton(checks, text="对流层改正", variable=self.tropo_var).pack(side="left")
        ttk.Checkbutton(checks, text="电离层改正", variable=self.iono_var).pack(side="left", padx=10)
        ttk.Checkbutton(checks, text="抗差估计", variable=self.robust_var).pack(side="left")

        controls = ttk.Frame(outer)
        controls.grid(row=2, column=0, sticky="ew", pady=8)
        self.run_button = ttk.Button(controls, text="开始解算", command=self._start_run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="停止", command=self._stop_run, state="disabled")
        self.stop_button.pack(side="left", padx=6)
        self.export_csv_button = ttk.Button(
            controls, text="导出 CSV", command=self._export_csv, state="disabled"
        )
        self.export_csv_button.pack(side="left", padx=(18, 0))
        self.export_json_button = ttk.Button(
            controls, text="导出 JSON", command=self._export_json, state="disabled"
        )
        self.export_json_button.pack(side="left", padx=6)
        self.progress = ttk.Progressbar(controls, mode="determinate", length=220)
        self.progress.pack(side="right", padx=(8, 0))
        ttk.Label(controls, textvariable=self.status_var, width=28).pack(side="right")

        notebook = ttk.Notebook(outer)
        notebook.grid(row=3, column=0, sticky="nsew")
        self.notebook = notebook

        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="运行日志")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap="word", height=18, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="统计摘要")
        summary_frame.rowconfigure(0, weight=1)
        summary_frame.columnconfigure(0, weight=1)
        self.summary_text = scrolledtext.ScrolledText(
            summary_frame, wrap="word", height=18, state="disabled"
        )
        self.summary_text.grid(row=0, column=0, sticky="nsew")

        self.enu_frame = ttk.Frame(notebook)
        notebook.add(self.enu_frame, text="ENU 误差")
        self.enu_frame.rowconfigure(0, weight=1)
        self.enu_frame.columnconfigure(0, weight=1)

        self.quality_frame = ttk.Frame(notebook)
        notebook.add(self.quality_frame, text="解算质量")
        self.quality_frame.rowconfigure(0, weight=1)
        self.quality_frame.columnconfigure(0, weight=1)

        self.sky_frame = ttk.Frame(notebook)
        notebook.add(self.sky_frame, text="天空图")
        self.sky_frame.rowconfigure(0, weight=1)
        self.sky_frame.columnconfigure(0, weight=1)

    # ------------------------------------------------------------------
    # 界面辅助
    # ------------------------------------------------------------------
    def _browse(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("GNSS 数据", "*.*"), ("所有文件", "*.*")],
        )
        if path:
            variable.set(path)

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_summary(self, text: str) -> None:
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", text)
        self.summary_text.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        self.run_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        if running:
            self.status_var.set("解算中 …")
            self.progress.configure(value=0, maximum=100)

    def _build_config(self) -> SolverConfig:
        return SolverConfig(
            frequency_mode=self.mode_var.get(),
            use_troposphere=bool(self.tropo_var.get()),
            use_ionosphere=bool(self.iono_var.get()),
            robust=bool(self.robust_var.get()),
            elevation_cutoff_deg=float(self.cutoff_var.get()),
            min_satellites=int(self.min_sats_var.get()),
            max_pdop=float(self.max_pdop_var.get()),
        )

    # ------------------------------------------------------------------
    # 运行
    # ------------------------------------------------------------------
    def _start_run(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        try:
            start = _parse_time(self.start_var.get())
            end = _parse_time(self.end_var.get())
            window = TimeWindow(start, end)
            config = self._build_config()
        except (ValueError, TypeError) as error:
            messagebox.showerror("参数错误", str(error))
            return

        for label, path in (
            ("SP3", self.sp3_var.get()),
            ("CLK", self.clk_var.get()),
            ("OBS", self.obs_var.get()),
        ):
            if not Path(path).is_file():
                messagebox.showerror("文件不存在", f"{label} 文件不存在：\n{path}")
                return

        self._stop_event.clear()
        self._set_running(True)
        self.export_csv_button.configure(state="disabled")
        self.export_json_button.configure(state="disabled")
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.progress.configure(value=0, maximum=100)
        self._append_log(f"开始处理：{window.describe()}")
        self._append_log(f"配置：{config.describe()}")

        self._worker = threading.Thread(
            target=self._worker_run,
            args=(
                self.sp3_var.get(),
                self.clk_var.get(),
                self.obs_var.get(),
                window,
                config,
                self.station_var.get().strip() or None,
            ),
            daemon=True,
        )
        self._worker.start()

    def _worker_run(
        self,
        sp3: str,
        clk: str,
        obs: str,
        window: TimeWindow,
        config: SolverConfig,
        station: Optional[str],
    ) -> None:
        try:
            result = run(
                sp3_path=sp3,
                clk_path=clk,
                obs_path=obs,
                window=window,
                config=config,
                logger=lambda message: self._queue.put(("log", message)),
                progress=lambda current, total: self._queue.put(("progress", (current, total))),
                should_stop=self._stop_event.is_set,
                station=station,
            )
            self._queue.put(("result", result))
        except Exception:
            self._queue.put(("error", traceback.format_exc()))
        finally:
            self._queue.put(("finished", None))

    def _stop_run(self) -> None:
        self._stop_event.set()
        self.status_var.set("正在停止 …")

    def _poll_queue(self) -> None:
        while True:
            try:
                kind, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log(str(payload))
            elif kind == "progress":
                current, total = payload
                percent = 100.0 * current / total if total else 0.0
                self.progress.configure(value=percent)
                self.status_var.set(f"解算中 {current}/{total}")
            elif kind == "result":
                self._on_result(payload)
            elif kind == "error":
                self._append_log(str(payload))
                messagebox.showerror("解算失败", str(payload).strip().splitlines()[-1])
                self.status_var.set("解算失败")
            elif kind == "finished":
                self._set_running(False)
                if self.status_var.get().startswith("解算中"):
                    self.status_var.set("就绪")

        self.root.after(120, self._poll_queue)

    def _on_result(self, result: RunResult) -> None:
        self.result = result
        self._append_log("解算完成。")
        self._set_summary(format_summary(result, use_truth=True))
        self.export_csv_button.configure(state="normal")
        self.export_json_button.configure(state="normal")
        self.status_var.set(f"完成：{result.solved_epochs}/{result.total_epochs} 历元")
        self._render_figures()

    # ------------------------------------------------------------------
    # 图形与导出
    # ------------------------------------------------------------------
    def _render_figures(self) -> None:
        if self.result is None:
            return
        try:
            import matplotlib

            matplotlib.use("TkAgg")
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            from .plotting import (
                create_enu_figure,
                create_quality_figure,
                create_skyplot_figure,
            )

            pairs = (
                ("enu", self.enu_frame, create_enu_figure),
                ("quality", self.quality_frame, create_quality_figure),
                ("sky", self.sky_frame, create_skyplot_figure),
            )
            for key, frame, factory in pairs:
                for child in frame.winfo_children():
                    child.destroy()
                figure = factory(self.result)
                if figure is None:
                    ttk.Label(frame, text="无可用数据").pack(pady=20)
                    continue
                canvas = FigureCanvasTkAgg(figure, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                self._figure_canvases[key] = canvas
        except Exception:
            self._append_log("绘图失败：\n" + traceback.format_exc())

    def _export_csv(self) -> None:
        if self.result is None:
            return
        default_name = f"spp_{self.result.dataset.station}_{self.result.solutions[0].time:%Y%m%d_%H%M%S}.csv"
        path = filedialog.asksaveasfilename(
            title="导出逐历元结果",
            defaultextension=".csv",
            initialfile=default_name,
            initialdir=str(DEFAULT_OUTPUT_DIR),
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        try:
            write_solutions_csv(path, self.result)
            self.status_var.set(f"已导出 {Path(path).name}")
        except Exception as error:
            messagebox.showerror("导出失败", str(error))

    def _export_json(self) -> None:
        if self.result is None:
            return
        default_name = f"spp_{self.result.dataset.station}_{self.result.solutions[0].time:%Y%m%d_%H%M%S}.json"
        path = filedialog.asksaveasfilename(
            title="导出统计摘要",
            defaultextension=".json",
            initialfile=default_name,
            initialdir=str(DEFAULT_OUTPUT_DIR),
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            write_summary_json(path, self.result)
            self.status_var.set(f"已导出 {Path(path).name}")
        except Exception as error:
            messagebox.showerror("导出失败", str(error))

    def _on_close(self) -> None:
        self._stop_event.set()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    SppApplication(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
