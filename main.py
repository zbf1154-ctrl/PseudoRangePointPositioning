#!/usr/bin/env python3
"""项目启动脚本。

- ``python main.py``：启动 Tkinter 图形界面；
- ``python main.py run ...``：转发到命令行；
- 也可以直接使用 ``python -m spp``。
"""

from __future__ import annotations

import sys

from spp.cli import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit(main(["gui"]))
    raise SystemExit(main())
