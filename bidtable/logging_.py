"""阶段进度、计数、告警收集。

产品感验收（agent.md 10.3）要求 ``[1/7] 解析投标范围… 4 个标的`` 这样的输出，
所以进度打印集中在这里，业务代码只调 :meth:`Progress.stage`。
"""

from __future__ import annotations

import sys
from typing import TextIO

TOTAL_STAGES = 7

STAGE_NAMES = (
    "解析投标范围",
    "解析厂商资料",
    "抽取招标要求",
    "范围裁剪与分节",
    "要求-证据匹配",
    "偏离判定",
    "写出结果",
)


class Progress:
    """把阶段进度打到 stderr（stdout 留给可被管道消费的结果摘要）。"""

    def __init__(self, stream: TextIO | None = None, *, quiet: bool = False) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.quiet = quiet
        self.log: list[str] = []

    def _write(self, line: str, *, force: bool = False) -> None:
        self.log.append(line)
        if force or not self.quiet:
            print(line, file=self.stream, flush=True)

    def stage(self, index: int, detail: str = "") -> None:
        """``index`` 从 1 开始。"""
        name = STAGE_NAMES[index - 1] if 1 <= index <= TOTAL_STAGES else f"阶段{index}"
        msg = f"[{index}/{TOTAL_STAGES}] {name}…"
        if detail:
            msg = f"{msg} {detail}"
        self._write(msg)

    def info(self, msg: str) -> None:
        self._write(f"      {msg}")

    def warn(self, msg: str) -> None:
        self._write(f"  ! 告警：{msg}")

    def error(self, msg: str) -> None:
        """错误永远打印：``-q`` 只该压掉进度，不能把非零退出码的原因一起吃掉。"""
        self._write(f"  × 错误：{msg}", force=True)
