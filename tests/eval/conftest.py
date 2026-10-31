# -*- coding: utf-8 -*-
"""eval 层共享的子进程约定（agent.md 6.8）。

前四层（unit / integration / golden / invariants）直接 ``import bidtable``，
跑在 pytest 自己的进程里。eval 层的唯一职责是**真跑**，所以它一律起一个
干净的 Python 解释器。只有这样才覆盖得到三件在进程内看不见的东西：
解释器启动时的 ``sys.path`` 解析、子进程 stdout 的编码形态、
以及「用户照着 README 敲的那一行命令」到底能不能跑通。

两个 eval 文件共用同一套调用约定，抽到这里：

* :func:`_child_env` 抹掉机器上真实存在的 Key 与模型覆盖项。
  开发机上 ``DASHSCOPE_API_KEY`` 是有值的，但账号已欠费、调用必失败；
  不清掉的话 ``load_config`` 会判成「在线」，成绩就随机器状态飘。
  清掉之后**离线降级是唯一可能的行为**，成绩才可复现 ——
  这也是为什么 eval 层不需要 ``network`` 标记：它一个包都不许联网。
 * ``child_run`` 统一 cwd、超时与解码方式。``utf8=False`` 时返回**原始字节**，
   专门给「控制台不是 UTF-8」那条用例用（Windows 上子进程的 stdout 编码
   跟随 ANSI 代码页，不是 UTF-8，只有拿到字节才验得出来）。
   ``extra_env=`` 用来**指定**某个码本，让那条用例在 Linux CI 上也不至于变成空断言。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING = REPO_ROOT / "训练集"
PACKS = ("P1", "P2", "P3", "P4", "P5")

# 与 tests/conftest.py 的 AMBIENT_ENV 保持同一份清单：那一份管进程内，这一份管子进程。
AMBIENT_ENV = (
    "LLM_API_KEY", "DASHSCOPE_API_KEY", "LLM_BASE_URL",
    "MODEL_TEXT", "MODEL_STRONG", "MODEL_VISION", "MODEL_OCR", "BIDTABLE_CACHE",
)


def _child_env(*, utf8: bool, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in AMBIENT_ENV}
    env["PYTHONPATH"] = str(REPO_ROOT)
    if utf8:
        env["PYTHONIOENCODING"] = "utf-8"
    else:
        # 两个逃生口都得摘掉。只摘 PYTHONUTF8 是不够的：跑 pytest 的 shell 通常自己
        # 就带着 PYTHONIOENCODING=utf-8，子进程会原样继承，于是"控制台不是 UTF-8"
        # 那条用例悄悄变成空断言（S43 实测踩过：报出来的码本是 utf-8 而不是 cp936）。
        env.pop("PYTHONUTF8", None)
        env.pop("PYTHONIOENCODING", None)
    if extra:
        env.update(extra)      # 放在最后：extra 是"我就要这个码本"的显式意图
    return env


@pytest.fixture(scope="session")
def child_run() -> Callable[..., subprocess.CompletedProcess]:
    """``child_run(["run.py", "--pack", ...])`` -> 跑完的子进程。

    默认 cwd 是仓库根、stdout/stderr 按 utf-8 解码；
    ``cwd=`` 用来验证命令行不依赖当前目录，``utf8=False`` 用来拿原始字节。
    """

    def _run(argv: list[str], *, utf8: bool = True, cwd: Path | None = None,
             timeout: int = 600,
             extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        decode: dict[str, Any] = {"encoding": "utf-8", "errors": "replace"} if utf8 else {}
        return subprocess.run(
            [sys.executable, *argv],
            cwd=str(cwd or REPO_ROOT), env=_child_env(utf8=utf8, extra=extra_env),
            capture_output=True, timeout=timeout, **decode,
        )

    return _run


@pytest.fixture(scope="session")
def training_dir() -> Path:
    if not TRAINING.is_dir():
        pytest.skip(f"训练集缺失：{TRAINING}")
    return TRAINING
