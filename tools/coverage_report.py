#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""``coverage`` 的替身：用标准库 ``sys.monitoring`` 量 ``bidtable/`` 的行覆盖率。

agent.md 6.9 的门禁之一是「``coverage``（``bidtable/`` 排除 ``llm/``）行覆盖 ≥70%」。
但这台机器上装不了 ``coverage`` / ``pytest-cov``（``ruff`` 同样装不了，
用 ``python -m compileall`` 顶替，见 README）。与其把门禁写成"跑不了"，
不如用 Python 3.12 自带的 PEP 669 监控接口自己量：它和 ``coverage.py``
用的是同一层机制（``sys.settrace`` 的现代替代），语义足够对齐。

两条必须知道的实现约束：

1. ``sys.monitoring`` **只能监控当前进程**。所以本工具不 spawn pytest，
   而是 ``import pytest`` 之后调 ``pytest.main([...])``，在同一进程里把离线套件跑完。
   推论：``tests/eval/`` 那层起子进程，本工具量不到——但离线套件本来也不跑它
   （``pytest.ini`` 的 addopts 会反选 eval/network/slow），所以口径是自洽的。
2. 行号回调第一次命中就返回 ``DISABLE``，把该 (代码对象, 位置) 的事件永久关掉。
   覆盖率只需要"这行跑过没有"、不需要"跑了几次"，于是稳态开销接近于零。

「可执行行」用 AST 静态数：所有语句节点的首行，扣掉 docstring 与占位省略号。
这与 ``coverage.py`` 的口径接近但不逐行相同（多行语句、``if TYPE_CHECKING``
块的处理各有出入），所以报出来的百分比应当读作**本工具口径下的门禁值**，
不要拿去跟 ``coverage report`` 对账。

用法::

    python tools/coverage_report.py                  # 跑离线套件、出表，未达门禁退出 1
    python tools/coverage_report.py --min 80
    python tools/coverage_report.py --json out/coverage.json
    python tools/coverage_report.py --missed         # 顺带列出未覆盖的行号
    python tools/coverage_report.py -- tests/unit -q # 只量一部分套件
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "bidtable"
EXCLUDE_PARTS = frozenset({"llm"})          # 6.9：bidtable/ 排除 llm/
DEFAULT_MIN = 70.0
DEFAULT_ARGS = ("tests", "-q")
_TOOL_NAME = "bidtable-coverage"

_EXECUTED: dict[str, set[int]] = {}
_WATCH: dict[str, bool] = {}


def _in_scope(fn: str) -> bool:
    """这个文件在测量范围内吗？每个文件名只判一次，结果进 ``_WATCH``。"""
    try:
        rel = Path(fn).resolve().relative_to(TARGET)
    except (OSError, ValueError, RuntimeError):
        return False
    return not EXCLUDE_PARTS.intersection(rel.parts[:-1])


def _on_line(code: Any, line_number: int) -> Any:
    fn = code.co_filename
    want = _WATCH.get(fn)
    if want is None:
        want = _WATCH[fn] = _in_scope(fn)
    if want:
        _EXECUTED.setdefault(fn, set()).add(line_number)
    return sys.monitoring.DISABLE        # 记过一次就够，关掉这个位置的事件


def _executable(path: Path) -> set[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            v = node.value.value
            if isinstance(v, str) or v is Ellipsis:
                continue                 # docstring 与 "..." 占位不产生可执行行
        out.add(node.lineno)
    return out


def _targets() -> list[Path]:
    return sorted(p for p in TARGET.rglob("*.py")
                  if not EXCLUDE_PARTS.intersection(p.relative_to(TARGET).parts[:-1]))


def _measure(argv: list[str]) -> int:
    """在当前进程里开监控、跑 pytest、关监控，返回 pytest 的退出码。"""
    mon = sys.monitoring
    tool = mon.COVERAGE_ID
    mon.use_tool_id(tool, _TOOL_NAME)
    try:
        mon.register_callback(tool, mon.events.LINE, _on_line)
        mon.set_events(tool, mon.events.LINE)
        import pytest                     # 必须在开监控之后 import，才量得到导入期的行
        return int(pytest.main(argv))
    finally:
        mon.set_events(tool, 0)
        mon.register_callback(tool, mon.events.LINE, None)
        mon.free_tool_id(tool)


def _collect() -> list[dict[str, Any]]:
    seen: dict[Path, set[int]] = {}
    for fn, lines in _EXECUTED.items():
        try:
            seen[Path(fn).resolve()] = lines
        except OSError:
            continue
    rows: list[dict[str, Any]] = []
    for path in _targets():
        exe = _executable(path)
        got = seen.get(path.resolve(), set())
        hit = exe & got
        rows.append({
            "file": path.relative_to(REPO).as_posix(),
            "statements": len(exe),
            "covered": len(hit),
            "percent": round(100.0 * len(hit) / len(exe), 1) if exe else 100.0,
            "missed": sorted(exe - hit),
        })
    return rows


def _dw(s: str) -> int:
    """终端显示宽度：东亚宽/全角算 2，其余算 1。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, width: int, right: bool = False) -> str:
    gap = max(0, width - _dw(s))
    return (" " * gap + s) if right else (s + " " * gap)


def _render(rows: list[dict[str, Any]], total: dict[str, Any], min_pct: float,
            show_missed: bool) -> str:
    name_w = max([_dw(r["file"]) for r in rows] + [_dw("文件")])
    num_w = max(len(str(total["statements"])), 6)
    out = ["bidtable/ 行覆盖率（排除 llm/）", ""]
    out.append(_pad("文件", name_w) + "  " + _pad("可执行", num_w, True) + "  "
               + _pad("已覆盖", num_w, True) + "  " + _pad("覆盖率", 7, True))
    out.append(_pad("", name_w, False) + "  " + _pad("", num_w) + "  "
               + _pad("", num_w) + "  " + _pad("", 7))
    for r in sorted(rows, key=lambda x: x["percent"]):
        out.append(_pad(r["file"], name_w) + "  "
                   + _pad(str(r["statements"]), num_w, True) + "  "
                   + _pad(str(r["covered"]), num_w, True) + "  "
                   + _pad(f"{r['percent']:.1f}%", 7, True))
        if show_missed and r["missed"]:
            out.append("    未覆盖行：" + _compress(r["missed"]))
    out.append("")
    out.append(_pad("合计", name_w) + "  " + _pad(str(total["statements"]), num_w, True)
               + "  " + _pad(str(total["covered"]), num_w, True) + "  "
               + _pad(f"{total['percent']:.1f}%", 7, True))
    ok = total["percent"] >= min_pct
    out.append("")
    out.append(f"  [{'PASS' if ok else 'FAIL'}] 行覆盖率 ≥{min_pct:g}%　实测 {total['percent']:.1f}%")
    return "\n".join(out)


def _compress(lines: list[int]) -> str:
    """``[3,4,5,9]`` -> ``3-5, 9``，免得长文件刷满屏。"""
    out: list[str] = []
    start = prev = lines[0]
    for n in lines[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return ", ".join(out)


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="coverage_report.py",
        description="用标准库 sys.monitoring 量 bidtable/（排除 llm/）的行覆盖率。",
        epilog="`--` 之后原样交给 pytest，例如：python tools/coverage_report.py -- tests/unit -q",
    )
    p.add_argument("--min", type=float, default=DEFAULT_MIN, dest="min_pct",
                   help=f"门禁百分比，默认 {DEFAULT_MIN:g}")
    p.add_argument("--json", default=None, help="把结果写到这个路径")
    p.add_argument("--missed", action="store_true", help="列出每个文件的未覆盖行号")
    p.add_argument("pytest_args", nargs="*", help="传给 pytest 的参数，默认 tests -q")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if not TARGET.is_dir():
        print(f"  × 错误：找不到测量目标 {TARGET}", file=sys.stderr)
        return 2
    if sys.version_info < (3, 12):
        print("  × 错误：sys.monitoring 需要 Python 3.12+。", file=sys.stderr)
        return 2
    os.chdir(REPO)                       # pytest 的 rootdir 与相对 testpaths 都依赖它
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    rc = _measure(list(args.pytest_args) or list(DEFAULT_ARGS))
    rows = _collect()
    total = {
        "statements": sum(r["statements"] for r in rows),
        "covered": sum(r["covered"] for r in rows),
    }
    total["percent"] = round(100.0 * total["covered"] / total["statements"], 1) if total["statements"] else 0.0
    print(_render(rows, total, args.min_pct, args.missed))

    if args.json:
        dest = Path(args.json)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"total": total, "files": rows, "min_pct": args.min_pct,
                                    "pytest_exit": rc}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"\n      覆盖率明细已写出：{dest}")

    if rc != 0:
        print(f"\n  × pytest 退出码 {rc}：套件本身没过，覆盖率数字不作数。", file=sys.stderr)
        return 1
    return 0 if total["percent"] >= args.min_pct else 1


if __name__ == "__main__":
    sys.exit(main())
