"""进度与告警输出层（EC-J8）。

产品感验收（agent.md 10.3）要求分阶段进度、告警、以及「非零退出码必须有可读原因」。
这里只测输出契约本身，不测业务：格式一旦漂移，复核报告和 CI 日志就同时失效。
"""

from __future__ import annotations

import io

import pytest

from bidtable.logging_ import STAGE_NAMES, TOTAL_STAGES, Progress

pytestmark = [pytest.mark.unit]


def _make(**kw) -> tuple[Progress, io.StringIO]:
    buf = io.StringIO()
    return Progress(stream=buf, **kw), buf


@pytest.mark.ec("J8")
def test_stage_lines_carry_index_name_and_detail():
    """七个阶段名按序对齐，detail 追加在省略号之后。"""
    prog, buf = _make()
    for i, name in enumerate(STAGE_NAMES, start=1):
        prog.stage(i)
    prog.stage(5, "17 条参与匹配")
    lines = buf.getvalue().splitlines()
    assert lines[0] == "[1/7] 解析投标范围…"
    assert lines[-1] == "[5/7] 要求-证据匹配… 17 条参与匹配"
    assert [ln.split("] ")[1].rstrip("…") for ln in lines[:7]] == list(STAGE_NAMES)
    assert TOTAL_STAGES == len(STAGE_NAMES) == 7


@pytest.mark.ec("J8")
def test_out_of_range_stage_falls_back_instead_of_crashing():
    """阶段号越界时降级成「阶段N」，不抛 IndexError——进度打印不该成为故障源。"""
    prog, buf = _make()
    prog.stage(0)
    prog.stage(99)
    assert buf.getvalue().splitlines() == ["[0/7] 阶段0…", "[99/7] 阶段99…"]


@pytest.mark.ec("J8")
def test_info_and_warn_have_distinct_gutters():
    """info 缩进对齐阶段正文，warn 有可 grep 的固定前缀。"""
    prog, buf = _make()
    prog.info("2 个标的")
    prog.warn("版面族未识别")
    assert buf.getvalue().splitlines() == ["      2 个标的", "  ! 告警：版面族未识别"]


@pytest.mark.ec("J8")
def test_quiet_suppresses_progress_but_keeps_the_log():
    """-q 只是不打屏，``.log`` 仍然完整——复核报告与落盘日志依赖它。"""
    prog, buf = _make(quiet=True)
    prog.stage(1, "细节")
    prog.info("信息")
    prog.warn("告警")
    assert buf.getvalue() == ""
    assert len(prog.log) == 3
    assert prog.log[0] == "[1/7] 解析投标范围… 细节"


@pytest.mark.ec("J7")
def test_error_prints_even_when_quiet():
    """``-q`` 不能把非零退出码的原因一起吃掉（N1~N15 的可读消息前提）。"""
    prog, buf = _make(quiet=True)
    prog.stage(1)
    prog.error("资料包不存在：D:\\nope")
    assert buf.getvalue() == "  × 错误：资料包不存在：D:\\nope\n"
    assert len(prog.log) == 2


@pytest.mark.ec("J8")
def test_default_stream_is_stderr():
    """进度走 stderr，stdout 留给可被管道消费的摘要行。"""
    prog = Progress()
    assert prog.stream is __import__("sys").stderr
    assert prog.quiet is False
