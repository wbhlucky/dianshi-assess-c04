"""异常层级与退出码分层（agent.md 4.5 / EC-J7）。

退出码是 CLI 与调用方之间唯一的机器可读契约：
``1`` = 你给的东西不对，改输入再跑；``2`` = 我产不出结果。
分错层，脚本就会在「用户笔误」上重试三次。
"""

from __future__ import annotations

import pytest

from bidtable import errors as E

pytestmark = [pytest.mark.unit]

INPUT_LEVEL = (E.InputError, E.PackNotFoundError, E.PackIncompleteError, E.VendorDocError)
FATAL_LEVEL = (E.TenderParseError, E.UnsupportedLayoutError, E.OutputError)
LLM_LEVEL = (
    E.LLMError, E.LLMAuthError, E.LLMQuotaError,
    E.LLMResponseError, E.LLMRateLimitError, E.LLMTimeoutError,
)


@pytest.mark.ec("J7")
def test_exit_code_layers_input_vs_unproducible():
    """输入类一律 1，产不出结果一律 2；基类兜底也是 1。"""
    assert E.BidTableError.exit_code == 1
    for cls in INPUT_LEVEL:
        assert cls.exit_code == 1, cls.__name__
    for cls in FATAL_LEVEL:
        assert cls.exit_code == 2, cls.__name__


@pytest.mark.ec("J7")
def test_every_error_is_catchable_at_one_place():
    """run.py / selfcheck 只 ``except BidTableError``，漏一个子类就会裸 traceback。"""
    for cls in (*INPUT_LEVEL, *FATAL_LEVEL, *LLM_LEVEL):
        assert issubclass(cls, E.BidTableError), cls.__name__


@pytest.mark.ec("J1")
def test_only_rate_limit_and_timeout_are_retryable():
    """401/403/404 重试只会烧钱刷日志；429/超时才值得退避。"""
    assert E.LLMRateLimitError.retryable is True
    assert E.LLMTimeoutError.retryable is True
    for cls in (E.LLMAuthError, E.LLMQuotaError, E.LLMResponseError):
        assert getattr(cls, "retryable", False) is False, cls.__name__


@pytest.mark.ec("J7")
def test_llm_errors_stay_degradable():
    """LLM 异常虽然 exit_code=2，但按 4.5 第 2 条绝不允许冒到顶层——
    这里断言它们与致命错误不在同一继承支，便于 ``except`` 顺序表达该约束。"""
    for cls in LLM_LEVEL:
        assert not issubclass(cls, E.InputError)
    assert issubclass(E.LLMAuthError, E.LLMError)


@pytest.mark.ec("J7")
def test_pack_incomplete_carries_the_missing_list():
    """N2 要求「列出缺什么」，所以缺失项必须挂在异常上而不是只写在消息里。"""
    exc = E.PackIncompleteError("资料包不完整", ("01_招标文件_*.pdf（招标 PDF）",))
    assert exc.missing == ("01_招标文件_*.pdf（招标 PDF）",)
    assert E.PackIncompleteError("x").missing == ()
    assert str(exc) == "资料包不完整"


@pytest.mark.ec("A4")
def test_unsupported_layout_is_declared_but_off_the_happy_path():
    """未知版面走降级抽取（N13），不抛这个异常；保留它只为显式严格模式与测试。"""
    assert E.UnsupportedLayoutError.exit_code == 2
    assert E.UnsupportedLayoutError.__doc__.strip().startswith("未识别的版面族")
