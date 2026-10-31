"""异常层级。

设计原则（agent.md 4.5）：
1. 面向用户的消息必须可行动，绝不裸露 traceback。
2. 能降级就不要崩；只有「无法产出结果」才向上抛。
"""

from __future__ import annotations


class BidTableError(Exception):
    """所有业务异常的基类。``str(e)`` 即可直接打印给用户。"""

    exit_code = 1


class InputError(BidTableError):
    """输入层面的问题（路径、缺文件、格式不对）。退出码 1。"""

    exit_code = 1


class PackNotFoundError(InputError):
    """``--pack`` 指向的目录不存在。"""


class PackIncompleteError(InputError):
    """资料包缺关键文件。``missing`` 列出缺失项。"""

    def __init__(self, message: str, missing: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.missing = missing


class TenderParseError(BidTableError):
    """招标 PDF 读不出来 / 0 页 / 加密。退出码 2：无法产出结果。"""

    exit_code = 2


class UnsupportedLayoutError(BidTableError):
    """未识别的版面族。

    正常流程不应抛这个异常：按 4.5 第 3 条，未知版面走降级抽取 + 显眼告警。
    保留它是为了测试与显式 ``--strict-layout`` 模式。
    """

    exit_code = 2


class VendorDocError(InputError):
    """厂商资料无法解析（无表格 / 表头列数不对）。单文件失败只告警不中断。"""


class OutputError(BidTableError):
    """结果写不出去（目标被 Excel 占用等）。退出码 2。"""

    exit_code = 2


class LLMError(BidTableError):
    """LLM 层异常基类。全部可降级，不应导致整体失败。"""

    exit_code = 2


class LLMAuthError(LLMError):
    """401 / 403 之外的鉴权类失败（401 无效令牌、403 配额耗尽）。"""


class LLMQuotaError(LLMError):
    """账户欠费、免费额度用尽、模型不存在。"""


class LLMRateLimitError(LLMError):
    """429。可重试。"""

    retryable = True


class LLMTimeoutError(LLMError):
    """超时。可重试。"""

    retryable = True


class LLMResponseError(LLMError):
    """返回体不是预期结构（例如要求 JSON 却给了散文）。可降级。"""
