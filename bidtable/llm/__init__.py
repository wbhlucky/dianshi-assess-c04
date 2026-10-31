"""LLM 窄口子（agent.md 4.3）。

只有三处允许调用模型：
L1 检测报告 OCR、L2 语义匹配兜底、L3 难判定裁决。
其余全部纯 Python、零网络。
"""

from .cache import DiskCache, cache_key
from .client import LLMClient
from .transport import (
    FakeTransport,
    LiveTransport,
    RecordingTransport,
    ReplayTransport,
    Request,
    Response,
)

__all__ = [
    "DiskCache",
    "cache_key",
    "LLMClient",
    "FakeTransport",
    "LiveTransport",
    "RecordingTransport",
    "ReplayTransport",
    "Request",
    "Response",
]
