"""OpenAI 兼容客户端：缓存优先 + 重试退避 + 错误分类 + 离线降级。

调用顺序：磁盘缓存 -> 传输层。命中缓存时 ``transport.calls`` 不增长，
这正是 N10 / B18「第二次运行 API 调用数 == 0」的断言依据。
"""

from __future__ import annotations

import time
from typing import Any

from .. import errors
from ..config import Config
from .cache import DiskCache, cache_key
from .transport import (
    FakeTransport,
    LiveTransport,
    RecordingTransport,
    ReplayTransport,
    Request,
    Response,
    Transport,
)

RETRYABLE = (errors.LLMRateLimitError, errors.LLMTimeoutError)


def build_transport(cfg: Config, *, replay_path: str | None = None) -> Transport | None:
    """按配置选传输。离线或无 Key 时返回 ``None``（调用方自行降级）。"""
    mode = (cfg.transport or "auto").lower()
    if mode == "fake":
        return FakeTransport()
    if mode == "replay":
        return ReplayTransport(replay_path or "tests/fixtures/llm/recorded.jsonl")
    if mode in ("live", "record") and not cfg.api_key:
        return None
    if mode == "auto" and not cfg.llm_available:
        return None
    if not cfg.api_key or not cfg.base_url:
        return None
    live = LiveTransport(cfg.api_key, cfg.base_url, timeout=cfg.timeout)
    if mode == "record":
        return RecordingTransport(live, replay_path or ".cache/llm/recorded.jsonl")
    return live


class LLMClient:
    """三个窄口子共用的门面。``role`` 取值 ``ocr`` / ``match`` / ``judge``。"""

    def __init__(
        self,
        cfg: Config,
        transport: Transport | None = None,
        cache: DiskCache | None = None,
        *,
        sleep: Any = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.transport = transport if transport is not None else build_transport(cfg)
        self.cache = cache if cache is not None else DiskCache(
            cfg.llm_cache_dir, enabled=not cfg.offline
        )
        self._sleep = sleep
        self.calls = 0            # 真实发出（含重试）的请求数
        self.cache_hits = 0
        self.degraded: list[str] = []  # 降级原因，进复核报告

    # -- 角色 -> 模型 -------------------------------------------------------
    def model_for(self, role: str) -> str:
        return {
            "ocr": self.cfg.model_ocr,
            "vision": self.cfg.model_vision,
            "match": self.cfg.model_text,
            "judge": self.cfg.model_strong,
            "text": self.cfg.model_text,
        }.get(role, self.cfg.model_text)

    @property
    def enabled(self) -> bool:
        return self.transport is not None

    def complete(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        params: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Response | None:
        """返回 ``Response``；不可用或最终失败时返回 ``None`` 并记录降级原因。

        **绝不向上抛 LLMError**：4.5 第 2 条要求能降级就不要崩。
        """
        if self.transport is None:
            self._note(role, "离线模式：未配置可用的 API Key 或指定了 --offline")
            return None

        use_model = model or self.model_for(role)
        prm = params or {}
        key = cache_key(role, use_model, messages, **prm)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return Response(text=cached.get("text", ""), usage=cached.get("usage", {}))

        req = Request(model=use_model, messages=messages, params=prm)
        attempt = 0
        while True:
            attempt += 1
            try:
                self.calls += 1
                resp = self.transport.send(req)
            except RETRYABLE as exc:
                if attempt > self.cfg.max_retries:
                    self._note(role, f"重试 {self.cfg.max_retries} 次后仍失败：{exc}")
                    return None
                self._sleep(min(2 ** (attempt - 1), 8))
                continue
            except errors.LLMError as exc:
                self._note(role, str(exc))
                return None
            except Exception as exc:  # noqa: BLE001 - 传输层意外错误也要降级
                self._note(role, f"传输层意外错误：{type(exc).__name__}: {exc}")
                return None
            self.cache.put(key, {"text": resp.text, "usage": resp.usage, "model": use_model})
            return resp

    def _note(self, role: str, reason: str) -> None:
        msg = f"[{role}] {reason}"
        if msg not in self.degraded:
            self.degraded.append(msg)
