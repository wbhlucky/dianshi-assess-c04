"""四种传输（agent.md 6.5）。

``LiveTransport``     真调 API（仅 ``-m eval`` / ``-m network``）
``RecordingTransport`` 真调并落盘 recorded.jsonl
``ReplayTransport``   从 recorded.jsonl 命中返回；未命中即失败并报出缺失键
``FakeTransport``     单元测试桩：按调用序返回预设值，或抛出指定 HTTP 错误

四种传输共享同一个 ``send(Request) -> Response`` 接口，
因此离线测试与真跑走完全相同的代码路径。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .. import errors


@dataclass(frozen=True)
class Request:
    model: str
    messages: list[dict[str, Any]]
    params: dict[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        blob = json.dumps(
            {"model": self.model, "messages": self.messages, "params": self.params},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Response:
    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)


class Transport:
    """传输基类。子类实现 :meth:`send`。"""

    name = "base"

    def send(self, request: Request) -> Response:  # pragma: no cover - 抽象
        raise NotImplementedError

    def close(self) -> None:
        return None


def classify_http(status: int, body: str) -> errors.LLMError:
    """把 HTTP 状态码 + 响应体分类到具体异常（N9 / B19）。

    实测过的响应体片段直接参与分类：``无效的令牌`` -> 401 鉴权，
    ``overdue-payment`` / ``Free quota exhausted`` -> 配额。
    """
    low = body.lower()
    if status == 401 or "无效的令牌" in body or "invalid api key" in low:
        return errors.LLMAuthError(
            f"模型 API 返回 401（令牌无效或已过期）。"
            f"请检查 LLM_API_KEY / DASHSCOPE_API_KEY 是否可用；"
            f"已缓存的调用不受影响，配置可用 Key 后重跑即可自动补全。原始响应：{body[:200]}"
        )
    if status == 403 or "quota" in low or "exhausted" in low:
        return errors.LLMQuotaError(
            f"模型 API 返回 403（额度用尽或无该模型权限）。原始响应：{body[:200]}"
        )
    if status == 404 or "not found" in low or "does not exist" in low:
        return errors.LLMQuotaError(
            f"模型 API 返回 404（模型名不存在）。请用 --model-ocr / --model-vision "
            f"换成当前账户可用的模型。原始响应：{body[:200]}"
        )
    if status == 429:
        return errors.LLMRateLimitError(f"模型 API 返回 429（限流），将退避重试。{body[:200]}")
    if 400 <= status < 500:
        if "overdue" in low or "欠费" in body or "access denied" in low or "arrearage" in low:
            return errors.LLMQuotaError(
                f"模型 API 返回 {status}（账户欠费或权限不足）。"
                f"已切换到离线模式：检测报告中的参数无法取证，相关行标记为「未响应」"
                f"并列入复核报告。配置可用的 LLM_API_KEY 后重跑即可自动补全，"
                f"已缓存的调用不会重复计费。原始响应：{body[:200]}"
            )
        return errors.LLMResponseError(f"模型 API 返回 {status}。原始响应：{body[:200]}")
    if status >= 500:
        return errors.LLMRateLimitError(f"模型 API 返回 {status}（服务端错误），将退避重试。")
    return errors.LLMResponseError(f"模型 API 返回意外状态 {status}。原始响应：{body[:200]}")


class LiveTransport(Transport):
    """OpenAI 兼容 HTTP 传输。用 ``openai`` SDK，失败时映射到 errors 层级。"""

    name = "live"

    def __init__(self, api_key: str, base_url: str, *, timeout: float = 60.0) -> None:
        if not api_key:
            raise errors.LLMAuthError("未配置 LLM_API_KEY，无法使用 LiveTransport。")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.calls = 0

    def send(self, request: Request) -> Response:
        self.calls += 1
        kwargs: dict[str, Any] = {"model": request.model, "messages": request.messages}
        kwargs.update(request.params)
        try:
            completion = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - SDK 异常类型跨版本不稳定
            raise _from_sdk_exception(exc) from exc
        choice = completion.choices[0].message
        usage = {}
        if getattr(completion, "usage", None):
            usage = {
                "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(completion.usage, "completion_tokens", 0) or 0,
            }
        return Response(text=choice.content or "", usage=usage)


def _from_sdk_exception(exc: Exception) -> errors.LLMError:
    status = getattr(exc, "status_code", None)
    body = str(getattr(exc, "response", None) or exc)
    name = type(exc).__name__
    if "Timeout" in name or "timed out" in body.lower():
        return errors.LLMTimeoutError(f"模型 API 调用超时。{body[:200]}")
    if "RateLimit" in name:
        return errors.LLMRateLimitError(f"模型 API 限流。{body[:200]}")
    if isinstance(status, int):
        return classify_http(status, body)
    return errors.LLMResponseError(f"模型 API 调用失败：{body[:200]}")


class RecordingTransport(Transport):
    """真调并把 (digest -> response) 追加进 jsonl，用于一次性录制回放语料。"""

    name = "record"

    def __init__(self, inner: Transport, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.calls = 0

    def send(self, request: Request) -> Response:
        resp = self.inner.send(request)
        self.calls += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "key": request.digest(),
            "request": {
                "model": request.model,
                "messages": request.messages,
                "params": request.params,
            },
            "text": resp.text,
            "usage": resp.usage,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return resp

    def close(self) -> None:
        self.inner.close()


class ReplayError(errors.LLMError):
    """回放未命中。消息里带缺失键，便于定位是哪次调用没录到。"""


class ReplayTransport(Transport):
    """从 recorded.jsonl 命中返回。未命中即失败并报出缺失的键。"""

    name = "replay"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.table: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.table[rec.get("key", "")] = rec
        self.calls = 0
        self.misses: list[str] = []

    def send(self, request: Request) -> Response:
        key = request.digest()
        rec = self.table.get(key)
        if rec is None:
            self.misses.append(key)
            raise ReplayError(
                f"回放语料未命中（key={key[:16]}…，model={request.model}）。"
                f"录制文件：{self.path}。请重跑录制或改用 --offline。"
            )
        self.calls += 1
        return Response(text=rec.get("text", ""), usage=rec.get("usage", {}))


class FakeTransport(Transport):
    """可编程桩。

    - ``responses`` 为字符串序列时按调用序返回；
    - 为 dict 时按 ``request.model`` 或 ``request.digest()`` 命中；
    - 为 callable 时直接调用；
    - 元素若是 ``Exception`` 实例则抛出（用于 N9 / B19）。
    """

    name = "fake"

    def __init__(
        self,
        responses: Sequence[Any] | dict[str, Any] | Callable[[Request], Any] | None = None,
        *,
        default: str = "",
    ) -> None:
        self.responses = list(responses) if isinstance(responses, (list, tuple)) else responses
        self.default = default
        self.calls = 0
        self.seen: list[Request] = []

    def send(self, request: Request) -> Response:
        self.seen.append(request)
        index = self.calls
        self.calls += 1
        item: Any = self.default
        if callable(self.responses):
            item = self.responses(request)
        elif isinstance(self.responses, dict):
            item = self.responses.get(request.digest(), self.responses.get(request.model, self.default))
        elif isinstance(self.responses, list):
            item = self.responses[index] if index < len(self.responses) else self.default
        if isinstance(item, Exception):
            raise item
        if isinstance(item, Response):
            return item
        return Response(text=str(item))
