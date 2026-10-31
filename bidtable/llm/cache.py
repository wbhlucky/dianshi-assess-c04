"""内容哈希 -> 磁盘 JSON 缓存。

键 = ``sha256(role + model + 归一化 messages + 关键 params)``。
第二次运行必须零 API 调用（agent.md 4.4 的硬测试 B18 / N10）。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_WS = re.compile(r"[ \t]+")


def _normalize_text(value: Any) -> Any:
    """抹掉不影响语义的空白差异，避免同一请求因排版抖动而 miss。"""
    if isinstance(value, str):
        return _WS.sub(" ", value).strip()
    if isinstance(value, list):
        return [_normalize_text(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_text(value[k]) for k in sorted(value)}
    return value


def cache_key(role: str, model: str, messages: list[dict[str, Any]], **params: Any) -> str:
    payload = {
        "role": role,
        "model": model,
        "messages": _normalize_text(messages),
        # 只纳入影响输出的参数，temperature 之类保留（不同值应产生不同条目）
        "params": _normalize_text({k: params[k] for k in sorted(params) if params[k] is not None}),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    """一个键一个 JSON 文件。损坏文件视为未命中，不抛异常。"""

    def __init__(self, directory: str | Path, *, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.directory / f"{key[:2]}" / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            self.misses += 1
            return None
        p = self._path(key)
        if not p.exists():
            self.misses += 1
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None
        if not isinstance(data, dict):
            # JSON 合法但形状不对（例如被别的工具写成了列表）——同样只能算未命中，
            # 否则调用方拿到的对象没有 .get()，会把整条流水线带崩。
            self.misses += 1
            return None
        self.hits += 1
        return data

    def put(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        p = self._path(key)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(p)
        except OSError:
            # 缓存写失败绝不能影响主流程
            pass

    def clear(self) -> None:
        import shutil

        if self.directory.exists():
            shutil.rmtree(self.directory, ignore_errors=True)
