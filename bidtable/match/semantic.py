"""L2 语义匹配兜底（agent.md 4.3 表格第 2 行）。

确定性打分器解决不了的只有两类：

1. **词面零重叠**——招标写「用于泌尿系结石的碎石」，厂商写「适用范围：暂不支持」。
    别名组能救一部分（:func:`bidtable.match.candidates.rescue_refusals`），
    但别名表必须保持短小，否则会引入误匹配（agent.md 8.2 R11 反过拟合）。
2. **一对多歧义**——同页几条指标名都沾边，词面分数咬得很近。

这两类都交给 LLM 二选一，且**只在确定性最高分低于 ``THRESHOLD`` 时才调用**。
离线（无可用 Key）时 :meth:`LLMClient.complete` 返回 ``None``，本模块随之返回
``None``，调用方保留确定性结果或判「未响应」，并打 ``llm_unavailable`` 旗标。

硬约束：本模块**不解析自由文本**，只认 ``{"index": <int|null>}``；
下标越界、JSON 损坏、返回 null 一律视作「LLM 没有意见」。
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Sequence

from ..llm.prompts import match_messages
from ..models import Requirement

if TYPE_CHECKING:  # 只为类型标注，运行期不需要 client 存在
    from ..llm.client import LLMClient
    from .candidates import Candidate

# 触发 LLM 兜底的最低确定性分数：低于 THRESHOLD 但高于它的候选才值得问一次。
# 再低的候选连「沾边」都算不上，问了只会引入幻觉匹配（agent.md N9）。
LLM_FLOOR = 0.35

_JSON_RE = re.compile(r"\{.*\}", re.S)


def _parse_index(text: str | None, n: int) -> int | None:
    """从模型回复里取出 ``index``。任何异常形态都返回 ``None``。"""
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj: Any = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    idx = obj.get("index")
    if idx is None or isinstance(idx, bool):
        return None
    try:
        i = int(idx)
    except (ValueError, TypeError):
        return None
    return i if 0 <= i < n else None


def suggest(
    req: Requirement,
    candidates: Sequence[Candidate],
    client: LLMClient | None,
) -> Candidate | None:
    """让 LLM 在 ``candidates`` 里挑一个；挑不出或不可用返回 ``None``。

    返回的 :class:`Candidate` 是**入参里的同一个对象**（已被就地打上
    ``llm_fallback`` 旗标），调用方不需要重新打分。
    """
    if client is None or not candidates:
        return None
    pool = [c for c in candidates if c.score >= LLM_FLOOR]
    if not pool:
        return None
    listing = [(c.evidence.name, c.evidence.value or "详见检测报告") for c in pool]
    resp = client.complete("match", match_messages(req.text, listing))
    if resp is None:
        return None
    idx = _parse_index(resp.text, len(pool))
    if idx is None:
        return None
    chosen = pool[idx]
    chosen.method = "llm"
    chosen.score = max(chosen.score, LLM_FLOOR)
    chosen.add_flag("llm_fallback")
    chosen.reason = f"确定性打分 {chosen.name_sim:.2f}/{chosen.value_sim:.2f} 未过阈值，由语义模型选定"
    return chosen
