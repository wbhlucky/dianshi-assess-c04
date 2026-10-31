# -*- coding: utf-8 -*-
"""厂商侧响应 -> 结构化事实（判定层第二段）。

与 :mod:`bidtable.judge.parse_req` 共用同一套扫描器，但**不抽比较符**：
响应里的 ``≥``/``以上`` 只是厂商在复述门槛，方向永远由要求侧决定。

两条来自训练集实证的设计约束：

* **拒答短语整体匹配**（agent.md X21）。``暂不支持`` 是负偏离，
  而 ``支持（水夹层过滤器）`` 只是把要求名重复了一遍，绝不能误伤。
  所以这里做的是 squash 之后的**整串相等**，不是子串包含。
* **只扫 ``value``，不扫 ``name``。** 证据名里也常带数字
  （``272μm光纤最大传输功率``、``3D内窥镜像素``），把它们当响应值
  会凭空造出一个 ``≥272μm`` 的假约束。名称只在对齐子句时用。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import NEGATIVE_PHRASES
from ..match.normalize import squash
from ..models import Evidence
from .parse_req import Bound, Scan, clauses, scan
from .units import Quantity, Resolution, period_of

#: 拒答短语的 squash 形。与 :mod:`bidtable.match.candidates` 的 ``_REFUSALS``
#: 必须保持一致（测试 ``test_refusal_vocabularies_agree`` 会盯住这一点）。
REFUSAL_KEYS: tuple[str, ...] = tuple(squash(p) for p in NEGATIVE_PHRASES)


@dataclass(frozen=True)
class ValSpec:
    """一条厂商响应的可比对结构。"""

    raw: str  # ``f"{name}：{value}"`` 原样，便于复核报告回溯
    name: str
    value: str
    text: str  # squash(value)
    refusal: str | None = None  # 命中的拒答短语原文（未命中为 None）
    is_pointer: bool = False
    quantities: tuple[Quantity, ...] = ()
    bounds: tuple[Bound, ...] = ()
    resolutions: tuple[Resolution, ...] = ()
    period: str = ""  # ``每年/每月/…`` 归一形，用于 ``次`` ↔ ``次/年`` 搭桥
    scans: tuple[Scan, ...] = field(default=(), repr=False)

    @property
    def resolution(self) -> Resolution | None:
        return self.resolutions[0] if self.resolutions else None

    @property
    def comparable(self) -> bool:
        """有没有任何可以和数值门槛对上的东西。"""
        return bool(self.quantities or self.bounds or self.resolutions)

    @property
    def is_text(self) -> bool:
        return not self.comparable and self.refusal is None


# --- 解析 -------------------------------------------------------------------


def refusal_of(value: str | None) -> str | None:
    """整串命中拒答短语时返回该短语原文，否则 ``None``。

    ``支持（水夹层过滤器）`` 的 squash 形不等于任何短语，所以返回 None（X21）；
    ``暂不支持`` 返回 ``"暂不支持"``。子串匹配会同时误伤这两类，故不用。
    """
    s = squash(value)
    if not s:
        return None
    for phrase, key in zip(NEGATIVE_PHRASES, REFUSAL_KEYS):
        if s == key:
            return phrase
    return None


def scan_value(value: str | None) -> tuple[str, ...]:
    """响应值按子句切开后的 squash 形，与 :func:`parse_value` 内部扫描同源。"""
    return tuple(squash(c) for c in clauses(value))


def parse_value(ev: Evidence | None) -> ValSpec:
    """把一条 :class:`~bidtable.models.Evidence` 解析成 :class:`ValSpec`。

    纯函数。``ev is None`` 时返回空壳（调用方一般在更早的分支就判了未响应，
    这里只是让它不至于抛异常）。
    """
    if ev is None:
        return ValSpec(raw="", name="", value="", text="")
    parts = clauses(ev.value)
    scans = tuple(scan(c) for c in parts)
    quantities: list[Quantity] = []
    bounds: list[Bound] = []
    resolutions: list[Resolution] = []
    for sc, clause in zip(scans, parts):
        quantities += [h.quantity for h in sc.hits]
        bounds += [Bound(iv, clause) for iv in sc.bounds]
        resolutions += list(sc.resolutions)
    return ValSpec(
        raw=ev.response_text or f"{ev.name}：{ev.value}",
        name=ev.name,
        value=ev.value,
        text=squash(ev.value),
        refusal=refusal_of(ev.value),
        is_pointer=bool(ev.is_pointer),
        quantities=tuple(quantities),
        bounds=tuple(bounds),
        resolutions=tuple(resolutions),
        period=period_of(f"{ev.name} {ev.value}"),
        scans=scans,
    )
