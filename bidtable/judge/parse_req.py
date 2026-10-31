# -*- coding: utf-8 -*-
"""招标侧要求 -> 结构化约束（判定层第一段）。

本模块**只解析、不比较**：把一句话要求拆成若干可独立判定的子句，再从每个子句里
取出「门槛值 + 达标方向 + 单位」。比较在 :mod:`bidtable.judge.decide`，
单位换算在 :mod:`bidtable.judge.units`。

三条设计约束，全部来自 agent.md 3.9 / 7.4 的实证：

* **裸数字按「至少」处理。** ``拥有2个夹管阀`` 而厂商给 ``1个`` 是负偏离，
  给 ``2个`` 是无偏离；训练集里没有任何一条裸数字要求是「不得超过」语义，
  而 ``3种/2种``、``4个/3个`` 这类正偏离全部出自裸数字或 ``≥``。
* **区间端点不再单独成为约束。** ``33℃-43℃`` 只产出一个 :class:`Bound`，
  否则会被读成「≥33 且 ≥43」，把 P5 标的1 序号9（响应 ``37~43℃``）判成正偏离。
* **子句是复合要求的判定单位（X20）。** ``血泵流速…；透析液流速…；脱水速率…``
  四个子参数只答一个时只判被答的那一句，所以每个约束都记住自己的子句原文，
  供 :mod:`~bidtable.judge.decide` 按名称对齐。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from ..match.normalize import NUM_UNIT_RE, squash, unit_key
from .units import (
    Interval,
    Quantity,
    Resolution,
    parse_intervals,
    parse_resolutions,
    period_of,
)

#: 子句分隔符。刻意**不含** ``.`` 与 ``、``：前者会切断 ``1.9.1`` 这类序号，
#: 后者是同一子句内的并列词（``性能检测、电气安全性检测``），切开会丢上下文。
CLAUSE_RE = re.compile(r"[;；。,，\n\r]+")

#: 前置比较符 -> 达标方向。``min`` = 越大越好（下界），``max`` = 越小越好（上界）。
#: 扫描时按长度倒序，保证 ``不低于`` 抢在 ``不`` 前、``不大于`` 抢在 ``大于`` 前。
_PREFIX_OPS: tuple[tuple[str, str], ...] = (
    ("≥", "min"), (">=", "min"), ("＞=", "min"), ("＞", "min"), (">", "min"),
    ("≤", "max"), ("<=", "max"), ("＜=", "max"), ("＜", "max"), ("<", "max"),
    ("不低于", "min"), ("不小于", "min"), ("不少于", "min"), ("至少", "min"),
    ("不大于", "max"), ("不超过", "max"), ("不多于", "max"), ("以内", "max"),
    ("高于", "min"), ("低于", "max"), ("大于", "min"), ("小于", "max"),
)
#: 后置比较符。中文习惯把方向写在数字后面：``10分钟以上``、``3mm以下``。
_SUFFIX_OPS: tuple[tuple[str, str], ...] = (
    ("以上", "min"), ("以下", "max"), ("以内", "max"),
)
_PREFIX_BY_LEN = tuple(sorted(_PREFIX_OPS, key=lambda t: -len(t[0])))
_SUFFIX_BY_LEN = tuple(sorted(_SUFFIX_OPS, key=lambda t: -len(t[0])))

#: 向前/向后找比较符的最大字符距离。窗口太宽会吃到上一个子句的比较符
#: （``≥2倍光学变焦，≥5倍电子放大`` 里两个 ``≥`` 各管各的）。
OP_WINDOW = 6

#: 严格不等号。判定上仍按 :func:`units.close` 放行取等——训练集里 ``＞100℃``
#: 对 ``105℃`` 是无偏离，没有「恰好等于严格界」的反例可依据。
STRICT_OPS = frozenset({"＞", ">", "＜", "<", "大于", "小于"})

#: 子句里出现这些词、**且整条要求里另有区间**时，门槛比的是区间**宽度**而不是端点
#: （``200μs～800μs可调…区间范围≥600μs``，agent.md 7.4.3）。
#: 两个条件缺一不可：``工作范围≥300mm`` 这种没有区间的，仍按下界判。
WIDTH_WORDS = ("区间范围", "可调范围", "调节范围", "跨度", "宽度", "范围")

#: 单字母单位后面紧贴这些字符时，它是**产品代际标签**而不是单位：
#: ``3D内窥镜``、``3D/2D切换``、``4K3D摄像系统`` 里的 ``3D`` 不是「3 屈光度」。
#: 真单位的右边只会是标点、数字、字符串结尾或另一个单位字符（``-24.5D～+20D``
#: 的 ``D`` 后面是 ``-`` 或结尾），绝不会直接贴汉字——所以这个判据不需要词表。
_DESIGNATOR_NEXT = re.compile(r"[\u4e00-\u9fff/]")


# --- 数据结构 ---------------------------------------------------------------


@dataclass(frozen=True)
class Hit:
    """子句里命中的一个「数值 + 单位」，连同它在 squash 文本里的字符区间。

    带上位置是为了两件事：向前/向后找比较符（:func:`_op_at`），
    以及把落在 ``33℃-43℃`` / ``3840x2160`` 字面区间**内部**的数字排除掉。
    """

    quantity: Quantity
    start: int
    end: int


@dataclass(frozen=True)
class Scan:
    """一个子句的解析结果。``text`` 是 squash 之后的子句原文。"""

    text: str
    hits: tuple[Hit, ...] = ()
    bounds: tuple[Interval, ...] = ()
    resolutions: tuple[Resolution, ...] = ()


@dataclass(frozen=True)
class Constraint:
    """一个「门槛 + 方向」约束。"""

    direction: str  # "min"（越大越好）/ "max"（越小越好）
    quantity: Quantity
    op: str  # 原文命中的比较符；"" = 裸数字，按 min 处理
    clause: str  # 子句原文，X20 复合要求按它对齐被回答的那一句

    @property
    def strict(self) -> bool:
        """是否严格不等号。目前只进复核报告，不改变判定（见 :data:`STRICT_OPS`）。"""
        return self.op in STRICT_OPS

    @property
    def dimension(self) -> str:
        return self.quantity.dimension

    def is_width(self, has_bound: bool) -> bool:
        """该约束比的是区间宽度还是端点值。"""
        return has_bound and any(w in self.clause for w in WIDTH_WORDS)


@dataclass(frozen=True)
class Bound:
    """要求里的一个区间约束，连同它所在的子句。"""

    interval: Interval
    clause: str


@dataclass(frozen=True)
class ReqSpec:
    """一条要求的完整可比对结构。"""

    raw: str
    text: str  # squash 之后
    constraints: tuple[Constraint, ...] = ()
    bounds: tuple[Bound, ...] = ()
    resolution: Resolution | None = None
    resolution_clause: str = ""
    period: str = ""  # `每年/每月/…` 归一后的周期词，用于 `次` ↔ `次/年` 搭桥

    @property
    def kind(self) -> str:
        """粗分类，只用于复核报告与测试断言；判定本身按结构逐项走。"""
        if self.resolution is not None:
            return "resolution"
        if self.bounds:
            return "interval"
        if self.constraints:
            return "numeric"
        return "text"

    @property
    def comparable(self) -> bool:
        return self.kind != "text"


# --- 解析 -------------------------------------------------------------------


def _dec(text: str) -> Decimal | None:
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError, ArithmeticError):
        return None


def clauses(text: str | None) -> list[str]:
    """按 :data:`CLAUSE_RE` 切子句，丢掉空白段（``≤3mm；`` 末尾会切出一个空串）。"""
    return [c.strip() for c in CLAUSE_RE.split(text or "") if c and c.strip()]


def _blocked(s: str, raws: tuple[str, ...]) -> list[tuple[int, int]]:
    """``raws`` 里每个字面片段在 ``s`` 中占据的字符区间（同一片段出现多次要全部记上）。"""
    out: list[tuple[int, int]] = []
    for raw in raws:
        if not raw:
            continue
        for m in re.finditer(re.escape(raw), s):
            out.append((m.start(), m.end()))
    return out


def _op_at(s: str, hit_start: int, hit_end: int) -> tuple[str, str]:
    """在数字前后 :data:`OP_WINDOW` 个字符内找比较符，返回 ``(方向, 命中的词)``。

    先前置、再后置、都没有则默认 ``min``（裸数字＝至少，见模块 docstring）。
    窗口很窄是刻意的：``≤±5%`` 的 ``≤`` 与 ``5`` 之间还夹着一个 ``±``，
    而 ``10分钟以上`` 的 ``以上`` 紧贴数字，两者都必须在 6 字符内命中。
    """
    head = s[max(0, hit_start - OP_WINDOW) : hit_start]
    for tok, direction in _PREFIX_BY_LEN:
        if tok in head:
            return direction, tok
    tail = s[hit_end : hit_end + OP_WINDOW]
    for tok, direction in _SUFFIX_BY_LEN:
        if tail.startswith(tok):
            return direction, tok
    return "min", ""


def _unit_of(s: str, m: re.Match) -> str:
    """``NUM_UNIT_RE`` 命中的单位归一形；产品代际标签按「无单位」处理。"""
    tok = m.group("unit")
    if not tok:
        return ""
    if len(tok) == 1 and tok.isascii() and tok.isalpha():
        if _DESIGNATOR_NEXT.match(s[m.end() : m.end() + 1]):
            return ""
    return unit_key(tok)


def scan(clause: str | None) -> Scan:
    """抽出一个子句里的全部数值信息，**不判方向**（方向由 :func:`parse_requirement` 补）。

    落在区间字面（``33℃-43℃``）与分辨率字面（``3840x2160``）内部的数字一律跳过，
    否则一个区间会被读成两个同向下界。``0.l-5ml`` 这类排版噪声由
    :func:`units.parse_intervals` 自己拒收，这里不会替它兜底。
    """
    s = squash(clause)
    if not s:
        return Scan(text="")
    bounds = parse_intervals(s)
    resolutions = parse_resolutions(s)
    blocked = _blocked(s, tuple(iv.raw for iv in bounds) + tuple(r.raw for r in resolutions))
    hits: list[Hit] = []
    for m in NUM_UNIT_RE.finditer(s):
        start = m.start("value")
        if any(a <= start < b for a, b in blocked):
            continue
        value = _dec(m.group("value"))
        if value is None:
            continue
        hits.append(
            Hit(
                quantity=Quantity(value=value, unit=_unit_of(s, m), raw=m.group(0)),
                start=start,
                end=m.end(),
            )
        )
    return Scan(text=s, hits=tuple(hits), bounds=tuple(bounds), resolutions=tuple(resolutions))


def parse_requirement(text: str | None) -> ReqSpec:
    """把一条招标要求解析成 :class:`ReqSpec`。**纯函数**，无 I/O、无副作用。"""
    raw = text or ""
    constraints: list[Constraint] = []
    bounds: list[Bound] = []
    resolution: Resolution | None = None
    resolution_clause = ""
    for clause in clauses(raw):
        sc = scan(clause)
        for hit in sc.hits:
            direction, op = _op_at(sc.text, hit.start, hit.end)
            constraints.append(Constraint(direction, hit.quantity, op, clause))
        bounds += [Bound(iv, clause) for iv in sc.bounds]
        if sc.resolutions and resolution is None:
            resolution, resolution_clause = sc.resolutions[0], clause
    return ReqSpec(
        raw=raw,
        text=squash(raw),
        constraints=tuple(constraints),
        bounds=tuple(bounds),
        resolution=resolution,
        resolution_clause=resolution_clause,
        period=period_of(raw),
    )
