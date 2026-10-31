# -*- coding: utf-8 -*-
"""偏离判定：把「要求结构 × 响应结构」对齐、比较、归并成四类偏离之一。

判定次序（agent.md 3.9）自上而下，命中即止：

1. 没有证据            -> ``未响应``（I5：响应/材料/位置三列同时留空）
2. 证据是检测报告指针  -> 按 ``--ocr-fallback`` 走「无偏离（推定）」或「未响应」
3. 响应整串是拒答短语  -> ``负偏离``（封闭词表，X21）
4. 逐字复述招标原文    -> ``无偏离``（复述即承诺满足，见 :data:`ECHO_MIN_LEN`）
5. 结构化比对          -> 逐子句对齐、单位换算、按方向与 τ 判
6. 无可比结构          -> ``无偏离``（X22：正偏离只可能来自数值）
7. LLM 仲裁（可选）    -> 只在低置信边界触发；离线时 ``complete`` 返回 None，静默跳过

**τ 是有方向的**，不是 ``abs(ratio-1)``。两条训练集实证把它钉死：
``＜4mJ`` 对 ``3mJ`` 是**无偏离**（比值 1.33 远超 τ），而 ``≤3mm`` 对 ``3.21mm``
是**负偏离**（比值仅 1.07）。也就是说上界方向**永远不会**判正偏离——
「越小越好」的指标做得更小只是达标，不构成对招标的「优于」。

经验间隙：训练集里最大的「无偏离」比值是 1.0786（755Lm/700Lm），
最小的「正偏离」比值是 1.1333（34秒/30秒），所以默认 τ=1.10 落在空隙正中。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Iterable, Sequence

from ..config import (
    DEV_NEGATIVE,
    DEV_NONE,
    DEV_NO_RESPONSE,
    DEV_POSITIVE,
    LAYOUT_UNKNOWN,
)
from ..ingest.scope import body_of
from ..llm.prompts import judge_messages
from ..match.candidates import blend
from ..match.normalize import squash
from ..models import Match
from . import units
from .parse_req import Constraint, ReqSpec, parse_requirement
from .parse_val import ValSpec, parse_value
from .units import Interval, Quantity, Resolution

#: τ 与低置信带宽的默认值。CLI ``--positive-margin`` 会覆盖 τ。
DEFAULT_MARGIN = 1.10
DEFAULT_BAND: tuple[float, float] = (1.05, 1.15)

#: 送给 LLM 仲裁的补充规则，与 :data:`JUDGE_SYSTEM` 不重复。
ARBITRATION_RULES = (
    "阈值 τ=1.10：只有数值型要求且实测值/要求值 > 1.10 才算正偏离；"
    "上界型要求（≤/不超过/以下）做得更小只算无偏离，绝不算正偏离；"
    "取等算无偏离；文字型要求一律无偏离。"
)

#: ``次`` 与 ``次/年`` 之间的桥。要求侧写「每年免费提供≥4次」而响应侧写
#: ``4.22次/年``：一个是 count 量纲、一个是 count_rate 量纲，不搭桥就永远不可比。
_RATE_TO_PERIOD = {"次/年": "年", "次/月": "月", "次/日": "日"}

#: 复核报告用的降信上限（旗标 -> 置信度天花板），按最小值生效。
_CONF_CAPS: tuple[tuple[str, float], ...] = (
    ("low_confidence", 0.55),
    ("low_margin", 0.60),
    ("ambiguous_evidence", 0.62),
    ("ocr_failed", 0.70),
    ("unit_assumed", 0.75),
)

#: 逐字复述招标原文的最小长度门槛（squash 后字符数）。
#: 低于它的短串很容易在长要求里"碰巧"出现，宁可交给结构化比对去判。
ECHO_MIN_LEN = 12

#: 未知版面族的置信度天花板（agent.md N13）。
#: 版面没认出来时 :func:`bidtable.ingest.tender._extract_unknown` 只能"全文捞参数行"，
#: 编号与列的对应关系都是猜的。这种行**必须**排在复核清单最前面，所以在这里统一
#: 压到 0.40 —— 低于 :data:`bidtable.output.review.CONF_REVIEW`(0.85)，
#: 不需要额外旗标就能保证每一行都进「优先复核」。
#: 不复用 ``low_confidence``：那条告警的措辞是"含数值门槛但找不到同量纲的量"，
#: 套在版面未识别上是假信息。响亮的告警由抽取层的【版面告警】负责。
UNKNOWN_LAYOUT_CAP = 0.40


# --- 数据结构 ---------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """一次判定运行需要的全部旋钮。:func:`settings_of` 从 Config 里取。"""

    tau: Decimal = Decimal("1.10")
    band: tuple[Decimal, Decimal] = (Decimal("1.05"), Decimal("1.15"))
    ocr_fallback: str = "assume"


@dataclass(frozen=True)
class Compare:
    """一次单点比较的结果。一条要求可能产出多个，最后由 :func:`reduce` 归并。"""

    deviation: str
    ratio: Decimal | None = None
    detail: str = ""
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Aligned:
    """两侧归一到同一量纲之后的数值对。"""

    need: Decimal
    got: Decimal
    unit: str
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    """一条要求的最终判定，直接供 :mod:`bidtable.pipeline` 组装 Row。"""

    deviation: str
    resp_text: str = ""
    evidence_file: str = ""
    location: str = ""
    confidence: float = 1.0
    flags: tuple[str, ...] = ()
    note: str = ""
    reason: str = ""
    method: str = "none"


# --- 旋钮与单位归一 ---------------------------------------------------------


def _dec(value: Any, default: float) -> Decimal:
    try:
        return Decimal(str(value))
    except (ArithmeticError, ValueError, TypeError):
        return Decimal(str(default))


def settings_of(cfg: Any) -> Settings:
    """从任意带同名属性的对象（通常是 :class:`~bidtable.config.Config`）取判定旋钮。

    用 ``getattr`` 而不是强类型，是为了让单元测试可以只造一个最小桩对象。
    """
    tau = _dec(getattr(cfg, "positive_margin", None), DEFAULT_MARGIN)
    band = tuple(getattr(cfg, "low_margin_band", None) or DEFAULT_BAND)
    lo, hi = (band + DEFAULT_BAND)[:2]
    fallback = getattr(cfg, "ocr_fallback", None) or "assume"
    return Settings(
        tau=tau,
        band=(_dec(lo, DEFAULT_BAND[0]), _dec(hi, DEFAULT_BAND[1])),
        ocr_fallback=fallback,
    )


def _bridge_period(ru: str, vu: str, rs_period: str, vs_period: str) -> tuple[str, str] | None:
    """``次`` ↔ ``次/年`` 搭桥。桥不通（周期词对不上）时返回 None。"""
    if ru == "次" and vu in _RATE_TO_PERIOD:
        want = _RATE_TO_PERIOD[vu]
        if not rs_period or rs_period == want:
            return (vu, vu)
    if vu == "次" and ru in _RATE_TO_PERIOD:
        want = _RATE_TO_PERIOD[ru]
        if not vs_period or vs_period == want:
            return (ru, ru)
    return None


def align_units(
    need: Quantity,
    got: Quantity,
    *,
    rs_period: str = "",
    vs_period: str = "",
) -> Aligned | None:
    """把「要求值 / 响应值」归一到同一量纲；不可比时返回 ``None``。

    三条通道，按代价从低到高：

    * 单位字面相同（含两侧都无单位）-> 直接比。
    * **B24 单向继承**：响应漏写单位（``触摸屏尺寸：6``）时继承要求侧单位，
      打 ``unit_assumed``。反方向**不继承**——要求侧的裸数字往往是「60 帧/秒」
      这类没有单位词的量，让它去继承 ``像素`` 会把 ``≥16万像素`` 判成 4000 倍正偏离。
    * 同量纲不同单位（X7/T12/B7）-> :func:`units.convert`，打 ``unit_converted``。
      ``9360mA`` vs ``≤10A`` 靠这条通道变成 ``9.36 vs 10``。
    """
    ru, vu = need.unit, got.unit
    if ru == vu:
        return Aligned(need.value, got.value, ru)
    if not vu and ru:
        return Aligned(need.value, got.value, ru, ("unit_assumed",))
    bridged = _bridge_period(ru, vu, rs_period, vs_period)
    if bridged is not None:
        ru, vu = bridged
        return Aligned(need.value, got.value, ru, ("unit_assumed",))
    if units.same_dimension(ru, vu):
        conv = units.convert(got.value, vu, ru)
        if conv is not None:
            return Aligned(need.value, conv, ru, ("unit_converted",))
    return None


def verdict(direction: str, need: Decimal, got: Decimal, st: Settings, *, detail: str = "") -> Compare:
    """单点判定。``direction`` = ``"min"``（越大越好）/ ``"max"``（越小越好）。"""
    ratio = units.ratio(got, need) if direction == "min" else units.ratio(need, got)
    if units.close(got, need):
        return Compare(DEV_NONE, Decimal(1), detail or f"实测 {got} 与要求取等，视为满足")
    if direction == "min":
        if got < need:
            return Compare(DEV_NEGATIVE, ratio, detail or f"实测 {got} 低于要求 {need}")
        if ratio is not None and ratio > st.tau:
            return Compare(DEV_POSITIVE, ratio, detail or f"实测 {got} 高出要求 {need}（{float(ratio):.2f} 倍 > τ）")
        return Compare(DEV_NONE, ratio, detail or f"实测 {got} 达标但未超 τ（{float(ratio or 0):.2f} 倍）")
    if got > need:
        return Compare(DEV_NEGATIVE, ratio, detail or f"实测 {got} 超过上限 {need}")
    return Compare(DEV_NONE, ratio, detail or f"实测 {got} 在上限 {need} 之内")


def margin_flag(ratio: Decimal | None, st: Settings) -> tuple[str, ...]:
    """落在低置信带宽里就打 ``low_margin``，供复核报告优先排序。"""
    if ratio is None:
        return ()
    return ("low_margin",) if st.band[0] <= ratio <= st.band[1] else ()


#: 归并优先级：负偏离 > 正偏离 > 无偏离。
_SEVERITY = {DEV_NONE: 0, DEV_POSITIVE: 1, DEV_NEGATIVE: 2, DEV_NO_RESPONSE: 3}


def reduce(compares: Iterable[Compare]) -> str:
    """多个子句判定的归并规则：任一负偏离即负偏离，否则任一正偏离即正偏离。"""
    worst = DEV_NONE
    for c in compares:
        if _SEVERITY.get(c.deviation, 0) > _SEVERITY.get(worst, 0):
            worst = c.deviation
    return worst


# --- 响应侧的可比对象 -------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """响应里一个可以拿来比的对象。``label`` 是它所在子句的 squash 形，供对齐。"""

    kind: str  # "point" | "interval" | "resolution"
    unit: str
    label: str
    point: Decimal | None = None
    interval: Interval | None = None


def response_items(vs: ValSpec) -> tuple[Item, ...]:
    """把 :class:`~bidtable.judge.parse_val.ValSpec` 摊平成对齐用的候选列表。"""
    name = squash(vs.name)
    items: list[Item] = []
    for sc in vs.scans:
        label = sc.text or name
        for iv in sc.bounds:
            items.append(Item("interval", iv.unit, label, interval=iv))
        for hit in sc.hits:
            items.append(Item("point", hit.quantity.unit, label, point=hit.quantity.value))
    for r in vs.resolutions:
        items.append(Item("resolution", r.unit, name, point=r.scalar))
    return tuple(items)


def endpoint(it: Item, direction: str, *, width: bool = False) -> Decimal | None:
    """取出该与门槛相比的那个数。

    * **宽度约束**（``区间范围≥600μs``）只能对区间比 ``hi-lo``；单值响应无从作答，
      返回 None 让它被跳过。若拿端点比，``200~800μs`` 会得到 800/600=1.33 的假正偏离。
    * **区间响应碰单点门槛**时取有利端点：下界要求看上端、上界要求看下端。
      厂商写 ``10~450ml/min`` 而要求 ``≥10ml/min``，能证明达标的是 450。
    """
    if width:
        return it.interval.width if it.interval is not None else None
    if it.interval is not None:
        return it.interval.hi if direction == "min" else it.interval.lo
    return it.point


def align_score(clause: str, name: str, it: Item) -> tuple[float, float]:
    """子句与响应项的对齐分，返回 ``(名字分, 数值分)``，**按字典序比较**。

    名字分必须排在第一键：厂商自己写的指标名才说明它在回答哪一句。
    反例是 P3 的 ``直径10mm，30°腹腔镜，长度≥300mm`` 对上 ``镜体长度：310mm``——
    ``直径10mm`` 与 ``310mm`` 的字符重合度（0.60）反而**高于** ``长度≥300mm``（0.35），
    把数值相似度当主键就会挑错子句，把无偏离判成 31 倍的正偏离。
    """
    return (blend(squash(name), clause), blend(clause, it.label))


def _rank(c: Compare) -> int:
    """单位保真度：同单位(0) > 换算(1) > 继承裸数字(2)。越小越可信。"""
    if "unit_assumed" in c.flags:
        return 2
    if "unit_converted" in c.flags:
        return 1
    return 0


def _exact(c: Compare) -> int:
    """取等优先：响应里原样写出要求值的，就是在回答这条要求。"""
    return 0 if (c.ratio is not None and units.close(c.ratio, Decimal(1))) else 1


def _key(c: Compare, sc: tuple[float, float]) -> tuple[int, int, float, float]:
    """对齐排序键，**越小越优**。单位保真 > 取等 > 名字分 > 数值分。"""
    return (_rank(c), _exact(c), -sc[0], -sc[1])


# --- 三类比较 ---------------------------------------------------------------


def _interval_in(iv: Interval, unit: str) -> Interval | None:
    """把响应区间换算到要求侧的单位。量纲不通时返回 None。"""
    if not iv.unit:
        return Interval(iv.lo, iv.hi, unit, iv.raw)  # B24：响应漏写单位
    if iv.unit == unit:
        return iv
    if not unit:
        return None
    lo, hi = units.convert(iv.lo, iv.unit, unit), units.convert(iv.hi, iv.unit, unit)
    if lo is None or hi is None:
        return None
    return Interval(lo, hi, unit, iv.raw)


def compare_intervals(need: Interval, got: Interval, st: Settings) -> Compare | None:
    """区间对区间：覆盖即达标，严格更宽且放宽侧超 τ 才算正偏离，不覆盖是负偏离。"""
    g = _interval_in(got, need.unit)
    if g is None:
        return None
    flags = () if got.unit == need.unit else (("unit_assumed",) if not got.unit else ("unit_converted",))
    if not g.covers(need):
        return Compare(
            DEV_NEGATIVE, None, f"响应区间 [{g.lo},{g.hi}] 未覆盖要求 [{need.lo},{need.hi}]", flags
        )
    if units.close(g.lo, need.lo) and units.close(g.hi, need.hi):
        return Compare(DEV_NONE, Decimal(1), f"区间 [{g.lo},{g.hi}] 与要求一致", flags)
    ratios = []
    if need.hi > 0:
        ratios.append(units.ratio(g.hi, need.hi))
    if need.lo > 0 and g.lo > 0:  # 负端点（-24.5D）取倒数没有物理意义，跳过
        ratios.append(units.ratio(need.lo, g.lo))
    best = max((r for r in ratios if r is not None), default=None)
    dev = DEV_POSITIVE if (best is not None and best > st.tau) else DEV_NONE
    detail = f"区间 [{g.lo},{g.hi}] 覆盖要求 [{need.lo},{need.hi}]"
    if dev == DEV_POSITIVE:
        detail += f"，放宽侧 {float(best):.2f} 倍 > τ"
    return Compare(dev, best, detail, flags + margin_flag(best, st))


def _point_in_interval(need: Interval, al: Aligned) -> Compare:
    """区间要求碰上单值响应：落在闭区间内即无偏离，否则负偏离（区间不会判正偏离）。"""
    if need.contains(al.got):
        return Compare(DEV_NONE, Decimal(1), f"实测 {al.got} 落在要求区间 [{need.lo},{need.hi}] 内", al.flags)
    return Compare(DEV_NEGATIVE, None, f"实测 {al.got} 不在要求区间 [{need.lo},{need.hi}] 内", al.flags)


def compare_resolution(need: Resolution, got: Decimal, unit: str, st: Settings) -> Compare | None:
    """分辨率比**宽度**（B4 / agent.md 3.9.1）：``3840*2160`` 与 ``3850像素`` 视为同一量。"""
    flags: tuple[str, ...] = ()
    value = got
    if unit and unit != need.unit:
        if not units.same_dimension(unit, need.unit):
            return None
        conv = units.convert(got, unit, need.unit)
        if conv is None:
            return None
        value, flags = conv, ("unit_converted",)
    elif not unit:
        flags = ("unit_assumed",)
    base = verdict("min", need.scalar, value, st, detail=f"分辨率宽度 {value} vs 要求 {need.scalar}")
    return Compare(base.deviation, base.ratio, base.detail, flags + margin_flag(base.ratio, st))


# --- 结构化比对主流程 -------------------------------------------------------


def _resolution_eligible(it: Item) -> bool:
    """哪些响应项可以拿去和 ``3840x2160`` 这种分辨率要求比。"""
    if it.point is None or it.kind == "interval":
        return False
    if it.kind == "resolution" or not it.unit:
        return True
    return it.unit == "像素" or units.dimension(it.unit) == "count"


@dataclass(frozen=True)
class _Atom:
    """要求侧的一个可比原子。``index`` 指回 ``rs`` 里对应的下标。"""

    kind: str  # "res" | "bound" | "cons"
    index: int
    clause: str


def _is_test_condition(con: Constraint, vs: ValSpec) -> bool:
    """这个"约束"其实是**测量条件**，不是门槛，应当跳过。

    ``软光纤最小弯曲半径：200μm 光纤≤4.0cm`` 里的 ``200μm`` 说的是
    "在哪种光纤上测"，厂商答 ``4cm`` 就是在回答这条要求；拿 200μm 去和 4cm 比
    会得到 200 倍的假正偏离。判据三条同时成立：
    无比较符的裸数值 + 带单位 + 该数值原样出现在响应的指标名里。
    """
    if con.op or not con.quantity.unit:
        return False
    for q in units.parse_quantities(squash(vs.name)):
        if q.unit == con.quantity.unit and units.close(q.value, con.quantity.value):
            return True
    return False


def _compare_atom(
    atom: _Atom, rs: ReqSpec, it: Item, vs: ValSpec, st: Settings
) -> Compare | None:
    """把一个要求原子和一个响应项比出结果。不可比（量纲不通等）返回 ``None``。"""
    if atom.kind == "res":
        if rs.resolution is None or not _resolution_eligible(it):
            return None
        return compare_resolution(rs.resolution, it.point or Decimal(0), it.unit, st)
    if it.kind == "resolution":
        return None  # 分辨率项只回答分辨率要求，不参与阈值/区间比较
    if atom.kind == "bound":
        need = rs.bounds[atom.index].interval
        if it.interval is not None:
            return compare_intervals(need, it.interval, st)
        if it.point is None:
            return None
        al = align_units(
            Quantity(need.lo, need.unit),
            Quantity(it.point, it.unit),
            rs_period=rs.period,
            vs_period=vs.period,
        )
        return None if al is None else _point_in_interval(need, al)
    con = rs.constraints[atom.index]
    if _is_test_condition(con, vs):
        return None
    got = endpoint(it, con.direction, width=con.is_width(bool(rs.bounds)))
    if got is None:
        return None
    al = align_units(
        con.quantity, Quantity(got, it.unit), rs_period=rs.period, vs_period=vs.period
    )
    if al is None:
        return None
    op = (con.op or "≥") if con.direction == "min" else (con.op or "≤")
    detail = f"{vs.name or '实测'} {al.got}{al.unit} vs 要求 {op}{al.need}{al.unit}"
    base = verdict(con.direction, al.need, al.got, st, detail=detail)
    return Compare(base.deviation, base.ratio, base.detail, al.flags + margin_flag(base.ratio, st))


def structural(rs: ReqSpec, vs: ValSpec, st: Settings) -> list[Compare]:
    """把要求结构与响应结构对齐后逐项比较，返回全部子判定（未归并）。

    **X20「复合要求只判被回答的那一句」在这里落地，且对分辨率 / 区间 / 阈值
    三类原子一视同仁。** 早先只对阈值原子做归并，结果 P5 血泵流速那条
    （要求 ``≥10ml/min``、``≤8000ml/h``，厂商答 ``10~450ml/min``）让 ml/h 门槛
    换算后压过了同单位的 ml/min 门槛，把无偏离判成负偏离。

    流程：每个「原子 × 响应项」组合算出比较结果与对齐键 ->
    若响应项数量少于原子数量（厂商只答了其中一部分），只保留对齐最优的那个原子 ->
    每个原子再只留对齐键最小的一项。没答的子句不该把整条要求拖成负偏离。
    """
    items = response_items(vs)
    if not items:
        return []
    name = squash(vs.name)

    atoms: list[_Atom] = []
    if rs.resolution is not None:
        atoms.append(_Atom("res", 0, squash(rs.resolution_clause)))
    atoms += [_Atom("bound", i, squash(b.clause)) for i, b in enumerate(rs.bounds)]
    atoms += [_Atom("cons", i, squash(c.clause)) for i, c in enumerate(rs.constraints)]

    cands: list[tuple[int, int, tuple[int, int, float, float], Compare]] = []
    for ai, atom in enumerate(atoms):
        for ii, it in enumerate(items):
            cmp_ = _compare_atom(atom, rs, it, vs, st)
            if cmp_ is None:
                continue
            cands.append((ai, ii, _key(cmp_, align_score(atom.clause, name, it)), cmp_))
    if not cands:
        return []

    if len({c[1] for c in cands}) < len({c[0] for c in cands}):
        top = min(cands, key=lambda c: c[2])[0]
        cands = [c for c in cands if c[0] == top]

    best: dict[int, tuple[tuple[int, int, float, float], Compare]] = {}
    for ai, _ii, key, cmp_ in cands:
        cur = best.get(ai)
        if cur is None or key < cur[0]:
            best[ai] = (key, cmp_)
    return [v[1] for _, v in sorted(best.items())]


# --- 置信度与 LLM 仲裁 ------------------------------------------------------


def confidence_of(base: float, flags: Sequence[str]) -> float:
    """按旗标折价。多个旗标同时命中取最严的那个上限。"""
    cap = 1.0
    for name, value in _CONF_CAPS:
        if name in flags:
            cap = min(cap, value)
    return round(min(base, cap), 3)


def _parse_deviation(text: str | None) -> str | None:
    """从模型输出里取偏离类型。优先 JSON，退化到子串扫描。"""
    if not text:
        return None
    s = text.strip()
    try:
        data = json.loads(s)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict):
        got = str(data.get("deviation") or "").strip()
        if got in (DEV_NONE, DEV_POSITIVE, DEV_NEGATIVE):
            return got
    for dev in (DEV_NEGATIVE, DEV_POSITIVE, DEV_NO_RESPONSE, DEV_NONE):
        if dev in s:
            return dev
    return None


def _needs_arbitration(flags: Sequence[str], rs: ReqSpec) -> bool:
    """只在两种情况下花模型额度：数值卡在 τ 附近，或要求可比却找不到可比结构。"""
    if "low_margin" in flags:
        return True
    return "low_confidence" in flags and rs.comparable


def arbitrate(dec: Decision, match: Match, rs: ReqSpec, vs: ValSpec, client: Any) -> Decision:
    """L3 仲裁。模型不可用或答非所问时**原样返回**确定性判定，绝不抛异常。"""
    if client is None or not getattr(client, "enabled", False):
        return dec
    if not _needs_arbitration(dec.flags, rs):
        return dec
    resp = client.complete(
        "judge",
        judge_messages(body_of(match.requirement), vs.raw or dec.resp_text, ARBITRATION_RULES),
    )
    if resp is None:
        return dec
    got = _parse_deviation(getattr(resp, "text", ""))
    if got is None or got == dec.deviation:
        return dec
    flags = tuple(dict.fromkeys((*dec.flags, "llm_arbitrated")))
    return Decision(
        deviation=got,
        resp_text=dec.resp_text,
        evidence_file=dec.evidence_file,
        location=dec.location,
        confidence=min(dec.confidence, 0.6),
        flags=flags,
        note=dec.note,
        reason=f"{dec.reason}｜模型仲裁改判为 {got}",
        method="llm",
    )


# --- 对外入口 ---------------------------------------------------------------


def decide(match: Match, *, cfg: Any = None, client: Any = None) -> Decision:
    """判定一条匹配结果。**纯函数**（除非传入了可用的 ``client``）。

    这一层只做一件包装工作：未知版面族（``layout == "?"``）的判定统一压到
    :data:`UNKNOWN_LAYOUT_CAP` 以下（N13）。放在这里而不是抽取层，是因为
    置信度是 :class:`Decision` 的字段，只有判定完成才知道该压到哪。
    """
    dec = _decide_one(match, cfg=cfg, client=client)
    if match.requirement.layout == LAYOUT_UNKNOWN:
        note = "招标版面未识别，本行按最低档置信度输出，请人工核对原文与列的对应关系。"
        dec = replace(
            dec,
            confidence=min(dec.confidence, UNKNOWN_LAYOUT_CAP),
            note=f"{dec.note} {note}".strip() if dec.note else note,
        )
    return dec


def _decide_one(match: Match, *, cfg: Any = None, client: Any = None) -> Decision:
    """判定主体，命中即止的七级次序见模块 docstring。"""
    req, ev = match.requirement, match.evidence
    st = settings_of(cfg)
    flags: list[str] = list(match.flags)

    if req.is_heading:
        return Decision("", method="heading", reason="标题/引导句，不参与判定")

    if ev is None:
        return Decision(
            deviation=DEV_NO_RESPONSE,
            confidence=match.confidence,
            flags=tuple(flags),
            reason=match.reason or "本标的的厂商资料里找不到可配对的指标",
            method="none",
        )

    if ev.is_pointer:
        flags.append("ocr_failed")
        if st.ocr_fallback == "no-response":
            return Decision(
                deviation=DEV_NO_RESPONSE,
                confidence=confidence_of(match.confidence, flags),
                flags=tuple(flags),
                reason="厂商把该指标指向检测报告，OCR 不可用，按未响应输出",
                method="pointer",
            )
        flags.append("pointer_assumed")
        return Decision(
            deviation=DEV_NONE,
            resp_text=ev.response_text,
            evidence_file=ev.filename,
            location=ev.locator,
            confidence=confidence_of(match.confidence, flags),
            flags=tuple(flags),
            note="检测报告 OCR 不可用：材料与位置已定位，实测值待人工补录后复核偏离类型。",
            reason="厂商响应为「详见检测报告」，离线状态下按「无偏离（推定）」输出",
            method="pointer",
        )

    vs = parse_value(ev)
    if vs.refusal:
        return Decision(
            deviation=DEV_NEGATIVE,
            resp_text=ev.response_text,
            evidence_file=ev.filename,
            location=ev.locator,
            confidence=confidence_of(match.confidence, flags),
            flags=tuple(flags),
            reason=f"厂商响应为「{vs.refusal}」，命中封闭拒答词表（agent.md 3.9.3）",
            method="phrase",
        )

    # 4. 逐字复述：厂商把招标原文抄回来，等于承诺"就按这个做"。
    #    只做**严格子串**判定，不用相似度——0.95 的比值版会在 P2 的
    #    ``支持。成像范围符合55º×55º…`` 上差 0.001 漏掉，那一条改由对齐键修。
    if len(vs.text) >= ECHO_MIN_LEN and vs.text in squash(body_of(req)):
        return Decision(
            deviation=DEV_NONE,
            resp_text=ev.response_text,
            evidence_file=ev.filename,
            location=ev.locator,
            confidence=confidence_of(match.confidence, flags),
            flags=tuple(flags),
            reason="厂商逐字复述了招标原文，视为承诺满足",
            method="echo",
        )

    rs = parse_requirement(body_of(req))
    compares = structural(rs, vs, st)
    for c in compares:
        flags += [f for f in c.flags if f not in flags]
    if not compares and rs.comparable:
        flags.append("low_confidence")
    deviation = reduce(compares) if compares else DEV_NONE
    reason = "；".join(c.detail for c in compares if c.detail) or (
        "要求与响应都没有可比数值，按文字型处理判无偏离（X22）"
    )
    dec = Decision(
        deviation=deviation,
        resp_text=ev.response_text,
        evidence_file=ev.filename,
        location=ev.locator,
        confidence=confidence_of(match.confidence, flags),
        flags=tuple(flags),
        reason=reason,
        method="compare" if compares else "text",
    )
    return arbitrate(dec, match, rs, vs, client)


def judge_all(
    matches: Sequence[Match], cfg: Any = None, client: Any = None
) -> tuple[list[Decision], list[str]]:
    """批量判定 + 汇总告警。告警是**去重的类别级**信息，不是每行一条。"""
    decisions = [decide(m, cfg=cfg, client=client) for m in matches]
    warnings: list[str] = []

    def _count(flag: str) -> int:
        return sum(1 for d in decisions if flag in d.flags)

    n_ocr = _count("ocr_failed")
    if n_ocr:
        mode = "未响应" if getattr(cfg, "ocr_fallback", "assume") == "no-response" else "无偏离（推定）"
        warnings.append(
            f"{n_ocr} 条要求指向检测报告，OCR 不可用，已按「{mode}」输出，需人工复核实测值。"
        )
    n_margin = _count("low_margin")
    if n_margin:
        lo, hi = (getattr(cfg, "low_margin_band", None) or DEFAULT_BAND)[:2]
        warnings.append(f"{n_margin} 条判定的比值落在 {lo}~{hi} 的低置信带内，建议优先复核。")
    n_nc = _count("low_confidence")
    if n_nc:
        warnings.append(f"{n_nc} 条要求含数值门槛，但响应侧找不到同量纲的量，已按无偏离输出。")
    n_conv = _count("unit_converted")
    if n_conv:
        warnings.append(f"{n_conv} 条判定做过单位换算（如 mA→A、g→kg），换算依据见复核报告。")
    n_arb = _count("llm_arbitrated")
    if n_arb:
        warnings.append(f"{n_arb} 条判定被模型仲裁改判，已在复核报告中标出。")
    return decisions, warnings
