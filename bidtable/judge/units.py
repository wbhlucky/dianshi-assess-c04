"""单位量纲表与换算（agent.md EC-G / X7 / B7 / B24）。

判定阶段的全部「先换算再比较」都发生在这里。三条硬要求：

1. **用 :class:`decimal.Decimal`，不用 float**（B7）。``20000g -> 20kg``、
   ``9360mA -> 9.36A`` 必须精确取等，否则 ``≥20kg`` 会被判成 ``19.999999...`` 而误判负偏离。
2. **量纲不同一律判「不可比」**，退回文字型判定（B4）。
   ``3840x2160P`` 与 ``3850像素`` 不是同一个量纲，硬比会得到 1.78 倍的假正偏离。
3. 单位同义词（``寸``/``英寸``）在 :mod:`bidtable.match.normalize` 里归一，
   本模块的键**已经是规范形**，只负责「属于哪个量纲」和「到基准单位的进率」。

依赖方向：本模块 import ``match.normalize``（叶子工具），反向不成立，无环。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from ..match.normalize import squash
from ..match.normalize import _UNIT_TOKENS as _UNIT_TOKENS
from ..match.normalize import unit_key as _unit_key

# 浮点/量化容差：换算后差值小于它就算相等（B7）。用 Decimal 后其实很少触发，
# 但 `3030000分钟 -> 50500小时` 这类多级换算仍需要一道保险。
EPS = Decimal("1e-9")

# unit_key -> (量纲, 到该量纲基准单位的进率)
# 基准单位：length=mm, mass=g, time=s, power=W, energy=J, volume=ml,
#           area=mm2, current=A, voltage=V, frequency=Hz, data=MB
_D = Decimal
UNIT_DIMENSION: dict[str, tuple[str, Decimal]] = {
    # 长度
    "nm": ("length", _D("0.000001")),
    "μm": ("length", _D("0.001")),
    "mm": ("length", _D("1")),
    "cm": ("length", _D("10")),
    "m": ("length", _D("1000")),
    "km": ("length", _D("1000000")),
    "英寸": ("length", _D("25.4")),
    "fr": ("fr", _D("1")),  # French 导管口径：训练集两侧同单位，不做进率假设
    # 面积
    "mm2": ("area", _D("1")),
    "cm2": ("area", _D("100")),
    "m2": ("area", _D("1000000")),
    # 质量
    "mg": ("mass", _D("0.001")),
    "g": ("mass", _D("1")),
    "kg": ("mass", _D("1000")),
    "t": ("mass", _D("1000000")),
    "吨": ("mass", _D("1000000")),
    # 功率 / 能量
    "μw": ("power", _D("0.000001")),
    "mw": ("power", _D("0.001")),
    "w": ("power", _D("1")),
    "kw": ("power", _D("1000")),
    "mj": ("energy", _D("0.001")),
    "j": ("energy", _D("1")),
    "kj": ("energy", _D("1000")),
    # 时间
    "ns": ("time", _D("0.000000001")),
    "μs": ("time", _D("0.000001")),
    "ms": ("time", _D("0.001")),
    "s": ("time", _D("1")),
    "min": ("time", _D("60")),
    "h": ("time", _D("3600")),
    "日": ("time", _D("86400")),
    "个月": ("time", _D("2592000")),  # 30 日；仅保证同单位比较时量纲一致
    "周年": ("time", _D("31536000")),  # 365 日
    # 电气
    "μa": ("current", _D("0.000001")),
    "ma": ("current", _D("0.001")),
    "a": ("current", _D("1")),
    "mv": ("voltage", _D("0.001")),
    "v": ("voltage", _D("1")),
    "kv": ("voltage", _D("1000")),
    "hz": ("frequency", _D("1")),
    "khz": ("frequency", _D("1000")),
    "mhz": ("frequency", _D("1000000")),
    "ghz": ("frequency", _D("1000000000")),
    # 压强 / 温度 / 角度 / 百分比 / 屈光度
    "mmhg": ("pressure", _D("1")),
    "kpa": ("pressure", _D("7.500616")),
    "mpa": ("pressure", _D("7500.616")),
    "bar": ("pressure", _D("750.0616")),
    "℃": ("temperature", _D("1")),
    "°": ("angle", _D("1")),
    "%": ("percent", _D("1")),
    "d": ("diopter", _D("1")),
    # 数据量
    "kb": ("data", _D("1") / _D("1024")),
    "mb": ("data", _D("1")),
    "gb": ("data", _D("1024")),
    "tb": ("data", _D("1048576")),
    # 流量 / 转速 / 声压
    "l/min": ("flow", _D("1")),
    "ml/min": ("flow", _D("0.001")),
    "l/h": ("flow", _D("1") / _D("60")),
    "ml/h": ("flow", _D("0.001") / _D("60")),
    "l/s": ("flow", _D("60")),
    "ml/s": ("flow", _D("0.06")),
    "rpm": ("spin", _D("1")),
    "转/分": ("spin", _D("1")),
    "db": ("db", _D("1")),
    "dpi": ("dpi", _D("1")),
    # 容量 / 浓度 / 辐照度
    "l": ("volume", _D("1000")),
    "ml": ("volume", _D("1")),
    "μl": ("volume", _D("0.001")),
    "mg/ml": ("concentration", _D("1")),
    "μg/ml": ("concentration", _D("0.001")),
    "g/l": ("concentration", _D("1")),
    "mw/cm2": ("irradiance", _D("1")),
    "w/cm2": ("irradiance", _D("1000")),
    "mw/m2": ("irradiance", _D("0.0001")),
    # 计数（同一量纲内进率差异巨大，但训练集只在同单位间比较）
    "个": ("count", _D("1")), "台": ("count", _D("1")), "套": ("count", _D("1")),
    "根": ("count", _D("1")), "副": ("count", _D("1")), "把": ("count", _D("1")),
    "束": ("count", _D("1")), "段": ("count", _D("1")), "次": ("count", _D("1")),
    "种": ("count", _D("1")), "倍": ("count", _D("1")), "条": ("count", _D("1")),
    "张": ("count", _D("1")), "片": ("count", _D("1")), "支": ("count", _D("1")),
    "只": ("count", _D("1")), "件": ("count", _D("1")), "项": ("count", _D("1")),
    "路": ("count", _D("1")), "档": ("count", _D("1")), "级": ("count", _D("1")),
    "孔": ("count", _D("1")), "人": ("count", _D("1")), "床": ("count", _D("1")),
    "通道": ("count", _D("1")), "自由度": ("count", _D("1")),
    "像素": ("count", _D("1")), "万像素": ("count", _D("10000")),
    "万": ("count", _D("10000")), "千": ("count", _D("1000")), "亿": ("count", _D("100000000")),
    # 频次（带周期后缀的计数）
    "次/年": ("count_rate", _D("1")),
    "次/月": ("count_rate", _D("12")),
    "次/日": ("count_rate", _D("365")),
}


def dimension(unit_key: str | None) -> str:
    """单位所属量纲；未知单位返回 ``""``（视作不可换算，只能同字符串比较）。"""
    if not unit_key:
        return ""
    return UNIT_DIMENSION.get(unit_key, ("", _D("1")))[0]


def same_dimension(a: str | None, b: str | None) -> bool:
    """两个单位是否可直接换算。任一侧未知量纲 -> ``False``（宁可不比，不可错比）。"""
    da, db = dimension(a), dimension(b)
    return bool(da) and da == db


def to_base(value: Decimal, unit_key: str | None) -> Decimal | None:
    """换算到该量纲的基准单位。未知单位返回 ``None``。"""
    if not unit_key or unit_key not in UNIT_DIMENSION:
        return None
    return value * UNIT_DIMENSION[unit_key][1]


def convert(value: Decimal, src: str | None, dst: str | None) -> Decimal | None:
    """``src`` 单位的数值换算到 ``dst`` 单位；量纲不同或未知则 ``None``。"""
    if src == dst:
        return value
    base = to_base(value, src)
    if base is None or not dst or dst not in UNIT_DIMENSION:
        return None
    if UNIT_DIMENSION[src][0] != UNIT_DIMENSION[dst][0]:
        return None
    return base / UNIT_DIMENSION[dst][1]


def ratio(a: Decimal, b: Decimal) -> Decimal | None:
    """``a / b``；分母为 0 返回 ``None``（判定侧据此退回文字型）。"""
    if b == 0:
        return None
    return a / b


def close(a: Decimal, b: Decimal) -> bool:
    """换算后是否相等（B7 的浮点/量化保险）。"""
    return abs(a - b) <= EPS * max(Decimal(1), abs(a), abs(b))


# --- 数值 / 区间 / 分辨率解析 ----------------------------------------------

_NUM = r"\d+(?:\.\d+)?"
_SIGNED = rf"[+-]?{_NUM}"
# 单位候选：最长优先，保证 `l/min` 抢在 `l` 前、`万像素` 抢在 `万` 前。
_UNIT_ALT = "|".join(re.escape(t) for t in _UNIT_TOKENS)
# 区间分隔符：squash 已把 `～ — – − ‥` 全折成 `-`，原文的 `——`/`---` 会变成连续多个 `-`。
# 两侧单位都可选：`8-10μm` 只在右侧带单位（左侧继承，B24），`33℃-43℃` 两侧都带。
_RANGE_RE = re.compile(
    rf"(?P<lo>{_SIGNED})(?P<u1>{_UNIT_ALT})?[-]+(?P<hi>{_SIGNED})(?P<u2>{_UNIT_ALT})?"
)
# `3840x2160` / `3840×2160P` / `1536×1536pixel`
_RES_RE = re.compile(rf"(?P<w>{_NUM})x(?P<h>{_NUM})(?P<tail>[a-z像素]*)")
# 负号只可能出现在这些边界之后（squash 已删空白，所以没有空格类）。
_SIGN_BOUNDARY = ":：;；(（[【,、/≥≤><=＞＜"


@dataclass(frozen=True)
class Quantity:
    """一个可比较的标量。``unit`` 是 :func:`bidtable.match.normalize.unit_key` 的规范形。"""

    value: Decimal
    unit: str = ""
    raw: str = ""

    @property
    def dimension(self) -> str:
        return dimension(self.unit)


@dataclass(frozen=True)
class Interval:
    """闭区间 ``[lo, hi]``，两侧单位归一到 ``unit``（缺失一侧继承另一侧，B24）。"""

    lo: Decimal
    hi: Decimal
    unit: str = ""
    raw: str = ""

    @property
    def width(self) -> Decimal:
        return self.hi - self.lo

    def contains(self, value: Decimal) -> bool:
        """闭区间包含判定（``response ⊇ required`` 的核心动作）。"""
        return self.lo <= value <= self.hi

    def covers(self, other: Interval) -> bool:
        """本区间是否**完整包含** ``other``。调用方保证两侧量纲一致。"""
        return self.lo <= other.lo and other.hi <= self.hi


@dataclass(frozen=True)
class Resolution:
    """``宽 x 高`` 形式的分辨率。与标量比较时**取宽**（agent.md B4 / 3.9.1）。"""

    width: Decimal
    height: Decimal
    unit: str = "像素"
    raw: str = ""

    @property
    def scalar(self) -> Decimal:
        return self.width


def parse_resolution(text: str | None) -> Resolution | None:
    """识别 ``3840*2160P`` / ``1536×1536pixel``。识别不到返回 ``None``。"""
    found = parse_resolutions(text)
    return found[0] if found else None


def parse_resolutions(text: str | None) -> list[Resolution]:
    """抽出**全部** ``宽x高`` 分辨率；:func:`parse_resolution` 只取第一个。

    判定层需要「全部」而不是「第一个」：P3 一条要求里同时写了
    ``4K视频信号（≥3840*2160）`` 和 ``全高清视频信号（≥1920*1080P）``，
    若只屏蔽第一个的字面区间，第二个会被当成两个裸数字门槛，凭空多出两条约束。
    """
    if not text:
        return []
    s = squash(text)
    out: list[Resolution] = []
    for m in _RES_RE.finditer(s):
        tail = m.group("tail") or ""
        out.append(
            Resolution(
                width=Decimal(m.group("w")),
                height=Decimal(m.group("h")),
                unit="像素" if ("像素" in tail or tail.startswith("p")) else "",
                raw=m.group(0),
            )
        )
    return out
    tail = m.group("tail") or ""
    unit = "像素" if ("像素" in tail or tail.startswith("p")) else ""
    return Resolution(
        width=Decimal(m.group("w")),
        height=Decimal(m.group("h")),
        unit=unit,
        raw=m.group(0),
    )


def parse_intervals(text: str | None) -> list[Interval]:
    """抽出全部区间。``8-10μm`` 这种只在右侧带单位的，左侧继承右侧单位（B24）。

    刻意**不**接受 ``0.l-5ml`` 这类被排版噪声污染的写法：`0.` 后面的 `.` 既不是
    单位也不是分隔符，正则整段匹配失败，返回空列表而不是硬凑一个 (0, 5) 出来。
    """
    if not text:
        return []
    s = squash(text)
    out: list[Interval] = []
    for m in _RANGE_RE.finditer(s):
        unit = _unit_key(m.group("u1") or m.group("u2") or "")
        lo, hi = Decimal(m.group("lo")), Decimal(m.group("hi"))
        if lo > hi:  # `43℃-33℃` 这类倒写：按数值大小归一
            lo, hi = hi, lo
        out.append(Interval(lo=lo, hi=hi, unit=unit, raw=m.group(0)))
    return out


def parse_quantities(text: str | None) -> list[Quantity]:
    """抽出全部带符号标量。**符号靠上下文分类**，不是靠字符：

    ``-`` 只有在「它是孤立的一个短横」且「前一个字符是边界标点或字符串开头」时才是负号；
    否则它是区间分隔符。三条反例，全都必须走「分隔符」分支：

    * ``370nm--1000nm``：连续两个 ``-``，最后那个属于分隔符串，不是负号。
    * ``100v-240v`` / ``33℃-43℃``：``-`` 前面是单位字符，是分隔符。
    * ``37-43℃``：``-`` 前面是数字，是分隔符。

    只有 ``-24.5~20d``（squash 后 ``-24.5-20d``）的第一个 ``-`` 落在字符串开头，
    以及 ``范围:(-24.5d)`` 这种落在边界标点之后的，才判为负号。
    """
    if not text:
        return []
    s = squash(text)
    out: list[Quantity] = []
    for m in re.finditer(rf"(?P<sign>-)?(?P<v>{_NUM})", s):
        negative = m.group("sign") is not None and _sign_is_negative(s, m.start("sign"))
        unit = _unit_key(_match_unit(s, m.end("v")))
        value = Decimal(m.group("v"))
        if negative:
            value = -value
        out.append(Quantity(value=value, unit=unit, raw=m.group(0) + (unit or "")))
    return out


def _match_unit(text: str, pos: int) -> str:
    """从 ``text[pos:]`` 起按最长优先吃掉一个单位 token，返回原始 token（未归一）。"""
    for tok in _UNIT_TOKENS:
        if text.startswith(tok, pos):
            return tok
    return ""


def _sign_is_negative(s: str, k: int) -> bool:
    """``s[k] == '-'`` 且后面紧跟数字时，这个 ``-`` 是负号还是区间分隔符。"""
    run = 1
    while k - run >= 0 and s[k - run] == "-":
        run += 1
    if run >= 2:  # `--` / `---`：分隔符串
        return False
    if k == 0:
        return True
    return s[k - 1] in _SIGN_BOUNDARY


# 周期后缀：`4.1次/年` 与要求里的 `每年免费提供≥4次` 必须能对上（agent.md 3.9.1）。
_PERIOD_WORDS: tuple[tuple[str, str], ...] = (
    ("每周年", "年"),
    ("每年", "年"),
    ("每个月", "月"),
    ("每月", "月"),
    ("每天", "日"),
    ("每日", "日"),
    ("每小时", "小时"),
    ("每周", "周"),
)


def period_of(text: str | None) -> str:
    """文本里出现的周期词，归一成 ``年/月/日/小时/周``；没有则 ``""``。"""
    if not text:
        return ""
    s = squash(text)
    # 长词优先：`每周年` 必须先于 `每周` 命中，否则会被归成「周」。
    for word, period in _PERIOD_WORDS:
        if squash(word) in s:
            return period
    return ""
