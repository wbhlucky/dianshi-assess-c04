"""指标名归一、别名表、单位指纹（agent.md 4.1 / EC-G / X7 / X21 / B23 / B24）。

匹配阶段只需要回答「这两段文字是不是在讲同一个指标」，**不需要**回答「谁大谁小」。
所以本模块只做*同一性*归一：全角半角、上标、空白、大小写、单位同义词。
量纲换算（``9360mA -> 9.36A``）留给 :mod:`bidtable.judge.units`；
依赖方向是 judge -> match.normalize（本模块不 import judge），无环。

三条设计约束，都是被训练集打出来的：

1. :func:`squash` 必须**删掉所有空白**而不是折叠成单空格。
   招标 PDF 里 `8 小时` / `4 根` / `1 天` 这类数字与量词之间的空格是排版产物，
   厂商文档里没有；不删就会让 `连续工作8 小时` 与 `连续工作8小时` 判为不同（P1 序号13）。
2. 单位同义词只做**改名**，不做进率换算。`寸` 和 `英寸` 归一到同一个指纹，
   但 `mA` 与 `A` 是两个不同指纹——换算会不会做对是判定阶段的责任，
   匹配阶段把它们判成「不同单位」反而是安全的（宁可少匹配，不可错匹配）。
3. 别名表是**领域通用词**（用途/配置/尺寸…），不是训练集字面量。
   它只用于把 `适用范围：暂不支持` 这类零词面重叠的证据挂到
   `用于泌尿系结石的碎石…` 这条要求上（P1 序号1），命中即记 ``alias`` 方法名，
   置信度低于词面匹配，并进复核报告。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

# --- 1. 文本归一 -----------------------------------------------------------

# NFKC 之后仍需手工统一的字符：NFKC 不动破折号家族与顿号/句号，
# 而 `～` 会被折成 `~`，两者必须落到同一个分隔符上（区间判定依赖它）。
_TRANSLATE = str.maketrans(
    {
        "~": "-",
        "～": "-",
        "—": "-",
        "–": "-",
        "−": "-",
        "‥": "-",
        "、": ",",
        "。": ".",
        "；": ";",
        "×": "x",
        "＊": "*",
        "　": "",
    }
)

_WS_RE = re.compile(r"\s+")
# 只有夹在两个数字之间的 `*` 才是乘号（`3840*2160P`）。
# 行首的 `*` 是 ★ 的半角写法，由 ingest.marks 在更早的阶段处理，这里不该碰。
_MULT_RE = re.compile(r"(?<=\d)[*](?=\d)")


def squash(text: str | None) -> str:
    """把任意文本压成**可逐字比较**的规范形。

    步骤：NFKC（全角→半角、``㎡``→``m2``、``²``→``2``、``µ``→``μ``）
    → 统一破折号/顿号/乘号 → 转小写 → **删除全部空白**。
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.translate(_TRANSLATE)
    s = _MULT_RE.sub("x", s)
    s = s.lower()
    return _WS_RE.sub("", s)


# --- 2. 字符二元组相似度 ---------------------------------------------------


def bigrams(text: str) -> set[str]:
    """字符二元组集合。单字符串退化成它自己，避免长度为 1 的指标名得 0 分。"""
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def containment(needle: str, haystack: str) -> float:
    """``needle`` 的二元组有多大比例出现在 ``haystack`` 里，值域 [0, 1]。

    用**包含系数**而不是 Dice/Jaccard：指标名通常比要求原文短得多
    （`指示光功率` vs `护眼指示光：绿光，波长520nm，功率≤5mW。`），
    对称系数会因为长度差被拉到 0.2 以下，包含系数才反映「名字被提到了没有」。
    """
    a = bigrams(needle)
    if not a:
        return 0.0
    b = bigrams(haystack)
    return len(a & b) / len(a)


def lcs_len(a: str, b: str) -> int:
    """最长公共**子串**长度（不是子序列）。O(len(a)*len(b))，两侧都是短文本。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


# --- 3. 单位指纹 -----------------------------------------------------------

# 只收「同一个东西的不同写法」，**不收进率关系**（见模块 docstring 约束 2）。
# 键是 squash 之后的形态，所以 `c ㎡` -> `cm2`、`cm²` -> `cm2` 自动等价（EC-G）。
ALIAS_UNITS: dict[str, str] = {
    # 长度
    "寸": "英寸",
    "吋": "英寸",
    "inch": "英寸",
    '"': "英寸",
    "um": "μm",
    "micron": "μm",
    "米": "m",
    "meter": "m",
    # 功率 / 能量
    "瓦": "w",
    "焦": "j",
    # 时间
    "秒": "s",
    "sec": "s",
    "分钟": "min",
    "小时": "h",
    "hr": "h",
    "天": "日",
    "day": "日",
    "月": "个月",
    "年": "周年",
    # 电气
    "安": "a",
    "伏": "v",
    "赫兹": "hz",
    # 温度 / 角度
    "°c": "℃",
    "celsius": "℃",
    "度": "°",
    # 计数 / 分辨率
    "pixel": "像素",
    "px": "像素",
    "p": "像素",
    # 容量
    "cc": "ml",
    "公升": "l",
}

# 参与分词的候选单位。按长度倒序，保证 `万像素` 抢在 `万` 前、`l/min` 抢在 `l` 前。
_EXTRA_UNITS = (
    "nm", "μm", "mm", "cm", "km", "m", "fr",
    "mm2", "cm2", "m2",
    "mg", "g", "kg", "t", "吨",
    "μw", "mw", "w", "kw",
    "mj", "j", "kj",
    "ns", "μs", "ms", "s", "min", "h", "日", "个月", "周年",
    "μa", "ma", "a", "mv", "v", "kv",
    "hz", "khz", "mhz", "ghz",
    "mmhg", "kpa", "mpa", "bar",
    "℃", "°", "%", "d",
    "gb", "mb", "tb", "kb",
    "个", "台", "套", "根", "副", "把", "束", "段", "次", "种", "倍",
    "条", "张", "片", "支", "只", "件", "项", "路", "档", "级", "孔",
    "万像素", "像素", "万", "亿", "千",
    "l/min", "ml/min", "l/h", "ml/h", "l/s", "ml/s", "l", "ml", "μl",
    "mg/ml", "μg/ml", "g/l",
    "mw/cm2", "w/cm2", "mw/m2",
    "db", "dpi", "rpm", "转/分", "次/年", "次/月", "次/日",
    "人", "床", "通道", "自由度",
)

_UNIT_TOKENS: tuple[str, ...] = tuple(
    sorted({*ALIAS_UNITS, *ALIAS_UNITS.values(), *_EXTRA_UNITS}, key=len, reverse=True)
)

_NUM = r"\d+(?:\.\d+)?"
NUM_UNIT_RE = re.compile(
    rf"(?P<value>{_NUM})(?P<unit>{'|'.join(re.escape(u) for u in _UNIT_TOKENS)})?"
)


def unit_key(unit: str | None) -> str:
    """单位同义归一（**不换算**）。未知单位原样返回，空值返回 ``""``。"""
    if not unit:
        return ""
    u = squash(unit)
    return ALIAS_UNITS.get(u, u)


@dataclass(frozen=True)
class NumTok:
    """文本里抽出的一个「数值 + 单位」指纹。"""

    value: Decimal
    unit: str  # 归一后的单位指纹；裸数字为 ""
    raw: str  # squash 之后的原文片段，用于复核报告
    index: int  # 出现次序（B23：要求里有多个数值时按量纲挑那一个）

    @property
    def has_unit(self) -> bool:
        return bool(self.unit)


def _dec(text: str) -> Decimal | None:
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError, ArithmeticError):
        return None


def numbers(text: str | None) -> list[NumTok]:
    """抽出全部「数值 + 单位」指纹。单位可选；``3840x2160`` 产出两个裸数字。"""
    if not text:
        return []
    s = squash(text)
    out: list[NumTok] = []
    for i, m in enumerate(NUM_UNIT_RE.finditer(s)):
        v = _dec(m.group("value"))
        if v is None:
            continue
        out.append(NumTok(value=v, unit=unit_key(m.group("unit")), raw=m.group(0), index=i))
    return out


def unit_fingerprint(text: str | None) -> frozenset[str]:
    """文本里出现过的**带单位**数值的单位集合（裸数字不算，噪声太大）。"""
    return frozenset(t.unit for t in numbers(text) if t.unit)


def shares_unit(a: str | None, b: str | None) -> bool:
    """两段文本是否至少共享一个单位指纹。EC-G 的 `c ㎡` 与 `cm²` 靠它对齐。"""
    fa, fb = unit_fingerprint(a), unit_fingerprint(b)
    return bool(fa and fb and (fa & fb))


# --- 4. 别名表（领域通用概念词，非训练集字面量） ---------------------------

# 形如 ``{组名: (同义/近义关键词…)}``。指标名与要求原文**都命中同一组**才算别名匹配。
# 只放医疗器械招标里的通用概念词，任何一条都能在不看训练集的前提下写出来；
# 新增一组的代价是可能引入误匹配，所以宁可少而准。
ALIAS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("用途", ("临床用途", "用途", "适应症", "适用于", "用于", "临床")),
    ("配置", ("配置清单", "标准配置", "配置要求", "配置", "配备", "标配", "附件清单")),
    ("尺寸", ("外形尺寸", "尺寸", "规格尺寸", "长宽高")),
    # 「多大」这一族在招标里写作 ``瞳孔大小`` / ``通光孔径`` / ``有效口径``，
    # 在厂商表里写作 ``最小瞳孔直径`` / ``过滤器有效口径``——词面不同，问的是同一件事。
    # 只收**几何尺度**类词，不收 ``范围``（太泛，会把参数区间也拖进来）。
    ("尺度", ("直径", "内径", "外径", "口径", "孔径", "大小", "幅度", "周长")),
    ("重量", ("重量", "净重", "毛重")),
    ("电源", ("工作电源", "输入电压", "电源要求", "供电", "电压")),
    ("功率", ("输出功率", "额定功率", "功率", "最大输出")),
    ("分辨率", ("分辨率", "像素", "清晰度", "成像分辨")),
    ("数量", ("数量", "台数", "件数", "数目")),
    ("认证", ("认证", "ce认证", "fda", "注册证")),
    ("保修", ("保修", "质保", "免费维保", "维护保养", "运维", "维保")),
    ("培训", ("培训", "教学", "操作指导")),
    ("开机率", ("开机率", "使用率", "完好率", "可用率")),
    ("存储", ("存储", "记录", "内存", "硬盘", "参数保存")),
    ("温度", ("温度", "控温", "加温", "恒温")),
    ("光斑", ("光斑", "照射面积", "光斑大小", "光斑直径")),
)


def alias_group(name: str | None, text: str | None) -> str | None:
    """指标名与要求原文是否落在同一个别名组；命中返回组名，否则 ``None``。

    比较全部在 :func:`squash` 之后做，所以 `CE 认证` 与 `ce认证` 等价。
    """
    if not name or not text:
        return None
    n, t = squash(name), squash(text)
    for group, keywords in ALIAS_GROUPS:
        if any(k in n for k in keywords) and any(k in t for k in keywords):
            return group
    return None
