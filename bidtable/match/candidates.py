"""确定性候选生成、打分与消歧（agent.md 4.3 L1、7.1 EC-C/EC-D、7.3 X3/X4、7.4 B23/B25）。

这一层是整条流水线里**唯一**决定「哪条要求由哪份材料证明」的地方，因此它的每一个
常数都必须能被训练集之外的理由解释。五条设计原则：

1. **名称相似度用「包含系数 + 字符包含」的加权混合**，不用 Dice/Jaccard。
    指标名（``指示光功率``）远比要求原文（``护眼指示光：绿光，波长520nm，功率≤5mW。``）短，
    对称系数会被长度差拉到 0.2 以下，而「名字被提到了没有」才是这里要问的问题。
2. **中心语剥离**（``年技术培训次数`` -> ``年技术培训``）单独算一次并取最大值。
    中文指标名的信息量集中在修饰语上，中心语（次数/范围/功率…）几乎不带区分度。
3. **值逐字回显直接判满分**。厂商把招标原文整段抄进「规格/说明」列是常态
    （P1 序号14/15、标的2 序号1），此时任何相似度模型都不该有发言权。
4. **量纲冲突一票否决式扣分**（B23）。``84W`` 与 ``200μm、272μm…可选`` 的数值指纹
    看起来沾边，但一边是功率一边是长度——扣 ``UNIT_PENALTY`` 让它掉到阈值以下。
    注意判据是**量纲集合是否相交**，不是单位字符串是否相等：``9360mA`` 与 ``≤10A``
    必须算「同量纲」，否则 X7 会被误杀。
5. **同值折叠优先于取最大**（X4「规格书胜出」）。同一条要求的候选里，值 squash 后
    完全相同的只保留文档优先级最高的那一个。这一步必须排在 argmax 之前：
    产品综合样本的指标名往往就是招标原文的前缀，名称分反而更高，
    不先折叠就会让优先级低的样本抢走技术规格书的位置。

消歧次序（``resolve`` 的五个 pass）同样不是随意排的，见各 pass 的 docstring。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ..config import NEGATIVE_PHRASES, DOC_REPORT
from ..ingest.scope import body_of, section_context
from ..judge.units import dimension
from ..models import Evidence, Match, Requirement
from . import semantic
from .normalize import (
    alias_group,
    bigrams,
    containment,
    lcs_len,
    numbers,
    shares_unit,
    squash,
)

# --- 阈值常数 ---------------------------------------------------------------
# 下面每个值都对应一个可复述的判据，调参时请连同注释一起改。

THRESHOLD = 0.62  # 接受匹配的最低综合分。P1 序号10（0.78）与序号4（0.48）分居两侧。
STRONG = 0.85  # 高置信度门槛；低于它的匹配在复核报告里折价并列出
PRIORITY_BAND = 0.05  # 「咬得很近」的判定带宽，用于 B25 歧义标记
ALIAS_CAP = 0.75  # 名称相似度已高于它就不再给别名加分，避免把强匹配推成满分
ALIAS_BONUS = 0.25  # 别名组命中的加分（P1 序号15「配置清单」↔「配置要求」靠它过线）
UNIT_AGREE_BONUS = 0.10  # 两侧共享单位指纹的小幅加分
UNIT_PENALTY = 0.15  # 量纲完全不相交的扣分（B23）
PREFIX_SCORE = 0.90  # 指标名是要求正文的前缀时的保底分
MIN_NAME_LEN = 4  # 前缀规则的最短指标名，防止「数量」这种双字词命中一切
MIN_ECHO_LEN = 6  # 「逐字回显」直接判满分的最短值长度
MODIFIER_GATE = 0.70  # 原名 blend 高于它就不再启用中心语剥离，见 name_similarity
REFUSAL_WEIGHT = 0.90  # 拒绝式证据只有名称一路信号，相对「名称+值双印证」打九折
POINTER_BAND = 0.10  # 指针与实值证据分差在此之内时让位（见 :func:`preference`）
# 候选池的收录下限：远低于 THRESHOLD，因为拒绝式挂接（rescue_refusals）与 LLM 兜底
# 都要能看到「沾边但没过线」的候选。综合分为 0 的直接不进池，省内存也省噪声。
POOL_FLOOR = 0.05

# 综合分里名称 / 数值相似度的权重。指针证据（值为「详见检测报告」）不带数值，
# 此时权重全部归到名称上，见 score_candidate。
W_NAME = 0.60
W_VALUE = 0.40

# 中心语表：中文指标名的通用后缀，剥离后剩下的修饰语才带区分度。
# 全部是领域通用词，不含任何训练集字面量（agent.md 8.2 R11）。
HEAD_NOUNS: tuple[str, ...] = (
    "次数", "数量", "数目", "时长", "时间", "范围", "直径", "面积", "尺寸",
    "功率", "长度", "重量", "容量", "电压", "电流", "频率", "浓度", "速度",
    "精度", "方式", "功能", "效率", "模式", "周期", "角度", "温度", "压力",
    "率", "值", "量",
)
_HEAD_NOUNS_BY_LEN: tuple[str, ...] = tuple(sorted(HEAD_NOUNS, key=len, reverse=True))

# 「X 数量」这类指标名的计数中心语，与要求正文里的「≥N 个/台/…」问句配对用。
COUNT_HEADS: tuple[str, ...] = ("数量", "数目", "台数", "件数", "个数")
# 要求正文里的「问几个」句式：一个数 + 一个量词。量词表与 normalize._EXTRA_UNITS 同源，
# 只取其中真正的**可数**量词，不收 ``mm``/``W`` 这种物理单位。
_COUNT_ASK = re.compile(
    r"\d+(?:个|台|套|件|副|条|根|只|张|片|支|把|束|段|次|种|路|档|级|孔|人|床|辆)"
)

# 拒绝式响应的规范形集合：厂商明确表示「做不到 / 要加钱」的固定说法。
# 与 judge 层共用同一份词表（config.NEGATIVE_PHRASES），这里只做**整句全等**比较。
_REFUSALS: frozenset[str] = frozenset(squash(p) for p in NEGATIVE_PHRASES)

# 指标名末尾的括号限定语，如 ``3D眼镜数量(眼镜式、夹片式各)`` 里的 ``(…)``。
_PAREN_TAIL = re.compile(r"[(（][^()（）]*[)）]\s*$")
# 回显判满分的附加条件：值里必须含汉字。纯 ``40L/min`` 这种「数字+单位」在长要求里
# 撞见是常态（P3「设置流量的调节范围：1L/min～40L/min」里就同时躺着 ``最大流速：40L/min``），
# 把它当成「厂商抄了原文」是误读——抄原文抄的一定是**话**，不是一个数。
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


# --- 相似度原语 -------------------------------------------------------------


def char_containment(needle: str, haystack: str) -> float:
    """``needle`` 的**字符集**有多大比例出现在 ``haystack`` 里，值域 [0, 1]。

    二元组包含对语序敏感，会漏掉 ``控制方式/软硬镜模式`` 这种带分隔符的写法；
    字符包含作为兜底维度补上「字都出现了，只是顺序/标点不同」的情形。
    """
    a = set(needle)
    if not a:
        return 0.0
    return len(a & set(haystack)) / len(a)


def blend(needle: str, haystack: str) -> float:
    """二元组包含（主）与字符包含（辅）的加权混合。"""
    return 0.7 * containment(needle, haystack) + 0.3 * char_containment(needle, haystack)


def modifier_of(name: str) -> str:
    """剥掉指标名的中心语后缀，只留修饰语。

    ``年技术培训次数`` -> ``年技术培训``（把名称相似度从 0.61 抬到 0.83）。
    长度不足 ``len(noun) + 2`` 时不剥，避免把 ``次数`` 这种纯中心语削成空串。
    """
    for noun in _HEAD_NOUNS_BY_LEN:
        if name.endswith(noun) and len(name) >= len(noun) + 2:
            return name[: -len(noun)]
    return name


def core_name(name: str) -> str:
    """剥掉指标名**末尾**的括号限定语，只留主干。

    厂商常把计量口径写进括号里（``3D眼镜数量(眼镜式、夹片式各)``），
    那段是「怎么数」的说明，不是指标名本身。留着它会让二元组集合凭空多出一倍
    噪声，把真正有区分度的 ``3D眼镜`` 稀释到 0.5 以下。
    只剥末尾一层、且剥完不能变空——``支持（水夹层过滤器）`` 这种**值**不归这里管，
    而 ``(选配)`` 这种整个名字都在括号里的写法必须原样保留。
    """
    stripped = _PAREN_TAIL.sub("", name).strip()
    return stripped or name


def name_variants(name: str | None) -> tuple[str, ...]:
    """指标名的几种规范形：原名、去括号、各自再去中心语。取最大相似度用。"""
    if not name:
        return ()
    raw, core = squash(name), squash(core_name(name))
    out: list[str] = [raw]
    if core and core != raw:
        out.append(core)
    for base in (raw, core):
        mod = modifier_of(base)
        if mod and mod != base and mod not in out:
            out.append(mod)
    return tuple(out)


def name_similarity(name: str | None, body: str | None) -> float:
    """指标名与要求正文（含分节限定语）的相似度。**两档**，不是一路取最大。

    第一档是「完整指标名」的两种规范形：原名与去括号名。它们都不损失区分度，
    直接取较大值；只要这一档已经不低于 :data:`MODIFIER_GATE`，就地返回。

    第二档才是中心语剥离，而它是一条**救济**规则：``年技术培训次数`` 在要求原文里
    写作「…每年度提供技术培训…」，剥掉 ``次数`` 能把 0.61 抬到 0.83。但剥离同时
    抹掉了唯一的区分度——``3D内窥镜工作长度`` 与 ``3D内窥镜工作直径`` 剥完都变成
    ``3D内窥镜工作``，于是 P3 的「工作长度≥300mm」和「工作直径10mm——12mm」
    会对两条证据打出**完全相同**的名称分，把一对正确答案整体对调。
    所以只在原名已经对不上（< MODIFIER_GATE）时才允许剥离来救。
    """
    rb = squash(body)
    if not name or not rb:
        return 0.0
    forms = (squash(name), squash(core_name(name)))
    full = max(blend(v, rb) for v in forms)
    if full >= MODIFIER_GATE:
        return min(1.0, full)
    return min(1.0, max(full, *(blend(modifier_of(v), rb) for v in forms)))


def grounded_name_similarity(name: str | None, body: str | None, context: str = "") -> float:
    """带分节限定语的名称相似度，但要求指标名**在正文里有落点**才允许分节加持。

    分节只能补「哪一台设备」，不能凭空造出「哪一个指标」。少了这道闸，
    ``3D内窥镜数量`` 这类以设备名开头的指标会对该分节下的**每一条**要求都拿到
    满分——因为剥掉中心语后剩下的 ``3D内窥镜`` 整个来自分节，正文一个字都没贡献
    （P3「4、工作长度≥300mm」就是这样被 ``检测报告`` 的数量指针抢走的）。

    落点判据用**二元组**而不是单字：``300mm`` 里的 ``3`` 会让单字包含误命中，
    而 ``3D内窥镜数量`` 与 ``数量：2 条`` 共享 ``数量`` 这个二元组，与
    ``工作长度≥300mm`` 一个都不共享。
    """
    plain = name_similarity(name, body)
    if not context or not name:
        return plain
    rb = squash(body)
    target = bigrams(rb)
    if not any(bigrams(v) & target for v in name_variants(name)):
        return plain
    return max(plain, name_similarity(name, f"{context} {body}"))


def count_modifier(name: str | None) -> str:
    """若指标名是「X + 计数中心语」，返回 X；否则返回 ``""``。

    ``泵数量`` -> ``泵``，``电刀数量`` -> ``电刀``，``触摸屏尺寸`` -> ``""``。
    """
    ns = squash(name)
    for head in COUNT_HEADS:
        if ns.endswith(head) and len(ns) > len(head):
            return ns[: -len(head)]
    return ""


def count_question(name: str | None, body: str | None) -> str:
    """证据名在报「X 的数量」、要求正文在问「X 有几个」时，返回 X。

    中文招标里这是同一个问题的两种写法：``具有≥4 个泵`` ↔ ``泵数量：3个``。
    此时指标名的全部信息量都在 X 上，计数中心语由正文的量词句式承担，
    所以名称相似度应当按 ``blend(X, 正文)`` 重算（-> 1.0），
    而不是被 ``泵数量`` 这个三字符串的 0.10 拖死。

    两个条件都要满足，缺一个都不算：X 必须**逐字**出现在正文里，
    正文里必须真的有「数 + 量词」。这样 ``电刀数量`` 不会挂到
    正文只写着 ``数量：1 台`` 的行上——那类行由分节限定语负责。
    """
    mod = count_modifier(name)
    if not mod:
        return ""
    rb = squash(body)
    return mod if (mod in rb and _COUNT_ASK.search(rb)) else ""


def num_similarity(value: str | None, req_text: str | None) -> float:
    """响应值里的「数值+单位」指纹有多大比例在要求原文里也出现过。

    ``4cm`` vs ``≤4.0cm`` 靠它对齐：``Decimal("4") == Decimal("4.0")`` 为真，
    单位指纹同为 ``cm``。响应侧没有数值时返回 0（不是 1），避免空值白拿满分。
    """
    toks = numbers(value)
    if not toks:
        return 0.0
    want = {(t.value, t.unit) for t in numbers(req_text)}
    return sum(1 for t in toks if (t.value, t.unit) in want) / len(toks)


def value_similarity(value: str | None, req_text: str | None) -> tuple[float, bool]:
    """返回 ``(相似度, 是否逐字回显)``。

    逐字回显的门槛是 ``len >= 4``：``8小时`` 只有 3 个字符，短到这个程度时
    「值出现在原文里」几乎不能说明任何事，必须回退到 LCS / 数值指纹。
    """
    vs = squash(value)
    if not vs:
        return 0.0, False
    rf = squash(req_text)
    if len(vs) >= MIN_NAME_LEN and vs in rf:
        return 1.0, True
    return max(lcs_len(vs, rf) / len(vs), num_similarity(value, req_text)), False


def unit_conflict(value: str | None, req_text: str | None) -> bool:
    """两侧都有带单位的数值，且量纲集合**完全不相交** -> 讲的不是同一件事（B23）。

    判据落在量纲而不是单位字符串上，所以 ``9360mA`` 与 ``≤10A`` 不算冲突（同为 current），
    ``万像素`` 与 ``万`` 也不算（同为 count）；而 ``84W`` 与 ``200μm…可选`` 算（power vs length）。
    任一侧没有带单位的数值时返回 ``False``——宁可不扣分，也不要在信息不足时乱扣。
    """
    da = {dimension(t.unit) for t in numbers(value) if t.unit}
    db = {dimension(t.unit) for t in numbers(req_text) if t.unit}
    da.discard("")
    db.discard("")
    return bool(da) and bool(db) and not (da & db)


def is_refusal(value: str | None) -> bool:
    """响应值是不是一句**完整的**拒绝式话术（agent.md X21）。

    整句全等，不是子串匹配：``支持（水夹层过滤器）`` 里含「不含」二字吗？不含。
    但 ``标准配置不含，可按需定制`` 的子串「不含」如果按包含匹配，就会把
    P5 标的5 那三条 ``支持（…）`` 全部误判成负偏离。
    """
    vs = squash(value)
    return bool(vs) and vs in _REFUSALS


# --- 候选 -------------------------------------------------------------------


@dataclass
class Candidate:
    """一条 ``(要求, 证据)`` 配对的打分结果。可变：消歧各 pass 会就地打旗标。"""

    requirement: Requirement
    evidence: Evidence
    score: float
    method: str = "none"
    name_sim: float = 0.0
    value_sim: float = 0.0
    echo: bool = False  # 值逐字回显于要求原文
    prefix: bool = False  # 指标名是要求正文的前缀
    alias: str = ""  # 命中的别名组名
    unit_agree: bool = False
    unit_clash: bool = False
    flags: tuple[str, ...] = ()
    reason: str = ""

    def add_flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags = (*self.flags, flag)

    @property
    def eligible(self) -> bool:
        """综合分是否过线，可以参与 argmax。"""
        return self.score >= THRESHOLD

    @property
    def confidence(self) -> float:
        """映射到 [0, 1] 的置信度：**取分数与各降信因子上限的较小值**。

        降信因子的排序反映「这一类匹配错了会有多难发现」：
        别名挂接（0.55）> 歧义候选（0.62）> LLM 兜底 / 报告指针（0.70）> 词面匹配（1.0）。
        未过 ``STRONG`` 的再统一折 0.85，让「刚过线」与「几乎确定」在复核报告里分得开。
        """
        cap = 1.0
        if "alias_rescue" in self.flags:
            cap = 0.55
        elif "ambiguous_evidence" in self.flags:
            cap = 0.62
        elif self.method == "llm" or self.evidence.is_pointer:
            cap = 0.70
        elif self.method == "alias":
            cap = 0.65
        conf = min(self.score, cap)
        if self.score < STRONG:
            conf *= 0.85
        return round(min(conf, 1.0), 3)


def _row_no(ev: Evidence) -> int:
    """从 ``locator`` 里取最后一个整数，用于同分时的稳定 tie-break。

    与 ``grade.py`` 的位置比对规则同源（都取 ``re.findall(r"\\d+", loc)[-1]``），
    这样「取表行号较小者」这条规则在评分口径下也是可解释的。
    """
    digits = re.findall(r"\d+", ev.locator or "")
    return int(digits[-1]) if digits else 0


def score_candidate(
    req: Requirement,
    ev: Evidence,
    *,
    body: str | None = None,
    context: str = "",
) -> Candidate:
    """给一个 ``(要求, 证据)`` 对打分。**纯函数**，不查表、不调用 LLM、无副作用。

    ``body`` 是去掉序号/标记后的要求正文；``context`` 是分节限定语
    （:func:`~bidtable.ingest.scope.section_context`），只参与**名称相似度**的
    比对，不参与前缀判定、别名判定与数值判定——那三件事问的都是
    「这条要求自己说了什么」，把父级标题混进去只会制造假命中。
    """
    text = body if body is not None else body_of(req)
    rb = squash(text)
    ns = squash(ev.name)
    vs = squash(ev.value)

    n_sim = grounded_name_similarity(ev.name, text, context)
    counted = count_question(ev.name, text)
    if counted:
        # 「X数量」对上「有 N 个 X」：信息量全在 X 上，按 X 重算名称相似度。
        n_sim = max(n_sim, blend(counted, text))
    if ev.is_pointer or not vs:
        # 指针证据的值是「详见检测报告」，样本条目行的值是空串，两者都**不携带
        # 任何可与要求原文比对的词面信息**（agent.md 4.3 L1 降级路径）。
        # 此时把 40% 的权重压在一个注定为 0 的值相似度上，只会让正确配对整体
        # 掉到阈值以下，所以名称相似度就是全部证据。
        v_sim, echo = 0.0, False
        base = n_sim
    elif is_refusal(ev.value):
        # 拒绝式响应（选配/暂不支持/标准配置不含…）同样是在**回绝**而不是复述指标，
        # 值里没有可比对的词面信息，也只能靠名称。但要打 :data:`REFUSAL_WEIGHT` 的折扣：
        # 名称匹配是单路信号，比「名称对得上 + 值也对得上」的双路印证弱一档，
        # 折扣让后者在同等名称分下稳定胜出（P3「摄像头可低温等离子消毒灭菌」
        # 就该输给实值证据 ``摄像头低温等离子消毒灭菌：支持（…）``，
        # 而不是输给分节里恰好提到 ``4K 腹腔镜`` 的 ``4K腹腔镜灭菌方式：选配``）。
        v_sim, echo = 0.0, False
        base = REFUSAL_WEIGHT * n_sim
    else:
        v_sim, echo = value_similarity(ev.value, req.text)
        base = W_NAME * n_sim + W_VALUE * v_sim

    prefix = len(ns) >= MIN_NAME_LEN and rb.startswith(ns)
    alias = alias_group(ev.name, text) or ""
    agree = shares_unit(ev.value, req.text)
    clash = unit_conflict(ev.value, req.text)

    if echo and len(vs) >= MIN_ECHO_LEN and _CJK_RE.search(vs):
        score, method = 1.0, "contain"
    else:
        score = base
        if agree:
            score += UNIT_AGREE_BONUS
        if clash:
            score -= UNIT_PENALTY
        if alias and n_sim < ALIAS_CAP and not ev.is_pointer:
            score += ALIAS_BONUS
        score = max(0.0, min(score, 1.0))
        if n_sim >= 0.90:
            method = "exact"
        elif v_sim >= 0.90:
            method = "fingerprint"
        elif alias:
            method = "alias"
        else:
            method = "exact" if score >= THRESHOLD else "none"
    if prefix:
        score = max(score, PREFIX_SCORE)
        method = "exact"

    return Candidate(
        requirement=req,
        evidence=ev,
        score=round(score, 4),
        method=method,
        name_sim=round(n_sim, 4),
        value_sim=round(v_sim, 4),
        echo=echo,
        prefix=prefix,
        alias=alias,
        unit_agree=agree,
        unit_clash=clash,
        flags=("count_alias",) if counted else (),
        reason=_reason(method, n_sim, v_sim, ev, alias),
    )


def _reason(method: str, n_sim: float, v_sim: float, ev: Evidence, alias: str) -> str:
    """人类可读的匹配依据，直接进复核报告（agent.md 10.3 第 2 条）。"""
    if ev.is_pointer:
        return f"检测报告指针项「{ev.name}」，实测值需 OCR 才能取得（名称相似度 {n_sim:.2f}）"
    if method == "contain":
        return f"响应值逐字回显于招标要求原文（长度 {len(squash(ev.value))} 字符）"
    if method == "alias" and alias:
        return f"指标名经「{alias}」别名组与要求对齐（名称 {n_sim:.2f} / 数值 {v_sim:.2f}）"
    if method == "fingerprint":
        return f"数值与单位指纹一致（名称 {n_sim:.2f} / 数值 {v_sim:.2f}）"
    return f"指标名与要求正文一致（名称 {n_sim:.2f} / 数值 {v_sim:.2f}）"


# --- Pass 1：同值折叠 -------------------------------------------------------


def collapse_values(cands: Sequence[Candidate]) -> list[Candidate]:
    """同一条要求下，**值 squash 后完全相同**的候选只保留文档优先级最高的那一个（X4）。

    必须排在 argmax 之前。产品综合样本里的指标名常常就是招标原文的前缀
    （``钬激光治疗系统配置要求``），名称分反而比技术规格书的 ``配置清单`` 高得多；
    不先折叠，优先级 2 的样本就会抢走优先级 1 的规格书，`证明材料文件` 列直接判错。

    两条硬性例外：

    * **指针证据不参与折叠**。它们的 ``value`` 都是空串，一折叠就只剩一条，
        而 P1 一个标的就有 3 条指向检测报告不同表行的指针。
    * 折叠只在**同一条要求内部**发生，跨要求不折叠（那是 claim 的职责）。
    """
    groups: dict[str, list[Candidate]] = {}
    for c in cands:
        if c.evidence.is_pointer:
            continue
        vs = squash(c.evidence.value)
        if vs:
            groups.setdefault(vs, []).append(c)

    dropped: set[int] = set()
    for grp in groups.values():
        if len(grp) < 2:
            continue
        best_pri = min(c.evidence.priority for c in grp)
        keep = [c for c in grp if c.evidence.priority == best_pri]
        # 同优先级还有多条（极少见）：留分数最高的那一条，其余一并折叠掉。
        keep.sort(key=lambda c: -c.score)
        survivor = keep[0]
        for c in keep[1:]:
            dropped.add(id(c))
        for c in grp:
            if c is not survivor:
                dropped.add(id(c))
        if len({c.evidence.doc_type for c in grp}) > 1:
            survivor.add_flag("conflict_resolved")
    return [c for c in cands if id(c) not in dropped]


# --- Pass 2：候选偏好序 + 歧义标记 ------------------------------------------


def _pointer_yields(c: Candidate, best_value_score: float | None) -> bool:
    """这条指针候选是否应当给实值证据让路。

    指针（``详见检测报告``）是所有证据里最弱的一档：它没有值，还要靠 OCR 才能
    变成可核对的实测值。但**不能**因此在打分阶段一刀切打折——训练集里 28 行的
    正确答案就是指针，其中不少（``年免费维护保养次数`` 对 ``每年免费提供≥4 次…``）
    名称分只比阈值高一点点，打折就直接掉出候选池。

    所以让位只发生在**排序**阶段，而且只针对「与某条实值证据几乎同样贴合」的指针：
    分差在 :data:`POINTER_BAND` 之内时，可核对的那条排前面。指针一枝独秀
    （本条要求下没有任何合格实值证据）或明显更贴合（分差超过带宽）时，原序不动。
    这条规则问的是「两份材料一样说得通时先信谁」，与具体标的无关。
    """
    if best_value_score is None or not c.evidence.is_pointer:
        return False
    return c.score - best_value_score <= POINTER_BAND


def _pref_key(c: Candidate, best_value_score: float | None) -> tuple[int, int, float, int, int]:
    """报告优先（X3）-> 近分指针让位 -> 分数降序 -> 文档优先级升序 -> 表行号升序。"""
    report_wins = 0 if (c.evidence.doc_type == DOC_REPORT and not c.evidence.is_pointer) else 1
    return (
        report_wins,
        1 if _pointer_yields(c, best_value_score) else 0,
        -c.score,
        c.evidence.priority,
        _row_no(c.evidence),
    )


def preference(cands: Sequence[Candidate]) -> list[Candidate]:
    """一条要求的**合格**候选偏好序；``[0]`` 即无争用时的胜出者。"""
    eligible = [c for c in cands if c.eligible]
    value_scores = [c.score for c in eligible if not c.evidence.is_pointer]
    best_value_score = max(value_scores) if value_scores else None
    return sorted(eligible, key=lambda c: _pref_key(c, best_value_score))


def mark_ambiguity(cands: Sequence[Candidate]) -> None:
    """就地打 ``ambiguous_evidence`` / ``close_call`` 旗标（B25、agent.md 10.3 第 2 条）。

    * ``ambiguous_evidence``：完全同分、同文档类型、同优先级的**多条**候选。
        此时无法从材料本身分辨该用哪一条，规则上取表行号较小者，但必须告诉复核人。
    * ``close_call``：次优候选来自**另一份材料**且分差在 ``PRIORITY_BAND`` 内。
        比同分歧义常见得多，也更值得在复核报告里露出来。
    """
    eligible = [c for c in cands if c.eligible]
    if len(eligible) < 2:
        return
    top = max(c.score for c in eligible)
    exact = [c for c in eligible if abs(c.score - top) <= 1e-9]
    if len(exact) > 1:
        head = exact[0]
        same = [
            c
            for c in exact
            if c.evidence.doc_type == head.evidence.doc_type
            and c.evidence.priority == head.evidence.priority
        ]
        if len(same) > 1:
            for c in same:
                c.add_flag("ambiguous_evidence")
    for c in eligible:
        if 0 < top - c.score <= PRIORITY_BAND and c.evidence is not exact[0].evidence:
            for d in eligible:
                if top - d.score <= PRIORITY_BAND:
                    d.add_flag("close_call")
            break


# --- Pass 3：独占声明竞争 ---------------------------------------------------


def claim(order: dict[int, list[Candidate]]) -> dict[int, Candidate]:
    """把证据**独占**地分派给要求：一条证据最多服务一条要求，没有共享通道。

    这不是保守，是数据本身的要求——训练集 206 行里 ``(证明材料文件, 位置, 投标响应)``
    三元组零重复，没有任何一条证据被两行共用。一旦允许「分数接近就共享」，
    一条通用证据（``电刀数量：1台``）会被同标的下所有 ``数量：1 台`` 同时认领，
    错一处就是错一片。

    算法是**稳定提议**（Gale–Shapley 的单向版）而不是简单的强者先挑：

    1. 要求按「自身最优候选分数」降序入队；
    2. 队首沿自己的偏好序往下提议，遇到已被占用的证据时，**只有自己的分数严格更高**
        才把它抢过来；抢不到就继续试下一条候选；
    3. 被抢走证据的要求重新入队，游标停在被抢那条**之后**，于是它接着去争自己的次优。

    单纯「强者先挑」会留下经典的贪心破口：A（0.95，只能要 X）先拿走 X，
    B（0.92，X/Y 都行）就只能空手，而 A 其实还有 0.90 的 Y 可用。
    提议-抢占把这种死结解开。

    终止性：游标只前进不回退，每轮循环要么推进某个游标、要么把该要求永久移出队列，
    总步数不超过所有偏好序长度之和；抢占要求分数**严格**更高，所以同一条证据上
    的持有分数单调递增，不会出现环。
    """
    assigned: dict[int, Candidate] = {}
    owners: dict[int, tuple[int, float]] = {}  # id(evidence) -> (持有者键, 持有者分数)
    cursor: dict[int, int] = dict.fromkeys(order, 0)
    # 平分时的次序必须**稳定**：早先用 str(id(req))（内存地址）打破平局，
    # 同一份资料包两次运行会在 P3 那种大量同分「数量：1 台」上给出不同结果。
    # 改用要求自身的 (标的, 分节, 序号)——调用方已过滤掉空偏好序。
    queue = sorted(
        order,
        key=lambda k: (-max(c.score for c in order[k]), order[k][0].requirement.key),
    )
    while queue:
        k = queue.pop(0)
        prefs = order[k]
        while cursor[k] < len(prefs):
            c = prefs[cursor[k]]
            cursor[k] += 1
            eid = id(c.evidence)
            owner = owners.get(eid)
            if owner is not None and c.score <= owner[1]:
                continue  # 占用者更贴合，让位；接着试自己的下一条候选
            if owner is not None:
                del assigned[owner[0]]
                queue.append(owner[0])  # 游标已越过被抢那条，它会去争次优
            assigned[k] = c
            owners[eid] = (k, c.score)
            break
    return assigned


# --- Pass 4：拒绝式证据的唯一挂接 -------------------------------------------


def rescue_refusals(
    reqs: Sequence[Requirement],
    evidence: Sequence[Evidence],
    assigned: dict[int, Candidate],
    warnings: list[str],
) -> dict[int, Candidate]:
    """把「厂商明确拒绝」的证据挂到它**唯一可能**对应的那条未匹配要求上（P1 序号1）。

    招标写「用于泌尿系结石的碎石，泌尿系肿瘤的汽化和凝固。」，厂商写「适用范围：暂不支持」，
    词面零重叠，任何相似度模型都过不了线；但这条证据显然就是在回答它。

    挂接条件刻意苛刻，三条全中才动手：

    1. 值**整句全等**于 ``NEGATIVE_PHRASES`` 里的某一条（X21：``支持（水夹层过滤器）`` 不算）；
    2. 该证据还没被任何要求认领；
    3. 同标的下仍未匹配的要求里，**恰好只有一条**与该指标名同属一个别名组。

    命中 0 条 -> 保持未响应；命中 ≥2 条 -> 保持未响应并告警，绝不猜（N9）。
    """
    used = {id(c.evidence) for c in assigned.values()}
    open_reqs = [r for r in reqs if not r.is_heading and id(r) not in assigned]
    out: dict[int, Candidate] = {}
    for ev in evidence:
        if id(ev) in used or not is_refusal(ev.value):
            continue
        hits = [r for r in open_reqs if r.lot_no == ev.lot_no and alias_group(ev.name, body_of(r))]
        if len(hits) == 1:
            req = hits[0]
            cand = score_candidate(req, ev, context=section_context(req))
            cand.method = "alias"
            cand.add_flag("alias_rescue")
            cand.reason = (
                f"厂商对该项明确回复「{ev.value}」；本标的内仅本条要求与指标名"
                f"「{ev.name}」同属「{cand.alias}」别名组，据此唯一挂接"
            )
            out[id(req)] = cand
            open_reqs.remove(req)
            used.add(id(ev))
        elif len(hits) > 1:
            nos = "、".join(r.no for r in hits)
            warnings.append(
                f"标的{ev.lot_no}：拒绝式证据「{ev.name}：{ev.value}」可能对应序号 {nos} "
                f"共 {len(hits)} 条要求，无法唯一确定，已保持未响应"
            )
    return out


# --- 编排 -------------------------------------------------------------------


def build_pool(
    reqs: Sequence[Requirement],
    evidence: Sequence[Evidence],
    *,
    floor: float = POOL_FLOOR,
) -> dict[int, list[Candidate]]:
    """为每条要求生成候选池：**同标的内**两两打分 -> 同值折叠 -> 歧义标记。

    跨标的绝不配对（I12 的匹配层对应物）：P5 四个标的各有自己的型号与资料，
    让标的1 的要求去匹标的5 的证据只会制造假阳性。
    标题/引导句（``is_heading``）不进池——它们不判分，也不该占用证据。

    名称相似度的比对文本是 ``分节限定语 + 要求正文``。P3 的招标表里有大量
    ``1、数量：1 台`` 这种正文完全不带主语的行，设备名只存在于分节标题
    （``四、配套LED冷光源``）；不把它拼进来，一条 ``电刀数量：1台`` 就会同时
    和五条「数量：1 台」打平。分节为空时（P1/P2/P4 全是）此路径退化为原样。

    返回的字典以 ``id(req)`` 为键，不用 ``req.key``：``(标的,分节,序号)`` 的唯一性
    是**输出层**要保证的不变式（I1），抽取层在异常资料包上可能产出重复键，
    用对象身份做键可以让重复键安全地各自成池，最后由 writer.dedupe 收敛并告警。
    """
    by_lot: dict[str, list[Evidence]] = {}
    for ev in evidence:
        by_lot.setdefault(ev.lot_no, []).append(ev)

    pool: dict[int, list[Candidate]] = {}
    for req in reqs:
        if req.is_heading:
            pool[id(req)] = []
            continue
        body = body_of(req)
        ctx = section_context(req)
        scored = [
            score_candidate(req, ev, body=body, context=ctx)
            for ev in by_lot.get(req.lot_no, ())
        ]
        kept = sorted((c for c in scored if c.score >= floor), key=lambda c: -c.score)
        kept = collapse_values(kept)
        mark_ambiguity(kept)
        pool[id(req)] = kept
    return pool


def _no_match_reason(req: Requirement, cands: Sequence[Candidate]) -> str:
    """未匹配时给复核报告的一句话解释（agent.md 10.3 第 3 条：跳过内容必须有理由）。

    三种情形必须分开说，否则复核人无法判断该不该手工补一条：

    * 本标的压根没有候选（跨标的不配对，或该标的没有任何厂商资料）；
    * 最高分没过 :data:`THRESHOLD`；
    * **过了阈值但证据被同标的内另一条要求独占认领**（:func:`claim` 的排他约束）。
        最后这种最常见于 P3 那种一堆「数量：1 台」，笼统说「低于阈值」是假话。
    """
    if not cands:
        return "本标的的厂商资料里没有任何可配对的指标"
    best = max(cands, key=lambda c: c.score)
    where = f"「{best.evidence.name}」（{_locator_of(best.evidence)}）"
    if best.score >= THRESHOLD:
        return (
            f"最贴合的候选是{where}，综合分 {best.score:.2f} 已过阈值，"
            f"但该证据已被同标的内更贴合的另一条要求独占认领（一条证据不服务两条要求）"
        )
    return f"最接近的候选是{where}，综合分 {best.score:.2f} 低于阈值 {THRESHOLD:.2f}"


def _locator_of(ev: Evidence) -> str:
    """locator 本身已带资料类型前缀，只在缺前缀时补上，避免「检测报告 检测报告 第1页」。"""
    loc = (ev.locator or "").strip()
    if loc.startswith(ev.doc_type):
        return loc
    return f"{ev.doc_type} {loc}".strip()


def match_all(
    reqs: Sequence[Requirement],
    evidence: Sequence[Evidence],
    *,
    client: object | None = None,
) -> tuple[list[Match], list[str]]:
    """匹配层总入口：``(要求, 证据) -> Match``，每条要求**恰好**一个 Match。

    五个 pass 的次序不可调换，理由分别写在各自 docstring 里：
    同值折叠 -> 偏好序与歧义标记 -> 独占声明竞争 -> 拒绝式唯一挂接 -> LLM 兜底。

    ``client`` 为 ``None`` 或未启用时，第五步静默跳过：离线跑通是硬要求
    （agent.md 4.3「即使 Key 不可用，整套流程仍能离线跑通」）。
    """
    warnings: list[str] = []
    pool = build_pool(reqs, evidence)
    order = {k: preference(v) for k, v in pool.items()}
    order = {k: v for k, v in order.items() if v}
    assigned = claim(order)
    assigned.update(rescue_refusals(reqs, evidence, assigned, warnings))

    enabled = client is not None and bool(getattr(client, "enabled", False))
    if enabled:
        for req in reqs:
            k = id(req)
            if req.is_heading or k in assigned:
                continue
            pick = semantic.suggest(req, pool.get(k, ()), client)  # type: ignore[arg-type]
            if pick is not None:
                assigned[k] = pick
                warnings.append(
                    f"标的{req.lot_no} 序号{req.no}：确定性打分未过阈值，改用语义模型匹配"
                    f"「{pick.evidence.name}」（{pick.evidence.locator}）"
                )

    out: list[Match] = []
    for req in reqs:
        cand = assigned.get(id(req))
        if cand is None:
            reason = "标题/引导句，不参与匹配" if req.is_heading else _no_match_reason(req, pool.get(id(req), ()))
            out.append(
                Match(
                    requirement=req,
                    evidence=None,
                    score=0.0,
                    method="none",
                    confidence=0.0,
                    reason=reason,
                )
            )
            continue
        out.append(
            Match(
                requirement=req,
                evidence=cand.evidence,
                score=cand.score,
                method=cand.method,
                confidence=cand.confidence,
                reason=cand.reason,
                flags=cand.flags,
            )
        )
    return out, warnings
