"""``分节`` 策略（agent.md 3.5）——**整个项目最大的覆盖率风险点**。

对齐键是 ``(标的, 分节, 序号)``，``分节`` 错一个字符该行所有指标归零。
而标准答案的 ``分节`` 约定**跨包不自洽**：P2 的单元格里明明有 ``1、眼底成像``
这样的层级标题，gold 却全不填；P5 标的5 填、同标的的序号 4/5 又不填。
所以这里不追求语义完美，而是把拟合出来的规则做成**可插拔策略 + 可切换 + 带诊断**。

============ ================================================== =========
策略          描述                                                训练集上限
============ ================================================== =========
``s4``       下述派生规则集（默认）                              1.000
``s2``       仅在 ``(标的,序号)`` 键冲突时才填（安全兜底）        0.946
``s1``       总是填                                              ≈0.72
``off``      从不填                                              ≈0.72
============ ================================================== =========

**S4 规则集**（逐条对应 agent.md 3.5）：

1. 参数来自独立的 ``序号`` 列表格（B 族）→ **永不**输出 ``分节``。
2. 中文序数标题（``^[一二三四五六七八九十]+、``）→ **总是**进路径。
3. 整数 ``N、Title`` 标题（有 ``N.k`` 子项）→ 仅在 (a) 路径里已有中文序数祖先，
   或 (b) 该整数层级是**混合层**（同层既有标题又有真参数项）时进路径。
   P3 走 (a)；P5 标的5 走 (b)（``1、治疗源`` 是标题而 ``4、欧盟CE 认证`` 是真参数）；
   P2 的一级层**纯是标题** → 两条都不成立 → 不填。
4. 引导句（以 ``：`` 结尾、无编号）→ 仅当它**包含该标的的设备名称**
   且**块内至少存在一个整数标题**时，算作 0 级标题。
   P3 服务标的两个条件都满足 → 填；P1 服务标的没有整数标题 → 不填。
5. 同层出现**兄弟参数项**时，该层及以下的路径重置。
   P5 标的5 的 ``4、欧盟CE 认证``（1 级真参数）把 ``3、过滤器`` 顶掉，于是序号 4/5 的
   ``分节`` 回到空——这正是 gold 的样子。
6. 标题文本**逐字保留，含末尾标点**（``十一、妇科电动检查床：`` 的冒号必须在）。

路径用 ``" / "`` 拼接（X31：写成 ``/`` 无空格虽然 ``norm()`` 会吃掉、不影响评分，
但影响人工复核，所以统一带空格）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from ..config import LAYOUT_B, SECTION_POLICIES
from ..errors import InputError
from ..models import Requirement, Scope
from . import numbering
from .numbering import int_parts

SEP = " / "
_CN = re.compile(r"^[一二三四五六七八九十]+$")

# 条目形态。分节只关心「它是哪一层的标题」，所以从 ``序号`` 反推即可，
# 不必把 ``Numbered`` 一路带到这一层。
CHINESE = "chinese"
LEAD = "lead"
ITEM = "item"

# 填充率诊断阈值（agent.md 风险 R1）：填得太多或该填却没填，都要在复核报告里喊一声。
FILL_RATE_HIGH = 0.8


@dataclass(frozen=True)
class PathEntry:
    """路径上的一级标题。"""

    level: int
    kind: str
    text: str


def classify(req: Requirement) -> tuple[str, int]:
    """返回 ``(形态, 层级)``。中文序数与引导句都算 0 级（它们是最外层的节）。"""
    no = (req.no or "").strip()
    if not no:
        return LEAD, 0
    if _CN.match(no):
        return CHINESE, 0
    parts = int_parts(no)
    return ITEM, (len(parts) if parts else 1)


def heading_text(req: Requirement) -> str:
    """标题的逐字原文（规则 6）。C 族的 ``text`` 剥掉了编号，所以优先用 ``head_text``。"""
    return (req.head_text or req.text or "").strip()


def _contiguous(reqs: list[Requirement]) -> list[list[Requirement]]:
    """按**连续同标的**切块。跨块必须重置路径（P1 标的1 与标的2 互不相干）。"""
    blocks: list[list[Requirement]] = []
    for req in reqs:
        if blocks and blocks[-1][0].lot_no == req.lot_no:
            blocks[-1].append(req)
        else:
            blocks.append([req])
    return blocks


def _paren_parents(block: list[Requirement]) -> set[int]:
    """``（N）`` 展开**合成**出来的父标题在块内的下标。

    这些行在原文里并不是标题（原文只有 ``2、设备运行维护：`` 加三个 ``（N）`` 项目符号），
    是 :mod:`bidtable.ingest.fold` 为了「展开时不把父项写进 xlsx」才打上 ``is_heading`` 的。
    所以它们**不能**参与 S4 的规则 3(b)「混合层」判据，也不能满足规则 4 的
    「块内至少存在一个整数标题」——否则 P1/P2 标的2 的分节会从 gold 的空值
    变成 ``设备运维技术服务要求如下：``，4 行对齐键全错。
    P3 标的2 的 ``2.1`` 是原文真编号，不在这里，所以 P3 照填不误。
    """
    out: set[int] = set()
    for i, req in enumerate(block):
        if not req.is_heading or not req.no:
            continue
        prefix = req.no + "."
        for other in block[i + 1 :]:
            if other.no.startswith(prefix) and numbering.starts_with_paren(
                other.head_text or other.text or ""
            ):
                out.add(i)
                break
    return out


def _mixed_levels(block: list[Requirement], skip: set[int]) -> set[int]:
    """规则 3(b) 的判据：哪些整数层级是「混合层」（同层既有标题又有真参数项）。"""
    seen: dict[int, list[bool]] = {}
    for i, req in enumerate(block):
        kind, level = classify(req)
        if kind != ITEM:
            continue
        slot = seen.setdefault(level, [False, False])
        heading = req.is_heading and i not in skip
        slot[0 if heading else 1] = True
    return {level for level, (has_head, has_param) in seen.items() if has_head and has_param}


def _gate(
    req: Requirement,
    kind: str,
    level: int,
    path: list[PathEntry],
    mixed: set[int],
    device_name: str,
    has_int_heading: bool,
    policy: str,
) -> bool:
    """这一条候选标题**是否真的进路径**。``is_heading`` 只是形态初筛，S4 还要再过一道。"""
    if not req.is_heading:
        return False
    if policy == "s1":
        return True  # S1：总是填
    if kind == CHINESE:
        return True  # 规则 2
    if kind == LEAD:
        # 规则 4：含设备名 + 块内有整数标题。两个条件缺一不可（P1 服务标的就靠后者挡掉）。
        text = heading_text(req)
        return bool(device_name) and device_name in text and has_int_heading
    # 规则 3：整数标题
    if any(p.kind == CHINESE for p in path):
        return True  # 3(a)
    return level in mixed  # 3(b)


def _walk(
    block: list[Requirement],
    scope: Scope | None,
    policy: str,
) -> list[str]:
    """走一遍一个标的块，产出与 ``block`` 等长的 ``分节`` 列表。"""
    device_name = (scope.device_name if scope else "") or ""
    paren_parents = _paren_parents(block)
    mixed = _mixed_levels(block, paren_parents)
    has_int_heading = any(
        req.is_heading and classify(req)[0] == ITEM and i not in paren_parents
        for i, req in enumerate(block)
    )
    path: list[PathEntry] = []
    out: list[str] = []

    for i, req in enumerate(block):
        kind, level = classify(req)
        if i not in paren_parents and _gate(
            req, kind, level, path, mixed, device_name, has_int_heading, policy
        ):
            path = [p for p in path if p.level < level]
            path.append(PathEntry(level, kind, heading_text(req)))
            out.append("")  # 标题行不写进 xlsx，它的分节无意义
            continue
        if kind == ITEM:
            # 规则 5：真参数项把同层及更深的标题顶掉。
            # 引导句（level 0）不参与重置，否则一句 ``具体技术参数：`` 会把整条路径清空。
            path = [p for p in path if p.level < level]
        out.append(SEP.join(p.text for p in path))
    return out


def _only_conflicts(reqs: list[Requirement]) -> list[Requirement]:
    """S2：只给 ``(标的, 序号)`` 撞键的行保留 ``分节``，其余清空。"""
    seen: dict[tuple[str, str], int] = {}
    for req in reqs:
        if req.is_heading:
            continue
        key = (req.lot_no, req.no)
        seen[key] = seen.get(key, 0) + 1
    return [
        req if (req.is_heading or seen.get((req.lot_no, req.no), 0) > 1) else replace(req, section="")
        for req in reqs
    ]


def fill_rate(reqs: list[Requirement]) -> float:
    """``分节`` 填充率诊断（写进复核报告，便于在未见包上人工快速核查）。"""
    body = [r for r in reqs if not r.is_heading]
    if not body:
        return 0.0
    return sum(1 for r in body if r.section) / len(body)


def diagnose(reqs: list[Requirement]) -> list[str]:
    """填充率异常时给出**显眼**告警（agent.md 风险 R1 的缓解措施）。"""
    warnings: list[str] = []
    body = [r for r in reqs if not r.is_heading]
    if not body:
        return warnings
    rate = fill_rate(reqs)
    if rate > FILL_RATE_HIGH:
        warnings.append(
            f"【分节告警】分节填充率 {rate:.0%} 偏高（>{FILL_RATE_HIGH:.0%}）。"
            "若本包的参数区确实由多级标题组织（训练集 P3 就是 100%），这属正常；"
            "否则通常意味着标题判别过宽，会让对齐键 (标的,分节,序号) 整体错位，"
            "可改用 --section-policy s2 或 off 复核。"
        )
    elif rate == 0.0 and _has_duplicate_keys(body):
        # 只在「不填分节就真的撞键」时才喊。P1/P2 的 gold 分节本来就全空，
        # 单纯存在多级序号不足以说明策略选错了。
        warnings.append(
            "【分节告警】分节填充率为 0，但正文里出现了重复的 (标的,序号) 键。"
            "这说明标题没被识别成分节，重复键会让 grade.py 静默丢行，"
            "可改用 --section-policy s1 复核。"
        )
    return warnings


def _has_duplicate_keys(body: list[Requirement]) -> bool:
    seen: set[tuple[str, str]] = set()
    for r in body:
        key = (r.lot_no, r.no)
        if key in seen:
            return True
        seen.add(key)
    return False


def apply(
    reqs: list[Requirement],
    scopes: list[Scope],
    *,
    policy: str = "s4",
) -> tuple[list[Requirement], list[str]]:
    """主入口：给每条要求填 ``分节``。返回 ``(新列表, 告警)``（``Requirement`` 是 frozen）。"""
    if policy not in SECTION_POLICIES:
        raise InputError(
            f"未知的分节策略 {policy!r}，可选：{'/'.join(SECTION_POLICIES)}。"
        )
    if policy == "off":
        return [replace(r, section="") for r in reqs], []

    # 规则 1：B 族的参数来自独立的「序号｜技术要求」双列表，永不输出分节。
    if all(r.layout == LAYOUT_B for r in reqs):
        return [replace(r, section="") for r in reqs], []

    by_lot = {sc.lot_no: sc for sc in scopes}
    out: list[Requirement] = []
    for block in _contiguous(reqs):
        scope = by_lot.get(block[0].lot_no)
        sections = _walk(block, scope, policy)
        out.extend(replace(r, section=s) for r, s in zip(block, sections))

    if policy == "s2":
        out = _only_conflicts(out)
    return out, diagnose(out)
