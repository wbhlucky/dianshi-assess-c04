"""``（N）`` 子项的**折叠决策**：证据驱动的 fold / split（agent.md 3.5 的 ``（N）`` 策略）。

这是训练集里最刁钻的一处**跨包不自洽**：P1 标的2 与 P2 标的2 的招标原文
**逐字相同**（只差换行位置），gold 却一个是折叠的、一个是展开的::

    2、设备运行维护：
    （1）日常巡检维护：…    （2）设备性能维护：…    （3）应用维护：…

    P1 gold 标的2 = 序号 {1, 2}          <- 折叠：（1）（2）（3）并进序号 2
    P2 gold 标的2 = 序号 {1, 2.1, 2.2, 2.3}  <- 展开：父项「2」连行都没有

⇒ **判据不在招标 PDF 里，而在厂商证据里**。实测三份 ``*_服务方案.docx`` 的 ``项目`` 列：

===== ================================= ========== ==================
包     服务方案的 ``项目``                    命中的是    gold
===== ================================= ========== ==================
P1    ``技术培训`` / ``设备运行维护``         **父项**    折叠
P2    ``日常巡检维护``                       **子项**    展开
P3    ``常巡检维护``                         **子项**    展开
===== ================================= ========== ==================

道理很直白：投标人**按什么粒度应答**，评标就按什么粒度拆行。厂商只给了一句
「设备运行维护」把整个服务包概括过去，就没有可对齐的子项；厂商逐项列出
「日常巡检维护」这样的子项，就必须拆开才能逐条判偏离。

所以规则是：

* 只有**由 ``（N）`` 展开出来的**子项才是折叠候选（P3 的 ``2.1`` 是原文里真写着的
  独立编号，折叠它是有损的，永不参与）；
* 只在**服务标的**上做（设备标的的 ``（N）`` 在抽取层就已 fold，见
  :func:`tender_layout_a.paren_mode`，P1 标的1 序号14 即此例）；
* 证据名命中父项标题而不命中任何子项标题 → **折叠**；否则 → **保持展开**。
* 拿不到证据时**默认展开**并告警：``多余条目`` 不计分，覆盖率计分，
  多输出永远比少输出安全（agent.md 4.5）。

``--paren-policy`` 可强制覆盖：``fold`` / ``split`` 跳过证据判断，``auto`` 走上述规则。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from ..models import Evidence, Requirement
from . import marks
from . import numbering
from .numbering import int_parts
from .scope import body_of

# ``（1）`` / ``(1)`` 前缀，用于把它从短标题里剥掉。
_PAREN_HEAD = re.compile(r"^\s*[（(]\s*\d{1,2}\s*[）)]\s*")
# 标题取到第一个冒号为止（``日常巡检维护：提供设备运行巡检服务…`` -> ``日常巡检维护``）。
_TITLE_CUT = re.compile(r"[：:]")


@dataclass
class FoldResult:
    """折叠结果。``notes`` 进复核报告，说明为什么这样拆/合。"""

    reqs: list[Requirement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", marks.strip_all(s or ""))


def title_of(req: Requirement) -> str:
    """取条目的**短标题**：剥掉编号/``（N）``/标记，截到第一个冒号。"""
    body = body_of(req)
    body = _PAREN_HEAD.sub("", body).strip()
    body = re.sub(r"^\d+(?:[.．]\d+)*\s*[、.．]?\s*", "", body).strip()
    return _TITLE_CUT.split(body, 1)[0].strip()


def is_paren_child(req: Requirement) -> bool:
    """这一条是不是由 ``（N）`` 展开出来的（``head_text`` 以 ``（N）`` 开头）。"""
    return numbering.starts_with_paren(req.head_text or req.text or "")


def _names_for(evidence: list[Evidence], lot_no: str) -> set[str]:
    return {_norm(e.name) for e in evidence if e.lot_no == lot_no and e.name}


def _hit(title: str, names: set[str]) -> bool:
    """标题与证据名互相包含即算命中（``常巡检维护`` vs ``日常巡检维护``）。"""
    t = _norm(title)
    if not t:
        return False
    return any(t in n or n in t for n in names if n)


def decide(
    parent: Requirement,
    children: list[Requirement],
    names: set[str],
) -> tuple[bool, str]:
    """返回 ``(是否折叠, 理由)``。纯函数，方便单测穷举。"""
    if not names:
        return False, "该标的没有任何厂商证据可参照，默认保持展开（多输出不计分，漏输出丢覆盖率）"
    p_title = title_of(parent)
    c_titles = [title_of(c) for c in children]
    p_hit = _hit(p_title, names)
    c_hit = [t for t in c_titles if _hit(t, names)]
    if p_hit and not c_hit:
        return True, f"厂商证据按父项「{p_title}」整体应答，未逐项应答子项，故折叠"
    if c_hit:
        return False, f"厂商证据逐项应答了子项「{'、'.join(c_hit)}」，故保持展开"
    return False, f"厂商证据既未命中父项「{p_title}」也未命中任何子项，保持展开"


def _blocks(reqs: list[Requirement]) -> list[list[int]]:
    """按连续同标的切块，返回下标分组。"""
    groups: list[list[int]] = []
    for i, req in enumerate(reqs):
        if groups and reqs[groups[-1][0]].lot_no == req.lot_no:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def resolve(
    reqs: list[Requirement],
    evidence: list[Evidence],
    scopes: list[object] | None = None,
    *,
    policy: str = "auto",
) -> FoldResult:
    """主入口。``reqs`` 必须是**已展开 ``（N）``** 的抽取结果（service 标的用 split 模式）。"""
    res = FoldResult()
    if policy not in ("auto", "fold", "split"):
        res.warnings.append(f"未知的 --paren-policy {policy!r}，已按 auto 处理。")
        policy = "auto"

    service_lots = {sc.lot_no for sc in (scopes or []) if getattr(sc, "is_service", False)}
    drop: set[int] = set()
    patch: dict[int, Requirement] = {}

    for group in _blocks(reqs):
        lot_no = reqs[group[0]].lot_no
        if service_lots and lot_no not in service_lots:
            continue  # 设备标的的 ``（N）`` 在抽取层已经 fold，这里不插手
        # 找「父项 + 其 （N） 展开子项」组
        for i in group:
            parent = reqs[i]
            parts = int_parts(parent.no)
            if len(parts) != 1 or not parent.is_heading:
                continue
            prefix = parent.no + "."
            kids = [j for j in group if j > i and reqs[j].no.startswith(prefix) and is_paren_child(reqs[j])]
            if not kids:
                continue
            children = [reqs[j] for j in kids]
            if policy == "split":
                do, why = False, "--paren-policy split 强制保持展开"
            elif policy == "fold":
                do, why = True, "--paren-policy fold 强制折叠"
            else:
                do, why = decide(parent, children, _names_for(evidence, lot_no))
            label = f"标的{lot_no} 序号{parent.no}（{title_of(parent)}）+ {len(children)} 个（N）子项"
            if not do:
                res.notes.append(f"{label}：保持展开 —— {why}。")
                continue
            merged = replace(
                parent,
                is_heading=False,
                text=parent.head_text + "".join(c.head_text or c.text for c in children),
                head_text=parent.head_text + "".join(c.head_text or c.text for c in children),
                mark=parent.mark or marks.find_marks("".join(c.mark for c in children)),
            )
            patch[i] = merged
            drop.update(kids)
            res.notes.append(f"{label}：已折叠成一行 —— {why}。")

    for i, req in enumerate(reqs):
        if i in drop:
            continue
        res.reqs.append(patch.get(i, req))
    return res
