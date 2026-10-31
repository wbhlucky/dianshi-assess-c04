"""范围裁剪（agent.md 3.7）：抽出来的条目里，哪些**不写进**响应偏离表。

四条规则，按危险程度从高到低：

1. **设备标的：从第一个 ``售后服务`` 条目起，整段裁掉。**
   P5 标的1 裁 ``21、售后服务要求`` 及 ``21.1``~``21.7``；标的5 裁 ``6、`` 及 6.1/6.2；
   标的8 裁 ``13、`` 及 ``13.1``~``13.6``，**连孤儿子项 ``14.7`` 一起裁**；
   标的12 有两个触发点（``16、售后服务：…`` 与 ``17、售后服务要求``），**从第一个裁**。
   ⇒ 裁剪是「从触发点起到本标的结束」的**顺序性**操作，不是逐条判断，
     这样孤儿子项（编号跳回前面的 ``14.7``）会自然被扫掉（X11）。
2. **服务标的不裁**（X15）。P1/P2/P3 的 ``标的2 = 设备运维技术服务``，
   售后服务要求**就是**正文，裁了覆盖率直接掉。判据只认 ``00_投标范围.txt`` 的 ``is_service``。
3. **永远排除**页尾套话与商务条款：``本项目核心产品为…``、
   ``本采购清单中所列技术规格及主要参数为最低要求…``、``三、商务要求``。
   （A 族的 :data:`tender_layout_a.CUT_MARKERS` 已在抽取层截断，这里是第二道保险，
   为的是 UNKNOWN 版面走 :func:`tender_pdf.extract_flat_requirements` 时也有兜底。）
4. **永远排除**投标文件格式章节里的空白表格模板标题（P1 p77 ``4.2 技术规格偏离表``）。
   它长得和参数表一模一样，但里面全是空的 ``1/2/…`` 行。

裁剪结果同时产出 ``dropped``（给测试断言）与 ``reasons``（给复核报告的「跳过清单」，
agent.md 11 节要求写明「裁掉了什么、为什么」）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import Requirement, Scope
from . import marks
from .section import SEP

# 规则 1 的触发前缀。判在**剥掉编号与标记之后**的正文上，
# 所以 ``21、售后服务要求``、``*16、售后服务：省内有常驻售后机构…`` 都能命中。
SERVICE_HEAD = "售后服务"

# 规则 3：整条以这些串开头就是套话/商务条款。用 ``startswith`` 而不是 ``in``，
# 免得把正文里恰好提到「交货期」的真参数误杀（过度裁剪比漏裁危险得多）。
BOILERPLATE_STARTS = (
    "本项目核心产品为",
    "本采购清单中所列技术规格及主要参数为最低要求",
    "三、商务要求",
    "商务要求",
    "付款方式和条件",
    "合同履行期限",
)

# 规则 4：空白格式模板的标题词。
TEMPLATE_HEADS = (
    "技术规格偏离表",
    "响应偏离表",
    "开标一览表",
    "投标函",
    "投标文件封面",
)

# 行首的「标记 + 编号 + 分隔符 + 标记」，剥掉它才拿到语义正文。
# 两种标记位置都要覆盖：``▲2.`` / ``*1.2`` 在前，``1.7★`` 在后（B 族把 ★ 追加在序号单元格里）。
_LEAD_NUM = re.compile(
    r"""^\s*
        [★▲*]?                      # 前置标记
        \s*
        (?:\d+(?:[.．]\d+)*|[一二三四五六七八九十]+)?   # 编号（可能没有）
        \s*
        (?:、|[.．])?                # 分隔符
        \s*
        [★▲*]?                      # 后置标记
        \s*""",
    re.X,
)

# 裁剪原因码（测试断言用，稳定字符串）。
R_SERVICE_TAIL = "service_tail"
R_BOILERPLATE = "boilerplate"
R_TEMPLATE = "blank_template"
R_EMPTY = "empty"

# 分节路径里每段的序号前缀：``四、`` / ``2.`` / ``（3）`` / ``1、``。
# 序号是**结构**信息，不是设备名的一部分，拼进匹配文本前必须剥掉，
# 否则 ``1、4K 采集摄像头`` 会凭空给指标名多送一个 ``1、`` 二元组。
_SECTION_ORDINAL = re.compile(
    r"^(?:[一二三四五六七八九十百零]+|\d{1,3}|[（(]\d{1,3}[)）])\s*[、.．)）:：]\s*"
)

# 原因码 -> 人读说明（复核报告的「跳过清单」用）。
_WHY = {
    R_SERVICE_TAIL: "设备标的的售后服务段",
    R_BOILERPLATE: "页尾套话 / 商务条款",
    R_TEMPLATE: "投标文件格式章节里的空白表格模板",
    R_EMPTY: "正文为空",
}


def body_of(req: Requirement) -> str:
    """剥掉编号与标记，拿到用于**语义判断**的正文。

    A/B 族的 ``text`` 是逐字原文（含 ``▲2.`` 这样的编号），C 族已经剥过，
    所以这里对两种形态都要能工作——正则匹配不上就原样返回。
    """
    raw = (req.head_text or req.text or "").strip()
    m = _LEAD_NUM.match(raw)
    body = raw[m.end() :].strip() if m else raw
    return body or marks.strip_all(raw).strip()


def section_context(req: Requirement) -> str:
    """把分节路径整理成一段**设备限定语**，供匹配层补足正文缺失的主语。

    P3 的招标表里有大量 ``1、数量：1 台`` 这类正文完全不带设备名的行，
    设备名只写在分节标题（``四、配套LED冷光源``）里。不把它拼进匹配文本，
    一条 ``电刀数量：1台`` 就会和同标的下所有「数量：1 台」打成平手。

    三条规则：

    * **保留完整路径**，不只取最深一段。最深一层有时比父级更窄
        （``二、配套4K 采集摄像头，4K 腹腔镜内窥镜。 / 1、4K 采集摄像头``），
        只取末段会漏掉 ``4K 腹腔镜`` 这个关键限定语。
    * **剥掉每段的序号前缀**，并去掉结尾的 ``：``——后者会和要求正文自己的
        冒号粘成一个假二元组。
    * **去重且保序**：路径里同名段重复出现（``4K 采集摄像头`` 出现两次）时只留一次，
        免得二元组集合被同一串字放大。

    分节为空时返回 ``""``，调用方据此退化成「只用正文」，P1/P2/P4 全走这条路。
    """
    if not req.section:
        return ""
    parts = []
    for seg in req.section.split(SEP):
        seg = _SECTION_ORDINAL.sub("", seg.strip()).rstrip("：:。 、").strip()
        if seg and seg not in parts:
            parts.append(seg)
    return " ".join(parts)


def is_service_trigger(req: Requirement) -> bool:
    """这一条是不是「售后服务」段的开头。"""
    return body_of(req).startswith(SERVICE_HEAD)


def single_reason(req: Requirement) -> str | None:
    """规则 3/4 的逐条判断。命中返回原因码，否则 ``None``。"""
    body = body_of(req)
    if not body:
        return R_EMPTY
    for marker in BOILERPLATE_STARTS:
        if body.startswith(marker):
            return R_BOILERPLATE
    if req.is_heading and any(h in body for h in TEMPLATE_HEADS):
        return R_TEMPLATE
    return None


@dataclass
class Cut:
    """裁剪结果。"""

    kept: list[Requirement] = field(default_factory=list)
    dropped: list[tuple[Requirement, str]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)  # 人读的跳过清单
    warnings: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for _, code in self.dropped:
            out[code] = out.get(code, 0) + 1
        return out


def _describe(run: list[Requirement]) -> str:
    """把一段被裁的条目描述成 ``21、售后服务要求 + 21.1~21.7（共 8 条）``。"""
    first = run[0].no or body_of(run[0])[:12]
    if len(run) == 1:
        return f"{first}（共 1 条）"
    return f"{first} + {run[1].no}~{run[-1].no}（共 {len(run)} 条）"


def apply(reqs: list[Requirement], scopes: list[Scope]) -> Cut:
    """主入口：按 ``00_投标范围.txt`` 的标的属性做范围裁剪。"""
    cut = Cut()
    service_lots = {sc.lot_no for sc in scopes if sc.is_service}
    cutting_lot: str | None = None  # 正在被顺序裁剪的标的
    run: list[Requirement] = []  # 当前标的被售后服务段裁掉的条目

    def flush() -> None:
        nonlocal run
        if run:
            cut.reasons.append(
                f"标的{run[0].lot_no} 的售后服务段 {_describe(run)} 已裁剪："
                f"设备标的的售后服务要求不在标准答案范围内（agent.md 3.7 第 1 条）。"
            )
            run = []

    for req in reqs:
        if cutting_lot is not None and req.lot_no != cutting_lot:
            flush()
            cutting_lot = None

        if cutting_lot == req.lot_no:
            cut.dropped.append((req, R_SERVICE_TAIL))
            run.append(req)
            continue

        if req.lot_no not in service_lots and is_service_trigger(req):
            cutting_lot = req.lot_no
            cut.dropped.append((req, R_SERVICE_TAIL))
            run.append(req)
            continue

        reason = single_reason(req)
        if reason is not None:
            cut.dropped.append((req, reason))
            cut.reasons.append(
                f"标的{req.lot_no} 序号{req.no or '(无编号)'} 已裁剪：{_WHY[reason]}。"
            )
            continue

        cut.kept.append(req)

    flush()
    if cutting_lot is not None and cutting_lot in service_lots:  # pragma: no cover
        cut.warnings.append(f"标的{cutting_lot} 是服务标的，售后服务段未被裁剪（符合预期）。")
    return cut
