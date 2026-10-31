"""招标 PDF 抽取的**统一入口**：按版面族分派到 A/B/C 三个实现。

分派表刻意做得很薄——三个版面模块的签名完全一致
（``extract(t, scopes, *, paren_policy) -> (list[Requirement], list[str])``），
所以这里只负责选一个、并在选不出来时走降级路径。

``UNKNOWN`` 版面的降级策略（agent.md 4.5）：**不抛异常**，改用
:func:`tender_pdf.extract_flat_requirements` 把整份 PDF 里像参数的行捞出来，
全部挂到第一个设备标的下，并给出**响亮**的告警 + 把置信度打到很低，
让复核报告第一眼就能看到「这份招标文件没被真正解析」。
宁可交一份覆盖率低但结构合法的表，也不要在考核现场崩掉。
"""

from __future__ import annotations

from ..config import LAYOUT_A, LAYOUT_B, LAYOUT_C, LAYOUT_UNKNOWN
from ..models import Requirement, Scope
from . import numbering, tender_layout_a, tender_layout_b, tender_layout_c, tender_pdf

_DISPATCH = {
    LAYOUT_A: tender_layout_a.extract,
    LAYOUT_B: tender_layout_b.extract,
    LAYOUT_C: tender_layout_c.extract,
}


def extract(
    t: tender_pdf.Tender,
    scopes: list[Scope],
    *,
    paren_policy: str = "auto",
) -> tuple[list[Requirement], list[str]]:
    """按 :attr:`tender_pdf.Tender.layout` 分派抽取。"""
    fn = _DISPATCH.get(t.layout)
    if fn is not None:
        return fn(t, scopes, paren_policy=paren_policy)
    return _extract_unknown(t, scopes)


def _extract_unknown(
    t: tender_pdf.Tender,
    scopes: list[Scope],
) -> tuple[list[Requirement], list[str]]:
    """UNKNOWN 版面的降级路径。"""
    devices = [sc for sc in scopes if not sc.is_service]
    scope = devices[0] if devices else (scopes[0] if scopes else None)
    lot_no = scope.lot_no if scope else "1"
    device_name = scope.device_name if scope else ""
    warnings = [
        f"【版面告警】招标 PDF「{t.path.name}」不属于已支持的 A/B/C 三种版面族，"
        f"已降级为「全文捞参数行」模式：全部条目都挂到标的{lot_no}"
        f"（{device_name or '设备名未知'}）下，分节留空，置信度按最低档处理。"
        "覆盖率会明显下降，请人工核对招标文件的参数表结构。"
    ]
    # 降级路径返回的是 ``(整行原文, 页码)``：编号与标记在这一层现解，
    # 原文列保留整行（含编号），因为没有版面信息可以安全地剥离它。
    reqs: list[Requirement] = []
    for text, page in tender_pdf.extract_flat_requirements(t):
        n = numbering.parse_line(text)
        reqs.append(
            Requirement(
                lot_no=lot_no,
                device_name=device_name,
                section="",
                no=n.no if n else "",
                mark=n.mark if n else "",
                text=text,
                page=page,
                is_heading=False,
                layout=LAYOUT_UNKNOWN,
                head_text=text,
            )
        )
    if not reqs:
        warnings.append("降级模式也没有捞到任何像参数的行，输出将为空表。")
    return reqs, warnings
