"""C 族版面：``N-科室-设备名技术参数`` 多块结构（P5，13 个块，只投其中 4 个）。

一个块的实测解剖（P5 p12，标的5）::

    5-耳鼻喉-红外辅照治疗仪技术参数      <- 块标题，``N`` 就是标的号
    采购人：新郑市公立人民医院
    年度计划序号                        <- 表头标签（有的块写「参数号」）
    设备名称
    红外辐照治疗装置                    <- **权威设备名**（块标题里是「红外辅照治疗仪」）
    质量层次
    原装进口
    数量
    1 台
    设备配置要求及用途：该设备主要用于…
    具体技术参数：                      <- 参数区起点
    1、治疗源
    1.1 治疗光源：卤素光源
    …
    产品配置清单：                      <- 参数区终点（其后是另一张配置表）

四条必须单独处理的边界：

1. **块标题的设备名不可信**（X5）：标的5 标题写 ``红外辅照治疗仪``，
   表头写 ``红外辐照治疗装置``，gold 用后者。优先级：投标范围 > 块内表头 > 块标题。
2. **块标题写法不统一**：``…技术参数`` / ``…参数`` / ``…高清关节镜``（没有「参数」二字）。
   所以 :data:`tender_pdf.C_BLOCK_RE` 刻意放宽，这里只按 ``Tender.blocks`` 的页序切块。
3. **参数区起点写法不统一**：``具体技术参数：``（标的1/5/8/12）与
   ``一、具体技术参数``（标的2）。:data:`PARAMS_RE` 允许前置中文序数。
4. **编号可能独占一行**（标的8 的 ``1.``~``9.``，正文在下一行）。
   :func:`numbering.split_items` 天然处理：``1.`` 的 ``rest`` 为空，下一行解析不出编号
   就成为它的续行；:func:`_body_text` 再把两段拼起来。

**跳号是真实的**：标的1 的源文件从 ``15、`` 直接跳到 ``17、``（16 不存在），
gold 也只有 18 行（1–15、17、19、20）。⇒ 绝不合成缺失序号。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import LAYOUT_C
from ..models import Requirement, Scope
from . import numbering, tender_pdf
from .tender_layout_a import paren_mode

# 参数区起点：``具体技术参数：`` / ``一、具体技术参数``。
PARAMS_RE = re.compile(r"^\s*(?:[一二三四五六七八九十]+\s*、\s*)?具体技术参数\s*[：:]?\s*$")
# 参数区终点：``产品配置清单：`` 及其后的配置表表头。
CONFIG_HEAD = "产品配置清单"
# 块内表头标签（它们本身不是参数，必须在 body 之前被跳过）。
DEVICE_LABEL = "设备名称"
_HEADER_LABELS = ("采购人", "参数号", "年度计划序号", "质量层次", "数量", DEVICE_LABEL)
_PURPOSE_PREFIX = "设备配置要求及用途"


@dataclass
class Block:
    """一个 ``N-科室-设备`` 参数块。"""

    lot_no: str
    dept: str
    title: str
    title_device: str  # 块标题里的设备名（**不可信**，仅兜底）
    header_device: str = ""  # 块内「设备名称」表头的值（权威）
    page: int = 0
    lines: list[tuple[str, int]] = field(default_factory=list)  # 参数区行 (文本, 页码)
    quantity: str = ""
    has_params: bool = False  # 是否找到了明确的「具体技术参数」起点


def _iter_lines(t: tender_pdf.Tender, start: int, stop: int):
    """逐行产出 ``(文本, 页码)``，页码 1 起，闭区间 ``[start, stop]``。"""
    for pg in range(max(1, start), min(t.npages, stop) + 1):
        for ln in t.text(pg).split("\n"):
            yield ln, pg


def _find_title(lines: list[tuple[str, int]], title: str, cursor: int) -> int:
    """从 ``cursor`` 起找块标题行的下标；找不到返回 ``-1``。"""
    want = title.strip()
    for i in range(cursor, len(lines)):
        if lines[i][0].strip() == want:
            return i
    return -1


def _body_window(lines: list[tuple[str, int]]) -> tuple[int, int, bool]:
    """定位参数区 ``[start, end)``。返回 ``(start, end, 是否找到明确起点)``。

    起点判据按可靠度递降：``具体技术参数`` > ``设备配置要求及用途`` > 最后一个表头标签。
    终点判据：第一行以 ``产品配置清单`` 开头（含 ``产品配置清单：`` 与配置表表头）。
    """
    start, found = 0, False
    for i, (text, _) in enumerate(lines):
        s = text.strip()
        if PARAMS_RE.match(s):
            return i + 1, _body_end(lines, i + 1), True
        if s.startswith(_PURPOSE_PREFIX):
            start, found = i + 1, True
        elif s in _HEADER_LABELS or (s.startswith("采购人") and "：" in s):
            start, found = i + 1, found
    return start, _body_end(lines, start), found


def _body_end(lines: list[tuple[str, int]], start: int) -> int:
    for i in range(start, len(lines)):
        if lines[i][0].strip().startswith(CONFIG_HEAD):
            return i
    return len(lines)


def _header_device(lines: list[tuple[str, int]], body_start: int) -> tuple[str, str]:
    """从块内表头取 ``(设备名称, 数量)``：标签行的下一非空行就是值。"""
    device = qty = ""
    for i in range(min(body_start, len(lines))):
        s = lines[i][0].strip()
        if s == DEVICE_LABEL and not device:
            device = _next_value(lines, i + 1, body_start)
        elif s == "数量" and not qty:
            qty = _next_value(lines, i + 1, body_start)
    return device, qty


def _next_value(lines: list[tuple[str, int]], i: int, stop: int) -> str:
    for j in range(i, min(stop, len(lines))):
        s = lines[j][0].strip()
        if s and s not in _HEADER_LABELS:
            return s
    return ""


# 最后一个块之后还要多看几页：块可以跨页，而它没有「下一个块标题」来收口。
TAIL_PAGES = 6


def split_blocks(t: tender_pdf.Tender) -> tuple[list[Block], list[str]]:
    """把 C 族正文切成块。块的边界是**下一个块标题**，块内再按参数区窗口收窄。

    同一个标的号出现两次时（目录里也印了块标题的未见包），保留**有「具体技术参数」
    起点**的那一个：目录行后面跟的是页码而不是参数，窗口里必然找不到起点。
    """
    warnings: list[str] = []
    titles = t.blocks
    if not titles:
        warnings.append("C 族招标文件里没有找到任何「N-科室-设备名」块标题，抽取将返回空结果。")
        return [], warnings

    first = min(b.page for b in titles)
    last = max(b.page for b in titles)
    lines = list(_iter_lines(t, first, min(t.npages, last + TAIL_PAGES)))

    best: dict[str, Block] = {}
    order: list[str] = []
    cursor = 0
    for idx, bt in enumerate(titles):
        i = _find_title(lines, bt.title, cursor)
        if i < 0:
            warnings.append(f"招标 PDF 第{bt.page}页 的块标题「{bt.title}」在正文里找不到，已跳过该块。")
            continue
        nxt = len(lines)
        if idx + 1 < len(titles):
            j = _find_title(lines, titles[idx + 1].title, i + 1)
            if j > i:
                nxt = j
        raw = lines[i + 1 : nxt]
        start, end, found = _body_window(raw)
        device, qty = _header_device(raw, start)
        block = Block(
            lot_no=bt.lot_no,
            dept=bt.dept,
            title=bt.title,
            title_device=bt.device_name,
            header_device=device,
            page=bt.page,
            lines=raw[start:end],
            quantity=qty,
            has_params=found,
        )
        prev = best.get(bt.lot_no)
        if prev is None:
            best[bt.lot_no] = block
            order.append(bt.lot_no)
        else:
            warnings.append(
                f"标的{bt.lot_no} 的块标题在招标 PDF 里出现了多次（第{prev.page}页与第{bt.page}页），"
                f"已保留含「具体技术参数」的那一处。"
            )
            if not prev.has_params and block.has_params:
                best[bt.lot_no] = block
        cursor = i + 1

    out = [best[k] for k in order]
    for block in out:
        if not block.has_params:
            warnings.append(
                f"标的{block.lot_no}（{block.title}）里没有找到「具体技术参数」起点，"
                f"已改用块内表头之后的全部行，请人工复核是否混入了配置清单内容。"
            )
    return out, warnings


def device_name(block: Block, scope: Scope | None) -> str:
    """设备名优先级：投标范围 > 块内表头 > 块标题（X5：块标题可能是错别字）。"""
    if scope is not None and scope.device_name:
        return scope.device_name
    return block.header_device or block.title_device


def _body_text(item: numbering.Item) -> str:
    """C 族的 ``招标要求（原文）``：编号与标记**不进原文**（gold 实测如此）。

    编号独占一行时（标的8 的 ``1.``）``Numbered.text`` 为空，正文全在续行里，
    所以取 ``numbered.text`` + 续行拼接，而不是 :attr:`Item.text`（后者含编号与标记）。
    """
    n = item.numbered
    if n is None:
        return item.text  # 无编号的孤儿续行，原样保留
    return n.text + "".join(item.lines[1:])


def extract(
    t: tender_pdf.Tender,
    scopes: list[Scope],
    *,
    paren_policy: str = "auto",
) -> tuple[list[Requirement], list[str]]:
    """C 族主入口。**只输出投标范围声明过的标的**（不变式 I12）。"""
    blocks, warnings = split_blocks(t)
    by_no = {sc.lot_no: sc for sc in scopes}
    declared = set(by_no)
    out: list[Requirement] = []

    for block in blocks:
        if declared and block.lot_no not in declared:
            continue  # 未投标的标的：不输出（I12）
        scope = by_no.get(block.lot_no)
        if scope is None and declared:
            warnings.append(
                f"招标 PDF 里的标的{block.lot_no}（{block.title}）未在投标范围中声明，已跳过。"
            )
            continue
        if not block.lines:
            warnings.append(f"标的{block.lot_no}（{block.title}）的参数区是空的，未产出任何要求。")
            continue
        dev = device_name(block, scope)
        items = numbering.split_items(
            block.lines,
            paren=paren_mode(paren_policy, bool(scope and scope.is_service)),
        )
        headings = numbering.mark_headings(items)
        for it, heading in zip(items, headings):
            n = it.numbered
            if n is None:
                warnings.append(
                    f"标的{block.lot_no} 第{it.page}页 有一段无法归属到任何编号的文本，已并入前一条目。"
                )
                continue
            out.append(
                Requirement(
                    lot_no=block.lot_no,
                    device_name=dev,
                    section="",
                    no=n.no,
                    mark=n.mark,
                    text=_body_text(it),
                    page=it.page,
                    is_heading=heading or n.kind == numbering.KIND_LEAD,
                    layout=LAYOUT_C,
                    head_text=it.text,
                )
            )

    if scopes and not out:
        warnings.append(
            "C 族抽取没有产出任何要求：招标 PDF 的块标题标的号与投标范围声明的标的号完全对不上，"
            f"（PDF: {sorted({b.lot_no for b in blocks})} / 投标范围: {sorted(declared)}）。"
        )
    return out, warnings
