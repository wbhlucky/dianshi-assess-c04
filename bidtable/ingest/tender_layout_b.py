"""B 族版面：``五、技术要求`` 的 ``序号｜技术要求`` 双列表（P4）。

B 族与 A 族的三处结构差异，每一处都对应一个必须单独处理的边界：

1. **没有竖线**（P4 实测 ``vs == 0``）。列分割线只能从横线推：表体横线的
   ``x0`` 就是分割线（实测 ``x=123.2..534.4``），``序号`` 词在左侧
   （``x0∈[69.2,102.2]``、``x1<=111.2``），``技术要求`` 词在右侧（``x0=128.4``）。
   所以左列取 ``[页左边距, 横线x0]``，右列取 ``[横线x0, 横线x1]``。
2. **序号在单元格里垂直居中**，正文从行带顶部开始（P4 ``1.9.3``：序号 ``y=109.0``，
   正文两行 ``y=98.0`` / ``y=121.0``，同属行带 ``(95.4,142.6)``）。
   ⇒ 必须按**行带**裁切取词，不能用 ``get_text`` 的线性行。
3. **★ 追加在序号单元格里**（``1.7★`` / ``1.9.1★`` / ``1.16★``），
   而 A 族的标记在正文行首（``▲2.最大单脉冲能量5J。``）。
   :func:`marks.strip_trailing_mark` 负责这一种，:func:`numbering.parse_item` 两种都认。

**同页异表的排除**靠几何而不是文字：P4 p25 下半是 ``六、配置要求`` 的另一张表
（``序号|配置内容|数量``），它的横线 x 区间是 ``118.7..325.0``，与技术要求表的
``123.2..534.4`` 不同，:func:`_table_geom` 只收「与主区间端点吻合」的横线，
于是那张表**根本不会进入行带列表**。页眉单线（``y=41.0, x=61.6..531.7``）同理被剔除。
序号递增闸门是第二道保险（``六、配置要求`` / ``1 热蒸汽泊疗设备 1 台`` 都不构成合理递增）。

**裸整数行是分组标题**：``1 主机系统要求`` / ``2 热蒸汽治疗系统专用软件要求``
在 gold 里没有对应行。判别式沿用 agent.md 3.5 的确定性规则——存在 ``N.k`` 子项
即为标题（:func:`_mark_headings`），而不是「level==1 就是标题」，
这样未见包里真的用 ``1``/``2``/``3`` 编号的 B 族表不会被误吞。

B 族**从不产出 ``分节``**：gold P4 的 37 行 ``分节`` 全为空，分组标题只体现在序号前缀里。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import LAYOUT_B
from ..models import Requirement, Scope
from . import cellstitch as cs
from . import numbering, tender_pdf

IDX_NO, IDX_TEXT = 0, 1
NO_HEADER = "序号"
# 横线端点与主区间的容差（pt）。同一张表的线在 PDF 里会有 0.x pt 的抖动。
SPAN_TOL = 2.0
# ``一、设备名称及数量：热蒸汽治疗设备/壹台`` —— 没有投标范围文件时的设备名兜底。
_DEVICE_LINE = re.compile(r"设备名称及数量[：:]\s*([^/／\n]+)")


@dataclass
class RawRow:
    """技术要求表的一行（标题行也在内，靠 :attr:`is_heading` 区分）。"""

    page: int
    no: str
    mark: str
    text: str
    no_cell: str = ""
    is_heading: bool = False


def _table_geom(page) -> tuple[float, float, list[float]] | None:
    """恢复 B 族表格几何：``(列分割x, 表右边界x, 行分界y列表)``。

    返回 ``None`` 表示这一页没有可识别的表体（调用方跳过该页）。
    """
    hs, _ = cs.page_rules(page)
    span = cs.dominant_hspan(page, hs)
    if span is None:
        return None
    x0, x1 = span
    height = page.rect.height
    ys = sorted(
        {
            round(h.y, 1)
            for h in hs
            if h.span > cs.MIN_H_SPAN
            and abs(h.x0 - x0) <= SPAN_TOL
            and abs(h.x1 - x1) <= SPAN_TOL
            and cs.EDGE_MARGIN < h.y < height - cs.EDGE_MARGIN
        }
    )
    if len(ys) < 2:
        return None
    return x0, x1, ys


def page_bands(page, page_no: int) -> list[cs.BandCells]:
    """把一页切成 ``[序号 | 技术要求]`` 两列的行带。"""
    geom = _table_geom(page)
    if geom is None:
        return []
    x0, x1, ys = geom
    # 左列一直伸到页左边距：B 族的序号词比横线 x0 小约 50pt，
    # 用页边距而不是 magic number，横排/竖排、不同页宽的未见包都成立。
    cols = [int(page.rect.x0), int(round(x0)), int(round(x1) + 1)]
    return cs.band_cells(page, cols, ys, page_no, sep="\n")


def extract_rows(t: tender_pdf.Tender) -> tuple[list[RawRow], list[str]]:
    """扫过 B 族的参数页，产出原始行 + 告警。"""
    seq = numbering.Sequencer()
    rows: list[RawRow] = []
    warnings: list[str] = []
    started = False

    for page_no in t.param_pages:
        bands = page_bands(t.page(page_no), page_no)
        if not bands:
            continue
        for band in bands:
            no_cell = cs.join_lines(band.get(IDX_NO)).strip()
            text = cs.join_lines(band.get(IDX_TEXT)).strip()
            if not no_cell and not text:
                continue
            n = numbering.parse_item(no_cell) if no_cell else None
            if n is None:
                if not no_cell and rows and text:
                    rows[-1].text += text  # 序号列为空的续带，直接拼接（不补空格）
                    continue
                if started:
                    warnings.append(
                        f"招标 PDF 第{page_no}页 序号「{no_cell or '(空)'}」不构成编号，"
                        f"判定技术要求表到此结束，其后内容属于其它章节，已排除。"
                    )
                    return _mark_headings(rows), warnings
                if no_cell and no_cell != NO_HEADER:
                    warnings.append(
                        f"招标 PDF 第{page_no}页 技术要求表之前出现了无法识别的序号「{no_cell}」，已跳过。"
                    )
                continue
            if not seq.plausible(n):
                warnings.append(
                    f"招标 PDF 第{page_no}页 序号「{no_cell}」不构成合理递增（上一条为 "
                    f"{seq.last_no or '无'}），"
                    f"判定技术要求表到此结束，其后内容属于其它章节，已排除。"
                )
                return _mark_headings(rows), warnings
            seq.advance(n)
            started = True
            rows.append(
                RawRow(page=page_no, no=n.no, mark=n.mark, text=text, no_cell=no_cell)
            )

    if not rows:
        warnings.append(
            "B 族「五、技术要求」表里没有识别出任何行（序号列未能解析出递增编号）。"
        )
    return _mark_headings(rows), warnings


def _mark_headings(rows: list[RawRow]) -> list[RawRow]:
    """``N`` 是分组标题 **当且仅当** 后面存在 ``N.k`` 子项（agent.md 3.5）。"""
    for i, r in enumerate(rows):
        if len(numbering.int_parts(r.no)) > 1:
            continue
        prefix = r.no + "."
        r.is_heading = any(o.no.startswith(prefix) for o in rows[i + 1 :])
    return rows


def device_name_fallback(t: tender_pdf.Tender) -> str:
    """没有 ``00_投标范围.txt`` 时，从 ``一、设备名称及数量：X/壹台`` 兜底取设备名。"""
    for page_no in range(max(1, t.chapter_page - 1), min(t.npages, t.chapter_page + 2) + 1):
        m = _DEVICE_LINE.search(t.text(page_no))
        if m:
            return m.group(1).strip()
    return ""


def pick_lot(scopes: list[Scope]) -> tuple[str, str, list[str]]:
    """B 族整张技术要求表只对应**一个**设备标的，返回 ``(标的号, 设备名, 告警)``。

    B 族的表结构里没有标的列（``序号`` 是 ``N.k`` 参数号，不是标的号），
    所以标的号只能来自 ``00_投标范围.txt``。多个设备标的时无法从表里区分，
    取第一个并给出显眼告警（宁可信 scope 也不静默猜）。
    """
    warnings: list[str] = []
    devices = [sc for sc in scopes if not sc.is_service]
    if not devices:
        warnings.append(
            "B 族招标文件的「五、技术要求」表没有标的列，且投标范围里未声明设备标的，"
            "已按标的 1 输出，请人工确认标的号与设备名称。"
        )
        return "1", "", warnings
    if len(devices) > 1:
        warnings.append(
            "B 族招标文件的「五、技术要求」表把全部参数写在一张表里，无法按标的拆分，"
            f"已把全部 {len(devices)} 个设备标的的要求都归到标的{devices[0].lot_no}"
            f"（{devices[0].device_name}），请人工复核。"
        )
    return devices[0].lot_no, devices[0].device_name, warnings


def extract(
    t: tender_pdf.Tender,
    scopes: list[Scope],
    *,
    paren_policy: str = "auto",  # noqa: ARG001 —— 与 A/C 族保持同一签名
) -> tuple[list[Requirement], list[str]]:
    """B 族主入口：产出 :class:`Requirement` 列表（``分节`` 恒为空）。"""
    lot_no, device_name, warnings = pick_lot(scopes)
    if not device_name:
        device_name = device_name_fallback(t)
    rows, w = extract_rows(t)
    warnings.extend(w)
    out: list[Requirement] = []
    for r in rows:
        out.append(
            Requirement(
                lot_no=lot_no,
                device_name=device_name,
                section="",
                no=r.no,
                mark=r.mark,
                text=r.text,
                page=r.page,
                is_heading=r.is_heading,
                layout=LAYOUT_B,
                head_text=r.text,
            )
        )
    return out, warnings
