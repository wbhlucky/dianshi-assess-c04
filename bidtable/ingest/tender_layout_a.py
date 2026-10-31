"""A 族版面：``二、采购清单`` 的 6 列表，参数全挤在一个跨页大单元格里（P1/P2/P3）。

表格列：``序号｜货物名称｜技术规格及主要参数｜单位｜数量｜行业``。

行带（band）的四类形态，实测自 P1 p8–p10 / P3 p8–p14：

============= ===================================== =========================
形态           实测                                   处理
============= ===================================== =========================
表头带         ``c0='序号' c1='货物名称'``            跳过
横幅带         ``c0='适用本' c1='本国产品标准'``       跳过（分组变更，不是新标的）
              ``c0='其他'``
标的带         ``c0='1' c1='钬激光治/疗机'``          新标的；设备名 = c1 各行拼接
续页带         ``c0='' c1='' c2='（1）分辨率≥…'``     并入上一个标的的参数文本
============= ===================================== =========================

**横幅带的坑**：它是合并单元格，文字被列切分线切成三段
（``适用本`` + ``本国产品标准`` + ``准的货物``），逐列读会得到乱码。
所以这里**不依赖横幅文字**判断分组：「新标的」的判据只有一条——
``c0`` 是纯整数。服务标的由 ``00_投标范围.txt`` 的 ``is_service`` 决定，
横幅带（``c0`` 既非整数也非空）自然落入「跳过」。

**跨页拼接**：续页只有 ``x >= 参数列起点`` 的词，``c0``/``c1`` 天然为空，
因此按行带顺序处理即可完成拼接，不需要额外的状态机。
"""

from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass, field

from ..config import LAYOUT_A
from ..models import Requirement, Scope
from . import cellstitch as cs
from . import numbering, tender_pdf
from .numbering import KIND_PAREN, PAREN_FOLD, PAREN_SPLIT

IDX_NO, IDX_NAME, IDX_PARAM = 0, 1, 2
# 至少要能切出「序号 / 货物名称 / 技术规格及主要参数 / 单位」四列才认为是采购清单表。
MIN_COLS = 4
# 参数区之后必须裁掉的页尾套话与下一节标题（agent.md 3.7 第 3、4 条）。
CUT_MARKERS = (
    tender_pdf.A_TAIL_BOILERPLATE,
    "本项目核心产品为",
    tender_pdf.A_NEXT_SECTION,
)


@dataclass
class LotBlock:
    """采购清单里的一行（= 一个标的）及其全部参数行。"""

    index_no: str  # 表格 ``序号`` 列的原值
    device_name: str  # ``货物名称`` 列，已去换行
    lines: list[tuple[str, int]] = field(default_factory=list)  # (行文本, 页码)
    start_page: int = 0
    end_page: int = 0

    @property
    def raw_param(self) -> str:
        return "".join(text for text, _ in self.lines)


def _page_bands(page, page_no: int) -> tuple[list[int], list[cs.BandCells]]:
    """在一页上恢复列边界与行带，并把行带裁剪到表格的 y 范围内。"""
    hs, vs = cs.page_rules(page)
    cols = cs.column_edges(page, vs)
    if len(cols) < MIN_COLS + 1:
        return cols, []
    long_vs = [v for v in vs if v.span > cs.MIN_V_SPAN]
    band_ys = cs.row_bands(page, hs)
    if long_vs:
        y_lo = min(v.y0 for v in long_vs)
        y_hi = max(v.y1 for v in long_vs)
        band_ys = [y for y in band_ys if y_lo - 2.0 <= y <= y_hi + 2.0]
    return cols, cs.band_cells(page, cols, band_ys, page_no, sep="\n")


def _cell(band: cs.BandCells, idx: int) -> str:
    return band.get(idx).strip()


def _param_lines(band: cs.BandCells, page_no: int) -> list[tuple[str, int]]:
    """参数列按 ``\\n`` 拆回行，并附带页码（供 :func:`numbering.split_items` 用）。"""
    raw = band.get(IDX_PARAM)
    return [(ln, page_no) for ln in raw.split("\n")]


def _cut_at_marker(lines: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], bool]:
    """遇到页尾套话 / ``三、商务要求`` 就截断。返回 ``(保留行, 是否已截断)``。"""
    out: list[tuple[str, int]] = []
    for text, page in lines:
        if any(marker in text for marker in CUT_MARKERS):
            return out, True
        out.append((text, page))
    return out, False


def extract_blocks(t: tender_pdf.Tender) -> tuple[list[LotBlock], list[str]]:
    """扫过 A 族的参数页，产出每个标的的参数行。"""
    warnings: list[str] = []
    blocks: list[LotBlock] = []
    current: LotBlock | None = None
    cut = False

    for page_no in t.param_pages:
        if cut:
            break
        cols, bands = _page_bands(t.page(page_no), page_no)
        if not bands:
            continue
        for band in bands:
            c0 = _cell(band, IDX_NO)
            c2 = _cell(band, IDX_PARAM)
            if not c0 and not c2:
                continue  # 空带
            if c0.isdigit():
                name = cs.join_lines(band.get(IDX_NAME)).strip()
                current = LotBlock(
                    index_no=c0,
                    device_name=name,
                    start_page=page_no,
                    end_page=page_no,
                )
                blocks.append(current)
            elif c0:
                continue  # 表头带（``序号``）或横幅带（``适用本`` / ``其他``）
            elif current is None:
                warnings.append(
                    f"招标 PDF 第{page_no}页出现了没有归属标的的参数文本（表格起始行未被识别），已跳过。"
                )
                continue
            if current is None or not c2:
                continue
            kept, hit = _cut_at_marker(_param_lines(band, page_no))
            current.lines.extend(kept)
            current.end_page = page_no
            if hit:
                cut = True
                break

    if not blocks:
        warnings.append("A 族采购清单里没有识别出任何标的行（``序号`` 列未出现纯整数）。")
    return blocks, warnings


def bind_lot(block: LotBlock, scopes: list[Scope]) -> tuple[str, str, list[str]]:
    """把采购清单的 ``序号``/``货物名称`` 绑定到 ``00_投标范围.txt`` 声明的标的。

    优先信 ``序号``（A 族里它就是标的号），再用设备名交叉校验；
    校验不上时以 scope 的设备名为准（scope 文件是权威来源，见 lotmap 的 X5 结论），
    并留一条告警供人工复核。
    """
    warnings: list[str] = []
    if not scopes:
        return block.index_no, block.device_name, warnings
    by_no = {sc.lot_no: sc for sc in scopes}
    sc = by_no.get(block.index_no)
    if sc is not None:
        if block.device_name and sc.device_name:
            a = sc.device_name.replace(" ", "")
            b = block.device_name.replace(" ", "")
            if a not in b and b not in a:
                warnings.append(
                    f"采购清单序号{block.index_no} 的货物名称「{block.device_name}」"
                    f"与投标范围声明的「{sc.device_name}」不一致，已按投标范围为准。"
                )
        return sc.lot_no, sc.device_name or block.device_name, warnings
    # 序号对不上：退回按设备名匹配
    for cand in scopes:
        if cand.device_name and cand.device_name.replace(" ", "") in block.device_name.replace(" ", ""):
            warnings.append(
                f"采购清单序号{block.index_no} 未在投标范围中声明，按货物名称匹配到标的{cand.lot_no}。"
            )
            return cand.lot_no, cand.device_name, warnings
    warnings.append(
        f"采购清单序号{block.index_no}（{block.device_name}）未在投标范围中声明，"
        f"仍按原序号输出，请人工确认该标的是否需要投标。"
    )
    return block.index_no, block.device_name, warnings


def paren_mode(policy: str, is_service: bool) -> str:
    """把 ``--paren-policy`` 解析成 :func:`numbering.split_items` 的模式。

    ``auto`` = 服务标的展开、设备标的并入（训练集实测：P2 服务标的的 gold 序号是
    ``2.1/2.2/2.3``，而 P1 设备标的的 ``（1）（2）（3）`` 在 gold 里并进序号14 一整条）。
    """
    if policy == "fold":
        return PAREN_FOLD
    if policy == "split":
        return PAREN_SPLIT
    return PAREN_SPLIT if is_service else PAREN_FOLD


def extract(
    t: tender_pdf.Tender,
    scopes: list[Scope],
    *,
    paren_policy: str = "auto",
) -> tuple[list[Requirement], list[str]]:
    """A 族主入口：产出 :class:`Requirement` 列表。

    这一步**只做抽取**，不做裁剪与分节：
    ``售后服务`` 裁剪在 :mod:`bidtable.ingest.scope`，分节在 :mod:`bidtable.ingest.section`。
    分层是为了让每一层都能被单独测试（agent.md 6.1 的「一个测试只考一个边界」）。

    两件在本层就必须定下来的事：

    1. ``（N）`` 策略取决于标的是不是服务标的（设备标的 fold、服务标的 split）。
       scope 信息只有这里拿得到，所以放在这里而不是 :mod:`numbering`。
    2. ``is_heading`` 用 :func:`numbering.mark_headings` 的确定性判别式打上；
       标题行**不写进 xlsx**（gold 里没有它们的对应行，写进去只会撑爆
       ``多余条目 <= 2`` 的门禁），只用来构造 ``分节`` 路径。
    """
    blocks, warnings = extract_blocks(t)
    out: list[Requirement] = []
    service_lots = {sc.lot_no for sc in scopes if sc.is_service}
    for block in blocks:
        lot_no, device_name, w = bind_lot(block, scopes)
        warnings.extend(w)
        items = numbering.split_items(
            block.lines,
            paren=paren_mode(paren_policy, lot_no in service_lots),
        )
        headings = numbering.mark_headings(items)
        parent_no = ""
        parent_at: dict[str, int] = {}
        paren_parents: set[int] = set()
        for it, heading in zip(items, headings):
            n = it.numbered
            if n is None:
                warnings.append(
                    f"标的{lot_no} 有一段无法归属到任何编号的文本（第{it.page}页），已并入前一条目。"
                )
                continue
            if n.kind == KIND_PAREN:
                # 服务标的：``（N）`` 展开成 ``父.N``（P2 标的2 的 2.1/2.2/2.3）。
                no = f"{parent_no}.{n.no}" if parent_no else n.no
                if parent_no in parent_at:
                    # 一旦 ``（N）`` 被展开成独立子项，父项就成了**纯标题**：
                    # gold P2 标的2 只有 2.1/2.2/2.3，没有「序号 2」这一行。
                    # 这里只打标记，要不要真的展开交给 fold.resolve 按厂商证据决定。
                    paren_parents.add(parent_at[parent_no])
            else:
                no = n.no
                if n.level == 1 and n.kind != numbering.KIND_LEAD:
                    parent_no = n.no
                    parent_at[no] = len(out)
            out.append(
                Requirement(
                    lot_no=lot_no,
                    device_name=device_name,
                    section="",
                    no=no,
                    mark=n.mark,
                    text=it.text,
                    page=it.page,
                    is_heading=heading,
                    layout=LAYOUT_A,
                    head_text=it.text,
                )
            )
        for idx in paren_parents:
            out[idx] = replace(out[idx], is_heading=True)
    return out, warnings
