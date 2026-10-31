"""招标 PDF：打开、章节定位、版面族探测、未知版面的降级抽取。

三个版面族（agent.md 3.6）与训练集的对应关系：

==== ====== ============================================ =================
族    包      章节路径                                    参数形态
==== ====== ============================================ =================
A     P1/P2/P3 ``第二章 项目需求`` -> ``二、采购清单``      6 列表，参数全挤在
                                                          ``技术规格及主要参数`` 一个跨页大单元格
B     P4      ``第八章 货物需求一览表及技术规格`` ->        独立 ``序号｜技术要求`` 双列表
              ``五、技术要求``
C     P5      ``N-科室-设备名技术参数`` 多块                行内编号项
==== ====== ============================================ =================

**目录页陷阱**：P1/P2/P3 的 p2 是目录，里面的 ``第二章项目需求`` 与正文章节标题
逐字相同。定位正文时一律要求页码 > 2，且必须同页出现表格列头 ``技术规格及主要参数``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from .. import errors
from ..config import LAYOUT_A, LAYOUT_B, LAYOUT_C, LAYOUT_UNKNOWN
from . import cellstitch as cs

# --- 各版面族的定位签名 ----------------------------------------------------
A_LIST_HEADING = "二、采购清单"
A_COL_HEADER = "技术规格及主要参数"
A_TAIL_BOILERPLATE = "本采购清单中所列技术规格及主要参数为最低要求"
A_NEXT_SECTION = "三、商务要求"
B_TECH_HEADING = re.compile(r"^\s*五\s*、\s*技术要求\s*$", re.M)
B_NEXT_SECTION = re.compile(r"^\s*六\s*、\s*配置要求", re.M)
B_LIST_HEADING = re.compile(r"^\s*四\s*、\s*采购清单", re.M)
# C 族块标题。写成容忍空格/全角连字符的形式，因为源文件里 ``2-…`` 那块带内部空格。
C_BLOCK_RE = re.compile(
    r"^\s*(?P<lot>\d{1,2})\s*[-—－–]\s*(?P<dept>.+?)\s*[-—－–]\s*(?P<name>.+?)\s*$",
    re.M,
)
C_PARAMS_HEADING = "具体技术参数"
C_CONFIG_HEADING = "产品配置清单"

# 目录页（含 ``....`` 点线或成串章节名）不参与正文定位
TOC_MAX_PAGE = 2
# 参数区最多向后扫多少页。训练集里 A 族最长 7 页（P3 p8–14）；
# 这个上限只是防走飞的保险丝，不是业务常量。
MAX_PARAM_PAGES = 40
# C 族块标题里「科室」字段的常见收尾字，给放宽后的正则加一道真伪闸门。
_DEPT_SUFFIXES = ("科", "中心", "室", "ICU", "部", "病区")


@dataclass(frozen=True)
class BlockTitle:
    """C 族一个 ``N-科室-设备名技术参数`` 块。"""

    lot_no: str
    dept: str
    device_name: str
    page: int  # 1 起
    title: str  # 逐字原文


@dataclass
class Tender:
    """招标 PDF 的定位结果。``doc`` 由调用方负责 :meth:`close`。"""

    path: Path
    doc: fitz.Document
    npages: int
    layout: str = LAYOUT_UNKNOWN
    chapter_page: int = 0
    param_pages: tuple[int, ...] = ()
    blocks: list[BlockTitle] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def close(self) -> None:
        try:
            self.doc.close()
        except Exception:  # pragma: no cover - fitz 偶发重复关闭
            pass

    def __enter__(self) -> "Tender":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def page(self, no: int) -> fitz.Page:
        """按 1 起页码取页。"""
        return self.doc[no - 1]

    def text(self, no: int) -> str:
        return self.doc[no - 1].get_text("text")


def open_pdf(path: str | Path) -> fitz.Document:
    """打开招标 PDF。失败一律 :class:`TenderParseError`（退出码 2，消息可行动）。"""
    p = Path(path)
    if not p.exists():
        raise errors.TenderParseError(f"招标 PDF 不存在：{p.resolve()}")
    try:
        doc = fitz.open(str(p))
    except Exception as exc:  # fitz 抛的是 C 层异常，类型不稳定
        raise errors.TenderParseError(
            f"招标 PDF 打不开：{p.name}\n原因：{exc}\n"
            f"请确认该文件未损坏、未加密，且是 PDF 格式（不是改了扩展名的图片）。"
        ) from exc
    if doc.needs_pass or doc.is_encrypted:
        doc.close()
        raise errors.TenderParseError(
            f"招标 PDF 已加密，无法读取：{p.name}\n请提供解密后的版本，或去掉打开密码后重跑。"
        )
    if doc.page_count == 0:
        doc.close()
        raise errors.TenderParseError(
            f"招标 PDF 是 0 页的空文件：{p.name}\n没有正文就无法抽取招标要求。"
        )
    return doc


def find_pages(doc: fitz.Document, needle, *, after: int = TOC_MAX_PAGE) -> list[int]:
    """返回文本命中 ``needle`` 的页码（1 起），跳过目录页。``needle`` 可为字符串或已编译正则。"""
    hits: list[int] = []
    for i in range(after, doc.page_count):
        text = doc[i].get_text("text")
        if isinstance(needle, str):
            if needle in text:
                hits.append(i + 1)
        elif needle.search(text):
            hits.append(i + 1)
    return hits


def find_blocks(doc: fitz.Document) -> list[BlockTitle]:
    """扫描 C 族块标题。同一页可能有多个块（训练集里没有，但未见包可能有）。

    源文件的块标题**并不统一**（P5 实测三种写法）：
    ``1-EICU-床旁血液净化系统技术参数``、``2-病理科-切片机参数``、``6-骨科-高清关节镜``。
    所以正则刻意放宽到 ``N-x-y``，再用 :func:`_looks_like_block` 做真伪闸门。
    """
    out: list[BlockTitle] = []
    for i in range(doc.page_count):
        text = doc[i].get_text("text")
        for m in C_BLOCK_RE.finditer(text):
            dept, name = m.group("dept").strip(), m.group("name").strip()
            if not _looks_like_block(dept, name):
                continue
            out.append(
                BlockTitle(
                    lot_no=m.group("lot"),
                    dept=dept,
                    device_name=name,
                    page=i + 1,
                    title=m.group(0).strip(),
                )
            )
    return out


def _looks_like_block(dept: str, name: str) -> bool:
    """``N-x-y`` 是不是一个真的参数块标题（挡掉 ``1.19 可选配弹性定量分析技术…`` 这类散文）。"""
    if len(dept) > 12 or len(name) > 40 or "\n" in dept or "\n" in name:
        return False
    if name.endswith(("参数", "技术参数")):
        return True
    return any(dept.endswith(s) for s in _DEPT_SUFFIXES)


def detect_layout(doc: fitz.Document) -> tuple[str, list[str]]:
    """探测版面族。返回 ``(layout, warnings)``。

    判定顺序 C -> A -> B：C 的块标题最独特；A 的列头文字最独特；
    B 是兜底（``序号｜技术要求`` 双列表 + ``四、采购清单``）。
    """
    warnings: list[str] = []
    blocks = find_blocks(doc)
    if len({b.lot_no for b in blocks}) >= 2:
        return LAYOUT_C, warnings
    if find_pages(doc, A_COL_HEADER):
        return LAYOUT_A, warnings
    if find_pages(doc, B_TECH_HEADING):
        return LAYOUT_B, warnings
    warnings.append(
        "未识别的招标 PDF 版面（既不是「采购清单大单元格」，也不是「序号|技术要求」双列表，"
        "也没有「N-科室-设备名技术参数」块）。已降级为「把每个检测到的编号行当作一条要求」，"
        "这些行全部标为低置信度，请务必人工复核。"
    )
    return LAYOUT_UNKNOWN, warnings


def locate(path: str | Path) -> Tender:
    """打开 + 探测 + 定位，一次到位。"""
    p = Path(path)
    doc = open_pdf(p)
    t = Tender(path=p, doc=doc, npages=doc.page_count)
    t.layout, warns = detect_layout(doc)
    t.warnings.extend(warns)
    if t.layout == LAYOUT_A:
        _locate_a(t)
    elif t.layout == LAYOUT_B:
        _locate_b(t)
    elif t.layout == LAYOUT_C:
        t.blocks = find_blocks(doc)
        t.param_pages = tuple(sorted({b.page for b in t.blocks}))
    return t


def _locate_a(t: Tender) -> None:
    """A 族：定位 ``二、采购清单`` 所在页，向后扫到表格结束。

    **含尾**：出现 ``三、商务要求`` / 页尾套话的那一页**本身仍可能有表格续页**
    （P1 p10、P3 p14 实测如此：表格底边在 y≈244 / y≈316，散文在其下方）。
    所以这一页要收进来，真正的切断由 :mod:`tender_layout_a` 在行带层面按 y 做。
    """
    starts = [pg for pg in find_pages(t.doc, A_COL_HEADER) if A_LIST_HEADING in t.text(pg)]
    if not starts:
        starts = find_pages(t.doc, A_COL_HEADER)
    if not starts:
        t.warnings.append("未在招标 PDF 中找到「二、采购清单」表格，A 族抽取将返回空结果。")
        return
    t.chapter_page = starts[0]
    t.param_pages = _scan_table_pages(t, t.chapter_page, (A_NEXT_SECTION, A_TAIL_BOILERPLATE))


def _locate_b(t: Tender) -> None:
    """B 族：定位 ``五、技术要求``。

    同样**含尾**：P4 p25 上半是 ``2.4``~``2.7`` 的收尾行，下半才是 ``六、配置要求``
    的另一张表。行带层面的停止判据（序号必须合理递增）负责区分两者。
    """
    starts = find_pages(t.doc, B_TECH_HEADING)
    if not starts:
        t.warnings.append("未在招标 PDF 中找到「五、技术要求」，B 族抽取将返回空结果。")
        return
    t.chapter_page = starts[0]
    t.param_pages = _scan_table_pages(t, t.chapter_page, (B_NEXT_SECTION,), need_vrule=False)


def _scan_table_pages(
    t: Tender,
    start: int,
    stop_markers: tuple,
    *,
    need_vrule: bool = True,
) -> tuple[int, ...]:
    """从 ``start`` 页起向后收表格页；命中停止标记的那一页收进来后结束。"""
    pages = [start]
    limit = min(t.npages, start + MAX_PARAM_PAGES)
    for pg in range(start + 1, limit + 1):
        if not _page_has_table(t.page(pg), need_vrule=need_vrule):
            break
        pages.append(pg)
        text = t.text(pg)
        if any(m.search(text) if hasattr(m, "search") else m in text for m in stop_markers):
            break
    return tuple(pages)


def _page_has_table(page: fitz.Page, *, need_vrule: bool = True) -> bool:
    """几何判据：这一页还有没有表体。

    要求至少 2 条长横线（构成一个行带）；A 族还要求至少一条贯穿竖线。
    B 族没有竖线，所以放宽。
    """
    hs, vs = cs.page_rules(page)
    long_h = [h for h in hs if h.span > cs.MIN_H_SPAN]
    if len(long_h) < 2:
        return False
    if need_vrule:
        return any(v.span > cs.MIN_V_SPAN for v in vs)
    return True


def extract_flat_requirements(t: Tender) -> list[tuple[str, int]]:
    """未知版面的**降级**抽取（agent.md 4.5 第 3 条）：把每个像编号项的行都当成一条要求。

    返回 ``(文本, 页码)`` 列表。调用方负责打低置信度旗标 + 显眼告警。
    这里刻意**不做**任何裁剪，因为未知版面上「哪些该裁」没有依据；
    宁可多输出（``多余条目`` 不计分），也不要静默丢行。
    """
    from . import numbering

    out: list[tuple[str, int]] = []
    seq = numbering.Sequencer()
    for pg in range(1, t.npages + 1):
        for line in t.text(pg).split("\n"):
            s = line.strip()
            if not s:
                continue
            n = numbering.parse_item(s)
            if n is None or not seq.plausible(n):
                continue
            seq.advance(n)
            out.append((s, pg))
    return out
