# -*- coding: utf-8 -*-
"""集成层：A 族版面（``二、采购清单`` 的 6 列跨页大单元格表，P1/P2/P3）。

本文件 5 条用例，各盯一个**只有把真招标 PDF 从头读一遍才会暴露**的能力，
与单元测试互不重复：``test_cellstitch.py`` 只考几何切片，
``test_tender_layout_a.py``（unit）只考行带分类，这里考的是
「定位 -> 拼页 -> 裁尾」这条链在真文件上的端到端行为。

1. T2  章节定位：目录页与正文页含同一个标题串，靠什么不落到目录上。
2. T3  跨页拼接：P3 标的1 的参数单元格横跨 6 页，拼完不丢不重。
3. B13 续页识别：续页只有参数列的词，``序号``/``货物名称`` 天然为空，
       因此并入上一个标的，而不是被当成一个新标的。
4. B15 表格框线之外的散文：P3 p14 页尾正文永远不成为要求行。
5. X16 商务要求：五个包里都不允许出现 ``三、商务要求`` 的行。

全部读 ``训练集/`` 下的真文件与冻结中间态，因此整文件打 ``integration``。
"""
from __future__ import annotations

import pytest

from bidtable.ingest import cellstitch as cs
from bidtable.ingest import pack as pk
from bidtable.ingest import tender_layout_a as ta
from bidtable.ingest import tender_pdf as tp

pytestmark = [pytest.mark.integration]

# P3 p14 上落在表格最后一道横线之下、因此不属于任何行带的散文。
P14_BELOW_FRAME_PROSE = (
    "本项目核心产品为",
    tp.A_TAIL_BOILERPLATE,
    tp.A_NEXT_SECTION,
)


@pytest.fixture(scope="module")
def tenders(repo_root):
    """打开 P1 / P3 的真招标 PDF 并完成章节定位，模块级复用同一个 fitz 句柄。"""
    out = {}
    for name in ("P1", "P3"):
        root = repo_root / "训练集" / name
        pdf = pk.discover(root).tender_pdf if root.is_dir() else None
        if pdf is None or not pdf.is_file():
            pytest.skip(f"缺少 {name} 的招标 PDF，A 族集成测试无法进行")
        out[name] = tp.locate(pdf)
    yield out
    for t in out.values():
        t.close()


@pytest.mark.ec("A1")
@pytest.mark.case("T2")
def test_t2_chapter_locate_lands_on_the_body_table_not_the_toc(tenders):
    """章节定位落在正文参数表，而不是第 2 页那份同名目录。

    边界在哪：``第二章项目需求`` 在 p2 的目录里**逐字**出现（``after=0`` 时命中
    ``[2, 8]``）。把它挡在外面的不是模糊匹配，而是两道硬闸门——
    ``find_pages`` 的默认 ``after=2`` 与 ``TOC_MAX_PAGE == 2``；
    再加一条「列头 ``技术规格及主要参数`` 必须同页出现」，p8 才被唯一钉死。

    参数页为什么是 ``(8, 9, 10)`` 而不是 agent.md §7.2 T2 写的 8–9：
    p10 上表格还在（7 条竖线、2 个行带），页尾套话与 ``三、商务要求`` 也在 p10。
    裁剪发生在 :func:`ta._cut_at_marker` 的**行**级而不是页级，
    所以 p10 必须先进参数页集合，随后由 ``CUT_MARKERS`` 把它的尾巴切掉。
    这是文档与实现的一处偏差，以实现为准。
    """
    t = tenders["P1"]
    assert (t.layout, t.npages, t.chapter_page) == ("A", 89, 8)
    assert t.param_pages == (8, 9, 10)
    assert t.warnings == []

    heading = "第二章项目需求"
    assert tp.TOC_MAX_PAGE == 2
    assert tp.find_pages(t.doc, heading, after=0) == [2, 8]
    assert tp.find_pages(t.doc, heading) == [8]
    assert tp.find_pages(t.doc, tp.A_LIST_HEADING) == [8]
    assert tp.find_pages(t.doc, tp.A_COL_HEADER) == [8, 10]
    assert tp.find_pages(t.doc, tp.A_NEXT_SECTION) == [10]

    # 参数页在 p10 收尾：p7 与 p11 都切不出足够的列，扫描自然停住。
    for pno in (7, 11):
        page = t.page(pno)
        _, vs = cs.page_rules(page)
        assert len(cs.column_edges(page, vs)) < ta.MIN_COLS + 1


@pytest.mark.ec("A1")
@pytest.mark.case("T3")
def test_t3_p3_cross_page_param_cell_is_stitched_without_loss(tenders, frozen):
    """P3 标的1 的参数单元格横跨 p8–p13 六页，拼完 84 条既不丢也不虚增。

    边界在哪：一个跨页大单元格被切成 6 段，任何一段漏读都会静默少一批要求；
    重复读同一段则会凭空多出行。所以这条用例同时钉住三件事——
    页数连续、每页行数固定、总条数守恒（``cut.json`` 的 ``skipped`` 是空清单）。

    agent.md §7.2 T3 说「67 条，无重复无丢失」，实测两处都要修正：
    tender 阶段是 **84** 条（标的1）+ 6 条（标的2）= 90，输出才是 71 条；
    「无重复」也不成立——``1、数量：1 台`` 在 p10/p11/p12 各出现一次，
    84 条里只有 82 个不同文本。它没被误当重复行删掉，是因为三条各自挂在
    不同的分节下（``四、配套LED 冷光源`` / ``六、气腹机`` /
    ``九、配置适合腹腔镜及开腹手术使用电刀：``），
    ``(标的, 分节, 序号)`` 这个对齐键仍然唯一。这才是「无丢失」的正确含义。
    """
    tender = frozen("P3", "tender")
    cut = frozen("P3", "cut")
    reqs = tender["reqs"]
    lot1 = [r for r in reqs if r["lot_no"] == "1"]
    lot2 = [r for r in reqs if r["lot_no"] == "2"]

    assert (len(reqs), len(lot1), len(lot2)) == (90, 84, 6)
    assert tender["layout"] == "A"
    assert tender["param_pages"] == [8, 9, 10, 11, 12, 13, 14]

    pages = sorted({r["page"] for r in lot1})
    assert pages == list(range(8, 14))
    per_page = {p: sum(1 for r in lot1 if r["page"] == p) for p in pages}
    assert per_page == {8: 6, 9: 17, 10: 20, 11: 18, 12: 16, 13: 7}
    assert sum(per_page.values()) == 84

    # 唯一一处文本重复，且它靠分节差异保住了键唯一性。
    texts = [r["text"] for r in lot1]
    assert {t for t in texts if texts.count(t) > 1} == {"1、数量：1 台"}
    assert len(set(texts)) == 82
    dup_rows = [r for r in cut["reqs"] if r["text"].strip() == "1、数量：1 台"]
    assert [(r["page"], r["section"]) for r in dup_rows] == [
        (10, "四、配套LED 冷光源"),
        (11, "六、气腹机"),
        (12, "九、配置适合腹腔镜及开腹手术使用电刀："),
    ]
    assert len({(r["lot_no"], r["section"], r["no"]) for r in dup_rows}) == 3

    # 守恒：折叠与裁剪一条都没丢，90 进 90 出，71 行由分节阶段的标题行策略决定。
    assert len(cut["reqs"]) == 90
    assert cut["skipped"] == [], "skipped 是清单本身，P3 一条都没主动裁掉"
    assert cut["counts"]["参与判定"] == 71

    blocks, warns = ta.extract_blocks(tenders["P3"])
    assert warns == []
    assert [(b.index_no, b.start_page, b.end_page) for b in blocks] == [("1", 8, 13), ("2", 13, 14)]
    assert sorted({p for _, p in blocks[0].lines}) == list(range(8, 14))


@pytest.mark.ec("A1")
@pytest.mark.case("B13")
def test_b13_a_single_column_page_is_a_continuation_not_a_new_lot(tenders):
    """续页只有 ``x >= 参数列左界`` 的词，因此被并进上一个标的，而不是新开一个。

    边界在哪：``extract_blocks`` 判「新标的」的唯一依据是 ``序号`` 列（``c0``）为纯整数。
    续页上 ``c0``/``c1`` 天然为空——那两列的单元格在上页就已经跨页合并完了，
    本页只留下参数列的文字。于是 ``c0`` 既不是整数、又不是非空串，
    代码一路走到 ``current.lines.extend(...)``，行被追加到同一个 :class:`LotBlock`。
    反过来，如果用「本页有没有词」或「本页有没有横线」判新标的，
    P3 标的1 会被劈成 6 个标的，输出行数直接爆掉。

    p8 与 p9 的对照就是这条用例的全部内容：p8 三个行带的 ``c0`` 依次是
    表头「序号」、横幅「适用本」、标的「1」；p9 只有一个行带，``c0``/``c1`` 全空。
    """
    t = tenders["P3"]

    cols8, bands8 = ta._page_bands(t.page(8), 8)
    assert len(bands8) == 3
    assert [ta._cell(b, ta.IDX_NO).replace("\n", "") for b in bands8] == ["序号", "适用本", "1"]
    assert ta._cell(bands8[2], ta.IDX_NAME).replace("\n", "") == "4K3D 腹腔镜摄像系统"

    param_x = cols8[ta.IDX_PARAM]
    for pno in (9, 10, 11, 12):
        page = t.page(pno)
        words = page.get_text("words")
        assert words, f"p{pno} 不该是空白页"
        # 续页的词全部落在参数列及其右侧，序号/货物名称列一个词都没有。
        assert min(w[0] for w in words) >= param_x, f"p{pno} 出现了参数列左侧的词"
        _, bands = ta._page_bands(page, pno)
        assert len(bands) == 1, f"p{pno} 应当只有一个续页行带"
        assert ta._cell(bands[0], ta.IDX_NO) == ""
        assert ta._cell(bands[0], ta.IDX_NAME) == ""
        assert ta._cell(bands[0], ta.IDX_PARAM)

    # 结论：整份 P3 只识别出 2 个标的，不是 7 个。
    blocks, warns = ta.extract_blocks(t)
    assert warns == []
    assert len(blocks) == 2
    assert blocks[0].index_no == "1"
    assert sorted({p for _, p in blocks[0].lines}) == list(range(8, 14))


@pytest.mark.ec("A1")
@pytest.mark.case("B15")
def test_b15_prose_below_the_table_frame_never_becomes_a_requirement(tenders, frozen):
    """P3 p14 表格框线之下的页尾散文一条也不进要求集合。

    边界在哪：p14 画了两道**贯穿整页宽度**的框线（``x0=0.0, x1=595.3``，与页面同宽），
    它们不是表格线。``cs.row_bands`` 靠 ``EDGE_MARGIN == 5.0`` 把贴边横线剔掉，
    于是整页只剩 ``[99.4, 315.9]`` 一个行带；行带之下还有 24 个词
    （y 从 321.9 一直到 780.7 的页码），里面有 ``本项目核心产品为：采购清单序号1``（X25）、
    ``三、商务要求``（X16）和最低要求套话——它们全都不属于任何行带，
    因此不可能被切成单元格文本。

    顺带记一处实现内的分歧：同一页上 ``cs.table_xspan``（取横线交集）给
    ``(69.8, 531.2)``，``cs.dominant_hspan``（取众数跨度）给 ``(0.0, 595.0)``。
    两者在有整页宽框线的页面上必然不一致，本用例只依赖 ``row_bands`` 的结果。
    """
    page = tenders["P3"].page(14)
    hs, vs = cs.page_rules(page)
    band_ys = cs.row_bands(page, hs)
    assert band_ys == [99.4, 315.9]
    assert cs.EDGE_MARGIN == 5.0

    full_width = [h for h in hs if h.span >= page.rect.width - cs.EDGE_MARGIN * 2]
    assert len(full_width) == 2, "p14 应当正好有 2 道贯穿页宽的框线被剔除"
    assert cs.table_xspan(page, hs) != cs.dominant_hspan(page, hs)

    below = [w for w in page.get_text("words") if w[1] > band_ys[-1]]
    assert len(below) == 24
    below_text = " ".join(w[4] for w in below)
    for prose in P14_BELOW_FRAME_PROSE:
        assert prose in below_text, f"「{prose}」应当在框线之下"

    # 冻结中间态里，没有任何一条 P3 要求带这些散文，也没有把页码当成要求行。
    for stage in ("tender", "cut"):
        reqs = frozen("P3", stage)["reqs"]
        assert reqs
        blob = "\n".join(r["text"] for r in reqs)
        for prose in P14_BELOW_FRAME_PROSE:
            assert prose not in blob, f"{stage} 阶段混进了「{prose}」"
        assert not any(r["text"].strip().isdigit() for r in reqs)


@pytest.mark.ec("A1")
@pytest.mark.case("X16")
def test_x16_commercial_requirements_are_cut_from_every_pack(frozen):
    """五个包的冻结中间态里都不出现商务要求，尽管其中三个包的商务要求就在参数页上。

    边界在哪：``三、商务要求`` 在 P1/P2 的 p10、P3 的 p14 上**确实存在**，
    而且这些页都在 ``param_pages`` 里——也就是说页级过滤根本挡不住它。
    真正起作用的是 :data:`ta.CUT_MARKERS` 的行级截断：一旦某一行文本里出现
    套话、``本项目核心产品为`` 或 ``三、商务要求``，该行及其之后的内容全部丢弃，
    并且 ``extract_blocks`` 立刻 ``break`` 出页循环。
    P4/P5 走的是另一条路：它们压根没有这个标题（命中页为空），
    所以同一条断言在两种情形下都得成立。

    X17 的空白 ``技术规格偏离表`` 走第三种机制：它在 P1 的 p63 与 p77，
    远在 ``param_pages`` 之外，页级过滤天然排除，不需要 CUT_MARKERS 出手。
    """
    for name in ("P1", "P2", "P3", "P4", "P5"):
        tender = frozen(name, "tender")
        cut = frozen(name, "cut")
        assert tender["reqs"] and cut["reqs"], f"{name} 的冻结中间态是空的"
        for stage, doc in (("tender", tender), ("cut", cut)):
            for row in doc["reqs"]:
                text = row["text"]
                assert tp.A_NEXT_SECTION not in text, f"{name}/{stage} 混入商务要求标题"
                assert "商务" not in text, f"{name}/{stage} 混入商务要求正文：{text[:40]}"
                assert tp.A_TAIL_BOILERPLATE not in text
                assert "本项目核心产品为" not in text
                assert "技术规格偏离表" not in text, f"{name}/{stage} 混入空白格式模板"

        # A 族的三个包：商务要求标题所在页就在参数页集合内，证明靠的是行级裁剪。
        if name in ("P1", "P2", "P3"):
            inside = list(tender["param_pages"])
            assert inside and len(inside) <= tp.MAX_PARAM_PAGES
            assert inside[-1] == (10 if name in ("P1", "P2") else 14)
