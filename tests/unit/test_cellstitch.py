"""``bidtable/ingest/cellstitch.py``：表格几何的八个互不重叠的边界。

1. ``join_lines`` 默认 ``sep=""``——换行**直接删除**，不补空格（X19）。
2. ``page_rules`` 同时认 ``l`` 段与细长 ``re`` 矩形，且**绝不调用** ``find_tables()``（X18）。
3. ``column_edges`` / ``row_bands`` 的三道过滤：跨度、贴边、四舍五入去重。
4. ``bands_from`` 配对相邻 y；``filter_bands_to_table`` 的 ±1.0 松弛是**闭区间**。
5. ``cell_text`` 的 clip 四边各内缩 ``FLAT_TOL``，这是同页混排按 x 分流的唯一依据（B14）。
6. ``band_cells`` 是 ``cols × bands`` 网格；``BandCells`` 容器在越界/空带时安静降级。
7. ``table_xspan`` 取**交集**：一条杂线就能把区间收窄。
8. ``dominant_hspan`` 取**众数**：单条页眉线无法劫持表体区间。
"""

from __future__ import annotations

from types import SimpleNamespace

import fitz
import pytest

from bidtable.ingest import cellstitch as cs

pytestmark = [pytest.mark.unit]


class _StubPage:
    """``column_edges``/``row_bands`` 只读 ``.rect.width``/``.rect.height``，无需真页面。"""

    def __init__(self, width: float = 595.0, height: float = 842.0) -> None:
        self.rect = SimpleNamespace(width=width, height=height)


def _blank_page(width: float = 595.0, height: float = 842.0) -> fitz.Page:
    return fitz.open().new_page(width=width, height=height)


@pytest.mark.ec("A1")
@pytest.mark.case("X19")
def test_join_lines_deletes_the_newline_instead_of_replacing_it_with_a_space():
    """``find_tables()`` 会在折行处插一个空格，gold 原文里没有那个空格。

    对照实例（P1 序号2）：gold = ``▲2.最大单脉冲能量5J。（大于5J 为优于）``，
    PDF 原始 = ``▲2.最大单脉冲能量5J。（大于5J 为\\n优于）``。PyMuPDF 把行尾空格
    留在行内，所以默认 ``sep=""`` 直接拼接就能逐字对齐 gold；补空格会让
    「招标要求（原文）」列全线失配。
    """
    assert cs.join_lines("（大于5J 为\n优于）") == "（大于5J 为优于）"
    # 行内原有的空格一个字都不能动
    assert cs.join_lines("每年提供≥\n4 次的设备维护保养服务") == "每年提供≥4 次的设备维护保养服务"
    # PyMuPDF 的取词结果总以 \n 收尾：默认 sep 下它被吃掉，非空 sep 下会留一个尾巴
    assert cs.join_lines("1.7\n") == "1.7"
    assert cs.join_lines("1.7\n", "\n") == "1.7\n"
    assert cs.join_lines("a\nb\n", "/") == "a/b/"
    # 显式 sep 是 B 族用来保留行结构的（它要按行再切一次编号）
    assert cs.join_lines("1.\n基础率时段：≥48 段。", "\n") == "1.\n基础率时段：≥48 段。"
    # 退化输入
    assert cs.join_lines("") == ""
    assert cs.join_lines(None) == ""
    assert cs.join_lines(None, "\n") == ""


@pytest.mark.ec("A2")
@pytest.mark.case("X18")
def test_page_rules_reads_thin_rectangles_as_rules_and_never_calls_find_tables(monkeypatch):
    """X18：一律自建几何，不用 ``find_tables()``——P1 p8 用它只能拿到 15 条参数里的 3 条。

    这里把 ``fitz.Page.find_tables`` 换成会炸的桩，再跑 ``page_rules``：
    只要实现偷偷调用它就立刻失败。同时钉住四条判定：``l`` 段、细长 ``re`` 矩形、
    ``FLAT_TOL`` 之内的轻微倾斜算水平、超出容差的斜线**既不算横也不算竖**。
    """

    def _boom(self, *args, **kwargs):  # pragma: no cover - 被调用即为失败
        raise AssertionError("cellstitch 不允许调用 find_tables()（X18）")

    monkeypatch.setattr(fitz.Page, "find_tables", _boom, raising=False)

    page = _blank_page()
    page.draw_line(fitz.Point(123.2, 100.0), fitz.Point(534.4, 100.0))  # 横线段
    page.draw_line(fitz.Point(200.0, 100.0), fitz.Point(200.0, 400.0))  # 竖线段
    page.draw_rect(fitz.Rect(123.2, 300.0, 534.4, 300.8), color=None, fill=(0, 0, 0))  # 扁矩形->横
    page.draw_rect(fitz.Rect(400.0, 100.0, 400.9, 400.0), color=None, fill=(0, 0, 0))  # 窄矩形->竖
    page.draw_line(fitz.Point(123.2, 500.0), fitz.Point(534.4, 500.4))  # 倾斜 0.4 < FLAT_TOL
    page.draw_line(fitz.Point(123.2, 600.0), fitz.Point(534.4, 601.5))  # 倾斜 1.5 > FLAT_TOL
    page.draw_rect(fitz.Rect(50.0, 700.0, 300.0, 750.0), color=None, fill=(0, 0, 0))  # 粗矩形

    hs, vs = cs.page_rules(page)
    assert [round(h.y, 1) for h in hs] == [100.0, 300.0, 500.0]
    assert [round(v.x, 1) for v in vs] == [200.0, 400.0]
    # 端点取自绘图指令，带浮点噪声，只能近似比较
    assert hs[0].x0 == pytest.approx(123.2, abs=1e-3)
    assert hs[0].x1 == pytest.approx(534.4, abs=1e-3)
    assert hs[0].span == pytest.approx(411.2, abs=1e-2)
    # 倾斜线取的是**起点** y，不是两端平均
    assert hs[2].y == pytest.approx(500.0, abs=1e-6)
    # re 分支的竖线取矩形左边 x0，跨度是矩形高度
    assert vs[1].span == pytest.approx(300.0, abs=1e-3)


@pytest.mark.ec("A1")
def test_column_edges_and_row_bands_keep_only_rules_that_cross_the_table_body():
    """两道过滤 + 一次四舍五入去重，就是把页眉页脚装饰线挡在切分之外的全部机制。

    列取 ``round(x)`` 取整（P1 p8 的七条竖线：62/94/159/372/408/443/540），
    行取 ``round(y, 1)``（P1 p8 的四条横线：255.7/311.2/336.3/741.7）。
    两个比较都是**严格**的：``span > min_span``、``EDGE_MARGIN < x < w - EDGE_MARGIN``，
    所以刚好等于阈值的线会被丢掉——这不是疏忽，页眉线常常正好贴着页边。
    """
    page = _StubPage(width=595.0, height=842.0)
    body_x = [62.2, 62.4, 94.0, 159.0, 372.0, 408.0, 443.0, 540.0]  # 62.2/62.4 应去重成一个
    vs = [cs.VRule(x=x, y0=100.0, y1=500.0) for x in body_x]
    vs.append(cs.VRule(x=300.0, y0=100.0, y1=150.0))  # 跨度 50 <= 60 -> 丢
    vs.append(cs.VRule(x=310.0, y0=100.0, y1=160.0))  # 跨度正好 60 -> 也丢（严格大于）
    vs.append(cs.VRule(x=5.0, y0=100.0, y1=500.0))  # 正好贴边 -> 丢
    vs.append(cs.VRule(x=3.0, y0=100.0, y1=500.0))  # 左出界 -> 丢
    vs.append(cs.VRule(x=592.0, y0=100.0, y1=500.0))  # 右出界 -> 丢
    assert cs.column_edges(page, vs) == [62, 94, 159, 372, 408, 443, 540]

    body_y = [255.74, 255.66, 311.2, 336.3, 741.7]  # 前两个应去重成 255.7
    hs = [cs.HRule(y=y, x0=62.0, x1=540.0) for y in body_y]
    hs.append(cs.HRule(y=400.0, x0=62.0, x1=212.0))  # 跨度 150 <= 200 -> 丢
    hs.append(cs.HRule(y=3.0, x0=62.0, x1=540.0))  # 贴上边 -> 丢
    hs.append(cs.HRule(y=840.0, x0=62.0, x1=540.0))  # 贴下边 -> 丢
    assert cs.row_bands(page, hs) == [255.7, 311.2, 336.3, 741.7]
    # 阈值可调：把 min_span 降到 100 就放行那条 150pt 的短线
    assert 400.0 in cs.row_bands(page, hs, min_span=100.0)


@pytest.mark.ec("A2")
def test_bands_from_pairs_neighbours_and_the_slack_filter_is_inclusive_at_one_point():
    """行带是**相邻两条线**构成的区间，所以 N 条线只产出 N-1 个带。

    ``filter_bands_to_table`` 的 ±1.0 松弛是给 ``round(y, 1)`` 留的余量，
    区间两端都是闭的：B 族靠它把 y=41.0 的页眉线挡在表体之外，
    同时不能把表体自己的第一条线（正好等于 lo）误删。
    """
    assert cs.bands_from([]) == []
    assert cs.bands_from([5.0]) == []  # 单条线撑不出一个带
    # P1 p9 的真实分界
    assert cs.bands_from([99.4, 369.9, 400.2, 743.1]) == [
        (99.4, 369.9),
        (369.9, 400.2),
        (400.2, 743.1),
    ]

    ys = [41.0, 99.0, 100.0, 200.0, 300.0, 301.0, 400.0]
    assert cs.filter_bands_to_table(ys, 100.0, 300.0) == [99.0, 100.0, 200.0, 300.0, 301.0]
    # 松弛刚好 1.0 是闭的，再多一点点就出局
    assert cs.filter_bands_to_table([98.9], 100.0, 300.0) == []
    assert cs.filter_bands_to_table([301.1], 100.0, 300.0) == []
    assert cs.filter_bands_to_table([], 100.0, 300.0) == []
    # lo > hi（调用方几何探测失败时会传进来）=> 空区间，不能抛
    assert cs.filter_bands_to_table([100.0], 300.0, 100.0) == []


@pytest.mark.ec("A1")
@pytest.mark.case("B14")
def test_cell_text_routes_words_by_x_and_truncates_a_glyph_that_straddles_the_clip():
    """同页混排（P3 p13 左半是标的1、右半是标的2）唯一靠得住的分流手段就是 x 裁切。

    同时钉住两处锋利边，它们解释了为什么列/行边界**必须**来自真实矢量线：
    clip 切进字形时 PyMuPDF 是**截断**而不是丢弃（``.7``、``1.``），
    而 y 方向切进字形腰部时它会**凭空插一个空格**（``1 7``）——正是 X19 要躲的产物。
    """
    page = _blank_page()
    page.insert_text(fitz.Point(70, 120), "1.7")
    page.insert_text(fitz.Point(140, 120), "work current")

    # 分界线落在两列之间：每个词恰好归一列
    assert cs.cell_text(page, 60, 130, 100, 140) == "1.7"
    assert cs.cell_text(page, 130, 400, 100, 140) == "work current"
    # 空白区域 -> 空串（不抛、不返回 None）；sep 透传，行尾那个 \n 会留下一个空片
    assert cs.cell_text(page, 60, 130, 300, 400) == ""
    assert cs.cell_text(page, 60, 400, 100, 140, sep="|") == "1.7|work current|"
    # 锋利边：clip 切进字形 -> 截断
    assert cs.cell_text(page, 75, 130, 100, 140) == ".7"
    assert cs.cell_text(page, 60, 78, 100, 140) == "1."
    # 锋利边：y 切进字形腰部 -> PyMuPDF 插空格（X19 的现场复现）
    assert cs.cell_text(page, 60, 130, 100, 115) == "1 7"
    # 四边各内缩 FLAT_TOL 后仍完整包住字形时，结果不受影响
    assert cs.cell_text(page, 60, 130, 112, 124) == "1.7"


@pytest.mark.ec("A1")
@pytest.mark.ec("A2")
def test_band_cells_is_a_grid_whose_container_degrades_quietly_instead_of_raising():
    """``band_cells`` 是 ``len(cols)-1`` 列 × ``len(band_ys)-1`` 行的网格。

    调用方（A/B 两族的版面抽取）拿到的是**形状不可预知**的页面：列可能探不到、
    行带可能为空、某一带可能整片空白。容器必须安静降级——越界 ``get`` 回空串、
    负索引也回空串（不沿用 list 的环绕语义）、空网格回 ``[]``——
    因为一条抽不出来的行只该少一行，不该让整个资料包失败。
    """
    page = _blank_page()
    page.insert_text(fitz.Point(70, 120), "1.7")
    page.insert_text(fitz.Point(140, 120), "work current")

    bands = cs.band_cells(page, [60, 130, 400], [100.0, 140.0, 220.0, 260.0], 7, sep="\n")
    assert len(bands) == 3  # 4 条分界线 -> 3 个带
    first, blank = bands[0], bands[1]
    assert (first.page_no, first.y0, first.y1, first.height) == (7, 100.0, 140.0, 40.0)
    assert first.cells == ["1.7\n", "work current\n"]
    assert not cs.is_empty_band(first)
    # 无字的带：cells 形状仍在，只是全空 -> 交给 is_empty_band 判定后跳过
    assert blank.cells == ["", ""]
    assert cs.is_empty_band(blank)
    # 越界与负索引都回空串
    assert first.get(2) == ""
    assert first.get(9) == ""
    assert first.get(-1) == ""
    # 只有空白字符也算空带
    assert cs.is_empty_band(cs.BandCells(1, 0.0, 1.0, ["", "  ", "\n"]))
    assert cs.is_empty_band(cs.BandCells(1, 0.0, 1.0, []))
    assert cs.BandCells(1, 10.5, 20.25, []).height == 9.75
    # 网格塌掉时安静返回空列表：探不到列 / 探不到行
    assert cs.band_cells(page, [60], [100.0, 140.0], 1) == []
    assert cs.band_cells(page, [], [100.0, 140.0], 1) == []
    assert cs.band_cells(page, [60, 130], [], 1) == []
    assert cs.band_cells(page, [60, 130], [100.0], 1) == []


@pytest.mark.ec("A2")
def test_the_table_extent_is_a_vote_not_an_intersection():
    """B 族没有竖线，只能拿横线的 x 区间当列分割依据；两个算法在这里给出不同答案。

    ``table_xspan`` 取**交集**（``max(x0)``、``min(x1)``）：一条杂线就能把它收窄。
    ``dominant_hspan`` 取**众数**：只有多数派才算表体。
    P4 p25 是活教材——同一页上 ``六、配置要求`` 的小表（118.7..325.0，4 条线）
    与主参数表（123.2..534.4，12 条线）并存：交集会把右边界错拉到 325.0，
    众数则稳稳给出 123.0..534.0。``tender_layout_b._table_geom`` 用的是后者。
    平票时 ``max`` 返回**先插入**的那个，即文档顺序，因此结果对同一份 PDF 恒定。
    """
    page = _StubPage()
    body = [cs.HRule(y=100.0 + i, x0=123.2, x1=534.4) for i in range(12)]
    header = cs.HRule(y=41.0, x0=61.6, x1=531.7)  # P4 的页眉单线
    small = [cs.HRule(y=200.0 + i, x0=118.7, x1=325.0) for i in range(4)]  # 六、配置要求

    # 众数：12 票的主表压过 1 票页眉与 4 票小表，且返回的是 float(round(x))
    assert cs.dominant_hspan(page, body + [header]) == (123.0, 534.0)
    assert cs.dominant_hspan(page, small + body) == (123.0, 534.0)
    # 交集：同一段输入被页眉线收窄（534.4 -> 531.7），被小表腰斩（-> 325.0）
    assert cs.table_xspan(page, body + [header]) == pytest.approx((123.2, 531.7))
    assert cs.table_xspan(page, small + body) == pytest.approx((123.2, 325.0))
    # 平票 -> 文档顺序里先出现的那个胜出（与输入顺序绑定，不是随机的）
    a, b = cs.HRule(y=1.0, x0=10.0, x1=400.0), cs.HRule(y=2.0, x0=20.0, x1=500.0)
    assert cs.dominant_hspan(page, [a, b]) == (10.0, 400.0)
    assert cs.dominant_hspan(page, [b, a]) == (20.0, 500.0)
    # 退化：没有长线时两者都回 None，调用方据此跳过该页而不是崩
    short = [cs.HRule(y=1.0, x0=0.0, x1=200.0)]  # span 正好等于 MIN_H_SPAN -> 不算长线
    assert cs.dominant_hspan(page, short) is None
    assert cs.table_xspan(page, short) is None
    assert cs.dominant_hspan(page, []) is None
    assert cs.table_xspan(page, []) is None
    # 交集特有：两条长线完全错开 -> x1 <= x0 -> None
    disjoint = [cs.HRule(y=1.0, x0=0.0, x1=300.0), cs.HRule(y=2.0, x0=400.0, x1=700.0)]
    assert cs.table_xspan(page, disjoint) is None
    assert cs.dominant_hspan(page, disjoint) == (0.0, 300.0)  # 众数不受错开影响，平票取先者
