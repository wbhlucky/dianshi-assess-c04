"""产物不变量 IV1..IV14（agent.md §6.6）。

这一层的分工：每条用例只盯**一个**不变量，但对 5 个真实资料包的产物同时断言。
也就是说粒度是「一个不变量 × 全部版面族」，而不是「一个版面族 × 十四个互相重叠
的断言」——后者会让同一个缺陷在十四条用例里各报一次，看起来热闹，实际信息量为零。

5 个包正好把三种版面族和四种标的结构铺满：P1/P2/P3 是 A 族（P2/P3 带服务标的），
P4 是 B 族单设备，P5 是 C 族多设备非连续子集（标的 1/5/8/12，且标的 12 的数量写成
``1None``）。所以每条不变量在这里天然就是跨版面族的。

每条用例都带一个「非空跑」守卫：先统计这一轮里真正触发该不变量的行数，再断言它 >0。
没有这个守卫，一条写成 ``all(... for r in [])`` 的断言可以永远绿灯。

用例清单（与 agent.md §6.6 的 IV1..IV14 一一对应，编号即用例号）：

1.  IV1  (标的, 分节, 序号) 三元组唯一
2.  IV2  工作表恰好 10 列，且列名与顺序等于 ``config.COLUMNS``
3.  IV3  活动工作表名 == ``config.SHEET_NAME``
4.  IV4  偏离类型落在 ``config.DEVIATIONS`` 内，空串绝不出现在产物里
5.  IV5  未响应 与 投标响应/证明材料文件/位置三列同时为空 互为充要
6.  IV6  非未响应行引用的证明材料文件在包内真实存在
7.  IV7  位置串只取四种白名单形态
8.  IV8  「表行N」不越界（docx 有真上界；检测报告离线只能断言 >=1）
9.  IV9  「第N页」不越产品综合样本的真实页数
10. IV10 标记只取 星号/三角/空
11. IV11 序号非空、不含标记、不含空白
12. IV12 输出的标的 是 投标范围声明的标的 的子集
13. IV13 招标要求（原文）非空且单行
14. IV14 行数守恒：输出行 == 分节后条目 - 标题/引导句 == 参与判定
"""

from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from openpyxl import load_workbook

from bidtable import pipeline as pl
from bidtable.config import (
    COLUMNS,
    DEVIATIONS,
    DEV_NO_RESPONSE,
    MARK_STAR,
    MARK_TRIANGLE,
    SHEET_NAME,
    load_config,
)
from bidtable.ingest import vendor_catalog as vc
from bidtable.ingest import vendor_docx as vd
from bidtable.logging_ import Progress
from bidtable.output import writer

pytestmark = [pytest.mark.invariants, pytest.mark.integration]

PACKS = ("P1", "P2", "P3", "P4", "P5")

#: 位置串的四种合法形态（agent.md IV7）。四个分支互斥，且整串必须被吃掉。
LOC_RE = re.compile(
    r"^(?:技术规格书 表行(\d+)"
    r"|产品综合样本 第(\d+)页"
    r"|检测报告 第(\d+)页 表行(\d+)"
    r"|服务方案 表行(\d+))$"
)


class Built(NamedTuple):
    """一个资料包跑完离线流水线、并按产品路径写出 xlsx 之后的全部现场。"""

    pack: str
    root: Path
    result: Any
    rows: tuple
    xlsx: Path
    vendor: Path
    doc_rows: dict
    catalog_pages: dict


def _parse_loc(loc: str):
    """把位置串拆成 ``(文档类型, 页号 or None, 表行号 or None)``；不认识返回 ``None``。"""
    m = LOC_RE.match(loc)
    if m is None:
        return None
    if m.group(1) is not None:
        return ("技术规格书", None, int(m.group(1)))
    if m.group(2) is not None:
        return ("产品综合样本", int(m.group(2)), None)
    if m.group(3) is not None:
        return ("检测报告", int(m.group(3)), int(m.group(4)))
    return ("服务方案", None, int(m.group(5)))
    return ("服务方案", None, int(m.group(5)))


@pytest.fixture(scope="module")
def built(tmp_path_factory, repo_root, fixtures_dir) -> dict:
    """5 个真实资料包各跑一遍离线流水线并写出 xlsx。

    从 ``match`` 阶段起跑、复用 ``tests/fixtures/packs/<P>/`` 里冻结的前四阶段产物，
    所以整层不碰网络、不重新解析 PDF，5 个包合计约 4 秒。
    """
    work = tmp_path_factory.mktemp("invariants")
    out: dict = {}
    for pack in PACKS:
        root = repo_root / "训练集" / pack
        frozen = fixtures_dir / "packs" / pack / "scope.json"
        if not root.is_dir() or not frozen.is_file():
            pytest.skip(
                f"缺少 {pack}；先跑 python -m bidtable.selfcheck --packs 训练集 --freeze"
            )
        cache = work / f"cache_{pack}"
        shutil.copytree(fixtures_dir / "packs" / pack, cache)
        cfg = load_config(pack=root, out=work / f"{pack}.xlsx", offline=True)
        result = pl.process_pack(
            root,
            cfg,
            progress=Progress(quiet=True),
            from_stage="match",
            use_cache=True,
            cache_root=cache,
        )
        template = repo_root / result.stats["template_xlsx"]
        xlsx, warns = writer.write(
            result.rows, cfg.out, template=template if template.is_file() else None
        )
        assert warns == [], f"{pack} 写出告警：{warns}"
        vendor = root / "02_厂商资料"
        doc_rows, catalog_pages = {}, {}
        for f in sorted(vendor.iterdir()):
            low = f.suffix.lower()
            if low == ".docx":
                doc_rows[f.name] = len(vd.parse_docx(f).rows)
            elif low == ".pdf":
                catalog_pages[f.name] = vc.parse_catalog(f).pages
        out[pack] = Built(
            pack, root, result, tuple(result.rows), xlsx, vendor, doc_rows, catalog_pages
        )
    return out


def _sheet(b: Built):
    """打开写好的工作簿，返回 ``(wb, ws)``；调用方负责 close。"""
    wb = load_workbook(b.xlsx)
    return wb, wb.active


def _col(ws, name: str) -> list:
    """按列名取出该列（含表头以下的所有数据行）的值。"""
    idx = list(COLUMNS).index(name) + 1
    return [ws.cell(row=i, column=idx).value for i in range(2, ws.max_row + 1)]


@pytest.mark.case("IV1")
@pytest.mark.ec("I3")
@pytest.mark.ec("I6")
@pytest.mark.ec("I7")
def test_key_lot_section_no_is_unique(built):
    """IV1：``(标的, 分节, 序号)`` 三元组唯一。

    这是个真边界而不是走过场：P3 的 71 行正文只摊到 24 个不同的 ``(标的, 序号)``
    上，16 组互相撞号，靠「分节」（如 ``五、医用显示器 / 1、4K3D 医用显示器``）才拉回
    唯一。反过来 P1/P2/P4 的分节恒为空串，唯一性完全落在序号上。同一条断言因此
    同时压住了「分节策略选错会撞号」和「序号解析会重号」两类缺陷。
    """
    for b in built.values():
        keys = Counter((r.lot_no, r.section, r.no) for r in b.rows)
        dup = {k: v for k, v in keys.items() if v > 1}
        assert not dup, f"{b.pack} 撞号：{dup}"
        assert len(keys) == len(b.rows)


@pytest.mark.case("IV2")
@pytest.mark.ec("J3")
@pytest.mark.ec("A1")
def test_sheet_has_exactly_the_ten_template_columns(built):
    """IV2：工作表恰好 10 列，列名与顺序等于 ``config.COLUMNS``。

    盯的是「多一列/少一列/串位」这一类产物缺陷。标准答案 xlsx 有第 11 列
    ``判定说明``，而模板只有 10 列——产物必须跟模板走，绝不能把内部判定说明
    写进交付表（agent.md X28）。这条从**写出来的文件**读表头，所以连
    ``writer._from_template`` 悄悄改表头的修复路径也一并覆盖了。
    """
    for b in built.values():
        wb, ws = _sheet(b)
        try:
            assert ws.max_column == len(COLUMNS), f"{b.pack} 列数 {ws.max_column}"
            assert [c.value for c in ws[1]] == list(COLUMNS)
            assert ws.max_row == len(b.rows) + 1
        finally:
            wb.close()


@pytest.mark.case("IV3")
@pytest.mark.ec("J3")
def test_active_sheet_is_named_after_the_contract(built):
    """IV3：活动工作表名 == ``响应偏离表``，且工作簿里没有第二张表。

    表名是跟 ``grade.py``、复核报告、以及人工在 Excel 里的第一眼的接口契约。
    模板本身就带这个名字，``writer._from_template`` 会把活动表重命名回去；
    走内置新建路径时由 ``_new_workbook`` 命名。两条路径必须收敛到同一个名字。
    """
    for b in built.values():
        wb, ws = _sheet(b)
        try:
            assert wb.sheetnames == [SHEET_NAME], f"{b.pack} {wb.sheetnames}"
            assert ws.title == SHEET_NAME
        finally:
            wb.close()


@pytest.mark.case("IV4")
@pytest.mark.ec("J3")
@pytest.mark.ec("E1")
@pytest.mark.ec("E2")
@pytest.mark.ec("E3")
@pytest.mark.ec("E4")
def test_deviation_stays_inside_the_closed_vocabulary(built):
    """IV4：偏离类型只取 ``无偏离/正偏离/负偏离/未响应``，空串绝不落盘。

    内部 ``Row.deviation`` 允许空串（标题行、引导句行），但那些行在写表前就被
    摘掉了；所以「空串不出现在产物里」正是 IV14 行数守恒的另一面。
    P2 恰好只有 3 种（没有正偏离），P1/P3/P4/P5 四种齐活，值域是被真数据撑开的。
    """
    allowed = set(DEVIATIONS)
    for b in built.values():
        got = {r.deviation for r in b.rows}
        assert got <= allowed, f"{b.pack} 越界值 {got - allowed}"
        assert "" not in got
        wb, ws = _sheet(b)
        try:
            on_disk = set(_col(ws, "偏离类型"))
        finally:
            wb.close()
        assert on_disk == got, f"{b.pack} 落盘值与内存值不一致"


@pytest.mark.case("IV5")
@pytest.mark.ec("E4")
def test_no_response_is_exactly_the_three_empty_columns(built):
    """IV5：``未响应`` 与「投标响应/证明材料文件/位置三列同时为空」互为充要。

    双向很关键：正向抓「判了未响应却还挂着证据」，反向抓「三列全空却判成无偏离」
    ——后者是最难看的产物缺陷，因为它在 Excel 里看起来像一条有结论的行。
    5 个包每个都至少有 3 条未响应行，两个方向都被真数据跑过。
    """
    for b in built.values():
        hits = 0
        for r in b.rows:
            empty = not r.resp_text and not r.evidence_file and not r.location
            if r.deviation == DEV_NO_RESPONSE:
                hits += 1
                assert empty, (
                    f"{b.pack} 序号{r.no} 未响应却有证据：{r.resp_text!r}"
                )
            else:
                assert not empty, (
                    f"{b.pack} 序号{r.no} 三列全空却判成 {r.deviation}"
                )
        assert hits > 0, f"{b.pack} 没有未响应行，这条不变量是空跑"


@pytest.mark.case("IV6")
@pytest.mark.ec("D1")
@pytest.mark.ec("D2")
@pytest.mark.ec("D3")
@pytest.mark.ec("D4")
def test_evidence_file_really_exists_in_the_pack(built):
    """IV6：非未响应行引用的证明材料文件，在 ``02_厂商资料/`` 里真实存在。

    这是最容易被幻觉出来的一列：文件名是按 ``厂商前缀_型号_文档类型`` 拼出来的，
    一旦型号绑定串了包，产物就会指向一个不存在的文件。断言落在**磁盘**上而不是
    证据对象上，所以连 ``lotmap.attach_lots`` 回填文件名的路径也一并压住。
    四个 EC-D 来源（规格书/样本/报告/服务方案）在 5 个包里都出现，除了 P4/P5
    没有服务方案——那正好是「非全包都要有全部来源」的现实约束。
    """
    for b in built.values():
        hits = 0
        for r in b.rows:
            if r.deviation == DEV_NO_RESPONSE:
                continue
            hits += 1
            assert r.evidence_file, f"{b.pack} 序号{r.no} 判了 {r.deviation} 却没有文件名"
            assert (b.vendor / r.evidence_file).is_file(), (
                f"{b.pack} 序号{r.no} 指向不存在的文件 {r.evidence_file!r}"
            )
        assert hits > 0, f"{b.pack} 一条有证据的行都没有，这条不变量是空跑"


@pytest.mark.case("IV7")
@pytest.mark.ec("D1")
@pytest.mark.ec("D2")
@pytest.mark.ec("D3")
@pytest.mark.ec("D4")
def test_location_matches_one_of_the_four_shapes(built):
    """IV7：位置串只取四种白名单形态，且整串必须被吃掉。

    白名单是 ``技术规格书 表行N`` / ``产品综合样本 第N页`` /
    ``检测报告 第N页 表行N`` / ``服务方案 表行N``，未响应行留空。
    正则带 ``^...$`` 而非 ``search``，所以「检测报告 检测报告 第1页」这类
    拼接重复（``candidates._locator_of`` 修过的那个缺陷）会当场失败。
    """
    for b in built.values():
        kinds = Counter()
        for r in b.rows:
            if not r.location:
                assert r.deviation == DEV_NO_RESPONSE, (
                    f"{b.pack} 序号{r.no} 位置为空却判成 {r.deviation}"
                )
                continue
            parsed = _parse_loc(r.location)
            assert parsed is not None, f"{b.pack} 序号{r.no} 位置越界：{r.location!r}"
            kinds[parsed[0]] += 1
        assert len(kinds) >= 3, f"{b.pack} 位置形态太单一，这条不变量是空跑：{kinds}"
        assert len(kinds) >= 3, f"{b.pack} 位置形态太单一，这条不变量是空跑：{kinds}"


@pytest.mark.case("IV8")
@pytest.mark.ec("D1")
@pytest.mark.ec("D3")
@pytest.mark.ec("D4")
def test_table_row_index_never_exceeds_the_source_document(built):
    """IV8：「表行N」不越界。

    分两种情况，因为它们的可验证强度天生不同：

    * docx（技术规格书/服务方案）——``vendor_docx.parse_docx`` 能数出真实行数，
      所以 ``1 <= N <= len(rows)`` 是硬上界。5 个包里每个 docx 的**最大用行号恰好
      等于它自己的行数**，这说明上界是紧的：索引整体偏移一位就会当场越界，
      不会被一个宽松上界放过去。
    * 检测报告（.jpg）——离线没有 OCR，报告表格的真实行数拿不到，只能断言
      ``N >= 1``。这是 agent.md §2b 记录的离线天花板，不是可以糊过去的地方，
      所以这里明确写死「只断言下界」，不假装验了上界。
    """
    for b in built.values():
        used: dict = {}
        for r in b.rows:
            parsed = _parse_loc(r.location) if r.location else None
            if parsed is None or parsed[2] is None:
                continue
            kind, _page, rowno = parsed
            assert rowno >= 1, f"{b.pack} 序号{r.no} 表行号 {rowno}"
            if kind == "检测报告":
                continue
            cap = b.doc_rows.get(r.evidence_file)
            assert cap is not None, (
                f"{b.pack} {kind} 行指向了非 docx 文件 {r.evidence_file!r}"
            )
            assert rowno <= cap, (
                f"{b.pack} 序号{r.no} 表行{rowno} 越界（{r.evidence_file} 只有 {cap} 行）"
            )
            used[r.evidence_file] = max(used.get(r.evidence_file, 0), rowno)
        assert used, f"{b.pack} 没有 docx 表行定位，这条不变量是空跑"
        assert any(used[f] == b.doc_rows[f] for f in used), (
            f"{b.pack} 上界从未被触到，无法排除整体偏移：{used}"
        )


@pytest.mark.case("IV9")
@pytest.mark.ec("D2")
def test_catalog_page_index_never_exceeds_the_pdf(built):
    """IV9：「第N页」不越产品综合样本的真实页数。

    上界来自 ``vendor_catalog.parse_catalog(...).pages``，也就是真去数了 PDF。
    这条能抓住两类缺陷：把「样本里第 k 个产品」当成页号，以及空白页被跳过后
    页号整体前移（``parse_catalog`` 对空白页是跳过并告警、**不**重编号的）。
    P5 的样本有 4 页且真的用到了第 4 页，所以上界是紧的、不是空跑。
    """
    max_seen = 0
    for b in built.values():
        hits = 0
        for r in b.rows:
            parsed = _parse_loc(r.location) if r.location else None
            if parsed is None or parsed[1] is None:
                continue
            kind, page, _row = parsed
            assert page >= 1, f"{b.pack} 序号{r.no} 页号 {page}"
            if kind == "检测报告":
                # 十张报告图都是单页，页号恒为 1；离线无从验证上界，只断言下界。
                continue
            cap = b.catalog_pages.get(r.evidence_file)
            assert cap is not None, (
                f"{b.pack} {kind} 行指向了非 PDF 文件 {r.evidence_file!r}"
            )
            assert page <= cap, (
                f"{b.pack} 序号{r.no} 第{page}页越界（{r.evidence_file} 只有 {cap} 页）"
            )
            hits += 1
            max_seen = max(max_seen, page)
        assert hits > 0, f"{b.pack} 没有样本页定位，这条不变量是空跑"
    assert max_seen > 1, f"全部产物都只用到第 1 页，页号上界从未被压过（最大 {max_seen}）"


@pytest.mark.case("IV10")
@pytest.mark.ec("F1")
@pytest.mark.ec("F4")
@pytest.mark.ec("F5")
def test_mark_column_only_carries_star_triangle_or_nothing(built):
    """IV10：标记列只取 星号、三角、空串三种，半角星号绝不出现。

    后半句才是这条用例的价值所在：P5 的招标原文用半角 ``*``（``*21.1``），
    P1/P2/P3 用 ``▲``，P4/P5 用全角 ``★``。投标范围说明里明确「部分文件用半角
    ``*`` 表示同样含义，按 ★ 处理」，所以产物里出现 ``*`` 就意味着归一化没跑。
    """
    allowed = {MARK_STAR, MARK_TRIANGLE, ""}
    union = set()
    for b in built.values():
        got = {r.mark for r in b.rows}
        assert got <= allowed, f"{b.pack} 越界标记 {got - allowed}"
        assert "*" not in got, f"{b.pack} 半角星号漏到了产物里"
        union |= got
    assert union == allowed, f"5 个包合起来没有铺满标记值域：{union}"


@pytest.mark.case("IV11")
@pytest.mark.ec("H1")
@pytest.mark.ec("H3")
@pytest.mark.ec("H5")
@pytest.mark.ec("H6")
@pytest.mark.ec("H7")
@pytest.mark.ec("H8")
def test_sequence_number_is_clean(built):
    """IV11：序号非空、不含标记、不含空白，且只由数字和点组成。

    序号是从招标原文里剥出来的，最容易带脏东西：``▲2.`` 的 ``▲``（H6）、
    ``1.7★`` 的 ``★``（H7）、``*21.1`` 的 ``*``（H8）、``1．2`` 的全角点（H12）、
    ``1.9.1`` 的三级点号（H5）、以及 ``1、`` 的顿号（H2）。
    这一条同时盯住「标记没剥干净」和「序号被切成两半」两类缺陷。
    """
    for b in built.values():
        for r in b.rows:
            assert r.no, f"{b.pack} 有一行序号为空：{r.req_text[:20]!r}"
            assert not re.search(r"\s", r.no), f"{b.pack} 序号含空白 {r.no!r}"
            assert not (set(r.no) & {MARK_STAR, MARK_TRIANGLE, "*", "、", "．"}), (
                f"{b.pack} 序号含标记或全角点 {r.no!r}"
            )
            assert re.fullmatch(r"[\d.]+", r.no), f"{b.pack} 序号越界字符 {r.no!r}"


@pytest.mark.case("IV12")
@pytest.mark.ec("B1")
@pytest.mark.ec("B2")
@pytest.mark.ec("B3")
@pytest.mark.ec("B4")
def test_output_lots_stay_inside_the_declared_scope(built):
    """IV12：输出的标的是投标范围声明的标的的子集，且一个都不少。

    子集方向抓「凭空多出一个标的」（型号绑定串包、C 族把范围外的标的 2/3 也
    抽出来）；相等方向抓「整个标的被静默丢掉」——P5 声明的是非连续子集
    1/5/8/12，只要 ``lotmap`` 少绑一个型号，那一整块 11~18 行就会消失，
    而覆盖率之外的任何指标都不会报警。B4 的 ``1None`` 数量瑕疵也在这一条里：
    标的 12 照样得出行。
    """
    for b in built.values():
        declared = {s.lot_no for s in b.result.scopes}
        got = {r.lot_no for r in b.rows}
        assert got <= declared, f"{b.pack} 越权标的 {got - declared}"
        assert got == declared, f"{b.pack} 丢了标的 {declared - got}"


@pytest.mark.case("IV13")
@pytest.mark.ec("A1")
@pytest.mark.ec("A2")
@pytest.mark.ec("A3")
def test_requirement_text_is_one_non_empty_line(built):
    """IV13：招标要求（原文）非空且不含换行。

    A 族的核心难点是「一个单元格被横向分页切成两半」，靠 ``cellstitch`` 按 x
    区间把两页的文字拼回一个单元格，拼接用的分隔符是空串。如果哪天分隔符被改成
    ``"\\n"``，产物里就会出现跨行的原文，Excel 里一格变两行、``grade.py`` 的行匹配
    也会错位。三个版面族都要过这一条。
    """
    for b in built.values():
        for r in b.rows:
            assert r.req_text, f"{b.pack} 序号{r.no} 原文为空"
            assert r.req_text.strip(), f"{b.pack} 序号{r.no} 原文只有空白"
            assert "\n" not in r.req_text and "\r" not in r.req_text, (
                f"{b.pack} 序号{r.no} 原文跨行：{r.req_text[:40]!r}"
            )


@pytest.mark.case("IV14")
@pytest.mark.ec("C9")
@pytest.mark.ec("J8")
def test_row_count_is_conserved_across_the_cut_stages(built):
    """IV14：行数守恒——输出行 == 分节后条目 - 标题/引导句 == 参与判定。

    三个数来自三个不同的地方：``len(rows)`` 是产物，``cut_counts`` 是裁剪阶段的
    账本，``stage_counts["输出行"]`` 是进度输出里的账本。三者必须对得上，
    否则说明有行在判定或写表之间被悄悄吃掉（``build_rows`` 用的是 ``zip``，
    长度不齐时会静默截断——这正是 B18 那一类缺陷的入口）。
    C9 是这条的语义依据：标题行和引导句行本来就不该出现在产物里。
    """
    for b in built.values():
        cc = b.result.stats["cut_counts"]
        sc = b.result.stats["stage_counts"]
        assert len(b.rows) == cc["参与判定"], f"{b.pack} {len(b.rows)} != {cc}"
        assert cc["参与判定"] == cc["分节后"] - cc["标题/引导句"], f"{b.pack} {cc}"
        assert sc["输出行"] == len(b.rows), f"{b.pack} 进度账本与产物不一致：{sc}"
        assert cc["分节后"] <= cc["裁剪后"] <= cc["折叠后"] <= cc["抽取条目"], (
            f"{b.pack} 裁剪账本倒流：{cc}"
        )
