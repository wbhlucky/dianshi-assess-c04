# -*- coding: utf-8 -*-
"""集成层：B 族版面（``序号 | 技术要求`` 双列表，``★`` 追加在序号列里，P4）。

B 族与 A 族的根本差别是**标记的存放位置**：A 族把 ``▲2.`` 整串塞在参数大单元格里，
B 族把 ``1.7★`` 塞在 ``序号`` 单元格里。两种形态都必须抽出干净的 ``序号``
与独立的 ``标记`` 列，但只有 B 族能让 ``招标要求（原文）`` 也保持干净——
A 族的原文列按契约是逐字回显，``▲`` 本来就在里面。这个不对称正是 B16 要钉的东西。

1. T5  ``1.7★`` -> ``序号 == "1.7"``、``标记 == "★"``、原文不含 ``★``。
2. B16 ``★`` 在序号列（P4）与 ``▲`` 在原文里（P1）两种形态并存时的对照。
3. X32 gold 标 ``不判分`` 的 ``2.1`` 行：按策略保留为 ``未响应``，接受 ``多余 == 1``。
"""
from __future__ import annotations

import json

import pytest

from bidtable.config import DEV_NO_RESPONSE, MARK_STAR, MARK_TRIANGLE
from bidtable.ingest import pack as pk
from bidtable.ingest import tender_layout_b as tb
from bidtable.ingest import tender_pdf as tp

pytestmark = [pytest.mark.integration]

MARK_CHARS = (MARK_STAR, MARK_TRIANGLE, "*")


@pytest.fixture(scope="module")
def p4_raw_rows(repo_root):
    """真 P4 招标 PDF 的 B 族原始行（标记尚未从序号里剥出来之前的形态在 ``no_cell``）。"""
    root = repo_root / "训练集" / "P4"
    pdf = pk.discover(root).tender_pdf if root.is_dir() else None
    if pdf is None or not pdf.is_file():
        pytest.skip("缺少 P4 的招标 PDF，B 族集成测试无法进行")
    t = tp.locate(pdf)
    try:
        rows, warns = tb.extract_rows(t)
        assert t.layout == "B"
        assert warns == []
    finally:
        t.close()
    return rows


@pytest.mark.ec("F3")
@pytest.mark.case("T5")
def test_t5_p4_star_glued_to_the_sequence_number_is_split_apart(p4_raw_rows, offline_rows):
    """``1.7★`` 被拆成干净的 ``1.7`` 与独立的 ``★``，且原文里不留标记。

    边界在哪：B 族的 ``★`` 与序号之间**没有分隔符**，靠的是标记词表而不是正则切分。
    如果直接把整个单元格当序号，对齐键就成了 ``1.7★``，与 gold 的 ``1.7`` 永远对不上，
    覆盖率会当场掉一档；更隐蔽的是 ``grade.py`` 的 ``_parse_star`` 读的是**输出的标记列**，
    标记没搬过去，关键项风险识别也会跟着失效。

    P4 一共 4 行带标记（``1.7 / 1.9.1 / 1.9.2 / 1.16``），其中 ``1.9.1``~``1.9.6``
    是三级序号，所以这条用例顺带证明「剥离标记」与「识别多级编号」互不干扰。
    """
    raw = {(r.page, r.no): r for r in p4_raw_rows}
    assert len(p4_raw_rows) == 39

    r = raw[(23, "1.7")]
    assert r.no_cell == "1.7★", "源单元格确实是把 ★ 直接粘在序号后面的"
    assert (r.no, r.mark) == ("1.7", MARK_STAR)
    assert r.text == "输出方式：连续冷凝蒸汽治疗"
    assert not any(ch in r.text for ch in MARK_CHARS)

    assert sorted(r.no for r in p4_raw_rows if r.mark) == ["1.16", "1.7", "1.9.1", "1.9.2"]
    assert all(r.mark == MARK_STAR for r in p4_raw_rows if r.mark)
    # 三级序号在剥掉标记之后仍然完整。
    assert [r.no for r in p4_raw_rows if r.no.startswith("1.9.")] == [
        "1.9.1", "1.9.2", "1.9.3", "1.9.4", "1.9.5", "1.9.6",
    ]

    # 端到端：这些性质一路保持到输出行。
    out = {(row.lot_no, row.no): row for row in offline_rows("P4")}
    assert (out[("1", "1.7")].mark, out[("1", "1.7")].req_text) == (
        MARK_STAR, "输出方式：连续冷凝蒸汽治疗",
    )
    assert not any(ch in out[("1", "1.9.1")].req_text for ch in MARK_CHARS)
    assert all(not any(ch in row.no for ch in MARK_CHARS) for row in out.values())


@pytest.mark.ec("F3")
@pytest.mark.ec("F4")
@pytest.mark.case("B16")
def test_b16_star_in_the_number_column_and_triangle_in_the_text_coexist(offline_rows):
    """两种标记存放形态并存时，``序号`` 都干净、``标记`` 都到位，而原文列**不该**被洗。

    边界在哪：这是一条很容易写错的断言。直觉上会想「输出的原文里不该再有标记字符」，
    但 ``招标要求（原文）`` 按契约是**逐字回显**招标文件的单元格内容——
    A 族的 ``▲`` 本来就长在参数大单元格里（``▲2.最大单脉冲能量5J。…``），
    把它从原文里抹掉等于篡改招标原文，反而会让归一化后的文本与 gold 对不上。
    所以正确的对照是：

    * P4（B 族）：标记原本在 ``序号`` 列，原文从头到尾就没有标记字符 -> 输出原文干净；
    * P1（A 族）：标记原本粘在原文的前缀里 -> 输出原文保留 ``▲2.``，
      且**只有** ``序号 2`` 与 ``序号 6`` 两行如此。

    两种情形下 ``序号`` 都必须是裸的、``标记`` 都必须非空，这才是 B16 的实质。
    """
    p1 = offline_rows("P1")
    p4 = offline_rows("P4")

    p1_marked = {r.no for r in p1 if any(ch in r.req_text for ch in MARK_CHARS)}
    p4_marked = {r.no for r in p4 if any(ch in r.req_text for ch in MARK_CHARS)}
    assert p1_marked == {"2", "6"}
    assert p4_marked == set()

    for row in p1:
        # 不能按 ``row.no in p1_marked`` 过滤：P1 标的2 也有一行 ``序号 == "2"``
        # 而标记为空。必须用「原文里含标记字符」这同一个判据回头筛，才不会误伤。
        if any(ch in row.req_text for ch in MARK_CHARS):
            assert row.mark == MARK_TRIANGLE
            assert row.req_text.startswith(MARK_TRIANGLE + row.no + ".")
    for row in p4:
        assert not any(ch in row.no for ch in MARK_CHARS)
    assert {r.no for r in p4 if r.mark} == {"1.7", "1.9.1", "1.9.2", "1.16"}
    assert {r.mark for r in p4 if r.mark} == {MARK_STAR}
    assert {r.mark for r in p1 if r.mark} == {MARK_TRIANGLE}

    # 两个包各自都只有「空」与一种标记，绝不混用。
    assert {r.mark for r in p1} == {"", MARK_TRIANGLE}
    assert {r.mark for r in p4} == {"", MARK_STAR}


@pytest.mark.ec("E5")
@pytest.mark.case("X32")
def test_x32_a_not_scored_gold_row_is_kept_and_costs_exactly_one_extra(offline_rows, fixtures_dir):
    """gold 把 P4 ``2.1`` 标成 ``不判分``；我们按策略保留它，代价是 ``多余条目 == 1``。

    边界在哪：``不判分`` 是 gold 独有的第五种取值，输出的 ``偏离类型`` 词表
    （``config.DEVIATIONS``）里**没有**它，所以任何实现都必须替它挑一个替身。
    两条路都合法：删掉这行（覆盖率不受影响，因为 ``grade.py`` 本来就把它剔出分母），
    或者保留成 ``未响应``（多出一条「多余条目」）。
    本项目选保留，理由是 ``2.1 主机系统内配置，具备功能有：`` 在招标文件里
    是一条真实存在的引导句，删掉它会让复核人在 xlsx 里对不上原文的行号；
    而 ``多余条目`` 的门禁是 ``<= 2``，一个包多 1 条完全在预算内。

    反面教材正是 X32 说的那种做法：把 gold 里没有的行当成「我们漏抽了」再补一条，
    那才会把多余条目推到 2 条以上。
    """
    gold = json.loads((fixtures_dir / "gold" / "P4.json").read_text(encoding="utf-8"))
    assert len(gold) == 37
    not_scored = {k: v for k, v in gold.items() if v["dev"] == "不判分"}
    assert list(not_scored) == ["1||2.1"]
    assert not_scored["1||2.1"]["req"] == "主机系统内配置，具备功能有："

    rows = offline_rows("P4")
    assert len(rows) == 37
    row = next(r for r in rows if (r.lot_no, r.no) == ("1", "2.1"))
    assert row.deviation == DEV_NO_RESPONSE
    # 未响应行的三列必须全空（I5），否则宽松命中会把它误判成有材料。
    assert (row.resp_text, row.evidence_file, row.location) == ("", "", "")

    scored = {k for k, v in gold.items() if v["dev"] != "不判分"}
    ours = {(r.lot_no, r.section, r.no) for r in rows}
    # gold 的键是 "标的||序号" 两段，三元组要从值里的 lot/sec/no 取。
    gold_keys = {(v["lot"], v["sec"], v["no"]) for v in gold.values() if v["dev"] != "不判分"}
    assert ours - gold_keys == {("1", "", "2.1")}, "多出来的应当正好是那一条不判分行"
    assert gold_keys - ours == set(), "覆盖率必须是满的，一条都不能少"
    assert len(scored) == 36
