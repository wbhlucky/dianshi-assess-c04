# -*- coding: utf-8 -*-
"""集成层：C 族版面（``N-科室-设备名技术参数`` 分块 + 正文流水行，P5）。

C 族与 A/B 族的根本差别是**参数区没有表格**：一行一条正文流水，块的边界靠
``5-耳鼻喉-红外辅照治疗仪技术参数`` 这种「序号-科室-设备名」抬头识别。
这带来三个 A/B 族根本不存在的失效面，本文件四条用例各盯一个：

1. X5  块抬头里的设备名是招标方自己打错的（``辅`` vs ``辐``），只能信结构化字段。
2. X8  C 族用半角 ``*`` 而不是全角 ``★`` 标关键项（``*1.2 输出功率：≥750w``）。
3. X12 源文件本身跳号（标的1 缺 16、18），不能当缺漏去补全。
4. X11 售后服务尾巴里粘着一个孤儿子项 ``14.7``，必须跟着父项一起被裁掉。

四条都跑「真 PDF 抽取 + 真资料包端到端输出」两层，因为 C 族的失效一半发生在
抽取期（块识别、标记归一），一半发生在裁剪期（标的范围、售后服务尾巴）。
"""
from __future__ import annotations

import json
from typing import NamedTuple

import pytest

from bidtable.config import MARK_STAR
from bidtable.ingest import pack as pk
from bidtable.ingest import tender_layout_c as tc
from bidtable.ingest import tender_pdf as tp
from bidtable.models import Scope

pytestmark = [pytest.mark.integration]


class P5(NamedTuple):
    """C 族抽取的中间产物（真 P5 招标 PDF）。"""

    reqs: tuple
    blocks: tuple
    scopes: tuple


@pytest.fixture(scope="module")
def p5(repo_root, fixtures_dir) -> P5:
    """真 P5：13 个块、85 条原始要求、4 个声明标的（1/5/8/12）。"""
    root = repo_root / "训练集" / "P5"
    pdf = pk.discover(root).tender_pdf if root.is_dir() else None
    if pdf is None or not pdf.is_file():
        pytest.skip("缺少 P5 的招标 PDF，C 族集成测试无法进行")
    raw = json.loads((fixtures_dir / "packs" / "P5" / "scope.json").read_text(encoding="utf-8"))
    scopes = [Scope.make(s["lot_no"], s["device_name"], s.get("quantity", "")) for s in raw["scopes"]]
    t = tp.locate(pdf)
    try:
        assert t.layout == "C"
        blocks, block_warns = tc.split_blocks(t)
        reqs, req_warns = tc.extract(t, scopes)
    finally:
        t.close()
    assert block_warns == [] and req_warns == []
    assert len(blocks) == 13 and len(reqs) == 85
    return P5(tuple(reqs), tuple(blocks), tuple(scopes))


@pytest.mark.ec("B3")
@pytest.mark.case("X5")
def test_x5_the_block_title_typo_never_reaches_the_device_name_column(p5, offline_rows):
    """块抬头写的是 ``红外辅照治疗仪``（错字），输出的 ``设备名称`` 必须是 ``红外辐照治疗装置``。

    边界在哪：C 族一个块里有**三个**候选设备名，优先级是 ``标的声明 > 块内表头 > 块抬头``。
    最省事的实现是直接拿块抬头（它就在同一行里，最好切），而 P5 标的5 的抬头恰好是
    招标方打错的「辅照治疗仪」——与结构化字段「辐照治疗装置」差一个字。
    一字之差不会让 grade.py 判错（它只对齐 ``(标的,分节,序号)``），但会让复核人在
    xlsx 里看到一个采购清单上根本不存在的设备名，这是产品感的硬伤。

    与 agent.md 的偏差：无。X5 的锚点（P5 标的5，辅 vs 辐）与实测完全一致。
    """
    b5 = next(b for b in p5.blocks if b.lot_no == "5")
    assert b5.title == "5-耳鼻喉-红外辅照治疗仪技术参数"
    assert "辅" in b5.title_device and "辐" not in b5.title_device, "块抬头确实带着错字"
    assert b5.header_device == "红外辐照治疗装置"

    scope5 = next(s for s in p5.scopes if s.lot_no == "5")
    assert scope5.device_name == "红外辐照治疗装置"
    assert tc.device_name(b5, scope5) == "红外辐照治疗装置"
    # 没有标的声明时，块内表头仍然压过块抬头——两级兜底都不能落到错字上。
    assert tc.device_name(b5, None) == "红外辐照治疗装置"

    rows = [r for r in offline_rows("P5") if r.lot_no == "5"]
    assert rows, "标的5 必须真的出现在输出里"
    assert {r.device_name for r in rows} == {"红外辐照治疗装置"}
    assert not any("辅" in r.device_name for r in rows)


@pytest.mark.ec("F2")
@pytest.mark.case("X8")
def test_x8_half_width_star_is_normalised_and_only_one_survives_the_cut(p5, offline_rows):
    """半角 ``*`` 归一成 ``★``；但 5 个原始关键项里只有 1 个能活到输出。

    边界在哪：P5 全文用半角 ``*`` 而不是 ``★``（``*1.2 输出功率：≥750w``），
    且写法不统一——``*1.2``（无空格）、``* 1.1、``（有空格）、``*16、``（顿号）三种都有。
    漏掉任何一种，``标记`` 列就是空的，grade.py 的 ``_parse_star`` 读的是**输出的标记列**，
    关键项风险识别会当场失效。

    第二个边界更容易被忽略：抽取期一共 5 个 ``★``，输出期只剩 1 个。
    另外 4 个（标的1 的 21.1/21.4/21.7、标的12 的 16）全长在售后服务尾巴上，
    被标的范围裁剪一并扫掉。也就是说「标记归一」与「尾巴裁剪」必须都正确，
    输出里才会正好剩 P5 唯一的那个真关键项。
    """
    raw_marked = {(r.lot_no, r.no): r.mark for r in p5.reqs if r.mark}
    assert raw_marked == {
        ("1", "21.1"): MARK_STAR, ("1", "21.4"): MARK_STAR, ("1", "21.7"): MARK_STAR,
        ("5", "1.2"): MARK_STAR, ("12", "16"): MARK_STAR,
    }
    assert not any("*" in r.no for r in p5.reqs), "半角 * 必须从序号里剥干净"
    for r in p5.reqs:
        if r.mark:
            assert not r.text.lstrip().startswith("*")
    r12 = next(r for r in p5.reqs if (r.lot_no, r.no) == ("5", "1.2"))
    assert r12.text == "输出功率：≥750w"

    out = offline_rows("P5")
    row = next(r for r in out if (r.lot_no, r.no) == ("5", "1.2"))
    assert (row.mark, row.section) == (MARK_STAR, "1、治疗源")
    assert row.req_text == "输出功率：≥750w"
    assert {r.mark for r in out} == {"", MARK_STAR}
    assert sum(1 for r in out if r.mark == MARK_STAR) == 1, "另外 4 个 ★ 都在售后服务尾巴上"


@pytest.mark.ec("H11")
@pytest.mark.case("X12")
def test_x12_a_gap_in_the_source_numbering_is_never_backfilled(p5, offline_rows, fixtures_dir):
    """标的1 源文件本身缺 16、18，输出必须原样跳号，不能补。

    边界在哪：「序号连续」是个很诱人的健康检查，但招标文件的编号是人手打的，
    P5 标的1 就是 1..15、17、19、20（缺 16、18），子项还缺 21.5。
    一旦实现自作主张重新编号或补全，对齐键 ``(标的,分节,序号)`` 会与 gold 整体错位，
    覆盖率不是掉几条而是掉一整段——这是 C 族最贵的一类 bug。

    本用例把「源 PDF / 我们的输出 / gold」三方摆在一起对照，跳号在三方一致，
    才证明它是源文件的属性而不是我们丢行。
    """
    raw1 = [r.no for r in p5.reqs if r.lot_no == "1"]
    assert raw1[:18] == [str(i) for i in range(1, 16)] + ["17", "19", "20"]
    assert {"16", "18", "21.5"} & set(raw1) == set(), "源文件本来就跳号"

    out1 = [r.no for r in offline_rows("P5") if r.lot_no == "1"]
    assert out1 == [str(i) for i in range(1, 16)] + ["17", "19", "20"]
    assert len(set(out1)) == len(out1), "序号不许重复"
    assert [int(n) for n in out1] == sorted(int(n) for n in out1), "也不许被重排"

    gold = json.loads((fixtures_dir / "gold" / "P5.json").read_text(encoding="utf-8"))
    gold1 = {v["no"] for v in gold.values() if v["lot"] == "1"}
    assert {"16", "18"} & gold1 == set(), "gold 同样没有 16、18，说明这不是漏抽"
    assert set(out1) == gold1


@pytest.mark.ec("H10")
@pytest.mark.case("X11")
def test_x11_an_orphan_sub_item_inside_the_service_tail_is_swept_with_it(p5, offline_rows, fixtures_dir):
    """孤儿子项 ``14.7`` 粘在 ``13.6`` 的正文里，随售后服务尾巴一起被裁掉。

    边界在哪：P5 标的8 的参数区止于 ``12``，之后是 ``13 售后服务要求`` 加 13.1~13.6。
    源 PDF 在 13.6 那一行少了一个换行，于是 ``14.7 提供操作手册和维修手册`` 被**粘进**
    13.6 的正文，从来没有成为独立一行。裁剪必须按「售后服务标题之后全裁」的语义走，
    而不是按序号连续性走，否则这半句会作为一条幽灵要求留在输出里。

    与 agent.md 的偏差：X11 原文写的是「保留 ``14.7 提供操作手册和维修手册`` ✗」，
    暗示它曾是一条独立的行。实测它并不是行，而是 13.6 正文的后半截；
    真正被裁掉的是整个 13.x 尾巴。本用例因此额外拿标的12 的 ``17.7`` 做对照——
    那一句在标的12 里**是**独立成行的，同样被裁掉。粘连态与独立态都覆盖，才算钉死。
    """
    raw8 = [r for r in p5.reqs if r.lot_no == "8"]
    assert [r.no for r in raw8] == [str(i) for i in range(1, 13)] + [
        "13", "13.1", "13.2", "13.3", "13.4", "13.5", "13.6",
    ]
    assert not any(r.no == "14.7" for r in p5.reqs), "孤儿从来没有独立成行"
    assert next(r for r in raw8 if r.no == "13.6").text.endswith("14.7 提供操作手册和维修手册")

    out = offline_rows("P5")
    out8 = [r.no for r in out if r.lot_no == "8"]
    assert out8 == [str(i) for i in range(1, 13)], "13 及其全部子项一并裁掉"
    assert not any("操作手册" in r.req_text or "14.7" in r.req_text for r in out)

    # 对照：标的12 的同一句话是独立的 17.7 行，同样被尾巴裁掉。
    assert any((r.lot_no, r.no) == ("12", "17.7") for r in p5.reqs)
    assert [r.no for r in out if r.lot_no == "12"] == [str(i) for i in range(1, 16)]

    gold = json.loads((fixtures_dir / "gold" / "P5.json").read_text(encoding="utf-8"))
    assert {v["no"] for v in gold.values() if v["lot"] == "8"} == {str(i) for i in range(1, 13)}
