# -*- coding: utf-8 -*-
"""P4 的金标准对照：B 族版面（两列表 + ★ 追加在序号格里），也是唯一偏离全对的包。

P4（热蒸汽治疗设备，国际招标，37 行）在五个包里独有三件事：

1. 它是唯一的 **B 族版面**：招标参数表只有「序号 | 技术要求」两列，
   没有 A 族那种跨页大单元格，也没有 C 族那种"科室-设备名"标题块。
   于是两件事跟着变：分节列**恒为空**（B 族的表里没有层级标题可用），
   设备名只能从文档级兜底取，因此 37 行的 ``设备名称`` 完全一致。
2. 它是唯一把 ★ **写在序号单元格里**的包（原文是 ``1.7★``）。
   A 族的标记是写在正文开头的（``▲2.最大单脉冲能量5J。``，原文原样保留），
   B 族必须把 ★ 从序号里劈出来：序号回到干净的 ``1.7``，标记进第 5 列。
   这两种处理方式相反，所以 A/B 两族必须各自钉一次（A 族见 test_tender_layout_a）。
3. 它是唯一**偏离类型全对**（1.000，混淆矩阵非对角为空）的包，
   同时也是唯一 ``多余条目 == 1`` 的包 —— 而那一行恰恰是标准答案里
   标了 ``不判分`` 的行（E5）。这两件事放在一起，才是 P4 真正的边界。

与 agent.md 的偏差：无。7.1 的 E5（不判分）在这里第一次被真正触发。
"""

from __future__ import annotations

import pytest

from bidtable import config, selfcheck as selfcheck_mod

pytestmark = [pytest.mark.golden]

PACK = "P4"
N_ROWS = 37          # 我们输出的行数
N_SCORED = 36        # grade.py 计入分母的行数（少一行"不判分"）
_MARKED_NOS = {"1.7", "1.9.1", "1.9.2", "1.16"}
_THREE_LEVEL = {"1.9.1", "1.9.2", "1.9.3", "1.9.4", "1.9.5", "1.9.6"}
_DROPPED_HEADINGS = ("主机系统要求", "热蒸汽治疗系统专用软件要求")
_EXTRA_KEY = ("1", "", "2.1")


@pytest.mark.ec("A2")
@pytest.mark.ec("H5")
@pytest.mark.ec("H7")
@pytest.mark.ec("F3")
def test_p4_splits_the_star_out_of_the_number_cell_and_keeps_three_level_numbering(offline_run):
    """B 族：37 行、分节恒空、设备名恒定、4 个 ★ 全部从序号格里劈出来、三级序号完整。

    边界在哪：``1.7★`` 这一格同时藏着三种信息（序号 / 标记 / 正文的起点），
    劈错任何一刀都不会报错，只会让对齐键或标记列悄悄错位。
    所以这里把三刀分别钉住：
    序号必须是干净的 ``1.7``（不能带 ★，也不能因为 ★ 而被当成非法序号丢掉）、
    标记列必须拿到 ``★``（不能留在正文里）、
    正文必须不含 ★（不能两边都留一份）。

    第二条边界是三级序号 ``1.9.1``~``1.9.6``：它们必须与父项 ``1.9`` **共存**，
    既不能被当成 ``1.9`` 的重复而合并，也不能被 ``numbering`` 误读成新的顶层项。
    P4 能跑通还有个前提——一个 ``split_items`` 块不能以二级号开头，
    这里靠 ``1.1`` 那行先把 ``last_top`` 引导起来（已在 test_numbering 里单独证明）。

    第三条边界是 B 族"没有层级可用"的后果：分节恒为空、设备名恒定。
    这不是偷懒，是版面事实；把它钉住可以防止有人给 B 族硬套 A 族的分节策略。

    真实资料包实例：``训练集/P4``（热蒸汽泊疗设备国际招标采购项目，1 个标的）。

    与 agent.md 的偏差：无。
    """
    result = offline_run(PACK)
    rows = result.rows
    assert result.stats["layout"] == config.LAYOUT_B
    assert len(rows) == N_ROWS

    # B 族：单标的、分节恒空、设备名来自文档级兜底因此全表一致。
    assert {r.lot_no for r in rows} == {"1"}
    assert {r.section for r in rows} == {""}
    assert {r.device_name for r in rows} == {"热蒸汽治疗设备"}

    # 标记：正好 4 行有标记，全部是 ★，且 P4 一个 ▲ 都没有（与 P1/P2/P3 相反）。
    marked = {r.no for r in rows if r.mark}
    assert marked == _MARKED_NOS
    assert {r.mark for r in rows} == {"", config.MARK_STAR}
    assert config.MARK_TRIANGLE not in {r.mark for r in rows}
    # 三刀都干净：序号里没有 ★，正文里也没有 ★ / ▲。
    assert all(config.MARK_STAR not in r.no for r in rows)
    assert not any(config.MARK_STAR in r.req_text for r in rows)
    assert not any(config.MARK_TRIANGLE in r.req_text for r in rows)

    # 三级序号与父项共存。
    assert {r.no for r in rows if r.no.count(".") >= 2} == _THREE_LEVEL
    assert "1.9" in {r.no for r in rows}
    nos = [r.no for r in rows]
    assert len(set(nos)) == N_ROWS  # 键唯一（I3 在 P4 上的实例）
    assert nos == sorted(nos, key=lambda s: [int(p) for p in s.split(".")])

    # 39 条抽取里少的 2 条是纯标题行，它们不许出现在成品里。
    assert result.stats["cut_counts"] == {
        "抽取条目": 39,
        "折叠后": 39,
        "裁剪后": 39,
        "分节后": 39,
        "标题/引导句": 2,
        "参与判定": N_ROWS,
    }
    assert {r.req_text for r in rows}.isdisjoint(_DROPPED_HEADINGS)
    assert result.skipped == []


@pytest.mark.ec("E5")
@pytest.mark.ec("E1")
@pytest.mark.ec("E4")
def test_p4_scores_a_perfect_deviation_column_and_its_one_extra_row_is_the_unscored_one(
    score_pack, gold_map
):
    """P4：偏离类型 1.000、混淆矩阵非对角为空，多出来的那一行正是标准答案的"不判分"行。

    边界在哪：这是 E5（不判分）唯一被真正触发的地方，而它的表现很反直觉——
    标准答案的 xlsx 里**有**这一行（``标的1 / 序号2.1``，一句引导话
    "主机系统内配置，具备功能有："），但它的偏离类型写的是 ``不判分``，
    于是 ``grade.py`` 把它从分母里剔掉（条目总数 36），
    同时把我们输出的它记成 ``多余条目 1``。

    所以"多余 1"不是误报，而是**我们选择照着招标原文把引导句也输出一行**的代价。
    这个选择是有意的：压掉它需要判断"哪一句是引导话"，
    判错就会丢真条目，覆盖率直接掉；而多余条目的门禁是 ``≤2``（不计分），
    留一行的代价远小于丢一行的风险。这条测试把这个权衡钉成可执行的事实：
    多出来的键**恰好**是 ``('1','','2.1')``，且标准答案的键一个不少。

    另一半边界是"满分"本身：混淆矩阵非对角必须**为空字典**（不是"很小"），
    36 格全在对角线上。P4 是唯一做到的包，因此它是判定层的阳性对照——
    一旦这里出现任何一格非对角，说明改动伤到了最基本的那条路径，
    而不是只在 P3 那种难包上退化。

    真实资料包实例：``训练集/P4``，37 行输出 / 36 行计分 / 多余 1。

    与 agent.md 的偏差：无。
    """
    result, score = score_pack(PACK)

    assert score["条目总数"] == N_SCORED
    assert score["提交匹配条目"] == N_SCORED
    assert score["多余条目"] == 1
    assert score["多余条目"] <= selfcheck_mod.MAX_EXTRA  # 门禁是 ≤2，不计分
    assert score["覆盖率"] == pytest.approx(1.0)
    assert score["偏离类型准确率"] == pytest.approx(1.0)
    assert score["证明材料命中率"] == pytest.approx(1.0)
    assert score["位置命中率"] == pytest.approx(1.0)
    # P4 的 ★ 行（序号 1.9.1）我们判未响应，标准答案也是未响应 → 关键项风险命中。
    # 这与 P5 的 0/1 形成对照：同样是 ★ + 检测报告指针，P4 的答案不需要实测值。
    assert score["关键项风险识别"].startswith("1/1")

    matrix = score["混淆矩阵"]
    off = {k: v for k, v in matrix.items() if k.split("->")[0] != k.split("->")[1]}
    assert off == {}
    assert sum(matrix.values()) == N_SCORED
    assert matrix == {"无偏离->无偏离": 31, "未响应->未响应": 1, "正偏离->正偏离": 1, "负偏离->负偏离": 3}

    gold_keys = set(gold_map(PACK))
    ours = {(r.lot_no, r.section, r.no) for r in result.rows}
    assert len(gold_keys) == N_SCORED
    assert ours - gold_keys == {_EXTRA_KEY}
    assert gold_keys - ours == set()

    extra = [r for r in result.rows if (r.lot_no, r.section, r.no) == _EXTRA_KEY]
    assert len(extra) == 1
    row = extra[0]
    # 引导句没有可比的响应，因此落在"未响应 + 三列全空"这一档。
    assert row.req_text == "主机系统内配置，具备功能有："
    assert row.deviation == config.DEV_NO_RESPONSE
    assert (row.resp_text, row.evidence_file, row.location) == ("", "", "")
