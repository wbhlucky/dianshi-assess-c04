# -*- coding: utf-8 -*-
"""P2 的金标准对照：``（N）`` 子项的**拆分**分支（与 P1 的折叠分支互为对照）。

P2 独有的一件事：它的服务标的被拆成 ``1 / 2.1 / 2.2 / 2.3`` 四行，
而 P1 的服务标的被折叠成 ``1 / 2`` 两行。两者走的是同一段代码
（``ingest/fold.py``），差别只在**厂商文档怎么应答**：

* P1 的 ``YF_服务方案.docx`` 项目列写的是父项「设备运行维护」→ 整体应答 → 折叠；
* P2 的 ``BT_服务方案.docx`` 项目列写的是子项「日常巡检维护」→ 逐项应答 → 拆分。

这是 agent.md 3.5 原本没写清楚、施工中才发现的规则（原文只说 ``auto``
按"服务标的是否逐项应答"决定，没给判据）。本文件把判据的**两种输出**
同时钉住：同一条代码路径、两种证据形态、两种行数。
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.golden]


@pytest.mark.ec("B2")
@pytest.mark.ec("H9")
def test_p2_splits_the_service_lot_because_the_vendor_answered_item_by_item(score_pack):
    """P2 服务标的拆成 4 行，而 P1 折叠成 2 行 —— 同一段代码的两种输出。

    边界在哪：``（N）`` 子项折叠是个**二选一**决策，选错不会报错，
    只会让行数对不上标准答案（覆盖率直接掉）。所以边界不在"会不会崩"，
    而在"判据是否真的读到了厂商证据的形状"。把 P1 与 P2 放在同一条测试里
    对照，才能证明它不是被硬编码成某一种——单独测 P2 的话，
    一个"永远拆分"的实现也能过。

    另一层边界是 ``fold.resolve`` 的服务标的守卫：只有当资料包里
    **确实声明了**服务标的时才允许折叠。P1/P2 都声明了标的2 为
    「设备运维技术服务」，因此走的是正常分支，而不是"空集合陷阱"那条
    （``scopes`` 为空时会把所有标的的 ``（N）`` 子项都折叠，已在 README 记录）。

    真实资料包实例：``训练集/P2``（视网膜脉络膜造影仪 + 服务），25 行；
    对照包 ``训练集/P1``，17 行。

    与 agent.md 的偏差：3.5 只描述了 ``auto`` 的意图，没给判据；
    实现落在"证据名命中父项标题且不命中任何子项标题 → 折叠，否则拆分"。
    """
    p2, score2 = score_pack("P2")
    p1, _ = score_pack("P1")

    # P2：拆分。父项 1 保留，（1）（2）（3）三个子项各自成行。
    lot2_p2 = [r.no for r in p2.rows if r.lot_no == "2"]
    assert lot2_p2 == ["1", "2.1", "2.2", "2.3"]
    assert len(p2.rows) == 25

    # P1：折叠。同样有 3 个（N）子项，但厂商按父项整体应答，故只留 2 行。
    assert [r.no for r in p1.rows if r.lot_no == "2"] == ["1", "2"]
    # 折叠必须在"跳过清单"里留痕，不能悄悄少行。
    assert len(p1.skipped) == 1
    assert "已折叠成一行" in p1.skipped[0]
    # 而 P2 拆分了，就没有这条折叠记录。
    assert not any("已折叠成一行" in s for s in p2.skipped)

    # 两个包的设备标的都不受影响：折叠/拆分只动服务标的。
    assert len({r.lot_no for r in p2.rows}) == 2
    assert len({r.lot_no for r in p1.rows}) == 2
    # 折叠只碰服务标的：P1 标的1 的 15 行一条不少、序号连续。
    assert [r.no for r in p1.rows if r.lot_no == "1"] == [str(i) for i in range(1, 16)]


@pytest.mark.ec("E3")
def test_p2_scores_full_coverage_with_exactly_two_negative_deviation_misses(score_pack, gold_map):
    """P2：覆盖率 1.000，偏离 0.920，两格失误都出在"负偏离被我们看轻了"。

    边界在哪：P2 的混淆矩阵里，非对角格**只有负偏离出发**的两格
    （``负偏离->无偏离`` 与 ``负偏离->未响应`` 各 1）。这不是巧合，
    它暴露了判定层的一个结构性偏保守：负偏离依赖封闭拒答词表
    （``config.NEGATIVE_PHRASES``）整句命中，厂商换个说法就会漏。
    把"失误全部集中在负偏离"钉住，比只钉一个 0.920 的数字有用得多——
    数字涨了跌了都可能是噪声，但**失误的方向**变了就一定是逻辑变了。

    真实资料包实例：``训练集/P2``，25 条全部命中标准答案，多余 0。

    与 agent.md 的偏差：无。
    """
    result, score = score_pack("P2")

    assert score["条目总数"] == 25
    assert score["提交匹配条目"] == 25
    assert score["多余条目"] == 0
    assert score["覆盖率"] == pytest.approx(1.0)
    assert score["偏离类型准确率"] == pytest.approx(23 / 25, abs=1e-3)
    assert score["关键项风险识别"].startswith("1/1")

    matrix = score["混淆矩阵"]
    assert sum(matrix.values()) == 25
    off = {k: v for k, v in matrix.items() if k.split("->")[0] != k.split("->")[1]}
    assert off == {"负偏离->无偏离": 1, "负偏离->未响应": 1}
    # 失误全部源自"负偏离"：判定层对拒答式表达的召回是已知短板。
    assert {k.split("->")[0] for k in off} == {"负偏离"}

    gold_keys = set(gold_map("P2"))
    assert {(r.lot_no, r.section, r.no) for r in result.rows} == gold_keys
