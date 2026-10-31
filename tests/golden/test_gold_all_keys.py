# -*- coding: utf-8 -*-
"""五个包合起来看：205 个键一次性对齐，以及 agent.md 10.1 那六道门禁的实测结果。

前面五个文件各盯一个包的独有形状。本文件盯的是**只有合起来才存在**的性质：

1. 205 条要求（P1 17 + P2 25 + P3 71 + P4 36 + P5 56）的 ``(包,标的,分节,序号)``
   四元组必须两两不同，且与标准答案的键集合完全一致 —— 覆盖率 1.000 的**结构性原因**。
   单看任何一个包都证明不了"没有跨包串键"。
2. 总混淆矩阵必须**恰好等于**五个包的矩阵逐格相加。这条断言证明
   ``selfcheck.aggregate`` 不是另算一遍，而是真的在汇总；
   顺带把 9 个非零格的数值钉死（改判定逻辑就会红）。
3. 偏离类型准确率 0.927 要放在**基线**旁边才有意义：一个把所有行都填
   ``无偏离`` 的傻瓜实现能拿 145/205 = 0.707（因为标准答案里无偏离占 145 条）。
   基线不是拍出来的魔法数，而是从 ``gold_map`` 现场数出来的。
4. ``gate_lines`` 必须输出**恰好 6 行**，5 过 1 不过，
   且不过的那一行是 ``关键项风险识别 = 满分``（离线上限 4/5，见 test_gold_P5）。
   门禁的名字里带阈值与冲刺线，所以名字本身就是契约的一部分：
   有人偷偷把 0.85 改成 0.80，这里会红。

与 agent.md 的偏差：10.1 要求关键项风险识别 5/5，离线条件下上限是 4/5，
因此本文件断言的是"5 过 1 不过，且不过的正是这一项"，而不是"全过"。
"""

from __future__ import annotations

import pytest

from bidtable import selfcheck as selfcheck_mod

pytestmark = [pytest.mark.golden]

PACKS = ("P1", "P2", "P3", "P4", "P5")
N_PER_PACK = {"P1": 17, "P2": 25, "P3": 71, "P4": 36, "P5": 56}
N_TOTAL = 205

# 总分混淆矩阵（9 个非零格）。gold->ours。
_TOTAL_MATRIX = {
    "无偏离->无偏离": 139,
    "无偏离->未响应": 6,
    "未响应->未响应": 18,
    "正偏离->无偏离": 1,
    "正偏离->正偏离": 10,
    "负偏离->无偏离": 5,
    "负偏离->未响应": 2,
    "负偏离->正偏离": 1,
    "负偏离->负偏离": 23,
}
_GATE_NAMES = (
    "覆盖率 ≥0.95（每包 ≥0.90）",
    "偏离类型准确率 ≥0.85（冲刺 ≥0.92，已达）",
    "证明材料命中率 ≥0.9（冲刺 ≥0.95）",
    "位置命中率 ≥0.85（冲刺 ≥0.92，已达）",
    "关键项风险识别 = 满分",
    "每包多余条目 ≤2（不计分）",
)
_FAILING_GATE = "关键项风险识别 = 满分"


def _judged(matrix: dict[str, int]) -> int:
    """材料/位置命中率的分母：标准答案不是"未响应"的那些行。"""
    return sum(v for k, v in matrix.items() if k.split("->")[0] != "未响应")


@pytest.mark.ec("J3")
@pytest.mark.ec("J8")
@pytest.mark.ec("E5")
def test_205_keys_align_and_the_total_matrix_is_exactly_the_sum_of_the_five(score_pack, gold_map):
    """205 个键跨包唯一且与标准答案全等；总矩阵 == 五个包逐格相加；0.927 比基线高 0.2 以上。

    边界在哪：三处。

    第一处是**键的跨包唯一性**。每个包内部键唯一已经被 invariants 层证明过，
    但"加上包名之后 205 个四元组仍然两两不同"是另一件事——
    它排除了两个包共用同一份 fixture、或者 ``gold_map`` 读错文件这类串包错误。
    覆盖率 1.000 只有在这个前提成立时才是"一条不少"，否则可能是"错配也计分"。

    第二处是**汇总口径**。``aggregate`` 用了三个不同的分母：
    条目总数（覆盖率、偏离准确率）、矩阵对角线之和（偏离正确数）、
    ``judged`` = 标准答案非"未响应"的行数（材料/位置命中率，这里是 187）。
    三个分母混用是最容易写错的地方，所以这里不只贴结果，
    而是把"总矩阵 == 逐格相加"和"材料/位置 == 按 judged 加权"都重算一遍比对。

    第三处是**基线**。0.927 单独看没有意义；
    全填 ``无偏离`` 就有 0.707。基线从 ``gold_map`` 现场数出来，
    再与总矩阵的"标准答案侧"交叉验证一次（两条独立路径得到同一个 145）。

    真实资料包实例：``训练集`` 全部五个包，205 条要求。

    与 agent.md 的偏差：无。
    """
    scores: dict[str, dict] = {}
    results: dict[str, object] = {}
    for pack in PACKS:
        results[pack], scores[pack] = score_pack(pack)

    # --- 第一处：跨包键唯一，且与标准答案全等 --------------------------------
    # B21（见 test_gold_P3.py）：分节是五个包里唯一"两边写法不同"的字段——
    # 我们输出 section.SEP == " / " 并保留招标原文的空格，标准答案存的是无空格的裸 "/"。
    # 只有 P3 的 17 个分节非空，所以比键前必须两边都过 grade.norm（= 吃掉所有空白），
    # 否则 P3 的 71 个键会全部对不上。归一化在这里不丢信息：
    # P3 的 17 个分节归一化后仍是 17 个互不相同的值，"" 也还是 ""。
    _norm = selfcheck_mod.load_grade().norm
    ours: set[tuple[str, str, str, str]] = set()
    gold: set[tuple[str, str, str, str]] = set()
    for pack in PACKS:
        keys = {(r.lot_no, _norm(r.section), r.no) for r in results[pack].rows}
        gkeys = {(lot, _norm(sec), no) for lot, sec, no in gold_map(pack)}
        # P4 多一行"不判分"（E5），是唯一允许的多出来的键。
        assert keys - gkeys <= {("1", "", "2.1")}
        assert gkeys - keys == set()
        ours |= {(pack, *k) for k in keys}
        gold |= {(pack, *k) for k in gkeys}
    assert len(ours) == N_TOTAL + 1  # 205 条计分 + P4 那一行不判分
    assert len(gold) == N_TOTAL
    assert len(ours & gold) == N_TOTAL
    for pack in PACKS:
        assert scores[pack]["条目总数"] == N_PER_PACK[pack]

    # --- 第二处：总矩阵 == 逐格相加，且三个分母各自自洽 ----------------------
    total = selfcheck_mod.aggregate([scores[p] for p in PACKS])
    summed: dict[str, int] = {}
    for pack in PACKS:
        for k, v in scores[pack]["混淆矩阵"].items():
            summed[k] = summed.get(k, 0) + int(v)
    assert total["混淆矩阵"] == summed == _TOTAL_MATRIX
    assert sum(_TOTAL_MATRIX.values()) == N_TOTAL
    diag = sum(v for k, v in _TOTAL_MATRIX.items() if k.split("->")[0] == k.split("->")[1])
    assert diag == 190

    assert total["条目总数"] == N_TOTAL
    assert total["提交匹配条目"] == N_TOTAL
    assert total["多余条目"] == 1  # 全部来自 P4 的不判分行
    assert total["覆盖率"] == pytest.approx(1.0)
    assert total["偏离类型准确率"] == pytest.approx(0.927, abs=1e-3)
    assert total["偏离类型准确率"] == pytest.approx(diag / N_TOTAL, abs=1e-3)

    judged = _judged(_TOTAL_MATRIX)
    assert judged == 187  # 205 - 18 条标准答案"未响应"
    for metric, expected in (("证明材料命中率", 0.941), ("位置命中率", 0.931)):
        weighted = sum(
            float(scores[p][metric]) * _judged(scores[p]["混淆矩阵"]) for p in PACKS
        ) / judged
        assert total[metric] == pytest.approx(weighted, abs=2e-3)
        assert total[metric] == pytest.approx(expected, abs=1e-3)

    # --- 第三处：基线 --------------------------------------------------------
    baseline_hits = sum(
        1 for pack in PACKS for v in gold_map(pack).values() if v["dev"] == "无偏离"
    )
    assert baseline_hits == 145
    # 交叉验证：总矩阵的"标准答案侧"也数得出同一个 145。
    gold_side: dict[str, int] = {}
    for k, v in _TOTAL_MATRIX.items():
        g = k.split("->")[0]
        gold_side[g] = gold_side.get(g, 0) + v
    assert gold_side == {"无偏离": 145, "正偏离": 11, "负偏离": 31, "未响应": 18}
    baseline = baseline_hits / N_TOTAL
    assert baseline == pytest.approx(0.707, abs=1e-3)
    assert total["偏离类型准确率"] - baseline > 0.2

    # P4 是唯一偏离全对的包；P3 是唯一拖后腿的包（0.873）。
    devs = {p: float(scores[p]["偏离类型准确率"]) for p in PACKS}
    assert [p for p in PACKS if devs[p] == pytest.approx(1.0)] == ["P4"]
    assert min(devs, key=lambda p: devs[p]) == "P3"


@pytest.mark.ec("J8")
@pytest.mark.ec("J1")
def test_gate_lines_is_six_lines_with_five_passes_and_one_documented_failure(score_pack):
    """门禁输出恰好 6 行、5 过 1 不过；不过的是关键项风险；换成 P1 单包则 6 行全过。

    边界在哪：门禁的**描述文字里带着阈值**（``≥0.95``、``冲刺 ≥0.92``），
    所以它不只是布尔值，而是把 agent.md 10.1 的数字写进了输出契约。
    逐字断言这 6 个名字，等于给阈值上了一把锁：
    把 0.85 调成 0.80 会让断言红，把某一项悄悄删掉会让行数变成 5 而红。

    第二处边界是**失败必须被精确定位**。总分下唯一不过的是
    ``关键项风险识别 = 满分``（实测 4/5），其余 5 项全过。
    如果哪天变成 2 项不过，说明退化的不是 OCR 那一处，而是判定层整体。

    第三处边界是**失败的局部性**：把 P1 单独当"全部资料包"喂进去，
    6 行必须全过（P1 是关键项风险 1/1 的那个包）。
    这条断言把"4/5 是 P5 一个包造成的"从叙述变成了可执行事实——
    它同时也说明门禁函数本身没有把失败硬编码进去。
    注意此时第 3 行的名字会变成 ``…（冲刺 ≥0.95，已达）``，
    因为 P1 的材料命中率是 1.000：名字随实测值变化，这正是它该有的行为。

    真实资料包实例：``训练集`` 五包总分，以及 ``训练集/P1`` 单包。

    与 agent.md 的偏差：10.1 期望 6 项全过；离线上限是 5 过 1 不过。
    """
    per_pack = [score_pack(p)[1] for p in PACKS]
    total = selfcheck_mod.aggregate(per_pack)
    lines = selfcheck_mod.gate_lines(total, per_pack)

    assert len(lines) == 6
    assert [name for name, _ok, _detail in lines] == list(_GATE_NAMES)
    assert [ok for _name, ok, _detail in lines] == [True, True, True, True, False, True]
    assert [detail for _n, _o, detail in lines] == [
        "1.000／最差包 1.000",
        "0.927",
        "0.941",
        "0.931",
        "4/5",
        "最差包 1",
    ]
    failing = [name for name, ok, _d in lines if not ok]
    assert failing == [_FAILING_GATE]

    # 分包门禁（10.1）：每包覆盖率 ≥0.90，且 P3/P5 不得显著低于均值。
    for pack, score in zip(PACKS, per_pack):
        assert float(score["覆盖率"]) >= 0.90, pack
    worst = min(float(s["偏离类型准确率"]) for s in per_pack)
    assert float(total["偏离类型准确率"]) - worst <= 0.06
    # 每包多余条目 ≤2。
    assert max(int(s["多余条目"]) for s in per_pack) <= selfcheck_mod.MAX_EXTRA

    # 失败是局部的：P1 单独跑，6 行全过。
    p1 = score_pack("P1")[1]
    solo = selfcheck_mod.gate_lines(selfcheck_mod.aggregate([p1]), [p1])
    assert len(solo) == 6
    assert all(ok for _name, ok, _detail in solo)
    assert [name for name, _o, _d in solo][:4] == [
        "覆盖率 ≥0.95（每包 ≥0.90）",
        "偏离类型准确率 ≥0.85（冲刺 ≥0.92，已达）",
        "证明材料命中率 ≥0.9（冲刺 ≥0.95，已达）",  # P1 材料 1.000，冲刺线也算达到
        "位置命中率 ≥0.85（冲刺 ≥0.92，已达）",
    ]
