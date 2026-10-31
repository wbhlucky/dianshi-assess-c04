# -*- coding: utf-8 -*-
"""P3 的金标准对照：唯一「分节」列被真正填满的包，也是五个包里最弱的那一个。

P3（4K3D 腹腔镜摄像系统，71 条，最大包）在五个包里独有两件事：

1. 它是唯一**分节列一格都不为空**的包（P1/P2/P4 全空，P5 只有 9 行有值）。
   于是对齐键 ``(标的, 分节, 序号)`` 在 P3 上第一次真的是三维的：
   分节写错不会抛异常，只会让整行的键对不上标准答案，覆盖率直接掉到地板上。
   agent.md 7.1 的 I-2（中文序数单级）、I-3（中文序数 + 整数二级）、
   I-4（引导句单级）、I-5（引导句 + 整数二级）、I-8（标题带末尾标点）
   这五个等价类**只在 P3 里同时出现**，所以把它们逐个点名钉在这里，
   而不是笼统断言一个「17 种分节」。
2. 它的偏离类型准确率最低（0.873，总分 0.927）。agent.md 10.1 的分包门禁
   专门点了 P3 的名（「不得显著低于均值」）；本文件把 P3 自己那一半
   （≥ 全局下限 0.85）钉住，"与均值的差"留给 ``test_gold_all_keys.py``，
   因为只有那里才真的算出了均值。

与 agent.md 的偏差：7.4 的 B21 原以为分节能与标准答案逐字对齐。实测标准答案
把分节存成**斜杠不带空格、且吃掉原文空格**的形式（``七、专用屏幕台车/1、专用屏幕台车``），
而我们按 ``section.SEP == " / "`` 输出并保留招标原文里的空格
（``七、专用屏幕台车 / 1、专用屏幕台车``）。逐字比较会让 71 行**全部**判错；
是 ``grade.norm``（吃掉所有空白）把两边接上的。所以本文件断言的是
「norm 之后相等、norm 之前不相等」——这条断言本身就是"norm 是承重的"的证据，
任何人把分节改成逐字对齐标准答案、或者把 norm 换成 strip，都会在这里红。

另：分节取自招标文件自身的标题文字，属于**公开输入**而非答案
（答案只有「投标响应」与「证明材料文件」两列），因此在这里写出字面量不违反 O10。
"""

from __future__ import annotations

import pytest

from bidtable import selfcheck as selfcheck_mod

pytestmark = [pytest.mark.golden]

PACK = "P3"
N_ROWS = 71
SEP = " / "


def _norm(text: str) -> str:
    """``grade.py`` 自己的归一化（吃掉所有空白）。它是分节能对上的唯一原因。"""
    return selfcheck_mod.load_grade().norm(text)


# agent.md 7.1 EC-I 表里 P3 独有的五个等价类，各自挑一个真实代表（写成标准答案的无空格形态）。
_I2_SINGLE_CHINESE = "一、4K3D摄像系统"          # I-2 中文序数单级
_I3_TWO_LEVEL = "五、医用显示器/2、4K医用显示器"   # I-3 中文序数 + 整数二级
_I4_LEAD_SINGLE = "设备运维技术服务要求如下："     # I-4 引导句单级（不是序号，是一句引导话）
_I5_LEAD_TWO = "设备运维技术服务要求如下：/2、设备运行维护："  # I-5 引导句 + 整数二级
_I8_TRAILING_PUNCT = ("九、配置适合腹腔镜及开腹手术使用电刀：", "十一、妇科电动检查床：")


@pytest.mark.ec("I2")
@pytest.mark.ec("I3")
@pytest.mark.ec("I4")
@pytest.mark.ec("I5")
@pytest.mark.ec("I8")
def test_p3_section_column_is_three_dimensional_and_only_lines_up_after_normalization(
    score_pack, gold_map
):
    """P3 的 71 行分节全非空、17 种形态、五种等价类各就各位，且只有 norm 之后才对得上。

    边界在哪：分节是**对齐键的一段**，它的错误是静默的。
    一个"永远填空分节"的实现能让 P1/P2/P4/P5 全对，却会让 P3 的 71 行
    整块错位；反过来，一个"把标准答案的空格形态硬编码进来"的实现
    会绕过 ``grade.norm`` 这层契约。所以这里同时钉三件事：
    非空（不能偷懒）、17 种（不能过粗，例如只留一级标题）、
    norm 前不相等（不能抄答案的形态）。

    二级分节还有一条结构边界：``SEP`` 只能出现一次。
    P3 最深只到两级（``五、医用显示器 / 2、4K 医用显示器``），
    出现两个 SEP 就说明标题判别把正文里的序号也当成了层级。

    真实资料包实例：``训练集/P3``，71 行、17 种分节、0 行为空。

    与 agent.md 的偏差：见模块 docstring（B21 的逐字对齐假设不成立）。
    """
    result, _score = score_pack(PACK)
    assert len(result.rows) == N_ROWS

    raw = [r.section for r in result.rows]
    distinct = set(raw)
    # 一格都不为空 —— 这是 P3 区别于其他四个包的地方。
    assert "" not in distinct
    assert len(distinct) == 17

    two_level = sorted(s for s in distinct if SEP in s)
    single = sorted(s for s in distinct if SEP not in s)
    # 7 个两级 / 10 个单级；两级最多两级，不许出现三段。
    assert (len(two_level), len(single)) == (7, 10)
    assert all(s.count(SEP) == 1 for s in two_level)
    # 二级那一段必须是**整数序号**开头（I-3 / I-5 的"整数二级"），
    # 而不是又一个中文序数 —— 出现"一、/二、"就说明层级判反了。
    assert all(s.split(SEP)[1][0].isdigit() for s in two_level)
    # 我们输出里的 "/" 只可能来自 SEP，正文自带的斜杠不许混进来。
    assert all(s.count("/") == s.count(SEP) for s in distinct)

    norms = {_norm(s) for s in distinct}
    gold_norms = {_norm(v["sec"]) for v in gold_map(PACK).values()}
    assert len(norms) == 17, "norm 之后不许塌缩：17 种分节必须仍然两两不同"
    assert norms == gold_norms
    # norm 之前**不**相等：我们保留招标原文空格并用 " / "，标准答案两者都没有。
    assert distinct != gold_norms
    assert all(SEP in s for s in distinct if "/" in _norm(s))

    # 五个等价类各自点名（写成标准答案的无空格形态，因此比的是 norm）。
    assert _I2_SINGLE_CHINESE in norms
    assert _I3_TWO_LEVEL in norms
    assert _I4_LEAD_SINGLE in norms
    assert _I5_LEAD_TWO in norms
    assert set(_I8_TRAILING_PUNCT) <= norms
    # I-8 的要点是**末尾标点被保留**，没有被当成噪声削掉。
    assert any(s.endswith("：") for s in single)
    # I-4 的要点：10 个单级分节里恰好 1 个不是中文序数开头，而是一句引导话，
    # 它照样被当成一级分节用（服务标的整段挂在它下面）。
    ordinal_led = [s for s in single if s[0] in "一二三四五六七八九十"]
    assert len(ordinal_led) == len(single) - 1 == 9
    assert {_norm(s) for s in single if s not in ordinal_led} == {_I4_LEAD_SINGLE}

    # 分节填充率 100% 会触发 section.diagnose 的告警（>80% 阈值）。
    # P3 是**真的**多级标题组织，所以这条告警是误报里的良性那一类，
    # 但它必须响 —— 静音了就意味着换到别的包上也不会有人提醒。
    assert any(w.startswith("【分节告警】") and "100%" in w for w in result.warnings)
    # 71 行一条没被主动丢弃：90 条抽取里少的 19 条全是标题/引导句。
    assert result.skipped == []
    assert result.stats["cut_counts"] == {
        "抽取条目": 90,
        "折叠后": 90,
        "裁剪后": 90,
        "分节后": 90,
        "标题/引导句": 19,
        "参与判定": N_ROWS,
    }


@pytest.mark.ec("E1")
@pytest.mark.ec("E4")
def test_p3_is_the_weakest_pack_and_its_nine_misses_are_all_accounted_for(score_pack):
    """P3：覆盖率 1.000、偏离 0.873，9 格失误拆成三类且与输出的未响应行数自洽。

    边界在哪：P3 是 §10.1 分包门禁点名的包，也是全局最弱的一环。
    只钉一个 0.873 意义不大（分数会随调参漂移），有意义的是钉住
    **失误的构成**：``无偏离->未响应`` 6 格、``负偏离->无偏离`` 2 格、
    ``负偏离->正偏离`` 1 格。这三类的成因完全不同
    （第一类是匹配层没找到证据，后两类是判定层对拒答式表达召回不足），
    构成变了就说明改错了地方。

    另一层边界是数字自洽：输出里 ``未响应`` 共 10 行，
    必须正好等于 ``6（我们说未响应、答案说无偏离）+ 4（两边都说未响应）``。
    这条等式把混淆矩阵和成品行数绑在一起，任何一边单独被改都会红。

    真实资料包实例：``训练集/P3``，71 条全部命中标准答案，多余 0。
    位置命中率 0.836 **低于全局下限 0.85** —— 这不是漏网，
    §10.1 的分包门禁只约束覆盖率（每包 ≥0.90），
    材料/位置的下限是定义在**总分**上的（总分 0.931 达标）。
    这里把 0.836 显式钉住，是为了让"P3 拖了位置命中率的后腿"变成写在测试里的事实。

    与 agent.md 的偏差：无。
    """
    result, score = score_pack(PACK)

    assert score["条目总数"] == N_ROWS
    assert score["提交匹配条目"] == N_ROWS
    assert score["多余条目"] == 0
    assert score["覆盖率"] == pytest.approx(1.0)
    # §10.1 分包门禁：每包覆盖率 ≥0.90。P3 满分。
    assert score["覆盖率"] >= 0.90

    assert score["偏离类型准确率"] == pytest.approx(62 / N_ROWS, abs=1e-3)
    # §10.1 的全局下限 0.85，逐包也要站得住（"不得显著低于均值"在 all_keys 里查）。
    assert score["偏离类型准确率"] >= 0.85
    assert score["证明材料命中率"] == pytest.approx(0.866, abs=1e-3)
    assert score["位置命中率"] == pytest.approx(0.836, abs=1e-3)
    assert score["位置命中率"] < 0.85  # 已知的逐包短板，见 docstring
    assert score["关键项风险识别"].startswith("1/1")

    matrix = score["混淆矩阵"]
    assert sum(matrix.values()) == N_ROWS
    diag = sum(v for k, v in matrix.items() if k.split("->")[0] == k.split("->")[1])
    assert diag == 62
    off = {k: v for k, v in matrix.items() if k.split("->")[0] != k.split("->")[1]}
    assert off == {"无偏离->未响应": 6, "负偏离->无偏离": 2, "负偏离->正偏离": 1}

    # 数字自洽：成品里的未响应行数 == 混淆矩阵里落到"未响应"这一列的总数。
    ours_no_resp = sum(1 for r in result.rows if r.deviation == "未响应")
    assert ours_no_resp == off["无偏离->未响应"] + matrix["未响应->未响应"] == 10
    # 未响应的行必须是"三列全空"那一档（IV5 在五个包上统一证明，这里只用它的推论）。
    for row in result.rows:
        if row.deviation == "未响应":
            assert (row.resp_text, row.evidence_file, row.location) == ("", "", "")
