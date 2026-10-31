# -*- coding: utf-8 -*-
"""集成层：匹配与判定的跨阶段传递（招标要求 -> 证据 -> 偏离类型 -> 判定依据）。

单元层用合成 fixture 钉住了打分公式、阈值与拒答词表；集成层要证明的是这些规则落到
**真资料包的某一行**时，四个产物彼此一致且可被人工追溯：xlsx 的三列、``Row.confidence``、
``stats["decisions"]`` 的判定依据、以及复核报告要用的旗标。

本文件两条用例各盯一个方向的边界，互为反面：

1. T11 挂得上的一侧 —— 指标名与要求正文并不同名，只能靠「数值+单位指纹」搭桥，
   且 ``5mW`` 对 ``≤5mW`` 正落在取等边界上，必须判**无偏离**而不是正偏离。
2. T16 挂不上的一侧 —— 最近候选综合分 0.24 低于阈值 0.62，整行必须退化成
   ``未响应`` + 三列全空 + 置信度 0.0，而判定依据要**指名**那个落选候选去了哪里。
"""
from __future__ import annotations

import pytest

from bidtable.output import review as rv

pytestmark = [pytest.mark.integration]

ROW_LEVEL_DOCS = ("技术规格书", "服务方案", "检测报告")


def _index(rows, lot, no):
    """按 (标的, 序号) 定位行下标。P1 的标的2 自带 no=='1'/'2'，只按 no 过滤会串标的。"""
    hits = [k for k, r in enumerate(rows) if (r.lot_no, r.no) == (lot, no)]
    assert len(hits) == 1, f"标的{lot} 序号{no} 应当唯一，实际命中 {hits}"
    return hits[0]


@pytest.mark.ec("C1")
@pytest.mark.ec("D5")
@pytest.mark.case("T11")
def test_t11_a_renamed_indicator_is_bridged_by_the_numeric_fingerprint_and_ties_go_to_none(
    offline_run,
):
    """指标名不是要求正文的连续子串时，只有数值指纹能把它挂上；挂上后取等判无偏离。

    边界在哪：P1 标的1 序号9 的要求是「9.护眼指示光：绿光，波长520nm，功率≤5mW。」，
    厂商证据的指标名是「指示光功率」——这五个字在要求正文里**不连续**出现
    （正文只有「指示光」与「功率」两段）。纯名称匹配因此拿不到高分（实测名称分 0.82），
    真正把它救回来的是「5mW 与 ≤5mW 数值+单位指纹一致」。若指纹通道缺失或被降级，
    这一行会掉到阈值以下变成 ``未响应``，而它在标准答案里是 ``无偏离``。

    第二个边界是取等：响应值恰好等于门槛（5mW == ≤5mW）。判成正偏离是常见误实现，
    本项目约定「取等 -> 无偏离」。同一份输出里 序号5（84W vs ≥60W）判正偏离，
    两行放在一起就能证明这不是「一律无偏离」蒙对的。

    与 agent.md 的偏差：T11 原文称本行 ``method == "alias"``，实测 ``method == "compare"``、
    ``match_method == "fingerprint"``。P1 里真正走 alias 通道的是标的1 序号1，
    下面一并钉住，以免这条纠正只存在于文档里。
    """
    result = offline_run("P1")
    rows, decs = result.rows, result.stats["decisions"]
    assert len(rows) == len(decs) == 17

    i = _index(rows, "1", "9")
    row, dec = rows[i], decs[i]
    assert row.mark == "" and row.req_text.startswith("9.护眼指示光")
    assert "指示光功率" not in row.req_text, "指标名必须不是要求正文的连续子串，否则本用例失去意义"

    assert row.resp_text == "指示光功率：5mW"
    assert row.evidence_file == "YF_YF-9834Pro_技术规格书.docx"
    assert row.location == "技术规格书 表行6"
    assert row.deviation == "无偏离"
    assert row.confidence == 0.995

    assert dec["method"] == "compare"
    assert dec["match_method"] == "fingerprint"
    assert dec["match_score"] == 0.995
    assert dec["match_reason"] == "数值与单位指纹一致（名称 0.82 / 数值 1.00）"
    assert dec["reason"] == "指示光功率 5mw vs 要求 ≤5mw", "判定依据必须同时写出响应值与门槛"

    # 取等 -> 无偏离；严格超出 -> 正偏离。两行同属 compare 通道，方向相反。
    j = _index(rows, "1", "5")
    assert decs[j]["method"] == "compare"
    assert (rows[j].deviation, rows[j].resp_text) == ("正偏离", "272μm光纤最大传输功率：84W")

    # 纠正 agent.md：P1 里唯一走 alias 的是标的1 序号1，且它带 alias_rescue 旗标。
    alias = [k for k, d in enumerate(decs) if d["match_method"] == "alias"]
    assert len(alias) == 1 and (rows[alias[0]].lot_no, rows[alias[0]].no) == ("1", "1")
    assert decs[alias[0]]["method"] == "phrase"
    assert rows[alias[0]].deviation == "负偏离"
    assert tuple(rows[alias[0]].flags) == ("alias_rescue",)


@pytest.mark.ec("E4")
@pytest.mark.ec("D7")
@pytest.mark.case("T16")
def test_t16_a_below_threshold_candidate_becomes_no_response_with_an_honest_reason(offline_run):
    """综合分低于阈值时整行退化为未响应，且判定依据要指名落选候选被谁拿走了。

    边界在哪：``未响应`` 不是一个「什么都没发生」的兜底值，它有三个必须同时成立的后果——
    ``投标响应``/``证明材料文件``/``位置`` 三列**全空**（写任何占位符都会让复核人误以为有据可查）、
    置信度归零（从而必然进入复核清单）、旗标与备注为空（它不是降级推定，是真的没证据）。
    本用例逐条钉住这四件事，再钉住判定依据的**诚实性**：它必须说出最接近的候选是谁、
    在哪、差多少分，而不是只写一句「无匹配」。P1 标的1 序号2 的落选候选
    「脉冲宽度可调范围」（检测报告 第1页 表行1，0.24 < 0.62）恰好被序号3 以 0.80 拿走，
    所以这一行的 ``未响应`` 是仲裁的结果而非漏配——顺带证明行级定位不会被两行重复占用。

    与 agent.md 的偏差：T16 原锚点是「P1 标的2 序号1 -> 三列全空 + 未响应」，但该行在
    agent.md 自己的标准答案里是 ``无偏离`` / ``YF_服务方案.docx`` / ``服务方案 表行1``，
    本项目输出与标准答案一致，故原锚点自相矛盾，改锚到标的1 序号2（▲ 且真的未响应）。
    """
    result = offline_run("P1")
    rows, decs = result.rows, result.stats["decisions"]

    i = _index(rows, "1", "2")
    row, dec = rows[i], decs[i]
    assert row.mark == "▲" and row.req_text.startswith("▲2.最大单脉冲能量5J")

    assert (row.resp_text, row.evidence_file, row.location) == ("", "", ""), "三列必须全空"
    assert row.deviation == "未响应"
    assert row.confidence == 0.0
    assert tuple(row.flags) == () and row.note == ""

    assert dec["method"] == "none" and dec["match_method"] == "none" and dec["match_score"] == 0.0
    reason = dec["reason"]
    assert reason == dec["match_reason"], "未响应时判定依据与匹配依据应是同一句话"
    for token in ("脉冲宽度可调范围", "检测报告 第1页 表行1", "0.24", "0.62"):
        assert token in reason, f"判定依据缺少 {token}：{reason}"

    # 落选候选确实被别人以更高分拿走了 -> 未响应是仲裁结果，不是漏配。
    k = _index(rows, "1", "3")
    assert rows[k].location == "检测报告 第1页 表行1"
    assert decs[k]["match_score"] == 0.8 > 0.24

    # 行级定位（表行N）在整个 P1 里不许被两行重复占用；样本是页级定位，允许复用。
    row_locs = [r.location for r in rows if r.location.startswith(ROW_LEVEL_DOCS)]
    page_locs = [r.location for r in rows if r.location.startswith("产品综合样本")]
    empty = [r for r in rows if r.location == ""]
    assert len(row_locs) + len(page_locs) + len(empty) == len(rows) == 17
    assert (len(row_locs), len(set(row_locs))) == (12, 12), "行级定位不许被两行重复占用"
    assert len(empty) == sum(1 for r in rows if r.deviation == "未响应") == 3

    # ▲ + 未响应 -> 关键项风险，且这是 P1 唯一的一行。
    assert [r.no for r in rv.risk_rows(rows)] == ["2"]

    # 原锚点行的真实形态：与标准答案一致，故不能作为「未响应」的例子。
    m = _index(rows, "2", "1")
    assert (rows[m].deviation, rows[m].evidence_file, rows[m].location) == (
        "无偏离",
        "YF_服务方案.docx",
        "服务方案 表行1",
    )
