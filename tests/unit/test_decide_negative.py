"""判定层·负偏离与未响应的四条独立成因（EC-C1/C2/C5 + EC-D1 + EC-E3/E4）。

负偏离在 gold 里占 31/205，来源却只有四条互不重叠的路径：低于下界门槛、
超出上界门槛、区间未被覆盖、命中封闭拒答词表。未响应则是"根本没配上证据"，
它有自己的一套输出契约（三列全空、置信度不折价）。每个用例盯一条路径。
"""

from __future__ import annotations

import dataclasses

import pytest

from bidtable.config import DEV_NEGATIVE, DEV_NO_RESPONSE
from bidtable.ingest.scope import body_of
from bidtable.judge import decide as J

pytestmark = [pytest.mark.unit]


@pytest.mark.ec("C1")
@pytest.mark.ec("E3")
@pytest.mark.case("B2")
def test_shortfall_below_a_lower_bound_is_negative(mk_match):
    """B2：``具有≥3个探测器`` 对 ``探测器数量：2个``。

    指标名与要求原文一个字都不重合（"探测器数量" vs "具有…个探测器"），
    能配上完全靠匹配层的名字变体；判定层这里只确认 shortfall 走 compare 通道，
    且不带任何降信旗标——单位相同、比值远离低置信带。
    """
    dec = J.decide(mk_match("具有≥3个探测器", "探测器数量", "2个"))
    assert dec.deviation == DEV_NEGATIVE
    assert dec.method == "compare" and dec.flags == () and dec.confidence == 0.900
    assert dec.reason == "探测器数量 2个 vs 要求 ≥3个"


@pytest.mark.ec("C2")
@pytest.mark.ec("E3")
@pytest.mark.ec("H2")
@pytest.mark.case("B1")
def test_exceeding_an_upper_bound_is_negative_even_by_a_hair(mk_match):
    """B1：``≤3mm`` 对 ``3.21mm`` 只差 7%，上界方向照样是负偏离（对比 X22）。

    顺带钉住 H2：要求原文带序号 ``1.6、``，``body_of`` 必须先把它剥掉，
    否则 ``1.6`` 会被当成一个无单位的下界约束参与比对。
    """
    m = mk_match("1.6、瞳孔大小：≤3mm；", "最小瞳孔直径", "3.21mm")
    assert body_of(m.requirement) == "瞳孔大小：≤3mm；"

    dec = J.decide(m)
    assert dec.deviation == DEV_NEGATIVE and dec.confidence == 0.900
    assert dec.reason == "最小瞳孔直径 3.21mm vs 要求 ≤3mm"


@pytest.mark.ec("C1")
@pytest.mark.ec("G7")
@pytest.mark.case("B5")
def test_fractional_counts_are_compared_not_rounded(mk_match):
    """B5：``≥5副`` 对 ``3.8副``。小数计数不取整、不当文字型，仍按数值判负偏离。"""
    dec = J.decide(mk_match("≥5副", "3D眼镜数量", "3.8副"))
    assert dec.deviation == DEV_NEGATIVE and dec.method == "compare"
    assert dec.reason == "3D眼镜数量 3.8副 vs 要求 ≥5副"


@pytest.mark.ec("D1")
@pytest.mark.ec("E3")
@pytest.mark.case("T15")
def test_refusal_phrase_short_circuits_before_any_comparison(mk_match):
    """T15/X21：整串命中拒答词表即负偏离，**在结构化比对之前**返回。

    要求 ``临床用途：可用于泌尿外科手术`` 本身不可比（纯文字型），
    若走到底会被 X22 兜成无偏离——词表短路正是为了拦住这一类。
    置信度不经 ``confidence_of`` 折价，因为拒答是确定结论而非猜测。
    """
    dec = J.decide(mk_match("临床用途：可用于泌尿外科手术", "临床用途", "暂不支持"))
    assert dec.deviation == DEV_NEGATIVE and dec.method == "phrase"
    assert dec.confidence == 0.900 and dec.flags == ()
    assert "暂不支持" in dec.reason and "3.9.3" in dec.reason
    assert dec.resp_text == "临床用途：暂不支持"


@pytest.mark.ec("E4")
@pytest.mark.case("X23")
def test_unmatched_requirement_leaves_three_columns_empty(mk_match):
    """X23/I5：配不上证据就是未响应，响应/材料/位置三列必须同时为空。

    置信度**原样**透传 ``match.confidence``（0.42）而不是走 ``confidence_of``——
    没有证据时折价毫无意义，反而会把匹配层的低分信息抹平。
    理由字段有值就透传，为空才用兜底文案。
    """
    m = mk_match("某个指标：≥1台", ev_name=None, confidence=0.42, flags=("alias_rescue",))
    dec = J.decide(m)
    assert dec.deviation == DEV_NO_RESPONSE and dec.method == "none"
    assert (dec.resp_text, dec.evidence_file, dec.location) == ("", "", "")
    assert dec.confidence == 0.42 and dec.flags == ("alias_rescue",)
    assert dec.reason == "本标的的厂商资料里找不到可配对的指标"

    told = J.decide(dataclasses.replace(m, reason="找不到配对"))
    assert told.reason == "找不到配对"


@pytest.mark.ec("C5")
@pytest.mark.ec("E3")
def test_uncovered_interval_is_negative_and_carries_no_ratio(mk_match):
    """区间要求没被覆盖 -> 负偏离，且 ratio 为 None（"差多少"无从定义）。

    ratio 为 None 连带两个后果：不会被打 low_margin，复核报告里也没有比值可排序。
    """
    dec = J.decide(mk_match("温度控制范围：33℃-43℃", "温度控制范围", "37~43℃"))
    assert dec.deviation == DEV_NEGATIVE and dec.method == "compare"
    assert dec.flags == () and dec.confidence == 0.900
    assert dec.reason == "响应区间 [37,43] 未覆盖要求 [33,43]"
