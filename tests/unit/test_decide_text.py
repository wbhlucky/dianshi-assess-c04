"""判定层·非数值通道（EC-C8 文字型 / EC-C9 标题行 / EC-D8 指针型 / EC-E5 置信封顶）。

结构比对不是唯一出口：标题行根本不判分，逐字复述算承诺，文字型要求永远拿不到
正偏离，指针型证据只能按 --ocr-fallback 走推定或未响应。每条走一条独立支路。
"""

from __future__ import annotations

import types

import pytest

from bidtable.config import DEV_NEGATIVE, DEV_NONE, DEV_NO_RESPONSE
from bidtable.judge import decide as J

pytestmark = [pytest.mark.unit]


@pytest.mark.ec("C9")
@pytest.mark.ec("E5")
@pytest.mark.case("N15")
def test_heading_rows_are_not_judged(mk_match):
    """N15：标题/引导句必须留空 ``偏离类型``，且四列全空、置信度满分。

    标题行进了 grade.py 会因为 ``偏离类型`` 非空而被当成漏抽去补，
    所以这里的第一道闸是 ``req.is_heading``，排在证据检查之前——
    即使误配上了证据也不能判。
    """
    m = mk_match("一、具体技术参数", "配置", "支持", req_kw={"is_heading": True})
    d = J.decide(m)
    assert d.deviation == ""
    assert d.method == "heading" and d.confidence == 1.0
    assert (d.resp_text, d.evidence_file, d.location, d.note) == ("", "", "", "")
    assert d.reason == "标题/引导句，不参与判定"


@pytest.mark.ec("E1")
def test_verbatim_echo_of_the_tender_text_counts_as_a_commitment(mk_match):
    """厂商把招标原文抄回来等于承诺"就按这个做"，但只对**严格子串**且够长的响应生效。

    ``ECHO_MIN_LEN == 12`` 是防"满足"这类两字空话蒙混的闸：11 字符走文字型，
    12 字符才认作复述。两条路结论都是无偏离，差别在 method 与可追溯的理由。
    """
    req = "配置要求：abcdefghijkl"
    assert J.ECHO_MIN_LEN == 12

    short = J.decide(mk_match(req, "配置", "abcdefghijk"))
    assert (short.method, short.deviation) == ("text", DEV_NONE)

    exact = J.decide(mk_match(req, "配置", "abcdefghijkl"))
    assert (exact.method, exact.deviation, exact.confidence) == ("echo", DEV_NONE, 0.9)
    assert exact.reason == "厂商逐字复述了招标原文，视为承诺满足"


@pytest.mark.ec("C8")
@pytest.mark.ec("E1")
@pytest.mark.case("X22")
def test_a_textual_requirement_never_yields_positive_deviation(mk_match):
    """X22：正偏离只发给数值型要求。响应侧自带数字也不改变这一点。

    第二行响应含 ``＜1秒``，若按"响应有数就比"会凭空造出一个正偏离；
    实际要求侧不可比（``rs.comparable is False``），所以既不给正偏离，
    也不打 low_confidence——低置信标记只属于"要求有门槛而响应没数"。
    """
    req = "具备软硬镜模式切换功能"
    for value in ("支持软硬镜模式切换功能", "支持，且切换时间＜1秒，业内领先"):
        d = J.decide(mk_match(req, "控制方式/软硬镜模式", value))
        assert (d.deviation, d.method, d.confidence, d.flags) == (DEV_NONE, "text", 0.9, ())
    assert d.reason == "要求与响应都没有可比数值，按文字型处理判无偏离（X22）"

    refusal = J.decide(mk_match(req, "控制方式", "暂不支持"))
    assert refusal.deviation == DEV_NEGATIVE and refusal.method == "phrase"


@pytest.mark.ec("C8")
@pytest.mark.ec("E5")
@pytest.mark.case("B23")
def test_numeric_requirement_with_no_comparable_answer_is_flagged(mk_match):
    """B23：要求含数值门槛、响应侧却找不到同量纲的量时，只能"按无偏离输出+降置信"。

    这是离线判定的诚实边界：不猜，但把不确定性交给复核报告排序。
    ``优于招标要求`` 这种自夸话术同样落到这条支路，不会被当成正偏离。
    """
    for value in ("详见样机", "满足", "优于招标要求"):
        d = J.decide(mk_match("工作长度：≥670 mm", "工作长度", value))
        assert (d.deviation, d.method, d.flags) == (DEV_NONE, "text", ("low_confidence",))
        assert d.confidence == 0.55

    met = J.decide(mk_match("工作长度：≥670 mm", "工作长度", "670mm"))
    assert (met.method, met.flags, met.confidence) == ("compare", (), 0.9)


@pytest.mark.ec("D3")
@pytest.mark.ec("D8")
@pytest.mark.ec("E5")
def test_pointer_evidence_follows_the_ocr_fallback_knob(mk_match):
    """指针型响应（``详见检测报告``）在 OCR 不可用时有两种诚实出口，由配置决定。

    ``assume``：材料与位置都已定位，按「无偏离（推定）」输出并留 note 让人补录实测值；
    ``no-response``：不推定，三列全空。两者置信度都被 ``ocr_failed`` 封到 0.70。
    """
    m = mk_match("脉冲宽度：200μs～800μs", "脉冲宽度", "详见检测报告",
                 ev_kw={"is_pointer": True, "doc_type": "检测报告",
                        "filename": "X_检测报告.jpg"})

    d = J.decide(m)
    assert (d.deviation, d.method, d.confidence) == (DEV_NONE, "pointer", 0.7)
    assert d.flags == ("ocr_failed", "pointer_assumed")
    assert d.evidence_file == "X_检测报告.jpg" and d.location == "检测报告 第1页 表行1"
    assert d.resp_text == "脉冲宽度：详见检测报告"
    assert "实测值待人工补录" in d.note and "无偏离（推定）" in d.reason

    strict = J.decide(m, cfg=types.SimpleNamespace(ocr_fallback="no-response"))
    assert (strict.deviation, strict.method, strict.confidence) == (DEV_NO_RESPONSE, "pointer", 0.7)
    assert strict.flags == ("ocr_failed",)
    assert (strict.resp_text, strict.evidence_file, strict.location, strict.note) == ("", "", "", "")


@pytest.mark.ec("E5")
@pytest.mark.ec("J8")
def test_confidence_caps_take_the_strictest_flag_and_never_raise():
    """置信度是"取最严上限"，不是叠加惩罚；未知标记不扣分，封顶也永远不上调。

    ``0.4`` 配 ``low_confidence`` 仍是 ``0.4``：匹配阶段本来就弱，判定层没有资格美化。
    """
    caps = {(): 0.9, ("low_confidence",): 0.55, ("low_margin",): 0.6,
            ("ambiguous_evidence",): 0.62, ("ocr_failed",): 0.7, ("unit_assumed",): 0.75}
    for flags, want in caps.items():
        assert J.confidence_of(0.9, flags) == want, flags

    assert J.confidence_of(0.9, ("ocr_failed", "low_confidence")) == 0.55
    assert J.confidence_of(0.9, ("low_margin", "ambiguous_evidence")) == 0.6
    assert J.confidence_of(0.9, ("unknown_flag",)) == 0.9
    assert J.confidence_of(0.4, ("low_confidence",)) == 0.4
