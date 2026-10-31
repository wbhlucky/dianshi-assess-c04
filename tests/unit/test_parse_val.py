# -*- coding: utf-8 -*-
"""厂商侧响应解析 :mod:`bidtable.judge.parse_val` 的等价类与边界用例。

钉住的能力彼此不重叠：拒答短语整串匹配（X21）、拒答词表跨模块一致、
只扫 value 不扫 name、comparable/is_text 四态真值表、空证据不抛、
多子句响应的区间与分辨率、投标响应原文的回退格式、周期词来源、
指针标记透传、子句切分同源。
"""
from __future__ import annotations

import pytest

from bidtable.judge import parse_val as V
from bidtable.judge.parse_val import parse_value, refusal_of, scan_value
from bidtable.match import candidates as K

pytestmark = [pytest.mark.unit]


def _quantities(spec):
    return [(str(q.value), q.unit) for q in spec.quantities]


def _bounds(spec):
    return [(str(b.interval.lo), str(b.interval.hi), b.interval.unit) for b in spec.bounds]


@pytest.mark.ec("E3")
@pytest.mark.case("X21")
def test_refusal_requires_whole_string_equality():
    """拒答短语是**整串**匹配：``暂不支持`` 命中，``支持（水夹层过滤器）`` 绝不能误伤。"""
    assert refusal_of("暂不支持") == "暂不支持"
    assert refusal_of("支持（水夹层过滤器）") is None  # 子串匹配会把它误判成拒答
    assert refusal_of("暂不支持该功能") is None  # 前后缀扩写同样不命中
    assert refusal_of("不支持") is None  # 不在 config.NEGATIVE_PHRASES 词表里
    assert refusal_of("") is None and refusal_of(None) is None
    # squash 吃掉首尾空白与全角括号差异，所以排版抖动仍要命中，且返回**原文**短语
    assert refusal_of(" 选配（需另行采购） ") == "选配（需另行采购）"
    assert refusal_of("标准配置不含，可按需定制") == "标准配置不含，可按需定制"
    assert refusal_of("部分支持（完整功能需升级版本）") == "部分支持（完整功能需升级版本）"


@pytest.mark.ec("J3")
def test_refusal_vocabularies_agree_across_modules():
    """判定层与匹配层共用同一份拒答词表，否则会出现「匹配上了却判不出负偏离」。"""
    assert frozenset(V.REFUSAL_KEYS) == K._REFUSALS
    # 两边容器类型不同（tuple vs frozenset）且顺序不同，裸 == 会假失败——必须按集合比
    assert V.REFUSAL_KEYS != K._REFUSALS
    for text in ("暂不支持", "支持（水夹层过滤器）", "", None):
        assert K.is_refusal(text) is (refusal_of(text) is not None), text


@pytest.mark.ec("D1")
def test_only_the_value_is_scanned_never_the_name(mk_ev):
    """证据名里的数字（``272μm``）不能变成响应值，否则会凭空造出 ``≥272μm`` 假约束。"""
    spec = parse_value(mk_ev("272μm光纤最大传输功率", "84W"))
    assert _quantities(spec) == [("84", "w")]
    assert spec.text == "84w"
    assert spec.name == "272μm光纤最大传输功率"  # 名称原样保留，只在对齐子句时用
    assert spec.scans[0].text == "84w"


@pytest.mark.ec("E1")
@pytest.mark.ec("E3")
@pytest.mark.ec("E5")
def test_comparable_and_is_text_form_a_four_state_truth_table(mk_ev):
    """四态互斥：可比数值 / 拒答 / 纯文字 / 空值。空值连 Scan 都不产生。"""
    rows = {
        "84W": (True, False, None),
        "暂不支持": (False, False, "暂不支持"),
        "支持（水夹层过滤器）": (False, True, None),
        "": (False, True, None),
    }
    for value, expected in rows.items():
        spec = parse_value(mk_ev("指标", value))
        assert (spec.comparable, spec.is_text, spec.refusal) == expected, value
    # 空值没有任何子句可扫；纯文字值仍会产生一个 Scan（判定层要看它的原文）
    assert parse_value(mk_ev("指标", "")).scans == ()
    assert len(parse_value(mk_ev("指标", "支持（水夹层过滤器）")).scans) == 1


@pytest.mark.ec("E4")
def test_none_evidence_returns_an_empty_shell():
    """``parse_value(None)`` 返回空壳而不抛异常（调用方通常在更早的分支就判了未响应）。"""
    spec = parse_value(None)
    assert (spec.raw, spec.name, spec.value, spec.text, spec.period) == ("", "", "", "", "")
    assert spec.refusal is None and spec.is_pointer is False and spec.resolution is None
    assert spec.quantities == () and spec.bounds == () and spec.resolutions == ()
    assert spec.comparable is False and spec.is_text is True and spec.scans == ()


@pytest.mark.ec("C5")
@pytest.mark.ec("G8")
def test_multi_clause_values_yield_bounds_and_resolutions(mk_ev):
    """响应值跨子句时，区间 / 分辨率 / 散点各自归位，并保留每个子句的 Scan。"""
    span = parse_value(mk_ev("血泵流速", "10~450ml/min"))
    assert _bounds(span) == [("10", "450", "ml/min")] and _quantities(span) == []

    res = parse_value(mk_ev("输出分辨率", "3840*2160"))
    assert _quantities(res) == [] and _bounds(res) == []  # 内层数字被字面量屏蔽
    assert (res.resolution.width, res.resolution.height) == (3840, 2160)
    assert res.resolution is res.resolutions[0]

    mixed = parse_value(mk_ev("脉冲宽度可调范围", "200μs～800μs可调，区间范围≥600μs"))
    assert len(mixed.scans) == 2
    assert _bounds(mixed) == [("200", "800", "μs")]
    assert _quantities(mixed) == [("600", "μs")]  # 区间内部没有外溢成散点


@pytest.mark.ec("E1")
def test_raw_falls_back_to_the_bidding_response_format(mk_ev):
    """``raw`` 供复核报告回溯：有 response_text 就用它，否则拼 ``名称：值``。"""
    assert parse_value(mk_ev("治疗深度", "详见检测报告", response_text="")).raw == "治疗深度：详见检测报告"
    assert parse_value(mk_ev("治疗深度", "详见检测报告", response_text="自定义")).raw == "自定义"
    assert parse_value(mk_ev("", "", response_text="")).raw == "："
    # raw 只留档，判定用的始终是 squash 后的 text
    assert parse_value(mk_ev("治疗深度", "详见检测报告", response_text="")).text == "详见检测报告"


@pytest.mark.ec("G4")
def test_period_is_read_from_name_and_value_together(mk_ev):
    """周期词取自 ``f"{name} {value}"``：名称里的「每年」要能桥接 ``次`` ↔ ``次/年``。"""
    spec = parse_value(mk_ev("每年维护保养次数", "4.22次/年"))
    assert spec.period == "年" and _quantities(spec) == [("4.22", "次/年")]
    assert parse_value(mk_ev("血泵流速", "10~450ml/min")).period == ""


@pytest.mark.ec("D8")
def test_pointer_flag_is_carried_through_untouched(mk_ev):
    """指针型证据必须把 is_pointer 原样透传，判定层据此走 ocr_failed 分支。"""
    spec = parse_value(mk_ev("治疗深度", "详见检测报告", doc_type="检测报告", is_pointer=True))
    assert spec.is_pointer is True
    assert spec.comparable is False and spec.is_text is True and spec.refusal is None
    assert spec.scans[0].text == "详见检测报告"
    assert parse_value(mk_ev("治疗深度", "详见检测报告")).is_pointer is False


@pytest.mark.ec("C6")
def test_scan_value_shares_the_clause_splitter():
    """``scan_value`` 与 ``parse_value`` 内部扫描同源，复核报告可据此复现切分。"""
    assert scan_value("血泵流速10~450ml/min；透析液流速") == ("血泵流速10-450ml/min", "透析液流速")
    assert scan_value(None) == () and scan_value("") == () and scan_value("   ") == ()
