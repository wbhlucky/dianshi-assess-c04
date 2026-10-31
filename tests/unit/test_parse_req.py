# -*- coding: utf-8 -*-
"""招标侧要求解析 :mod:`bidtable.judge.parse_req` 的等价类与边界用例。

每个测试只钉住**一种**解析能力，彼此不重叠：裸数字语义、区间端点不外溢、
子句切分的取舍、比较符最长匹配、比较符窗口作用域、严格不等号、产品代际标签、
分辨率字面量屏蔽、宽度模式的双条件门、周期归一、排版噪声拒收、kind 优先级、
序号残留、子句溯源（X20）、单位词表覆盖与缺口。
"""
from __future__ import annotations

import pytest

from bidtable.judge import parse_req as R
from bidtable.judge.parse_req import clauses, parse_requirement

pytestmark = [pytest.mark.unit]


def _cons(spec):
    """把约束投影成可断言的元组：(方向, 值, 单位, 比较符, 量纲)。"""
    return [
        (c.direction, str(c.quantity.value), c.quantity.unit, c.op, c.dimension)
        for c in spec.constraints
    ]


def _bounds(spec):
    return [(str(b.interval.lo), str(b.interval.hi), b.interval.unit) for b in spec.bounds]


@pytest.mark.ec("C4")
def test_bare_number_means_at_least():
    """无比较符的裸数字一律按「至少」处理，op 留空以便复核报告区分。"""
    assert _cons(parse_requirement("拥有2个夹管阀")) == [("min", "2", "个", "", "count")]
    assert _cons(parse_requirement("圆锥角：18度")) == [("min", "18", "°", "", "angle")]
    # 同一条里显式写了 ≥ 的，op 必须被记下来，不能和裸数字混为一谈
    assert _cons(parse_requirement("具有≥4 个泵")) == [("min", "4", "个", "≥", "count")]


@pytest.mark.ec("C5")
def test_interval_endpoints_never_become_constraints():
    """``33℃-43℃`` 只产出一个 Bound；若端点各自成约束会被读成「≥33 且 ≥43」。"""
    spec = parse_requirement("温度控制范围：33℃-43℃")
    assert spec.kind == "interval"
    assert _bounds(spec) == [("33", "43", "℃")]
    assert spec.constraints == ()
    # 多种破折号写法归一后都能识别为同一个区间
    for text, unit in (("有效治疗波段：400nm---1400nm", "nm"), ("光斑大小：8-10μm", "μm")):
        assert parse_requirement(text).constraints == ()
        assert parse_requirement(text).bounds[0].interval.unit == unit


@pytest.mark.ec("C6")
@pytest.mark.ec("H5")
def test_clause_splitter_protects_serial_numbers_and_enumerations():
    """分隔符刻意不含 ``.`` 与 ``、``：前者会切断 ``1.9.1``，后者会丢掉并列上下文。"""
    assert clauses("1.9.1 主机；性能检测、电气安全性检测。尾句，最后") == [
        "1.9.1 主机",
        "性能检测、电气安全性检测",
        "尾句",
        "最后",
    ]
    assert clauses("软光纤最小弯曲半径：200μm 光纤≤4.0cm。") == [
        "软光纤最小弯曲半径：200μm 光纤≤4.0cm"
    ]
    assert clauses("临床用途：可用于泌尿外科、普外科手术") == [
        "临床用途：可用于泌尿外科、普外科手术"
    ]
    # 尾部分隔符会切出空串，必须丢掉
    assert clauses("≤3mm；") == ["≤3mm"]
    assert clauses(None) == [] and clauses("") == [] and clauses("   ") == []


@pytest.mark.ec("C1")
@pytest.mark.ec("C2")
def test_comparator_longest_prefix_wins_and_suffix_forms_are_read():
    """``不低于`` 必须抢在 ``不`` 前、``不大于`` 抢在 ``大于`` 前；中文后置词也要认。"""
    assert _cons(parse_requirement("不低于2.5米")) == [("min", "2.5", "m", "不低于", "length")]
    low = parse_requirement("不大于5kg").constraints[0]
    assert (low.direction, low.op) == ("max", "不大于")  # 不是 ("min", "大于")
    assert parse_requirement("后备电源续用时间：12分钟以上").constraints[0].op == "以上"
    assert _cons(parse_requirement("3mm以下")) == [("max", "3", "mm", "以下", "length")]
    # ``≤`` 与 ``5`` 之间还夹着一个 ``±``，窗口必须容得下
    assert _cons(parse_requirement("注射精度：≤±5%")) == [("max", "5", "%", "≤", "percent")]


@pytest.mark.ec("C1")
def test_comparator_window_governs_only_its_own_number():
    """窗口 6 字符：同一子句里各管各的，超出窗口就退回裸数字语义。"""
    both = parse_requirement("≥2倍光学变焦、≥5倍电子放大")
    assert _cons(both) == [
        ("min", "2", "倍", "≥", "count"),
        ("min", "5", "倍", "≥", "count"),
    ]
    far = parse_requirement("≥某某某某某某某某5mm")  # ≥ 距数字 8 字符
    assert far.constraints[0].op == "" and far.constraints[0].direction == "min"
    assert R.OP_WINDOW == 6


@pytest.mark.ec("C3")
def test_strict_operators_are_flagged_without_changing_direction():
    """严格不等号只进复核报告；判定仍按取等放行（训练集无反例可依据）。"""
    assert parse_requirement("空气爆破能量：＜4mJ").constraints[0].strict is True
    strict = parse_requirement("（大于100W 为优于）").constraints[0]
    assert strict.strict is True and strict.direction == "min"
    assert parse_requirement("治疗深度：≥7CM").constraints[0].strict is False
    assert R.STRICT_OPS == frozenset({"＞", ">", "＜", "<", "大于", "小于"})


@pytest.mark.ec("G5")
def test_product_generation_labels_are_not_units():
    """``3D``/``4K`` 里的单字母是代际标签：右边紧贴汉字或 ``/`` 就不算单位。"""
    for text, values in (("3D内窥镜", ["3"]), ("3D/2D切换", ["3", "2"]), ("4K3D摄像系统", ["4", "3"])):
        got = parse_requirement(text)
        assert [str(c.quantity.value) for c in got.constraints] == values
        assert all(c.quantity.unit == "" for c in got.constraints), text
    # 真单位：``D`` 右边是破折号或字符串结尾，屈光度必须保住
    assert _bounds(parse_requirement("-24.5D～+20D")) == [("-24.5", "20", "d")]
    # 已知缺口：``&`` 不在 _DESIGNATOR_NEXT 里，``3D&4K`` 的 3 被当成屈光度。
    # 这条断言钉住当前行为，缺口在 README「已知问题」里登记，不做静默修补。
    assert parse_requirement("3D&4K").constraints[0].quantity.unit == "d"


@pytest.mark.ec("G8")
def test_resolution_literal_masks_the_numbers_inside_it():
    """分辨率字面内部的数字不能外溢成约束，否则 ``1920*1080`` 会变成两个下界。"""
    single = parse_requirement("主机分辨率1536×1536pixel")
    assert single.kind == "resolution"
    assert (single.resolution.width, single.resolution.height) == (1536, 1536)
    assert single.resolution.unit == "像素" and single.constraints == ()

    multi = parse_requirement("输出分辨率：（≥3840*2160），主屏（≥1920*1080P），图像帧率：60 帧/秒")
    assert (multi.resolution.width, multi.resolution.height) == (3840, 2160)
    assert multi.resolution_clause == "输出分辨率：（≥3840*2160）"  # 只取第一个
    assert _cons(multi) == [("min", "60", "", "", "")]  # 帧率保留，1920/1080 被屏蔽


@pytest.mark.ec("C5")
def test_width_mode_needs_both_a_width_word_and_a_real_interval():
    """``区间范围≥600μs`` 比的是宽度，``工作范围≥300mm`` 比的是端点——门是 AND。"""
    width = parse_requirement("脉冲宽度：200μs～800μs可调，区间范围≥600μs")
    con = next(c for c in width.constraints if c.op == "≥")
    assert con.is_width(True) is True and con.is_width(False) is False
    assert _bounds(width) == [("200", "800", "μs")]

    point = parse_requirement("工作范围≥300mm").constraints[0]
    assert point.clause and any(w in point.clause for w in R.WIDTH_WORDS)
    assert point.is_width(True) is True  # 只满足词表这一半
    # 调用点传的 has_bound 来自 ReqSpec.bounds；这条没有区间，所以实际拿到 False
    assert point.is_width(False) is False


@pytest.mark.ec("G4")
def test_period_is_normalized_and_composite_rate_units_survive():
    """``每年`` 归一成周期词用于 ``次``↔``次/年`` 搭桥；``3次/年`` 自带复合单位。"""
    yearly = parse_requirement("（2）设备性能维护：每年免费提供≥4 次的日常巡检维护")
    assert yearly.period == "年"
    assert [c.quantity.unit for c in yearly.constraints] == ["", "次"]

    rate = parse_requirement("年技术培训次数：≥3次/年")
    assert _cons(rate) == [("min", "3", "次/年", "≥", "count_rate")]
    assert rate.period == ""  # 单位里已含周期，不再另抽


@pytest.mark.ec("C5")
def test_typesetting_noise_is_rejected_rather_than_guessed():
    """``0.l-5ml`` 是排版噪声：区间解析拒收，只留下能确定的裸数字，不猜端点。"""
    spec = parse_requirement("0.l-5ml／每次最大给药量")
    assert spec.bounds == ()
    assert _cons(spec) == [("min", "0", "", "", ""), ("min", "5", "ml", "", "volume")]


@pytest.mark.ec("C8")
@pytest.mark.ec("C9")
def test_kind_precedence_and_text_fallback():
    """kind 优先级 resolution > interval > numeric > text；纯文字与空输入都不可比。"""
    both = parse_requirement("输出分辨率：（≥3840*2160），输入电压范围：100V-240V")
    assert both.kind == "resolution" and both.bounds and both.comparable
    assert parse_requirement("温度控制范围：33℃-43℃").kind == "interval"
    assert parse_requirement("治疗深度：≥7CM").kind == "numeric"

    text = parse_requirement("临床用途：可用于泌尿外科、普外科手术")
    assert text.kind == "text" and text.comparable is False

    for empty in (None, "", "   "):
        spec = parse_requirement(empty)
        # raw 保留原样（None 才归一成 ""），text 是 squash 形，空白一律塌成空串
        assert spec.raw == (empty or "") and spec.text == "" and spec.kind == "text"
        assert spec.comparable is False and spec.constraints == () and spec.resolution is None


@pytest.mark.ec("H2")
def test_leading_serial_number_leaves_a_harmless_unitless_constraint():
    """``1.6、`` 的序号会被扫成一个无单位约束，判定层靠量纲过滤丢掉它。"""
    spec = parse_requirement("1.6、瞳孔大小：≤3mm；")
    assert _cons(spec) == [
        ("min", "1.6", "", "", ""),
        ("max", "3", "mm", "≤", "length"),
    ]
    residue = spec.constraints[0]
    assert residue.quantity.dimension == "" and residue.quantity.unit == ""


@pytest.mark.ec("C6")
def test_every_constraint_remembers_its_own_clause_for_x20():
    """X20 要求复合要求只按被回答的子句判，所以每个约束都得带上子句原文。"""
    spec = parse_requirement("直径10mm，30°腹腔镜，长度≥300mm。")
    assert [c.clause for c in spec.constraints] == ["直径10mm", "30°腹腔镜", "长度≥300mm"]
    # clause 存的是**未 squash** 的原文，便于在复核报告里回溯
    assert spec.text == "直径10mm,30°腹腔镜,长度≥300mm."


@pytest.mark.ec("G7")
def test_unit_vocabulary_covers_scaled_counts_and_composite_rates():
    """``万像素`` 的万倍缩放、``段``/``台`` 计数单位、``l/min`` 复合单位都要有量纲。"""
    assert _cons(parse_requirement("分辨率≥16万像素")) == [("min", "16", "万像素", "≥", "count")]
    assert _cons(parse_requirement("基础率时段数：≥48段")) == [("min", "48", "段", "≥", "count")]
    assert _bounds(parse_requirement("设置流量的调节范围：1L/min～40L/min")) == [("1", "40", "l/min")]
    # 缺口：``辆`` 不在单位词表里，只能留空单位（靠量纲过滤兜住，不误判）
    assert _cons(parse_requirement("配置清单：主机1台、台车1辆")) == [
        ("min", "1", "台", "", "count"),
        ("min", "1", "", "", ""),
    ]


@pytest.mark.ec("C7")
def test_long_configuration_lists_keep_every_clause_independently():
    """长配置清单：``、`` 不切子句，每个数量各带自己的单位与比较符，不互相吞并。"""
    spec = parse_requirement("配置：主机1台、台车1辆、脚踏开关1个、导光束2条（长度≥2米，通光孔径≥4.8mm）")
    assert len(clauses(spec.raw)) == 2  # 只有 ``，`` 处断开
    assert _cons(spec) == [
        ("min", "1", "台", "", "count"),
        ("min", "1", "", "", ""),  # 辆不在词表里
        ("min", "1", "个", "", "count"),
        ("min", "2", "条", "", "count"),
        ("min", "2", "m", "≥", "length"),
        ("min", "4.8", "mm", "≥", "length"),
    ]
    assert len({c.clause for c in spec.constraints}) == 2
    # ``至少包含100V-240V`` 这类「区间 + 前置词」不能被读成数值约束
    span = parse_requirement("输入电压范围：至少包含100V-240V")
    assert span.kind == "interval" and span.constraints == ()
    assert _bounds(span) == [("100", "240", "v")]
