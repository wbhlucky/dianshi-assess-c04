"""判定层·对齐与分辨率（EC-G8 复合分辨率 / EC-G7 万倍单位 / EC-C6 复合子句选谁）。

一条要求里往往有好几个数，一个响应里也可能有好几个数：谁跟谁比，比错了就是
几十倍的假正偏离。分辨率只比宽度、测量条件不是门槛、对齐键名字分优先，
这三条规则各自挡住一类真实错判。
"""

from __future__ import annotations

from decimal import Decimal as D

import pytest

from bidtable.config import DEV_NEGATIVE, DEV_NONE
from bidtable.judge import decide as J
from bidtable.judge.parse_req import parse_requirement
from bidtable.judge.parse_val import parse_value
from bidtable.judge.units import Resolution
from bidtable.models import Evidence

pytestmark = [pytest.mark.unit]

ST = J.settings_of(None)


def _res(w, h, unit: str = "像素") -> Resolution:
    return Resolution(D(str(w)), D(str(h)), unit)


def _cmp(dev, ratio=None, flags=()) -> J.Compare:
    return J.Compare(dev, ratio, "", flags)


def _ev(name: str, value: str = "4.0cm") -> Evidence:
    return Evidence(lot_no="1", model="M", doc_type="技术规格书", filename="f.docx",
                    locator="技术规格书 表行1", name=name, value=value,
                    response_text=f"{name}：{value}", priority=1)


@pytest.mark.ec("G8")
@pytest.mark.case("B3")
def test_resolution_is_compared_by_width_and_gates_on_eligibility():
    """B3：``1536×1536pixel`` 对 ``1370pixel`` 是负偏离——比的是**宽度**，不是面积或高度。

    准入闸 ``_resolution_eligible`` 决定哪些响应项能拿去和分辨率要求比：
    无量纲裸数、像素、计数量纲都收；长度/区间不收（否则 ``310mm`` 会变成 310 万像素）。
    ``point is None`` 一律不收，这是防 ``Item`` 半成品漏进来。
    """
    c = J.compare_resolution(_res(1536, 1536), D(1370), "像素", ST)
    assert (c.deviation, c.flags) == (DEV_NEGATIVE, ())
    assert c.ratio == D("0.8919270833333333333333333333")
    assert c.detail == "分辨率宽度 1370 vs 要求 1536"

    table = [("resolution", "像素", D(3840), None, True), ("point", "", D(3850), None, True),
             ("point", "像素", D(3850), None, True), ("point", "个", D(2), None, True),
             ("point", "mm", D(310), None, False), ("point", "像素", None, None, False),
             ("interval", "mm", None, J.Interval(D(1), D(2), "mm"), False)]
    for kind, unit, point, interval, want in table:
        it = J.Item(kind, unit, "x", point=point, interval=interval)
        assert J._resolution_eligible(it) is want, (kind, unit, point)


@pytest.mark.ec("G8")
@pytest.mark.case("B4")
def test_a_marginally_wider_panel_is_not_a_deviation(mk_match):
    """B4：``3840*2160P`` 对 ``3850像素`` 判无偏离，比值 1.0026 远在 τ 之下。

    同一个要求，响应写不写单位会改变置信度而不改变结论：``3850像素`` 同量纲
    拿满 0.90，裸 ``3850`` 触发 B24 单位继承 -> ``unit_assumed`` 封顶 0.75。
    """
    with_unit = J.decide(mk_match("输出分辨率：3840*2160P", "输出分辨率", "3850像素"))
    assert (with_unit.deviation, with_unit.method) == (DEV_NONE, "compare")
    assert with_unit.confidence == 0.9 and with_unit.flags == ()

    bare = J.decide(mk_match("输出分辨率：3840*2160P", "输出分辨率", "3850"))
    assert (bare.deviation, bare.confidence, bare.flags) == (DEV_NONE, 0.75, ("unit_assumed",))

    exact = J.compare_resolution(_res(1920, 1080), D(1920), "像素", ST)
    assert (exact.deviation, exact.ratio) == (DEV_NONE, D(1))


@pytest.mark.ec("G8")
@pytest.mark.ec("E5")
def test_unitless_resolution_requirement_refuses_a_unit_bearing_answer():
    """要求侧漏写单位时，``3850像素`` 反而**不可比**（``same_dimension("像素","")`` 为假）。

    单位继承只朝一个方向走：响应漏单位可以继承要求的，要求漏单位不能吸收响应的。
    于是 ``（≥3840*2160）`` 这种写法只能退回文字型并打 low_confidence，
    而裸 ``3850`` 因为两侧都无单位反倒能比。已知缺口，结论仍是无偏离，故不致命。
    """
    need = _res(3840, 2160, "")
    assert J.compare_resolution(need, D(3850), "像素", ST) is None
    assert J.compare_resolution(need, D(3850), "mm", ST) is None
    assumed = J.compare_resolution(need, D(3850), "", ST)
    assert (assumed.deviation, assumed.flags) == (DEV_NONE, ("unit_assumed",))

    fell = J.compare_resolution(_res(3840, 2160, ""), D(3850), "", ST)
    assert fell.ratio == D("1.002604166666666666666666667")


@pytest.mark.ec("G7")
def test_scaled_count_units_convert_before_the_ratio_is_taken():
    """``16万像素`` 与 ``160000像素`` 是同一个量：先换算再比，比值恰为 1 而非 1e4。

    换算发生过就打上 ``unit_converted``，复核报告第 5 节会列出换算依据。
    """
    c = J.compare_resolution(_res(16, 9, "万像素"), D(160000), "像素", ST)
    assert (c.deviation, c.ratio, c.flags) == (DEV_NONE, D(1), ("unit_converted",))
    assert c.detail == "分辨率宽度 16 vs 要求 16"


@pytest.mark.ec("G1")
@pytest.mark.ec("E1")
def test_the_alignment_key_ranks_unit_fidelity_above_scores():
    """对齐键 ``(单位保真, 取等, -名字分, -数值分)`` 越小越优，**前两键压过任何相似度**。

    同单位比换算可信、换算比裸数字继承可信；``unit_assumed`` 与 ``unit_converted``
    同时出现时按更差的算。取等优先是因为厂商原样写出要求值就是在回答那一句。
    """
    assert [J._rank(_cmp(DEV_NONE, D(1))), J._rank(_cmp(DEV_NONE, D(1), ("unit_converted",))),
            J._rank(_cmp(DEV_NONE, D(1), ("unit_assumed",))),
            J._rank(_cmp(DEV_NONE, D(1), ("unit_assumed", "unit_converted")))] == [0, 1, 2, 2]
    assert J._exact(_cmp(DEV_NONE, D(1))) == 0
    assert J._exact(_cmp(DEV_NONE, D("1.05"))) == 1
    assert J._exact(_cmp(DEV_NEGATIVE, None)) == 1  # 无比值 -> 谈不上取等

    key = J._key
    assert key(_cmp(DEV_NONE, D(1)), (0.1, 0.1)) < key(_cmp(DEV_NONE, D(1), ("unit_converted",)), (0.9, 0.9))
    assert key(_cmp(DEV_NONE, D(1)), (0.2, 0.9)) < key(_cmp(DEV_NONE, D("1.05")), (0.9, 0.1))
    assert key(_cmp(DEV_NONE, D(1)), (0.9, 0.1)) < key(_cmp(DEV_NONE, D(1)), (0.1, 0.9))


@pytest.mark.ec("C6")
@pytest.mark.case("X20")
def test_the_vendor_indicator_name_picks_the_clause(mk_match):
    """X20 的另一半：复合要求里选哪一句，由**厂商自己写的指标名**决定，不由数值重合度决定。

    ``直径10mm，30°腹腔镜，长度≥300mm`` 对上 ``4K腹腔镜长度：310mm``：
    ``直径10mm`` 与 ``310mm`` 的字符重合度（0.60）高于 ``长度≥300mm``（0.55 对 0.20），
    把数值分当主键就会挑错子句，判出 31 倍的假正偏离。名字分排在第一键正是为此。
    """
    it = J.Item("point", "mm", "4k腹腔镜长度310mm", point=D(310))
    by_name = J.align_score("长度≥300mm", "4k腹腔镜长度", it)
    by_value = J.align_score("直径10mm", "4k腹腔镜长度", it)
    assert by_name[0] > by_value[0] and by_value[1] > by_name[1]
    assert by_name[0] == pytest.approx(0.2024, abs=1e-4)
    assert J.align_score("直径10mm", "4k腹腔镜直径", J.Item("point", "mm", "直径10mm", point=D(10)))[1] == 1.0

    d = J.decide(mk_match("直径10mm，30°腹腔镜，长度≥300mm。", "4K腹腔镜长度", "310mm"))
    assert (d.deviation, d.method, d.confidence) == (DEV_NONE, "compare", 0.9)
    assert d.reason == "4K腹腔镜长度 310mm vs 要求 ≥300mm"


@pytest.mark.ec("C6")
def test_a_measurement_condition_is_not_a_threshold():
    """``200μm软光纤最小弯曲半径：≤4.0cm`` 里的 ``200μm`` 说的是"在哪种光纤上测"。

    判据三条同时成立才算测量条件：无比较符的裸数值 + 带单位 + 该数值原样出现在
    响应的指标名里。半角 ``um`` 也算命中（走单位别名表）。少任何一条都不豁免，
    所以 ``具有≥2个夹管阀`` 对 ``夹管阀数量`` 仍然是真门槛。
    """
    rs = parse_requirement("200μm软光纤最小弯曲半径：≤4.0cm")
    assert [(c.direction, c.op, str(c.quantity.value), c.quantity.unit) for c in rs.constraints] == [
        ("min", "", "200", "μm"), ("max", "≤", "4.0", "cm")]

    for name in ("200μm软光纤最小弯曲半径", "200um软光纤最小弯曲半径"):
        vs = parse_value(_ev(name))
        assert [J._is_test_condition(c, vs) for c in rs.constraints] == [True, False]
    for name in ("最小弯曲半径", "软光纤最小弯曲半径"):
        assert [J._is_test_condition(c, parse_value(_ev(name))) for c in rs.constraints] == [False, False]

    plain = parse_requirement("具有≥2个夹管阀")
    assert [J._is_test_condition(c, parse_value(_ev("夹管阀数量"))) for c in plain.constraints] == [False]


@pytest.mark.ec("G8")
@pytest.mark.ec("C6")
def test_a_resolution_response_only_answers_a_resolution_requirement(mk_match):
    """分辨率形状的响应项被排除在阈值/区间比较之外，反之亦然。

    ``工作长度：≥670 mm`` 对上 ``3840*2160``：结构化比对返回空，退回文字型；
    若放行，3840 会被当成 3840mm，判出 5.7 倍的假正偏离。
    """
    m = mk_match("工作长度：≥670 mm", "输出分辨率", "3840*2160")
    vs = parse_value(m.evidence)
    items = J.response_items(vs)
    assert len(items) == 1 and items[0].kind == "resolution" and items[0].point == D(3840)
    assert J.structural(parse_requirement("工作长度：≥670 mm"), vs, ST) == []

    ok = mk_match("输出分辨率：3840*2160P", "输出分辨率", "3840*2160")
    cs = J.structural(parse_requirement("输出分辨率：3840*2160P"), parse_value(ok.evidence), ST)
    assert [(c.deviation, c.ratio) for c in cs] == [(DEV_NONE, D(1))]

    with pytest.raises(TypeError):
        J.compare_resolution(_res(3840, 2160), _res(3850, 2160), "像素", ST)
