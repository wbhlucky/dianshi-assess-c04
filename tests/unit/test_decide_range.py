"""判定层·区间与宽度语义（EC-C5 区间包含 / EC-C6 复合多子句 / EC-G2 单位等价）。

区间判定不是"包含就完事"：覆盖之后还要看放宽了多少倍才决定无偏离还是正偏离，
负端点不能取倒数，单位按字面归一，宽度约束只能拿跨度去比。每个用例盯一条规则。
"""

from __future__ import annotations

from decimal import Decimal as D

import pytest

from bidtable.config import DEV_NEGATIVE, DEV_NONE, DEV_POSITIVE
from bidtable.judge import decide as J
from bidtable.judge.parse_req import parse_requirement
from bidtable.judge.parse_val import parse_value
from bidtable.judge.units import Interval

pytestmark = [pytest.mark.unit]

ST = J.settings_of(None)


def _iv(lo, hi, unit: str = "") -> Interval:
    return Interval(D(str(lo)), D(str(hi)), unit)


@pytest.mark.ec("C5")
@pytest.mark.ec("E1")
@pytest.mark.ec("E2")
@pytest.mark.ec("E3")
def test_interval_coverage_decides_relaxation_not_mere_containment():
    """覆盖只是入场券：再看放宽倍数决定无偏离还是正偏离，不覆盖直接负偏离。

    放宽倍数落在 [1.05,1.15] 里会连带打上 low_margin，供复核报告优先排序。
    """
    c = J.compare_intervals(_iv(33, 43, "℃"), _iv(33, 43, "℃"), ST)
    assert (c.deviation, c.ratio, c.flags) == (DEV_NONE, D(1), ())
    assert c.detail == "区间 [33,43] 与要求一致"

    c = J.compare_intervals(_iv(33, 43, "℃"), _iv(32, 44, "℃"), ST)
    assert (c.deviation, c.ratio, c.flags) == (DEV_NONE, D("1.03125"), ())

    c = J.compare_intervals(_iv(200, 800, "μs"), _iv(190, 810, "μs"), ST)
    assert (c.deviation, c.flags) == (DEV_NONE, ("low_margin",))

    c = J.compare_intervals(_iv(37, 43, "℃"), _iv(33, 43, "℃"), ST)
    assert c.deviation == DEV_POSITIVE and c.flags == ("low_margin",)
    assert c.ratio == D("1.121212121212121212121212121")

    c = J.compare_intervals(_iv(33, 43, "℃"), _iv(37, 43, "℃"), ST)
    assert (c.deviation, c.ratio, c.flags) == (DEV_NEGATIVE, None, ())

    assert J.compare_intervals(_iv(33, 43, "℃"), _iv(33, 43, "mm"), ST) is None


@pytest.mark.ec("C5")
@pytest.mark.ec("E2")
def test_negative_endpoints_are_excluded_from_the_relaxation_ratio():
    """屈光补偿 ``-24.5D～+20D``：对负端点取倒数没有物理意义，只能看正端。

    ``need.lo > 0 and g.lo > 0`` 这道闸两侧都挡：要求端为负（-24.5D）与
    响应端为负（-6.37 l/min）都只按上端算，比值因此是正的 1.25 / 1.20。
    """
    c = J.compare_intervals(_iv(-24.5, 20, "d"), _iv(-30, 25, "d"), ST)
    assert (c.deviation, c.ratio) == (DEV_POSITIVE, D("1.25"))
    c = J.compare_intervals(_iv(1, 40, "l/min"), _iv(-6.37, 48, "l/min"), ST)
    assert (c.deviation, c.ratio) == (DEV_POSITIVE, D("1.2"))


@pytest.mark.ec("G2")
@pytest.mark.ec("C5")
def test_interval_unit_normalization_is_identity_preserving_and_literal():
    """同单位直接返回**原对象**（不重建、不掉精度）；跨量纲或别名单位一律拒绝。"""
    iv = _iv(10, 450, "ml/min")
    assert J._interval_in(iv, "ml/min") is iv

    conv = J._interval_in(_iv(1, 2, "l/min"), "ml/min")
    assert (conv.lo, conv.hi, conv.unit) == (D(1000), D(2000), "ml/min")
    assert str(conv.lo) == "1E+3"  # Decimal 的规范指数形：断言必须按数值而非字面

    assert J._interval_in(_iv(1, 2, "mm"), "w") is None
    # 已知缺口：单位别名归一只发生在 normalize.unit_key，这里按字面比，
    # 所以 "μ" 与 "μm" 互不相认（P5 标的12 序号4 因此退回文字型，结论仍是无偏离）。
    assert J._interval_in(_iv(8, 10, "μm"), "μ") is None


@pytest.mark.ec("C5")
@pytest.mark.ec("E1")
@pytest.mark.ec("E3")
def test_point_response_against_an_interval_is_pure_containment():
    """闭区间：端点值算达标；出界一律负偏离，且**永不**判正偏离（哪怕超 11 倍）。"""
    need = _iv(1, 40, "l/min")
    for got, dev in [("1", DEV_NONE), ("10", DEV_NONE), ("40", DEV_NONE),
                     ("0.5", DEV_NEGATIVE), ("450", DEV_NEGATIVE)]:
        c = J._point_in_interval(need, J.Aligned(D(1), D(got), "l/min"))
        assert c.deviation == dev, got
        assert (c.ratio, c.flags) == ((D(1), ()) if dev == DEV_NONE else (None, ()))


@pytest.mark.ec("C5")
def test_endpoint_picks_the_number_that_faces_the_threshold():
    """区间响应碰单点门槛时取有利端点（下界看上端、上界看下端）；宽度只看跨度。

    单值响应无法回答宽度约束 -> None，该原子被跳过。若拿端点去比，
    ``200~800μs`` 对 ``区间范围≥600μs`` 会得到 800/600=1.33 的假正偏离。
    """
    iv = J.Item("interval", "ml/min", "x", interval=_iv(10, 450, "ml/min"))
    pt = J.Item("point", "μs", "x", point=D(300))
    assert J.endpoint(iv, "min") == D(450)
    assert J.endpoint(iv, "max") == D(10)
    assert J.endpoint(pt, "min") == D(300)
    assert J.endpoint(iv, "min", width=True) == D(440)
    assert J.endpoint(pt, "min", width=True) is None


@pytest.mark.ec("C5")
@pytest.mark.ec("C6")
def test_width_constraint_compares_the_span_and_skips_bare_points(mk_match):
    """P5 ``200μs～800μs可调，区间范围≥600μs``：跨度 = 800-200 = 600，恰好取等。

    厂商只给单值时宽度原子直接跳过（endpoint 返回 None），区间原子照常判，
    不会拿 ``500μs`` 去和 ``≥600μs`` 比出一个假负偏离。
    """
    req = "脉冲宽度：200μs～800μs可调，区间范围≥600μs"

    both = mk_match(req, "脉冲宽度", "200μs～800μs可调，区间范围700μs")
    cs = J.structural(parse_requirement(req), parse_value(both.evidence), ST)
    assert [(c.deviation, c.ratio) for c in cs] == [(DEV_NONE, D(1))] * 2
    assert "600μs vs 要求 ≥600μs" in cs[1].detail  # 跨度，不是 700 也不是 800

    point = mk_match(req, "脉冲宽度", "500μs")
    cs = J.structural(parse_requirement(req), parse_value(point.evidence), ST)
    assert len(cs) == 1 and "落在要求区间 [200,800] 内" in cs[0].detail
    assert J.decide(point).deviation == DEV_NONE


@pytest.mark.ec("C6")
@pytest.mark.case("X20")
def test_partial_answer_collapses_to_the_best_aligned_atom(mk_match):
    """X20：厂商只答了复合要求里的一部分时，只判被答的那一句。

    判据是"响应项数 < 要求原子数"：三个门槛只答 ``2条`` 就只留一条子判定；
    三个门槛全答（``10X、16X、25X``）就三条都留；完全不可比时一条都没有。
    分辨率原子与阈值/区间原子走同一套归并，不做特殊处理。
    """

    def _atoms(req: str, name: str, value: str) -> list:
        m = mk_match(req, name, value)
        return J.structural(parse_requirement(req), parse_value(m.evidence), ST)

    one = _atoms("通用导光束≥2条，长度≥2米，通光孔径≥4.8mm", "通用导光束数量", "2条")
    assert len(one) == 1 and (one[0].deviation, one[0].ratio) == (DEV_NONE, D(1))

    three = _atoms("配置：10X、16X、25X", "放大倍数", "10X、16X、25X")
    assert len(three) == 3 and all(c.deviation == DEV_NONE for c in three)

    res = _atoms("输出分辨率：（≥3840*2160），主屏（≥1920*1080P），图像帧率：60 帧/秒",
                 "输出分辨率", "3840*2160")
    assert len(res) == 1 and res[0].flags == ("unit_assumed",)

    assert _atoms("临床用途：可用于泌尿外科、普外科手术", "临床用途", "泌尿外科") == []
    assert _atoms("工作长度：≥670 mm", "工作长度", "") == []

    dec = J.decide(mk_match("通用导光束≥2条，长度≥2米，通光孔径≥4.8mm",
                            "通用导光束数量", "2条"))
    assert dec.deviation == DEV_NONE and dec.method == "compare"
