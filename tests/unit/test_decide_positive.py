"""判定层·方向性与 τ 边界（EC-C 要求类型 / EC-E 判定结果 / EC-G 单位等价 / EC-J3）。

每条用例只压一个能力：下界方向的 τ 严格性、上界方向永不产生正偏离、
低置信带的闭端点、单位换算、取等恒判无偏离、单位继承的单向性、周期桥接、
多子句归并。gold 的「判定说明」里逐字写着换算依据，所以这些不是实现细节。
"""

from __future__ import annotations

from decimal import Decimal as D

import pytest

from bidtable.config import DEV_NEGATIVE, DEV_NONE, DEV_POSITIVE
from bidtable.judge import decide as J
from bidtable.judge.units import Interval, Quantity

pytestmark = [pytest.mark.unit]

ST = J.settings_of(None)  # τ=1.10 / band=(1.05, 1.15) / ocr_fallback="assume"


def _v(direction: str, need, got) -> J.Compare:
    return J.verdict(direction, D(str(need)), D(str(got)), ST)


def _q(value, unit: str) -> Quantity:
    return Quantity(D(str(value)), unit)


def _cmp(dev: str) -> J.Compare:
    return J.Compare(dev)


@pytest.mark.ec("C1")
@pytest.mark.ec("E2")
@pytest.mark.case("T14")
def test_tau_is_strict_and_direction_aware():
    """T14：``≥30秒`` 对 ``34秒`` = 1.1333 倍，刚过 τ 就是正偏离。

    恰好等于 τ（1.10）仍算无偏离——判据是 ``ratio > tau`` 而不是 ``>=``。
    这条严格性正是 gold 里 1.0786（无偏离）与 1.1333（正偏离）之间的分界线，
    差一点点就会把整包的偏离类型准确率拖下去。
    """
    want = [
        ("min", 100, 99, DEV_NEGATIVE, "0.99"),
        ("min", 100, 100, DEV_NONE, "1"),
        ("min", 100, 101, DEV_NONE, "1.01"),
        ("min", 100, 105, DEV_NONE, "1.05"),
        ("min", 700, 755, DEV_NONE, "1.078571428571428571428571429"),
        ("min", 100, 110, DEV_NONE, "1.1"),  # == τ，不算正偏离
        ("min", 30, 34, DEV_POSITIVE, "1.133333333333333333333333333"),
        ("min", 100, 115, DEV_POSITIVE, "1.15"),
        ("min", 100, 120, DEV_POSITIVE, "1.2"),
    ]
    for direction, need, got, dev, ratio in want:
        c = _v(direction, need, got)
        assert (c.deviation, str(c.ratio)) == (dev, ratio), (direction, need, got)


@pytest.mark.ec("C2")
@pytest.mark.ec("C3")
@pytest.mark.ec("E1")
def test_upper_bound_never_yields_positive_deviation():
    """X22 的数值侧：上界方向（``≤``/``＜``）省得再多也只是无偏离，超上限才是负偏离。

    上界的比值取倒数（need/got），所以 ``max 3 vs 2`` 的 ratio 高达 1.5 却**不**带
    low_margin——`verdict` 自己从不打旗标，低置信带只由调用方按"刚过门槛"语义叠加。

    比值一律按 28 位有效数字写死：这条断言同时钉住"全程 Decimal、不碰 float"
    （B7 关心的量化误差），换成二进制浮点第 17 位就会漂。
    """
    cases = [
        (3, 3, DEV_NONE, "1"),
        (3, 2, DEV_NONE, "1.5"),
        (4, 3, DEV_NONE, "1.333333333333333333333333333"),
        (5, 3, DEV_NONE, "1.666666666666666666666666667"),
        (3, "3.21", DEV_NEGATIVE, "0.9345794392523364485981308411"),
        (3, 4, DEV_NEGATIVE, "0.75"),
        (100, 105, DEV_NEGATIVE, "0.9523809523809523809523809524"),
    ]
    for need, got, dev, ratio in cases:
        c = _v("max", need, got)
        assert (c.deviation, str(c.ratio), c.flags) == (dev, ratio, ()), (need, got)


@pytest.mark.ec("J3")
@pytest.mark.ec("E1")
def test_low_margin_band_is_closed_on_both_ends():
    """``[1.05, 1.15]`` 两端都闭；ratio 为 None（区间不覆盖）时无从打标。"""
    assert ST.band == (D("1.05"), D("1.15"))
    for ratio, want in [
        ("1.0499", ()), ("1.05", ("low_margin",)), ("1.10", ("low_margin",)),
        ("1.15", ("low_margin",)), ("1.1501", ()),
    ]:
        assert J.margin_flag(D(ratio), ST) == want, ratio
    assert J.margin_flag(None, ST) == ()


@pytest.mark.ec("G2")
@pytest.mark.case("T12")
def test_unit_conversion_makes_milliamps_comparable_with_amps(mk_match):
    """X7/T12：``9360`` 绝不能直接和 ``10`` 比。先归一量纲，再判方向。"""
    al = J.align_units(_q(10, "a"), _q(9360, "ma"))
    assert (al.need, al.got, al.unit, al.flags) == (D(10), D("9.360"), "a", ("unit_converted",))
    assert J.align_units(_q(20, "kg"), _q(20000, "g")).got == D(20)
    assert J.align_units(_q(10, "mm"), _q(5, "w")) is None  # 跨量纲宁可不比

    dec = J.decide(mk_match("工作电流：≤10A", "工作电流", "9360mA"))
    assert dec.deviation == DEV_NONE and dec.method == "compare"
    assert dec.flags == ("unit_converted", "low_margin") and dec.confidence == 0.600


@pytest.mark.ec("C4")
@pytest.mark.ec("E1")
@pytest.mark.case("T13")
def test_equality_is_always_no_deviation(mk_match):
    """取等即达标：T13 ``≥40L/min`` 对 ``40L/min``。ratio 恰为 1，且不进低置信带。"""
    eq = _v("min", 100, 100)
    assert str(eq.ratio) == "1" and eq.flags == () and "取等" in eq.detail
    assert J.margin_flag(D(1), ST) == ()

    dec = J.decide(mk_match("流量：≥40L/min", "流量", "40L/min"))
    assert (dec.deviation, dec.method, dec.flags) == (DEV_NONE, "compare", ())
    assert dec.confidence == 0.900


@pytest.mark.ec("G5")
@pytest.mark.case("B24")
def test_unit_inheritance_runs_one_way_only():
    """B24：响应漏写单位就继承要求侧单位（打 ``unit_assumed``）；反方向**不**继承。

    要求侧的裸数字往往是 ``60帧/秒`` 这类没有单位词的量，让它去继承 ``像素``
    会把 ``≥16万像素`` 判成几千倍的正偏离。区间通道遵循同一条不对称规则。
    """
    al = J.align_units(_q(7, "英寸"), _q(6, ""))
    assert (al.unit, al.flags) == ("英寸", ("unit_assumed",))
    assert J.align_units(_q(16, ""), _q(725, "像素")) is None
    assert J._interval_in(Interval(D(10), D(450), ""), "ml/min").unit == "ml/min"
    assert J._interval_in(Interval(D(1), D(2), "ml/min"), "") is None


@pytest.mark.ec("G4")
@pytest.mark.case("B6")
def test_period_bridge_connects_count_to_count_rate(mk_match):
    """``次`` ↔ ``次/年`` 靠要求侧的周期词搭桥；周期词冲突时桥断开（宁可不比）。"""
    need, got = _q(4, "次"), _q("4.22", "次/年")
    ok = J.align_units(need, got, rs_period="年")
    assert (ok.need, ok.got, ok.unit, ok.flags) == (D(4), D("4.22"), "次/年", ("unit_assumed",))
    assert J.align_units(need, got, rs_period="月") is None
    assert J.align_units(need, got).flags == ("unit_assumed",)  # 无周期词时放行
    rev = J.align_units(got, need, vs_period="年")
    assert (rev.need, rev.got, rev.unit) == (D("4.22"), D(4), "次/年")

    dec = J.decide(mk_match("每年免费提供≥4次的日常巡检维护", "每年维护保养次数", "4.22次/年"))
    assert dec.deviation == DEV_NONE
    assert dec.flags == ("unit_assumed", "low_margin") and dec.confidence == 0.600


@pytest.mark.ec("C6")
@pytest.mark.ec("E3")
def test_reduce_takes_the_worst_sub_verdict():
    """多子句归并：负偏离(2) > 正偏离(1) > 无偏离(0)；空列表退回无偏离。"""
    assert J.reduce([_cmp(DEV_NONE), _cmp(DEV_POSITIVE)]) == DEV_POSITIVE
    assert J.reduce([_cmp(DEV_POSITIVE), _cmp(DEV_NEGATIVE)]) == DEV_NEGATIVE
    assert J.reduce([_cmp(DEV_NONE), _cmp(DEV_POSITIVE), _cmp(DEV_NEGATIVE)]) == DEV_NEGATIVE
    assert J.reduce([]) == DEV_NONE
