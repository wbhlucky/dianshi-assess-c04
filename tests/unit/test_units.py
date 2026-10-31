"""量纲表与数值解析（EC-G 单位等价类）。

gold 的「判定说明」里逐字写着 ``单位换算：9360mA = 9.36A``，所以这张表不是
实现细节而是评分口径的一部分。每条测试盯一个不同的换算/解析能力。
"""

from __future__ import annotations

from decimal import Decimal as D

import pytest

from bidtable.judge import units as U
from bidtable.match import normalize as N

pytestmark = [pytest.mark.unit]


# --------------------------------------------------------------------------- #
# 换算表
# --------------------------------------------------------------------------- #


@pytest.mark.ec("G2")
@pytest.mark.parametrize("value,src,dst,want", [
    (20000, "g", "kg", 20),            # P4 1.17 gold：单位换算：20000g = 20kg
    (9360, "ma", "a", "9.36"),         # P4 1.4  gold：单位换算：9360mA = 9.36A
    (3030000, "min", "h", 50500),      # P3 服务 gold：3030000分钟 = 50500小时
    (12, "英寸", "mm", "304.8"),       # 长度基准 mm，25.4 系数
    (2, "m", "cm", 200),
    (4, "cm", "μm", 40000),
    (60, "h", "日", "2.5"),
    (3, "次/月", "次/年", 36),          # 频次按 12 倍桥接
    (1, "l/min", "ml/min", 1000),
    (3600, "s", "h", 1),               # 时间族精确整除
    (1000, "ml", "l", 1),              # 容量族基准 ml
    (16, "万像素", "像素", 160000),     # 万 = 1e4，最长词优先
    (200, "mw/cm2", "w/cm2", "0.2"),   # c ㎡ / cm² 归一后同量纲
    (1, "像素", "个", 1),              # 像素与计数同属 count
])
def test_convert_anchors_quoted_by_gold(value, src, dst, want):
    assert U.convert(D(value), src, dst) == D(str(want))


@pytest.mark.ec("G5")
def test_convert_takes_canonical_keys_because_aliases_die_in_normalize():
    """本模块的键**已经是规范形**：``寸`` 这种同义词由 ``normalize.unit_key`` 归一，
    直接喂给 ``convert`` 必须返回 ``None`` 而不是猜一个进率（宁可不比，不可错比）。"""
    assert U.convert(D(39), "寸", "英寸") is None
    assert N.unit_key("寸") == N.unit_key("英寸") == "英寸"
    assert U.convert(D(39), N.unit_key("寸"), "英寸") == D(39)   # P3 五、2.2：39寸 = 39英寸


@pytest.mark.ec("G2")
def test_convert_refuses_cross_dimension_and_unknown_units():
    """宁可不比，不可错比：量纲不通或单位未知一律 None，由判定侧退回文字型。"""
    assert U.convert(D(1), "kg", "m") is None
    assert U.convert(D(1), "zz", "mm") is None
    assert U.convert(D(1), "mm", "zz") is None
    assert U.same_dimension("fr", "mm") is False      # 法兰规格不是长度
    assert U.same_dimension("℃", "°") is False       # 温度与角度不同族
    assert U.same_dimension("db", "dpi") is False
    assert U.dimension("zz") == ""


@pytest.mark.ec("G2")
def test_same_unit_short_circuits_without_touching_precision():
    """同单位直接原样返回，避免 ``Decimal`` 除法白白引入量化误差。"""
    v = D("4.22")
    assert U.convert(v, "次/年", "次/年") is v
    assert U.convert(5, "mm", "mm") == 5


@pytest.mark.ec("G2")
@pytest.mark.case("B7")
def test_close_absorbs_repeating_decimal_artifacts_from_conversion():
    """``ml/h`` 的基准系数是 1/60，换算结果是循环小数：
    ``450 ml/min`` 换成 ``ml/h`` 得到 ``26999.999…`` 而不是 ``27000``。
    没有这层容差，取等判定会在换算后翻成负偏离（B7 的真正风险不在浮点而在量化）。"""
    got = U.convert(D(450), "ml/min", "ml/h")
    assert got != D(27000) and U.close(got, D(27000)) is True
    back = U.convert(D(27000), "ml/h", "ml/min")
    assert back > D(450) and U.close(back, D(450)) is True   # 裸比较会误判，close 才救得回来
    assert U.close(D("19.999999999"), D(20)) is True
    assert U.close(D("450.0001"), D(450)) is False           # 容差不是万能的，真差异仍要暴露


@pytest.mark.ec("G4")
def test_flow_and_time_families_share_a_dimension():
    """分钟↔小时、ml/min↔ml/h 必须可换算，否则 P3 服务与 P5 血泵流速都判不了。"""
    assert U.same_dimension("ml/min", "ml/h") is True
    assert U.same_dimension("min", "h") is True
    assert U.dimension("ma") == U.dimension("a") == "current"
    # `c㎡` / `cm²` 只有过 squash+unit_key 才是同一个键，量纲表本身不认识异形写法
    assert U.dimension("c㎡") == ""
    assert U.dimension(N.unit_key("c㎡")) == U.dimension(N.unit_key("cm²")) == "area"
    assert U.dimension("次/年") == U.dimension("次/月") == "count_rate"
    assert U.to_base(D(20000), "g") == D(20000) and U.to_base(D(1), "km") == D(1000000)


@pytest.mark.ec("G1")
def test_ratio_guards_the_zero_denominator_and_keeps_decimal_precision():
    """分母为 0 时返回 None，判定侧据此退回文字型而不是抛 ZeroDivisionError；
    非零分母走 ``Decimal`` 除法（28 位有效数字），比 float 多留 11 位——
    τ 卡在 1.0786/1.1333 这种贴边比值上时，少一位就可能翻判定。"""
    assert U.ratio(D(1), D(0)) is None
    r = U.ratio(D(755), D(700))                       # 训练集实测最大无偏离比值
    assert isinstance(r, D) and r == D("1.078571428571428571428571429")
    assert r != D(str(755 / 700))                     # float 只剩 1.0785714285714286
    assert U.ratio(D(34), D(30)) == D("1.133333333333333333333333333")  # 实测最小正偏离比值


@pytest.mark.ec("G4")
def test_period_of_prefers_the_longest_period_word():
    """``每周年`` 必须解析成「年」而不是先命中「周」——长短词顺序决定 B6 的桥接对不对。"""
    assert U.period_of("每周年提供一次保养") == "年"
    assert U.period_of("每年") == "年"
    assert U.period_of("每周巡检") == "周"
    assert U.period_of("每日") == "日" and U.period_of("每小时") == "小时"
    assert U.period_of("无周期词") == ""


# --------------------------------------------------------------------------- #
# 区间 / 数量 / 分辨率解析
# --------------------------------------------------------------------------- #


@pytest.mark.ec("C5")
def test_interval_contains_is_closed_and_covers_is_superset():
    """``covers(other)`` 的语义是「本区间 ⊇ other」，端点闭合。
    P5 标的1 序号9 的判定就压在这上面：要求 ``33℃-43℃``、响应 ``37~43℃``，
    ``got.covers(need)`` 为假 -> 负偏离；反过来 ``need.covers(got)`` 为真，方向搞反就漏判。"""
    iv = U.Interval(D(33), D(43), "℃")
    assert iv.contains(D(33)) and iv.contains(D(43)) and not iv.contains(D(45))
    assert iv.covers(U.Interval(D(35), D(40), "℃")) is True      # 严格内嵌
    assert iv.covers(U.Interval(D(37), D(43), "℃")) is True      # 右端点重合仍算完整包含
    assert iv.covers(U.Interval(D(32), D(43), "℃")) is False     # 左端越界
    assert U.Interval(D(37), D(43), "℃").covers(iv) is False     # 反方向：响应窄于要求
    assert iv.covers(iv) is True                                 # 自反
    assert iv.width == D(10)


@pytest.mark.ec("G1")
@pytest.mark.case("B24")
def test_parse_intervals_inherits_the_unit_from_whichever_side_has_it():
    """``8-10μm`` 只在右侧写单位，左侧必须继承，否则区间两端量纲不一致。"""
    iv = U.parse_intervals("8-10μm")
    assert [(str(i.lo), str(i.hi), i.unit) for i in iv] == [("8", "10", "μm")]
    both = U.parse_intervals("100V-240V")
    assert [(str(i.lo), str(i.hi), i.unit) for i in both] == [("100", "240", "v")]
    assert U.parse_intervals("370nm——1000nm")[0].hi == D(1000)   # 双破折号写法


@pytest.mark.ec("G1")
def test_parse_intervals_preserves_negative_endpoints():
    """``-24.5D～+20D`` 的负号是数值的一部分；丢了它屈光补偿范围就判不出来。"""
    iv = U.parse_intervals("-24.5~20d")[0]
    assert (iv.lo, iv.hi, iv.unit) == (D("-24.5"), D(20), "d")
    qs = U.parse_quantities("-24.5d～+20d")
    assert [(str(q.value), q.unit) for q in qs] == [("-24.5", "d"), ("20", "d")]


@pytest.mark.ec("H12")
def test_numbering_and_typos_never_become_intervals():
    """三级编号 ``1.9.1`` 与排版噪声 ``0.l-5ml`` 都不能被读成区间——
    后者上界为 0，物理上不成立，必须整体丢弃。"""
    assert U.parse_intervals("1.9.1") == []
    assert U.parse_intervals("0.l-5ml／每次最大给药量") == []
    assert [(str(q.value), q.unit) for q in U.parse_quantities("0.l-5ml")] == [("0", ""), ("5", "ml")]


@pytest.mark.ec("G4")
def test_parse_quantities_keeps_rate_units_whole():
    """``4.22次/年`` 的单位是 ``次/年`` 而非 ``次``，否则 B6 的周期桥接无从下手。"""
    qs = U.parse_quantities("4.22次/年")
    assert [(str(q.value), q.unit) for q in qs] == [("4.22", "次/年")]
    assert qs[0].dimension == "count_rate"
    assert [(str(q.value), q.unit) for q in U.parse_quantities("分辨率≥16万像素")] == [("16", "万像素")]


@pytest.mark.ec("G8")
def test_parse_resolution_normalises_separators_and_unit_spellings():
    """``*`` / ``×`` / ``x`` 三种写法与 ``P`` / ``pixel`` / ``像素`` 三种单位都要归一。"""
    assert U.parse_resolution("3840*2160p").scalar == D(3840)
    r = U.parse_resolution("1536×1536pixel")
    assert (r.width, r.height, r.unit) == (D(1536), D(1536), "像素")
    assert U.parse_resolution("5j") is None                 # 能量不是分辨率
    assert U.parse_resolution("分辨率≥16万像素") is None       # 万像素不是 WxH 形态


@pytest.mark.ec("G8")
def test_resolution_scalar_is_width_not_area():
    """B3/B4 的前提：比的是宽度而不是面积，``3840×2160`` 才能与 ``3850像素`` 同量相比。"""
    r = U.parse_resolution("3840*2160p")
    assert r.scalar == r.width == D(3840) and r.scalar != r.width * r.height


@pytest.mark.ec("G8")
def test_parse_resolutions_finds_every_literal_in_a_compound_clause():
    """复合要求里两个分辨率字面量都要抓到；``parse_resolution`` 只取第一个。"""
    text = "输出分辨率：（≥3840*2160），主屏（≥1920*1080P）"
    all_ = U.parse_resolutions(text)
    assert [(str(r.width), str(r.height)) for r in all_] == [("3840", "2160"), ("1920", "1080")]
    assert U.parse_resolution(text).width == D(3840)
