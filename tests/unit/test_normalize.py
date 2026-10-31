"""指标名与单位指纹归一层（EC-G 单位等价族 / EC-C5~C8 要求类型）。

这一层只回答「是不是在讲同一个东西」，**不回答「谁大谁小」**：量纲换算是
``judge.units`` 的责任。所以最有价值的断言恰恰是它*故意不做*的事——
``寸`` ≡ ``英寸``（改名）但 ``mA`` ≠ ``A``（换算），因为匹配阶段宁可少匹配
也不能错匹配（normalize.py 模块约束 2）。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from bidtable.match import normalize as N

pytestmark = [pytest.mark.unit]


# --------------------------------------------------------------------------- #
# squash：把任意文本压成可逐字比较的规范形
# --------------------------------------------------------------------------- #


@pytest.mark.ec("G1")
def test_squash_deletes_whitespace_instead_of_folding_it():
    """招标 PDF 写 `8 小时`、厂商文档写 `8小时`：折叠成单空格就永远对不上（P1 序号13）。"""
    assert N.squash("连续工作 8 小时") == "连续工作8小时"
    assert N.squash("　全　角　空　格　") == "全角空格"          # 全角空格同样删掉
    assert N.squash(None) == "" and N.squash("") == ""


@pytest.mark.ec("G6")
def test_squash_folds_fullwidth_superscript_and_punctuation_families():
    """NFKC 管全角与上标，翻译表管破折号/顿号/乘号；两者缺一不可。"""
    assert N.squash("≥200mW/c ㎡") == "≥200mw/cm2"
    assert N.squash("cm²") == "cm2"
    assert N.squash("33℃～43℃") == "33°c-43°c"                 # ℃ 经 NFKC 成 °C 再小写
    assert N.squash("配置A、B；C。") == "配置a,b;c."
    assert N.squash("1536×1536pixel") == "1536x1536pixel"
    assert N.squash("400nm---1400nm") == "400nm---1400nm"      # 半角连字符原样保留


@pytest.mark.ec("F2")
def test_a_star_between_digits_is_a_multiplication_sign_but_a_leading_one_survives():
    """`3840*2160` 的 `*` 是乘号，`*21.1` 的 `*` 是半角 ★——后者必须留给 ingest.marks。"""
    assert N.squash("3840*2160P") == "3840x2160p"
    assert N.squash("*21.1 输出功率") == "*21.1输出功率"
    assert N.squash("＊配置") == "*配置"                        # 全角星号只改名，不当乘号


# --------------------------------------------------------------------------- #
# 单位指纹：只改名，不换算
# --------------------------------------------------------------------------- #


@pytest.mark.ec("G5")
def test_unit_key_renames_spelling_variants_of_one_unit():
    for alias in ("寸", "吋", "inch", '"'):
        assert N.unit_key(alias) == "英寸", alias
    assert N.unit_key("米") == N.unit_key("meter") == "m"
    assert N.unit_key("pixel") == N.unit_key("px") == N.unit_key("P") == "像素"
    assert N.unit_key("cc") == "ml" and N.unit_key("公升") == "l"
    assert N.unit_key("°C") == N.unit_key("℃") == "℃"
    assert N.unit_key("瓦") == "w" and N.unit_key("秒") == "s"


@pytest.mark.ec("G2")
@pytest.mark.ec("G3")
@pytest.mark.ec("G4")
@pytest.mark.case("X7")
def test_unit_key_deliberately_refuses_to_convert_across_scales():
    """X7 的根：`9360mA` 与 `≤10A` 在匹配阶段就是**不同单位**，直接比数字必然出错。
    换算发生在 judge.units；这里判成不同反而安全（宁可少匹配，不可错匹配）。"""
    assert N.unit_key("mA") != N.unit_key("A")
    assert N.unit_key("g") != N.unit_key("kg")
    assert N.unit_key("分钟") != N.unit_key("小时")
    assert N.shares_unit("≤10A", "9360mA") is False
    assert N.unit_key("bogus单位") == "bogus单位"                # 未知单位原样返回
    assert N.unit_key("") == "" and N.unit_key(None) == ""


@pytest.mark.ec("G6")
def test_superscript_area_units_share_one_fingerprint():
    """`c ㎡` / `cm²` / `cm2` 三种写法必须落到同一指纹，否则辐照度要求挂不上样本。"""
    assert N.unit_key("c ㎡") == N.unit_key("cm²") == N.unit_key("cm2") == "cm2"
    assert N.shares_unit("≥200mW/c ㎡", "295mW/cm²") is True
    assert N.unit_fingerprint("4595cm²") == frozenset({"cm2"})


@pytest.mark.ec("G7")
def test_the_longest_unit_token_wins_the_scan():
    """分词按长度倒序：`16 万像素` 不能切成 `16 万`，`次/年` 不能切成 `次`。"""
    assert N._UNIT_TOKENS == tuple(sorted(N._UNIT_TOKENS, key=len, reverse=True))
    assert [(str(t.value), t.unit) for t in N.numbers("16 万像素")] == [("16", "万像素")]
    assert [(str(t.value), t.unit) for t in N.numbers("4.22次/年")] == [("4.22", "次/年")]
    assert [(str(t.value), t.unit) for t in N.numbers("295mW/cm²")] == [("295", "mw/cm2")]
    assert N.shares_unit("≥4 次", "4.22次/年") is False       # 速率与计数是不同指纹


@pytest.mark.ec("G8")
def test_unit_fingerprint_counts_only_unit_bearing_numbers():
    """分辨率字面量产出两个裸数字：裸数字不进指纹，否则任意两段带数字的文本都会
    「共享单位」，X7 这类陷阱就防不住了。"""
    assert [(str(t.value), t.unit) for t in N.numbers("3840*2160")] == [
        ("3840", ""), ("2160", "")]
    assert N.unit_fingerprint("3840*2160") == frozenset()
    assert N.shares_unit("3840*2160", "1920*1080") is False
    assert N.shares_unit("", "8小时") is False and N.shares_unit(None, None) is False


@pytest.mark.ec("C6")
def test_numbers_keeps_the_ordinal_index_for_dimension_picking():
    """复合要求里同时有 `3.6 Fr` 与 `670 mm`：判定阶段靠 index 与量纲挑那一个（B23）。"""
    toks = N.numbers("≥3.6 Fr，工作长度：≥670 mm")
    assert [(str(t.value), t.unit, t.index) for t in toks] == [("3.6", "fr", 0), ("670", "mm", 1)]
    assert toks[0].value == Decimal("3.6") and all(t.has_unit for t in toks)
    assert [(str(t.value), t.unit) for t in N.numbers("0.l-5ml")] == [("0", ""), ("5", "ml")]
    assert N.numbers("") == [] and N.numbers(None) == []


@pytest.mark.ec("C5")
def test_a_leading_minus_sign_is_dropped_on_purpose():
    """`-24.5D～+20D` 在这里只抽出无符号数值：正负号由 parse_req._sign_is_negative
    结合上下文判断，归一层擅自保留符号会让区间端点比对错位。"""
    assert [(str(t.value), t.unit) for t in N.numbers("-24.5~20D")] == [("24.5", ""), ("20", "d")]


# --------------------------------------------------------------------------- #
# 相似度：包含系数与最长公共子串
# --------------------------------------------------------------------------- #


@pytest.mark.ec("C8")
def test_containment_is_asymmetric_because_names_are_shorter_than_requirements():
    """用包含系数而非 Dice：`指示光功率` 对整句要求得 0.75，反过来只有 0.13。
    对称系数会因为长度差把正确的匹配压到阈值以下。"""
    req = "护眼指示光：绿光，波长520nm，功率≤5mW。"
    assert N.containment("指示光功率", req) == 0.75
    assert N.containment(req, "指示光功率") < 0.15
    assert N.containment("", req) == 0.0
    assert N.bigrams("a") == {"a"} and N.bigrams("") == set()   # 单字指标名不退化成 0 分


@pytest.mark.ec("C7")
def test_lcs_len_measures_a_substring_not_a_subsequence():
    """长配置清单靠连续片段判回声；子序列会把 `abc`/`acb` 也算成 3。"""
    assert N.lcs_len("abcdef", "xbcdy") == 3
    assert N.lcs_len("abc", "acb") == 1
    assert N.lcs_len("abc", "abc") == 3 and N.lcs_len("", "x") == 0


@pytest.mark.ec("C8")
def test_alias_group_needs_both_sides_to_hit_the_same_closed_group():
    """零词面重叠的证据（`临床用途：暂不支持`）只能靠这张闭集表挂上去。"""
    assert N.alias_group("临床用途", "用于泌尿系结石的碎石") == "用途"
    assert N.alias_group("最小瞳孔直径", "瞳孔大小：≤3mm") == "尺度"
    assert N.alias_group("CE 认证", "欧盟CE认证") == "认证"       # 比较全在 squash 之后
    assert N.alias_group("年开机率", "设备年开机率≥95%") == "开机率"
    assert N.alias_group("临床用途", "治疗深度：≥7CM") is None     # 一侧不命中就不算
    assert N.alias_group("", "x") is None and N.alias_group(None, "x") is None


@pytest.mark.ec("C8")
def test_the_alias_table_stays_domain_general_and_deliberately_narrow():
    """两组刻意的「不命中」：`范围` 太泛会把参数区间拖进尺度组；`具有≥4 个泵` 与
    `泵数量` 的词面缺口由 candidates.count_question 负责，不该塞进别名表。"""
    assert N.alias_group("范围", "调节范围") is None
    assert N.alias_group("泵数量", "具有≥4 个泵") is None
    assert [g for g, _ in N.ALIAS_GROUPS] == [
        "用途", "配置", "尺寸", "尺度", "重量", "电源", "功率", "分辨率",
        "数量", "认证", "保修", "培训", "开机率", "存储", "温度", "光斑"]
