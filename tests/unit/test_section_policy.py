"""bidtable.ingest.section —— ``分节`` 策略（agent.md 3.5）：整个项目最大的覆盖率风险点。

对齐键是 ``(标的, 分节, 序号)``，分节错一个字符该行所有指标归零；而标准答案的分节约定
**跨包不自洽**（P2 的单元格里明明有 ``1、眼底成像`` 这样的层级标题，gold 却全不填；
P5 标的5 填、同标的的序号 4/5 又不填）。所以这里测的不是「语义对不对」，而是
**拟合出来的那套规则是否还在原位**——十条用例各盯一条规则的判别力：

1. 规则 1：B 族永不填，连 ``s1`` 都翻不动（T4，P4 真实数据）。
2. 规则 3：整数标题只有「中文序数祖先」或「混合层」两个入口（P2 vs P3）。
3. 规则 3(b) + 规则 5：混合层放行 + 兄弟参数项顶掉路径（I-6/I-7，P5 标的5）。
4. 规则 4：引导句需要「含设备名」**且**「块内有整数标题」，缺一不可（P3 vs P1）。
5. ``_paren_parents``：fold 合成的父标题对两道闸都不可见（P2 标的2）。
6. 跨标的必须重置路径 + 规则 6 标题逐字保留含末尾标点 + 分隔符 ``" / "``（X31）。
7. ``s2`` 只在 ``(标的,序号)`` 真撞键时才保留分节。
8. ``fill_rate`` 诊断的两个方向，以及两条早退路径根本不跑诊断。
9. ``classify`` / ``heading_text``：层级从**序号**反推，不看正文措辞。
10. 策略取值域契约：未知策略报 ``InputError``（退出码 1）并列出可选值。

除第 9/10 条外全部用五个资料包裁剪阶段的冻结产物驱动，即真实招标文件的抽取结果。
"""

from __future__ import annotations

from collections import Counter

import pytest

from bidtable.config import LAYOUT_B, SECTION_POLICIES
from bidtable.errors import InputError
from bidtable.ingest import section as sec
from bidtable.models import Requirement, Scope

pytestmark = [pytest.mark.unit]


@pytest.fixture
def cut(frozen):
    """``cut("P3")`` -> ``(裁剪阶段的真实条目, 标的声明)``。"""

    def _get(pack: str):
        cd = frozen(pack, "cut")
        sc = frozen(pack, "scope")
        return (
            [Requirement.from_dict(x) for x in cd["reqs"]],
            [Scope.from_dict(x) for x in sc["scopes"]],
        )

    return _get


def R(text, no, *, lot="1", heading=False, layout="A", head_text=""):
    """合成条目。分节只看 ``no`` / ``is_heading`` / ``layout`` / ``head_text``。"""
    return Requirement(
        lot_no=lot, device_name="D", section="", no=no, mark="", text=text,
        page=8, is_heading=heading, layout=layout, head_text=head_text or text,
    )


def bare(text, no="1", *, head_text="", section="", lot="1", heading=False, layout="A"):
    """``R`` 的补充：``head_text`` / ``section`` 可以显式留空或预填。"""
    return Requirement(
        lot_no=lot, device_name="D", section=section, no=no, mark="", text=text,
        page=8, is_heading=heading, layout=layout, head_text=head_text,
    )


@pytest.mark.ec("A2")
@pytest.mark.ec("I1")
@pytest.mark.case("T4")
def test_the_b_family_never_gets_a_section_even_under_the_always_fill_policy(cut):
    """B 族（P4）的参数来自独立的「序号｜技术要求」双列表，没有层级标题可言。

    ``apply`` 在算任何东西之前先看 ``layout``：整块都是 B 族 ⇒ 无条件清空并**直接返回**，
    连 ``s1``（总是填）都翻不动它，连诊断都不跑。P4 的 gold 分节全空，而它的序号里有
    6 个三级号 ``1.9.1``~``1.9.6``——那正是最容易被误判成「有多级标题」的形状。
    """
    reqs, scopes = cut("P4")
    assert {r.layout for r in reqs} == {LAYOUT_B}
    assert [r.no for r in reqs if r.no.count(".") == 2] == [
        "1.9.1", "1.9.2", "1.9.3", "1.9.4", "1.9.5", "1.9.6",
    ]
    body = [r for r in reqs if not r.is_heading]
    assert len(body) == 37 and len(reqs) == 39          # 2 个裸整数标题不输出

    for policy in SECTION_POLICIES:
        out, warns = sec.apply(reqs, scopes, policy=policy)
        assert [r.section for r in out] == [""] * len(out), policy
        assert warns == [], policy                      # 早退 ⇒ 没有分节告警


@pytest.mark.ec("A1")
@pytest.mark.ec("I2")
@pytest.mark.ec("I3")
def test_an_integer_heading_enters_the_path_only_via_a_chinese_ancestor_or_a_mixed_level(cut):
    """S4 规则 3 是最大的覆盖率杠杆，两个入口缺一不可。

    P2 标的1 的一级层**纯是标题**（``1、眼底成像`` / ``2、检查模式`` / ``3、硬件配置要求``，
    下面全挂 ``N.k``）：既没有中文序数祖先，1 级也不是混合层 ⇒ 两条都不成立 ⇒ 分节全空，
    这正是 gold 的样子。同一批数据换成 ``s1`` 会填出 26 行，覆盖率从 1.000 掉到 ≈0.72。
    P3 走 3(a)：每个整数标题都挂在中文序数节下面，照填不误。
    """
    reqs, scopes = cut("P2")
    lot1 = [r for r in reqs if r.lot_no == "1"]
    assert [(r.no, r.head_text) for r in lot1 if r.is_heading] == [
        ("1", "1、眼底成像"), ("2", "2、检查模式"), ("3", "3、硬件配置要求"),
    ]
    assert sec._paren_parents(lot1) == set()
    assert sec._mixed_levels(lot1, set()) == set()      # 一级只有标题，没有真参数项

    s4, _ = sec.apply(reqs, scopes, policy="s4")
    s1, w1 = sec.apply(reqs, scopes, policy="s1")
    assert sec.fill_rate(s4) == 0.0
    diff = [(a.no, b.section) for a, b in zip(s4, s1) if a.section != b.section]
    assert len(diff) == 26                              # 规则 3 一放开，26 行的对齐键全变
    assert diff[0] == ("1.1", "1、眼底成像")
    assert len(w1) == 1 and "填充率 100%" in w1[0]

    reqs3, scopes3 = cut("P3")
    out3, _ = sec.apply(reqs3, scopes3, policy="s4")
    assert sec.fill_rate(out3) == 1.0                   # 3(a) 全放行
    secs = {r.section for r in out3}
    assert "七、专用屏幕台车 / 1、专用屏幕台车" in secs   # 整数标题挂进中文序数祖先下面
    assert "一、4K3D 摄像系统" in secs


@pytest.mark.ec("I6")
@pytest.mark.ec("I7")
def test_the_mixed_level_gate_and_the_sibling_reset_reproduce_p5_lot5_exactly(cut):
    """P5 标的5 是标准答案里**同一标的内部分填、部分不填**的唯一实例（I-7）。

    ``1、治疗源`` / ``2、照射光斑面积`` / ``3、过滤器`` 是标题（下面挂 ``N.k``），而
    ``4、欧盟CE 认证`` / ``5.配备：红光滤波片…`` 是**同层的真参数项**——1 级因此是「混合层」，
    规则 3(b) 放行前三个标题；接着规则 5 让序号 4 这个兄弟参数项把同层路径顶掉，于是
    4/5 的分节回到空。gold 就是这样，两条规则合起来才拟合得上。
    """
    reqs, scopes = cut("P5")
    lot5 = [r for r in reqs if r.lot_no == "5"]
    assert sec._paren_parents(lot5) == set()
    assert sec._mixed_levels(lot5, set()) == {1}        # 1 级既有标题又有真参数项
    # skip 排除的是「标题侧」：三个标题被当成合成父项后，1 级只剩参数项 ⇒ 不再混合。
    # 这正是 P2 标的2 靠的机制，也是为什么 ``skip`` 传空集会给出完全不同的结论。
    assert sec._mixed_levels(lot5, {0, 6, 9}) == set()

    out, warns = sec.apply(reqs, scopes, policy="s4")
    got = {r.no: r.section for r in out if r.lot_no == "5" and not r.is_heading}
    assert got == {
        "1.1": "1、治疗源", "1.2": "1、治疗源", "1.3": "1、治疗源",
        "1.4": "1、治疗源", "1.5": "1、治疗源",
        "2.1": "2、照射光斑面积", "2.2": "2、照射光斑面积",
        "3.1": "3、过滤器", "3.2": "3、过滤器",
        "4": "", "5": "",                               # 兄弟参数项把路径顶掉
    }
    assert [r.section for r in out if r.is_heading] == [""] * 3   # 标题行自身恒空
    assert warns == []
    assert sec.fill_rate(out) == pytest.approx(9 / 56)
    # s1 在 P5 上与 s4 完全同解：本包只有这 3 个标题，放开闸门也没有别的可填
    s1, _ = sec.apply(reqs, scopes, policy="s1")
    assert [(r.no, r.section) for r in s1] == [(r.no, r.section) for r in out]


@pytest.mark.ec("I4")
@pytest.mark.ec("I5")
@pytest.mark.ec("B2")
def test_a_lead_in_needs_both_the_device_name_and_a_surviving_integer_heading(cut):
    """规则 4 的两个条件缺一不可，而 P1/P3 两个服务标的正好把它们分开验证。

    P3 标的2：引导句 ``设备运维技术服务要求如下：`` **含设备名**，且块内有原文真编号的
    ``2、设备运行维护：`` ⇒ 两条件都满足 ⇒ 填，并作为 0 级路径一直带到底。
    P1 标的2：招标原文与 P2 逐字相同，但 fold 把 ``2、设备运行维护：`` 和三个 ``（N）``
    子项**合成了一行**（``is_heading=False``）⇒ 块内没有整数标题 ⇒ 不填 ⇒ gold 分节全空。
    """
    reqs3, scopes3 = cut("P3")
    out3, _ = sec.apply(reqs3, scopes3, policy="s4")
    svc3 = next(s for s in scopes3 if s.lot_no == "2")
    assert svc3.device_name == "设备运维技术服务"        # 确实是引导句的子串
    assert [(r.no, r.is_heading, r.section) for r in out3 if r.lot_no == "2"] == [
        ("", True, ""),                                              # 标题行自身恒空
        ("1", False, "设备运维技术服务要求如下："),
        ("2", True, ""),
        ("2.1", False, "设备运维技术服务要求如下： / 2、设备运行维护："),
        ("2.2", False, "设备运维技术服务要求如下： / 2、设备运行维护："),
        ("2.3", False, "设备运维技术服务要求如下： / 2、设备运行维护："),
    ]

    reqs1, scopes1 = cut("P1")
    lot2 = [r for r in reqs1 if r.lot_no == "2"]
    assert [r.is_heading for r in lot2] == [True, False, False]      # fold 之后只剩引导句
    assert [r.no for r in lot2] == ["", "1", "2"]
    out1, _ = sec.apply(reqs1, scopes1, policy="s4")
    assert [r.section for r in out1 if r.lot_no == "2"] == ["", "", ""]
    # 同一批数据 s1 会填 2 行：唯一挡住 s4 的就是「块内没有整数标题」这半个条件
    s1, _ = sec.apply(reqs1, scopes1, policy="s1")
    assert [r.section for r in s1 if r.lot_no == "2" and not r.is_heading] == [
        "设备运维技术服务要求如下："] * 2
    assert sec.fill_rate(s1) == pytest.approx(2 / 17)


@pytest.mark.ec("H9")
@pytest.mark.ec("I5")
def test_a_fold_synthesized_paren_parent_is_invisible_to_both_heading_gates(cut):
    """P2 标的2 的 ``2、设备运行维护：`` 在原文里**不是**标题——原文只有它加三个 ``（N）``
    项目符号；是 fold 为了「展开时不把父项写进 xlsx」才给它打上 ``is_heading`` 的。

    所以它既不能满足规则 4 的「块内有整数标题」，也不能让 2 级变成规则 3(b) 的混合层，
    否则 P1/P2 标的2 的分节会从 gold 的空值变成 ``设备运维技术服务要求如下：``，
    4~5 行的对齐键全错。P3 标的2 的 ``2.1`` 是原文真编号（不以 ``（`` 开头），不受影响。
    """
    reqs, scopes = cut("P2")
    lot2 = [r for r in reqs if r.lot_no == "2"]
    assert [(r.no, r.is_heading) for r in lot2] == [
        ("", True), ("1", False), ("2", True), ("2.1", False), ("2.2", False), ("2.3", False),
    ]
    assert [r.head_text[:3] for r in lot2[3:]] == ["（1）", "（2）", "（3）"]
    assert sec._paren_parents(lot2) == {2}              # 块内下标，不是序号
    assert sec._mixed_levels(lot2, {2}) == set()        # 合成父项被排除 ⇒ 1 级不再混合
    assert sec._mixed_levels(lot2, set()) == {1}        # 不排除就会把 1 级误判成混合层

    out, _ = sec.apply(reqs, scopes, policy="s4")
    assert [r.section for r in out if r.lot_no == "2"] == [""] * 6

    reqs3, scopes3 = cut("P3")
    lot2_3 = [r for r in reqs3 if r.lot_no == "2"]
    assert len(lot2_3) == 6                              # 引导句 + 1 + 2 + 2.1/2.2/2.3
    assert [r.head_text[:4] for r in lot2_3[:4]] == ["设备运维", "1、技术", "2、设备", "2.1 "]
    assert sec._paren_parents(lot2_3) == set()          # 原文真编号 ⇒ 不是合成父项

    # ``_paren_parents`` 喂三个地方：(1) ``_walk`` 的 ``if i not in paren_parents`` 短路、
    # (2) ``_mixed_levels(block, paren_parents)``、(3) ``has_int_heading`` 的统计。
    # 对 P2 标的2 真正吃劲的是 (3)：``has_int_heading is False`` ⇒ 引导句过不了规则 4
    # ⇒ 整块全空。(1)(2) 单独都观测不到，因为 ``_walk`` 的短路早就把这行挡在闸门外了。
    assert sec._gate(lot2[0], sec.LEAD, 0, [], set(), "设备运维技术服务", False, "s4") is False
    assert sec._gate(lot2[0], sec.LEAD, 0, [], set(), "设备运维技术服务", True, "s4") is True

    # 诚实记录：``i not in paren_parents`` 短路的是**整个**闸门分支，包括标题行的
    # ``out.append("")``。所以在 s1 下这个合成父项会拿到一个非空分节（它照样不进 xlsx，
    # 因为标题行在 build_rows 之前就被滤掉了），但「标题行分节恒空」只在 s4/s2 下成立。
    s1, _ = sec.apply(reqs, scopes, policy="s1")
    assert [(r.no, r.section) for r in s1 if r.lot_no == "2" and r.is_heading] == [
        ("", ""), ("2", "设备运维技术服务要求如下："),
    ]


@pytest.mark.ec("I8")
@pytest.mark.ec("B2")
@pytest.mark.ec("J3")
@pytest.mark.case("X31")
def test_the_path_resets_at_a_lot_boundary_and_heading_text_keeps_its_trailing_colon(cut):
    """分节是**按连续同标的切块**走的，路径绝不跨标的泄漏；标题文本一律逐字保留。

    P3 是最能同时验证两点的包：标的1 的最后四行还挂在 ``十一、妇科电动检查床：`` 下面，
    紧接着的标的2 引导句分节必须立刻回到空——否则服务标的的 4 行会带上设备标的的
    中文序数节，对齐键 ``(标的,分节,序号)`` 全错。规则 6 那一半同样要紧：``九、`` /
    ``十、`` / ``十一、`` 三个标题在招标原文里就带末尾中文冒号，gold 单元格也带，
    任何 ``rstrip`` 都会让这几行归零。
    """
    reqs, scopes = cut("P3")
    out, _ = sec.apply(reqs, scopes, policy="s4")
    lot1 = [r for r in out if r.lot_no == "1"]
    lot2 = [r for r in out if r.lot_no == "2"]
    assert (len(lot1), len(lot2)) == (84, 6)             # 两个连续块，不是交错

    assert [r.no for r in lot1[-4:]] == ["3", "4", "5", "6"]
    assert {r.section for r in lot1[-4:]} == {"十一、妇科电动检查床："}
    assert lot2[0].is_heading and lot2[0].section == ""   # 边界处路径清零
    assert not any("妇科电动检查床" in r.section for r in lot2)
    assert {r.section for r in lot2[1:] if r.section} == {
        "设备运维技术服务要求如下：",
        "设备运维技术服务要求如下： / 2、设备运行维护：",
    }

    secs = {r.section for r in out}
    # 规则 6：末尾标点逐字保留，连标题里的 ``，`` 和 ``。`` 都不动
    assert "九、配置适合腹腔镜及开腹手术使用电刀：" in secs
    assert "十、配置腹腔镜器械一套：" in secs
    assert "十一、妇科电动检查床：" in secs
    assert "九、配置适合腹腔镜及开腹手术使用电刀" not in secs      # 没有偷偷 rstrip
    assert "二、配套4K 采集摄像头，4K 腹腔镜内窥镜。 / 1、4K 采集摄像头" in secs

    # X31：分隔符全局统一 ``" / "``；拼出来的每一段都干净，层级最深两级
    assert sec.SEP == " / "
    assert len(secs) == 18 and len(secs - {""}) == 17
    assert max(len(s.split(sec.SEP)) for s in secs) == 2
    for s in secs - {""}:
        assert s == s.strip() and "  " not in s
        assert not s.startswith(sec.SEP) and not s.endswith(sec.SEP)
        parts = s.split(sec.SEP)
        assert all(p and p == p.strip() for p in parts)
        assert sec.SEP.join(parts) == s
    assert "" in secs                                 # 没命中的行留空，不塞占位符
    assert "设备运维技术服务要求如下： / 2、设备运行维护：".split(sec.SEP) == [
        "设备运维技术服务要求如下：", "2、设备运行维护：",
    ]


@pytest.mark.ec("J3")
@pytest.mark.ec("I3")
def test_s2_keeps_a_section_only_where_the_alignment_key_would_otherwise_collide(cut):
    """``s2`` 不是「少填一点」，而是**只在 ``(标的,序号)`` 真撞键时才保留分节**。

    P3 的序号在每个中文序数节里都从 ``1`` 重新开始（``一、`` 有 1~6，``三、`` 又有 1~7），
    所以 71 行正文里有 63 行的 ``(标的,序号)`` 是重复的——分节是它们唯一的区分手段。
    剩下 8 行序号本来就唯一，``s2`` 就把分节清掉，把对齐键收窄到最保险的形状。
    两个策略都必须做到 ``(标的,分节,序号)`` 零重复，否则 grade.py 会静默覆盖丢行。
    """
    reqs, scopes = cut("P3")
    s4, w4 = sec.apply(reqs, scopes, policy="s4")
    s2, w2 = sec.apply(reqs, scopes, policy="s2")

    cleared = [(r.lot_no, r.no) for a, r in zip(s4, s2) if a.section != r.section]
    assert cleared == [("1", "1.6"), ("1", "8"), ("1", "9"), ("1", "2.5"),
                       ("2", "1"), ("2", "2.1"), ("2", "2.2"), ("2", "2.3")]
    assert all(b.section == "" for a, b in zip(s4, s2) if a.section != b.section)
    assert sec.fill_rate(s2) == pytest.approx(63 / 71)   # 只清那 8 行，一填一清都不多
    assert sec.fill_rate(s4) == 1.0
    assert len({r.section for r in s2}) == 16            # 18 -> 16（含空串）

    heads = [r for r in s2 if r.is_heading]
    assert len(heads) == 19 and all(r.section == "" for r in heads)   # 标题行原样不动

    # 63 这个数不是凑出来的：它正好等于「撞键的行数」，一一对应
    pairs = Counter((r.lot_no, r.no) for r in s4 if not r.is_heading)
    assert sum(pairs.values()) == 71 and len(pairs) == 24
    assert sum(1 for v in pairs.values() if v > 1) == 16       # 16 个 (标的,序号) 真撞了
    assert sum(v for v in pairs.values() if v > 1) == 63       # 涉及 63 行
    for tag, out, kept in (("s4", s4, 71), ("s2", s2, 63)):
        rows = [r for r in out if not r.is_heading]
        keys = [(r.lot_no, r.section, r.no) for r in rows]
        assert len(keys) == len(set(keys)) == 71, tag          # 对齐键零重复
        assert sum(1 for r in rows if r.section) == kept, tag

    assert len(w4) == len(w2) == 1                        # 填充率仍 >80%，各出一条
    assert "填充率 100%" in w4[0] and "填充率 89%" in w2[0]


@pytest.mark.ec("J8")
@pytest.mark.ec("J3")
def test_the_fill_rate_diagnostic_shouts_both_ways_and_is_skipped_by_the_early_returns(cut):
    """``diagnose`` 是双向哨兵，而两条早退路径根本不跑它——这个不对称必须钉住。

    偏高（>80%）意味着标题判别可能过宽，为 0 又撞键意味着标题**没**被识别成分节；
    两个方向都只出告警、不改结果。但 ``policy="off"`` 和「整包 B 族」是在诊断之前
    就 ``return`` 的，所以哪怕正文里有重复键也一个字都不说——排查时容易被误导。
    """
    reqs3, scopes3 = cut("P3")
    out, warns = sec.apply(reqs3, scopes3, policy="s4")
    assert len(warns) == 1
    assert warns[0].startswith("【分节告警】分节填充率 100% 偏高（>80%）")
    assert "--section-policy s2 或 off" in warns[0]

    dup = [R("1、眼底成像", "1", heading=True), R("波长：532nm", "1.1"),
           R("2、检查模式", "2", heading=True), R("波长：532nm", "1.1")]   # (标的,序号) 撞键
    scopes = [Scope.make("1", "设备")]
    out, warns = sec.apply(dup, scopes, policy="s4")
    assert sec.fill_rate(out) == 0.0 and [r.section for r in out] == [""] * 4
    assert len(warns) == 1
    assert "分节填充率为 0" in warns[0] and "重复的 (标的,序号) 键" in warns[0]
    assert "--section-policy s1" in warns[0]
    assert sec.diagnose(out) == warns                     # apply 只是转发 diagnose
    _, w2 = sec.apply(dup, scopes, policy="s2")             # s2 清完还是 0，照喊
    assert len(w2) == 1

    uniq = [R("1、眼底成像", "1", heading=True), R("波长：532nm", "1.1"),
            R("2、检查模式", "2", heading=True), R("模式：彩色", "2.1")]
    _, w = sec.apply(uniq, scopes, policy="s4")
    assert w == []                                        # 既不偏高也不撞键 ⇒ 安静

    _, w = sec.apply(dup, scopes, policy="off")             # 早退 1：策略开关
    assert w == []
    reqs4, scopes4 = cut("P4")
    _, w = sec.apply(reqs4, scopes4, policy="s4")           # 早退 2：整包 B 族
    assert w == []

    assert sec.fill_rate([]) == 0.0                         # 空输入不炸
    filled = bare("波长：532nm", "1.1", section="一、眼底成像")
    assert sec.fill_rate([R("1、眼底成像", "1", heading=True), filled,
                          bare("模式：彩色", "2.1")]) == pytest.approx(0.5)   # 只数正文行


@pytest.mark.ec("H1")
@pytest.mark.ec("H3")
@pytest.mark.ec("H5")
@pytest.mark.ec("H12")
@pytest.mark.ec("I2")
def test_classify_reads_the_level_off_the_number_never_off_the_prose():
    """S4 的每一道闸都建立在 ``(kind, level)`` 上，而它**只从序号反推**，不看正文措辞。

    这是有意为之：``21、售后服务要求`` 和 ``2、设备运行维护：`` 措辞天差地别，层级判据
    却必须一致；反过来，一条写满「要求如下：一、二、三」的正文也不能因为里面有中文
    序数就被当成 ``CHINESE``。全角句点 ``1．2`` 在这一层还算 1 级——半角化发生在
    更上游的 numbering，分节拿到的是已经归一过的序号。
    """
    assert sec.classify(bare("设备运维技术服务要求如下：", "")) == (sec.LEAD, 0)
    assert sec.classify(bare("4K3D 摄像系统", "十一")) == (sec.CHINESE, 0)
    assert sec.classify(bare("治疗源", "一")) == (sec.CHINESE, 0)
    assert sec.classify(bare("数量：1 台", "1")) == (sec.ITEM, 1)
    assert sec.classify(bare("波长：532nm", "1.1")) == (sec.ITEM, 2)
    assert sec.classify(bare("清洗消毒方式", "1.9.1")) == (sec.ITEM, 3)
    assert sec.classify(bare("输出功率", "12")) == (sec.ITEM, 1)
    assert sec.classify(bare("输出功率", "1．2")) == (sec.ITEM, 1)      # 全角句点不分级
    assert sec.classify(bare("x", "bogus")) == (sec.ITEM, 1)            # 认不出来也不抛
    assert sec.classify(bare("要求如下：一、二、三", "2.1")) == (sec.ITEM, 2)
    assert {sec.LEAD, sec.CHINESE, sec.ITEM} == {"lead", "chinese", "item"}

    # 规则 6：``head_text`` 优先于 ``text``，只去首尾空白，末尾标点保留
    assert sec.heading_text(bare("正文", "1", head_text="  1、标题：  ")) == "1、标题："
    assert sec.heading_text(bare("  正文  ", "1")) == "正文"
    assert sec.heading_text(bare("", "1")) == ""


@pytest.mark.ec("J3")
@pytest.mark.ec("J7")
def test_an_unknown_policy_is_rejected_before_any_layout_shortcut(cut):
    """策略取值域是契约：写错必须**当场**报错并列出可选值，而不是静默按 s4 跑。

    校验排在 ``off`` 早退和「整包 B 族」早退**之前**——拿 P4（整包 B 族，本该静默清空）
    来喂非法策略，如果校验放在后面就会被早退吞掉，用户永远看不到自己的拼写错误。
    退出码是 1（配置/输入错误），不是 2（解析失败）：CLI 靠这个区分「你参数写错了」
    和「这份招标文件我读不了」。
    """
    reqs4, scopes4 = cut("P4")
    with pytest.raises(InputError) as ei:
        sec.apply(reqs4, scopes4, policy="bogus")
    assert str(ei.value) == "未知的分节策略 'bogus'，可选：s4/s2/s1/off。"
    assert "/".join(SECTION_POLICIES) in str(ei.value)     # 消息跟着常量走，不会烂掉
    assert ei.value.exit_code == 1
    with pytest.raises(InputError):
        sec.apply([], [], policy="off!")                   # 空输入也先校验

    reqs3, scopes3 = cut("P3")
    out, warns = sec.apply(reqs3, scopes3, policy="off")    # off 本身合法
    assert len(out) == len(reqs3) == 90 and warns == []
    assert all(r.section == "" for r in out)
    assert all(a.no == b.no and a.text == b.text and a.is_heading == b.is_heading
               for a, b in zip(reqs3, out))                # 只动 section，别的字段原样
