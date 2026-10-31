"""bidtable.ingest.fold —— ``（N）`` 子项折叠：**判据在厂商证据里，不在招标文件里**。

P1 标的2 与 P2 标的2 的招标原文在 ``squash`` 后逐字相同（PDF 文本层只差一个空格），
gold 却一个折叠（序号 ``{1, 2}``）、一个展开（序号 ``{1, 2.1, 2.2, 2.3}``）。唯一的
差别在 ``*_服务方案.docx`` 的 ``项目`` 列：P1 按父项「设备运行维护」整体应答，
P2/P3 逐项应答了「日常巡检维护」。八条用例各盯一条判别力，1/2/8 用五个资料包的
冻结产物驱动，其余是纯函数与合成边界：

1. 同一份招标原文，P1 折叠 / P2 展开，差别只在证据（X13）。
2. P3 的 ``2.1`` 是原文真编号，永不参与折叠；``_blocks`` 按连续同标的切块。
3. ``decide`` 的四个出口与理由措辞；``warnings`` 只来自策略校验，其余一律进 ``notes``。
4. ``title_of`` 的剥离链：``（N）`` → 编号 → 标记 → 截到第一个冒号。
5. ``_hit`` 是**互相包含**，既是它的力量，也是它的过匹配风险。
6. 服务标的闸门只在**真的声明了服务标的**时才生效（空集合等于不设防）。
7. 折叠产物的字段/标记继承，以及子项按序号前缀而非相邻位置查找。
8. ``--paren-policy`` 覆盖证据；未知取值**降级并告警**（与 section 的报错形成对照）。
"""

from __future__ import annotations

import pytest

from bidtable.ingest import fold
from bidtable.match.normalize import squash
from bidtable.models import Evidence, Requirement, Scope

pytestmark = [pytest.mark.unit]


@pytest.fixture
def pack(frozen):
    """``pack("P1")`` -> ``(抽取阶段条目, 厂商证据, 标的声明)``——折叠的三个输入。"""

    def _get(name: str):
        return (
            [Requirement.from_dict(x) for x in frozen(name, "tender")["reqs"]],
            [Evidence.from_dict(x) for x in frozen(name, "vendor")["evidence"]],
            [Scope.from_dict(x) for x in frozen(name, "scope")["scopes"]],
        )

    return _get


def R(text, no, *, lot="1", heading=False, head_text="", mark=""):
    return Requirement(
        lot_no=lot, device_name="D", section="", no=no, mark=mark, text=text,
        page=8, is_heading=heading, layout="A", head_text=head_text or text,
    )


def EV(name, lot="1"):
    """一条服务方案证据。折叠只看 ``lot_no`` 和 ``name``。"""
    return Evidence(
        lot_no=lot, model="M", doc_type="服务方案", filename="F_服务方案.docx",
        locator="服务方案 表行1", name=name, value="v", response_text=f"{name}：v",
        priority=1, is_pointer=False, confidence=1.0,
    )


# 父项 + 两个 ``（N）`` 子项：训练集里唯一的折叠形状
PARENT = R("2、设备运行维护：", "2", heading=True)
KIDS = [
    R("（1）日常巡检维护：a", "2.1", head_text="（1）日常巡检维护：a"),
    R("（2）设备性能维护：b", "2.2", head_text="（2）设备性能维护：b"),
]
SERVICE = Scope.make("1", "设备运维技术服务")      # is_service True


@pytest.mark.ec("H9")
@pytest.mark.ec("B2")
@pytest.mark.ec("D4")
@pytest.mark.case("X13")
def test_identical_tender_text_folds_in_p1_and_splits_in_p2_because_the_evidence_differs(pack):
    """X13 的正反两面：**招标 PDF 逐字相同，输出行数却不同**。

    这不是拟合出来的怪癖，而是评标口径本身——投标人按什么粒度应答，就按什么粒度拆行。
    P1 的服务方案只给了「设备运行维护」这一条，拆成三行等于逼评委去找不存在的应答；
    P2 的服务方案逐项给了「日常巡检维护」，不拆就没法逐条判偏离。判据只能来自证据侧。
    """
    r1, e1, s1 = pack("P1")
    lot2 = [r for r in r1 if r.lot_no == "2"]
    assert [(r.no, r.is_heading) for r in lot2] == [
        ("", True), ("1", False), ("2", True),          # 抽取层就把父项标成标题，折叠才找得到
        ("2.1", False), ("2.2", False), ("2.3", False),
    ]
    assert all(fold.is_paren_child(r) for r in lot2[3:])
    assert fold._names_for(e1, "2") >= {"设备运行维护", "技术培训"}

    res = fold.resolve(r1, e1, s1, policy="auto")
    assert len(res.reqs) == 18 == len(r1) - 3           # 父项 + 3 子项 -> 1 行
    assert [(r.no, r.is_heading) for r in res.reqs if r.lot_no == "2"] == [
        ("", True), ("1", False), ("2", False),         # 折叠后父项变成正文行，会进 xlsx
    ]
    assert len(res.notes) == 1 and "已折叠成一行" in res.notes[0]
    assert res.notes[0].startswith("标的2 序号2（设备运行维护）+ 3 个（N）子项：")
    assert res.warnings == []

    r2, e2, s2 = pack("P2")
    lot2b = [r for r in r2 if r.lot_no == "2"]
    # ★ 招标侧只差 PDF 文本层的一个空格（P1 ``≥4次`` / P2 ``≥4 次``），squash 后逐字相同。
    # 判据不可能来自招标侧——这正是 X13 的全部意义，也是这里不敢用原文直接相等的原因。
    assert [squash(r.text) for r in lot2b] == [squash(r.text) for r in lot2]
    raw_diff = [(a.text, b.text) for a, b in zip(lot2, lot2b) if a.text != b.text]
    assert len(raw_diff) == 1 and squash(raw_diff[0][0]) == squash(raw_diff[0][1])
    assert [(r.no, r.is_heading) for r in lot2b] == [(r.no, r.is_heading) for r in lot2]
    assert fold._names_for(e2, "2") & {"日常巡检维护", "常巡检维护"}
    res2 = fold.resolve(r2, e2, s2, policy="auto")
    assert len(res2.reqs) == 30 == len(r2)                        # 一行不少
    assert [r.no for r in res2.reqs if r.lot_no == "2"] == ["", "1", "2", "2.1", "2.2", "2.3"]
    assert "保持展开" in res2.notes[0] and "日常巡检维护" in res2.notes[0]


@pytest.mark.ec("H9")
@pytest.mark.ec("H3")
@pytest.mark.ec("I5")
def test_a_real_sub_number_is_never_a_fold_candidate_even_under_a_forced_policy(pack):
    """``is_paren_child`` 是折叠的第一道闸：**原文真写着的 ``2.1`` 永不参与**。

    P3 标的2 的 ``2.1 常巡检维护：…`` 在招标 PDF 里就是独立编号（``head_text`` 不以
    ``（`` 开头），折掉它等于凭空丢一行、直接掉覆盖率。所以 P3 在 ``auto`` 和 ``fold``
    下都纹丝不动，连一条 note 都不出——闸门前面的 ``continue`` 是静默的。
    P5/P4 没有服务标的，``_blocks`` 切出来的每一块也都无候选。
    """
    r3, e3, s3 = pack("P3")
    lot2 = [r for r in r3 if r.lot_no == "2"]
    assert [r.head_text[:4] for r in lot2] == ["设备运维", "1、技术", "2、设备", "2.1 ", "2.2 ", "2.3 "]
    assert not any(fold.is_paren_child(r) for r in lot2)
    for pol in ("auto", "fold", "split"):
        res = fold.resolve(r3, e3, s3, policy=pol)
        assert len(res.reqs) == 90 and res.notes == [], pol

    assert fold._blocks(r3) == [list(range(84)), list(range(84, 90))]
    r5, e5, s5 = pack("P5")
    assert [(r5[g[0]].lot_no, len(g)) for g in fold._blocks(r5)] == [
        ("1", 25), ("5", 17), ("8", 19), ("12", 24),      # 非连续标的 ⇒ 各自成块
    ]
    assert fold.resolve(r5, e5, s5, policy="fold").reqs == list(r5)
    assert fold._blocks([]) == []
    assert fold._blocks([R("a", "1"), R("b", "1", lot="2"), R("c", "2")]) == [[0], [1], [2]]


@pytest.mark.ec("D4")
@pytest.mark.ec("D7")
@pytest.mark.ec("H9")
def test_decide_has_exactly_four_exits_and_only_the_policy_check_ever_warns():
    """``decide`` 是纯函数，四个出口的理由串会**原样进复核报告**，所以措辞即契约。

    子项命中优先于父项命中（第 4 组）：投标人既然逐项应答了，就不能再按父项整体对齐。
    拿不到证据时默认**展开**——多余条目不计分，漏条目直接掉覆盖率（agent.md 4.5）。
    还要注意 ``FoldResult.warnings`` 只有策略校验会写，其余全走 ``notes``：告警和
    可解释性是两条不同的通道，混在一起会让复核报告失去重点。
    """
    names = fold._names_for([EV("设备运行维护")], "1")
    table = [
        (set(), False, "该标的没有任何厂商证据可参照"),
        (names, True, "厂商证据按父项「设备运行维护」整体应答，未逐项应答子项，故折叠"),
        ({"日常巡检维护"}, False, "厂商证据逐项应答了子项「日常巡检维护」，故保持展开"),
        ({"无关项目"}, False, "既未命中父项「设备运行维护」也未命中任何子项，保持展开"),
    ]
    for nm, want, frag in table:
        do, why = fold.decide(PARENT, KIDS, nm)
        assert do is want and frag in why, nm
    # 父项和子项同时命中 ⇒ 子项赢
    do, why = fold.decide(PARENT, KIDS, {"设备运行维护", "日常巡检维护"})
    assert do is False and "逐项应答了子项" in why

    res = fold.resolve([PARENT, *KIDS], [], [SERVICE], policy="auto")
    assert len(res.reqs) == 3 and res.warnings == [] and len(res.notes) == 1
    assert "没有任何厂商证据" in res.notes[0]
    assert fold.resolve([], [], [], policy="x").warnings == [
        "未知的 --paren-policy 'x'，已按 auto 处理。"]


@pytest.mark.ec("H1")
@pytest.mark.ec("H2")
@pytest.mark.ec("H6")
@pytest.mark.ec("F1")
def test_title_of_peels_paren_head_numbering_and_marks_then_cuts_at_the_first_colon():
    """``title_of`` 是匹配证据名的唯一桥梁，剥离链的每一环都对应一种真实编号形态。

    ``（1）`` 项目符号、``2、`` / ``1.`` 编号、``★`` / ``▲`` 标记都要先剥掉，再截到第一个
    冒号——因为服务方案的 ``项目`` 列写的是短标题，招标原文写的是「短标题：长描述」。
    没有冒号时整段正文就是标题（``▲2.最大单脉冲能量5J。``），这也是它最容易过长的地方。
    """
    assert fold.title_of(R("（1）日常巡检维护：提供巡检", "2.1",
                           head_text="（1）日常巡检维护：提供巡检")) == "日常巡检维护"
    assert fold.title_of(R("2、设备运行维护：", "2", heading=True)) == "设备运行维护"
    assert fold.title_of(R("1.技术培训：每年3 次", "1")) == "技术培训"
    assert fold.title_of(R("设备运维技术服务要求如下：", "", heading=True)) == "设备运维技术服务要求如下"
    assert fold.title_of(R("▲2.最大单脉冲能量5J。", "2")) == "最大单脉冲能量5J。"   # 无冒号
    assert fold.title_of(R("★1.2 输出功率：≥750w", "1.2")) == "输出功率"
    # 已知锋利边：``body_of`` 会吃掉裸的前导数字，``4K 采集摄像头`` -> ``K 采集摄像头``。
    # 折叠只在服务标的上跑，服务标的不以型号数字开头，所以训练集里踩不到。
    assert fold.title_of(R("4K 采集摄像头", "1")) == "K 采集摄像头"

    assert fold._norm("★日常巡检维护") == "日常巡检维护"
    assert fold._norm("  设备 运行维护  ") == "设备运行维护"
    assert fold._norm(None) == "" and fold._norm("") == ""


@pytest.mark.ec("D4")
@pytest.mark.ec("H9")
def test_the_hit_test_is_mutual_containment_which_is_both_its_power_and_its_risk():
    """``_hit`` 用**互相包含**而不是相等，因为三份服务方案的措辞并不一致。

    P3 的项目列写的是 ``常巡检维护``（掉了「日」字），P2 写的是 ``日常巡检维护``；
    只有双向包含才能把两者判成同一个东西，P3 的展开才立得住。代价是短标题会误命中：
    两字的 ``巡检`` 能匹配上一整串。这是有意的取舍——折叠判错的代价是多输出/少输出
    几行，而漏判会让整包的分节对齐键错位。
    """
    names = {"日常巡检维护", "设备运行维护", "技术培训"}
    assert fold._hit("常巡检维护", names) is True        # P3 的掉字写法
    assert fold._hit("日常巡检维护", names) is True
    assert fold._hit("巡检", names) is True              # ★ 过匹配：短标题被长名吃掉
    assert fold._hit("应用维护", names) is False
    assert fold._hit("设备运维技术服务", names) is False  # 标的名不会误命中项目名
    assert fold._hit("", names) is False and fold._hit("x", set()) is False
    # 证据名先过 _norm：空白和标记都不参与比较
    assert fold._names_for([EV("★ 设备 运行维护 "), EV("", "2")], "1") == {"设备运行维护"}
    assert fold._names_for([EV("设备运行维护", "2")], "1") == set()   # 只认本标的


@pytest.mark.ec("B2")
@pytest.mark.ec("B3")
@pytest.mark.ec("H9")
def test_the_service_lot_gate_only_bites_when_a_service_lot_is_actually_declared():
    """``if service_lots and lot_no not in service_lots`` ——空集合等于**不设防**。

    设计意图是「设备标的的 ``（N）`` 在抽取层已经折过了，这里不插手」。但这个保护
    依赖 ``scopes`` 里真有一个 ``is_service`` 标的：``scopes=None``（或整包都是设备标的）
    时 ``service_lots`` 为空，短路失效，**所有**标的的 ``（N）`` 子项都会被折叠。
    这正是 ``scopes_from_pack`` 在 ``00_投标范围.txt`` 缺失时必须告警的原因之一。
    """
    device = R("1、治疗源", "1", heading=True, lot="1")
    dkids = [R("（1）卤素光源：a", "1.1", lot="1", head_text="（1）卤素光源：a")]
    svc_parent = R("2、设备运行维护：", "2", heading=True, lot="2")
    svc_kids = [R("（1）日常巡检维护：b", "2.1", lot="2", head_text="（1）日常巡检维护：b")]
    rows = [device, *dkids, svc_parent, *svc_kids]
    ev = [EV("治疗源", "1"), EV("设备运行维护", "2")]

    both = [Scope.make("1", "治疗机"), Scope.make("2", "设备运维技术服务")]
    res = fold.resolve(rows, ev, both, policy="auto")
    assert [(r.lot_no, r.no) for r in res.reqs] == [
        ("1", "1"), ("1", "1.1"), ("2", "2")]            # 只折服务标的，设备标的原样

    for tag, sc in (("device-only", [Scope.make("1", "治疗机"), Scope.make("2", "别的")]),
                    ("none", None), ("empty", [])):
        res = fold.resolve(rows, ev, sc, policy="auto")
        assert [(r.lot_no, r.no) for r in res.reqs] == [("1", "1"), ("2", "2")], tag


@pytest.mark.ec("F1")
@pytest.mark.ec("F4")
@pytest.mark.ec("H9")
@pytest.mark.ec("H11")
def test_a_fold_inherits_the_children_text_and_marks_and_finds_them_by_number_prefix():
    """折叠不是删行，而是把子项正文**并**进父项——判偏离要靠这段合并文本。

    ``text`` 与 ``head_text`` 同步更新（下游 ``body_of`` 优先读 ``head_text``），
    ``is_heading`` 翻成 ``False`` 好让这一行进 xlsx，其余字段原样继承。标记取父项，
    父项没有才从子项里找，且 ``★`` 优先于 ``▲``。子项是按 ``no`` 前缀找的，不要求
    物理相邻——中间插一条 ``3、`` 照样能折；但只认父项**之后**的行。
    """
    res = fold.resolve([PARENT, *KIDS], [EV("设备运行维护")], [SERVICE], policy="auto")
    m = res.reqs[0]
    assert len(res.reqs) == 1
    assert m.no == "2" and m.is_heading is False
    assert m.text == m.head_text == "2、设备运行维护：（1）日常巡检维护：a（2）设备性能维护：b"
    assert (m.lot_no, m.device_name, m.page, m.section, m.layout) == ("1", "D", 8, "", "A")
    assert m.mark == ""                                   # 父子都没标记

    starred = [R("（1）日常巡检维护：a", "2.1", head_text="（1）日常巡检维护：a", mark="★"),
               R("（2）设备性能维护：b", "2.2", head_text="（2）设备性能维护：b", mark="▲")]
    assert fold.resolve([PARENT, *starred], [EV("设备运行维护")], [SERVICE]).reqs[0].mark == "★"
    tri = R("2、设备运行维护：", "2", heading=True, mark="▲")
    assert fold.resolve([tri, *starred], [EV("设备运行维护")], [SERVICE]).reqs[0].mark == "▲"

    # 中间隔着别的行也能折；子项在父项**之前**则折不了（只向前找）
    gap = [PARENT, R("3、其它要求", "3", heading=True), KIDS[0]]
    r = fold.resolve(gap, [EV("设备运行维护")], [SERVICE])
    assert [x.no for x in r.reqs] == ["2", "3"] and "已折叠成一行" in r.notes[0]
    rev = [KIDS[0], PARENT]
    r = fold.resolve(rev, [EV("设备运行维护")], [SERVICE])
    assert [x.no for x in r.reqs] == ["2.1", "2"] and r.notes == []

    # 父项形状不对就整体跳过：非标题 / 多级号 / 空号 / 没有 （N） 子项
    for parent in (R("2、设备运行维护：", "2"), R("2、设备运行维护：", "2.1", heading=True),
                   R("设备运维技术服务要求如下：", "", heading=True)):
        r = fold.resolve([parent, *KIDS], [EV("设备运行维护")], [SERVICE])
        assert len(r.reqs) == 3 and r.notes == [], parent.no
    real = [PARENT, R("2.1 常巡检维护：a", "2.1"), R("2.2 性能维护：b", "2.2")]
    assert len(fold.resolve(real, [EV("设备运行维护")], [SERVICE]).reqs) == 3


@pytest.mark.ec("J3")
@pytest.mark.ec("J7")
@pytest.mark.ec("H9")
def test_the_policy_switch_overrides_the_evidence_and_an_unknown_value_degrades_loudly(pack):
    """``--paren-policy`` 是给人复核用的逃生阀，必须能**推翻**证据判据。

    ``fold`` 让 P2 也折（27 行），``split`` 让 P1 也不折（21 行）——两者都跳过 ``decide``，
    理由串直接写明是强制的，方便在复核报告里区分「模型判的」和「人指定的」。
    未知取值不抛异常而是降级到 ``auto`` 并告警：这与 ``section.apply`` 对非法
    ``--section-policy`` 直接 ``InputError`` 的处理**故意不同**——分节错一个字符整行归零，
    折叠错了只是多输出几行不计分的条目，容错代价天差地别。
    """
    r1, e1, s1 = pack("P1")
    r2, e2, s2 = pack("P2")
    assert len(fold.resolve(r1, e1, s1, policy="split").reqs) == 21   # 证据说折，人说不折
    assert len(fold.resolve(r2, e2, s2, policy="fold").reqs) == 27    # 证据说不折，人说折
    assert len(fold.resolve(r1, e1, s1, policy="auto").reqs) == 18
    assert len(fold.resolve(r2, e2, s2, policy="auto").reqs) == 30
    n = fold.resolve(r2, e2, s2, policy="fold").notes[0]
    assert "--paren-policy fold 强制折叠" in n and "已折叠成一行" in n
    assert "--paren-policy split 强制保持展开" in fold.resolve(r1, e1, s1, policy="split").notes[0]

    res = fold.resolve(r1, e1, s1, policy="bogus")
    assert res.warnings == ["未知的 --paren-policy 'bogus'，已按 auto 处理。"]
    assert len(res.reqs) == 18                              # 降级到 auto，结果不变
    assert "已折叠成一行" in res.notes[0]                     # 证据判据照常生效
