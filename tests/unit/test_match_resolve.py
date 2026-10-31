"""匹配层编排的单元测试：``bidtable/match/candidates.py`` 的五个 pass 与 ``match_all``。

打分层（``test_match_scoring.py``）解决「这一对有多像」，本文件解决
「一堆分数怎么变成一个确定的、可复核的、离线的配对结果」。九个用例各盯一件事：

1. 同值折叠：文档优先级压过相似度分（X4）；
2. 偏好序：检测报告压过一切，指针给实值让路（X3）；
3. 歧义标记：同分同料 vs 异料擦边，两种告警不能混（B25）；
4. 独占声明：稳定提议能解开贪心死结，且结果与插入顺序无关；
5. 拒绝式唯一挂接：宁可保持未响应也不猜；
6. 候选池：跨标的绝不配对，标题不进池，重复键各自成池；
7. 未匹配理由：三种情形分开说，不许拿「低于阈值」搪塞；
8. ``match_all`` 的离线契约：每条要求恰好一个 Match；
9. ``match_all`` 的 LLM 兜底接线：只在过 ``LLM_FLOOR`` 且未过阈值时调用。
"""

from __future__ import annotations

import pytest

from bidtable.match import candidates as C

pytestmark = [pytest.mark.unit]


# --------------------------------------------------------------------------- #
# 1. 同值折叠
# --------------------------------------------------------------------------- #


@pytest.mark.ec("D5")
@pytest.mark.ec("D6")
@pytest.mark.case("X4")
def test_collapse_values_lets_document_priority_beat_a_higher_score(mk_req, mk_ev):
    """值相同时留文档优先级最高的那条，**哪怕它分数更低**（X4：规格书胜样本）。

    产品综合样本的指标名常常就是招标原文的前缀，名称分虚高；不先折叠，
    优先级 2 的样本会抢走优先级 1 的规格书，``证明材料文件`` 列直接判错。
    两条硬性例外也必须在这里钉住：指针的 value 全是空串，一折叠就只剩一条，
    而 P1 一个标的就有 3 条指向不同表行的指针。
    """
    r = mk_req("工作长度：≥670 mm")
    a = C.Candidate(r, mk_ev("工作长度", "670mm", doc_type="技术规格书", row_no=3), 0.70)
    b = C.Candidate(r, mk_ev("工作长度", "670 mm", doc_type="产品综合样本", page=1), 0.95)
    p1 = C.Candidate(r, mk_ev("工作长度", "", doc_type="检测报告", is_pointer=True, row_no=1), 0.80)
    p2 = C.Candidate(r, mk_ev("脉冲宽度", "", doc_type="检测报告", is_pointer=True, row_no=2), 0.75)

    out = C.collapse_values([a, b, p1, p2])
    assert [(c.evidence.doc_type, c.score) for c in out] == [
        ("技术规格书", 0.70), ("检测报告", 0.80), ("检测报告", 0.75),
    ]
    # 折叠跨了两份材料 -> 必须告诉复核人这里做过取舍
    assert out[0].flags == ("conflict_resolved",)
    assert out[1].flags == () and out[2].flags == ()
    # 保序保身份：返回的是原对象，不是副本
    assert [c is x for c, x in zip(out, [a, p1, p2])] == [True, True, True]

    # 同一份材料内部的重复不需要告警，留分数高的那条即可
    a2 = C.Candidate(r, mk_ev("工作长度", "670mm", row_no=3), 0.90)
    b2 = C.Candidate(r, mk_ev("工作长度", "670mm", row_no=9), 0.70)
    same = C.collapse_values([a2, b2])
    assert len(same) == 1 and same[0] is a2 and same[0].flags == ()


# --------------------------------------------------------------------------- #
# 2. 偏好序
# --------------------------------------------------------------------------- #


@pytest.mark.ec("D3")
@pytest.mark.ec("D6")
@pytest.mark.ec("D8")
@pytest.mark.case("X3")
def test_preference_puts_the_report_first_and_makes_pointers_yield(mk_req, mk_ev):
    """实测报告压过一切同值材料（X3），哪怕它的相似度分更低。

    检测报告是唯一可核对的第三方实测证据，规格书与样本都是厂商自述。
    指针（``详见检测报告``）是另一回事：它没有值，只在「与某条实值证据
    几乎同样贴合」时让路；一枝独秀或明显更贴合时原序不动，
    否则训练集里 28 行以指针为正解的答案会全部塌掉。
    """
    r = mk_req("工作长度：≥670 mm")
    rep = C.Candidate(r, mk_ev("工作长度", "670mm", doc_type="检测报告", row_no=1), 0.80)
    spec = C.Candidate(r, mk_ev("工作长度", "670mm", doc_type="技术规格书", row_no=3), 0.95)
    cat = C.Candidate(r, mk_ev("工作长度", "670mm", doc_type="产品综合样本", page=1), 0.95)
    weak = C.Candidate(r, mk_ev("整机长度", "600mm", doc_type="技术规格书", row_no=8), 0.63)

    got = [(c.evidence.doc_type, c.score) for c in C.preference([spec, cat, rep, weak])]
    assert got == [("检测报告", 0.80), ("技术规格书", 0.95), ("产品综合样本", 0.95), ("技术规格书", 0.63)]

    # 让位带宽：分差在 POINTER_BAND 内让路，超出则不让
    assert C.POINTER_BAND == 0.10

    def ptr(score):
        return C.Candidate(r, mk_ev("工作长度", "", doc_type="检测报告", is_pointer=True), score)

    assert C._pointer_yields(ptr(0.90), 0.95) is True
    assert C._pointer_yields(ptr(1.00), 0.95) is True
    # 边界是浮点敏感的：1.05-0.95 == 0.10000000000000009 > 0.10，故不让位
    assert C._pointer_yields(ptr(1.05), 0.95) is False
    assert C._pointer_yields(ptr(1.06), 0.95) is False
    # 没有任何合格实值证据 -> 指针一枝独秀，绝不让位
    assert C._pointer_yields(ptr(0.90), None) is False
    assert C._pointer_yields(spec, 0.95) is False  # 非指针永远不让位
    assert [(c.evidence.doc_type, c.score) for c in C.preference([ptr(1.00), weak])] == [
        ("检测报告", 1.00), ("技术规格书", 0.63),
    ]

    # 同分打平取表行号小者 —— 保证同一份资料包两次运行结果一致
    t9 = C.Candidate(r, mk_ev("工作长度", "670mm", row_no=9), 0.90)
    t3 = C.Candidate(r, mk_ev("工作长度", "670mm", row_no=3), 0.90)
    assert [c.evidence.locator for c in C.preference([t9, t3])] == [
        "技术规格书 表行3", "技术规格书 表行9",
    ]
    # 不合格候选根本不进偏好序
    assert C.preference([C.Candidate(r, mk_ev("工作长度", "670mm"), 0.61)]) == []


# --------------------------------------------------------------------------- #
# 3. 歧义标记
# --------------------------------------------------------------------------- #


@pytest.mark.ec("D3")
@pytest.mark.ec("D6")
@pytest.mark.case("B25")
def test_ambiguity_distinguishes_a_real_tie_from_a_close_call(mk_req, mk_ev):
    """同分同料才是「无法分辨」（B25），异料擦边只是「值得看一眼」。

    两种旗标在复核报告里走不同措辞，混淆会让复核人对着一堆
    ``ambiguous_evidence`` 无从下手。分差超过 ``PRIORITY_BAND`` 就谁也不标：
    0.90 与 0.80 之间没有悬念，标了等于没标。
    """
    r = mk_req("工作长度：≥670 mm")

    def pair(s1, s2, doc2="技术规格书", row2=2):
        x = C.Candidate(r, mk_ev("工作长度", "670mm", row_no=1), s1)
        y = C.Candidate(r, mk_ev("工作长度", "670mm", doc_type=doc2, row_no=row2), s2)
        C.mark_ambiguity([x, y])
        return x.flags, y.flags

    assert pair(0.90, 0.90) == (("ambiguous_evidence",), ("ambiguous_evidence",))
    # 同分但来自不同材料：优先级本身就能分辨，不算歧义
    assert pair(0.90, 0.90, "产品综合样本") == ((), ())
    assert C.PRIORITY_BAND == 0.05
    assert pair(0.90, 0.88, "产品综合样本") == (("close_call",), ("close_call",))
    assert pair(0.90, 0.80, "产品综合样本") == ((), ())

    solo = C.Candidate(r, mk_ev("工作长度", "670mm"), 0.90)
    C.mark_ambiguity([solo])
    assert solo.flags == ()


# --------------------------------------------------------------------------- #
# 4. 独占声明竞争
# --------------------------------------------------------------------------- #


@pytest.mark.ec("D5")
@pytest.mark.ec("D6")
def test_claim_is_exclusive_preemptive_and_order_independent(mk_req, mk_ev):
    """一条证据只服务一条要求；提议-抢占解开贪心死结；结果与插入顺序无关。

    独占性是数据事实而不是保守：训练集 206 行的 ``(证明材料文件, 位置, 投标响应)``
    三元组零重复。贪心的破口是 A（0.95，只要 X）先拿走 X，
    B（0.92，X/Y 都行）就空手，而 A 其实还有 0.90 的 Y 可用。
    同分平局用 ``requirement.key`` 而不是 ``id()`` 打破，否则同一份资料包
    两次运行会在 P3 那堆同分「数量：1 台」上给出不同结果。
    """
    X = mk_ev("工作长度", "670mm", row_no=1)
    Y = mk_ev("整机长度", "600mm", row_no=2)
    ra = mk_req("工作长度：≥670 mm", no="1")
    rb = mk_req("整机长度：≥600 mm", no="2")

    fwd = {id(ra): [C.Candidate(ra, X, 0.95), C.Candidate(ra, Y, 0.90)],
           id(rb): [C.Candidate(rb, X, 0.92), C.Candidate(rb, Y, 0.88)]}
    rev = {id(rb): [C.Candidate(rb, X, 0.92), C.Candidate(rb, Y, 0.88)],
           id(ra): [C.Candidate(ra, X, 0.95), C.Candidate(ra, Y, 0.90)]}
    for order in (fwd, rev):
        res = C.claim(order)
        assert res[id(ra)].evidence is X  # 强者拿到首选
        assert res[id(rb)].evidence is Y  # 被让位后拿到次优，而不是空手
        assert len({id(v.evidence) for v in res.values()}) == 2  # 独占

    # 抢占：R 以 0.80 严格高于 O 的 0.70，把 E 抢走；O 的游标已越过 E，落空
    P = mk_ev("P", "p", row_no=7)
    E = mk_ev("E", "e", row_no=8)
    S, O, R = mk_req("S", no="1"), mk_req("O", no="2"), mk_req("R", no="3")
    res = C.claim({id(S): [C.Candidate(S, P, 1.00)],
                   id(O): [C.Candidate(O, P, 0.99), C.Candidate(O, E, 0.70)],
                   id(R): [C.Candidate(R, E, 0.80)]})
    assert res[id(S)].evidence is P
    assert res[id(R)].evidence is E
    assert id(O) not in res

    # 同分平局：两种插入顺序都判给序号小的那条
    Z = mk_ev("数量", "1台", row_no=5)
    t1, t2 = mk_req("数量：1台", no="1"), mk_req("数量：1台", no="2")
    r1 = C.claim({id(t1): [C.Candidate(t1, Z, 0.9)], id(t2): [C.Candidate(t2, Z, 0.9)]})
    r2 = C.claim({id(t2): [C.Candidate(t2, Z, 0.9)], id(t1): [C.Candidate(t1, Z, 0.9)]})
    assert set(r1) == set(r2) == {id(t1)}
    assert r1[id(t1)].evidence is Z

    assert C.claim({}) == {}
    # 前置条件：空偏好序会让 max() 抛 ValueError，调用方（match_all）必须先过滤
    with pytest.raises(ValueError):
        C.claim({id(t1): []})


# --------------------------------------------------------------------------- #
# 5. 拒绝式证据的唯一挂接
# --------------------------------------------------------------------------- #


@pytest.mark.ec("C8")
@pytest.mark.ec("D7")
def test_rescue_refusals_hangs_a_refusal_only_where_it_cannot_be_wrong(mk_req, mk_ev):
    """词面零重叠的拒绝式证据可以救回来，但只在**唯一可能**的那条要求上。

    P1 序号1 招标写「用于泌尿系结石的碎石…」，厂商写「临床用途：暂不支持」，
    任何相似度模型都过不了线，可它显然就是在回答这条。挂接条件刻意苛刻：
    值整句全等于拒绝词表、证据还没被认领、同标的下恰好只有一条要求与该指标名
    同属一个别名组。命中 2 条时保持未响应并**告警**——猜错一条就是一条负偏离漏判，
    比留空更贵。
    """
    ru = mk_req("临床用途：可用于泌尿外科手术", no="1")
    evr = mk_ev("临床用途", "暂不支持")
    w: list[str] = []
    out = C.rescue_refusals([ru], [evr], {}, w)
    c = out[id(ru)]
    assert c.method == "alias" and c.flags == ("alias_rescue",)
    assert c.score == 0.9 and c.alias == "用途"
    assert c.confidence == 0.55  # alias_rescue 封顶：这是猜出来的挂接，必须最显眼
    assert w == []
    assert c.reason == (
        "厂商对该项明确回复「暂不支持」；本标的内仅本条要求与指标名"
        "「临床用途」同属「用途」别名组，据此唯一挂接"
    )

    # 证据已被别的要求认领 -> 不动
    w = []
    taken = {id(ru): C.Candidate(ru, mk_ev("别的", "x"), 0.9)}
    assert C.rescue_refusals([ru], [evr], taken, w) == {} and w == []
    # 值不是整句拒绝（X21）-> 不动
    w = []
    assert C.rescue_refusals([ru], [mk_ev("临床用途", "支持（水夹层过滤器）")], {}, w) == {} and w == []
    # 命中 2 条 -> 保持未响应 + 一条可读的告警
    w = []
    ru2 = mk_req("用途：泌尿系肿瘤的汽化和凝固", no="2")
    assert C.rescue_refusals([ru, ru2], [evr], {}, w) == {}
    assert w == [
        "标的1：拒绝式证据「临床用途：暂不支持」可能对应序号 1、2 共 2 条要求，"
        "无法唯一确定，已保持未响应"
    ]
    # 标题不挂接；跨标的也不挂接
    w = []
    assert C.rescue_refusals([mk_req("临床用途：x", is_heading=True)], [evr], {}, w) == {} and w == []
    w = []
    assert C.rescue_refusals([mk_req("临床用途：x", lot_no="2")], [evr], {}, w) == {} and w == []


# --------------------------------------------------------------------------- #
# 6. 候选池
# --------------------------------------------------------------------------- #


@pytest.mark.ec("D7")
@pytest.mark.ec("B1")
@pytest.mark.ec("B3")
def test_build_pool_never_pairs_across_lots_and_never_merges_duplicate_keys(mk_req, mk_ev):
    """跨标的绝不配对，标题不进池，重复键各自成池。

    「跨标的不配对」是输出不变式 IV12（输出标的是声明标的的子集）在匹配层的对应物：
    P5 四个标的各有型号与资料，让标的1 的要求去匹标的5 的证据只会制造假阳性。
    键用 ``id(req)`` 而不是 ``req.key``：``(标的,分节,序号)`` 的唯一性是**输出层**
    的不变式，抽取层在异常资料包上可能产出重复键，用对象身份做键能让重复键
    安全地各自成池，最后由 ``writer.dedupe`` 收敛并告警。
    """
    h = mk_req("一、具体技术参数", no="1", is_heading=True)
    k1 = mk_req("触摸屏尺寸：≥10英寸", no="2")
    k2 = mk_req("触摸屏尺寸：≥10英寸", no="2")  # 与 k1 同 key，但是不同对象
    k3 = mk_req("工作长度：≥670 mm", no="3", lot_no="9")
    evs = [mk_ev("触摸屏尺寸", "10.4英寸"), mk_ev("工作长度", "670mm")]

    pool = C.build_pool([h, k1, k2, k3], evs)
    assert set(pool) == {id(h), id(k1), id(k2), id(k3)}
    assert pool[id(h)] == []          # 标题不进池，也不该占用证据
    assert pool[id(k3)] == []         # 标的9 没有任何证据
    assert len(pool[id(k2)]) == 2     # 同 key 不合并
    assert [(c.evidence.name, c.score, c.eligible) for c in pool[id(k1)]] == [
        ("触摸屏尺寸", 0.90, True), ("工作长度", 0.08, False),
    ]
    # 池底：低于 floor 的候选不进池（省下 claim 的无谓比较），0.05 是默认值
    assert C.POOL_FLOOR == 0.05
    assert len(C.build_pool([k1], evs, floor=0.0)[id(k1)]) == 2
    assert len(C.build_pool([k1], evs, floor=0.99)[id(k1)]) == 0


# --------------------------------------------------------------------------- #
# 7. 未匹配理由
# --------------------------------------------------------------------------- #


@pytest.mark.ec("D7")
@pytest.mark.ec("E4")
def test_the_no_match_reason_tells_the_truth_about_which_of_three_things_happened(mk_req, mk_ev):
    """跳过内容必须有理由，而三种未匹配的处置方式完全不同（agent.md 10.3 第 3 条）。

    空池 = 该标的压根没有厂商资料，复核人要去查资料包；
    低于阈值 = 打分不认，可能要调阈值；
    **过了阈值却被独占抢走** = 分数没问题，是同标的另一条要求更贴合。
    最后这种在 P3 那堆「数量：1 台」里最常见，笼统说「低于阈值」是假话，
    会把复核人引向完全错误的排查方向。
    """
    r = mk_req("触摸屏尺寸：≥10英寸")
    ev = mk_ev("触摸屏尺寸", "详见样机", row_no=1)
    assert C._no_match_reason(r, []) == "本标的的厂商资料里没有任何可配对的指标"
    assert C._no_match_reason(r, [C.Candidate(r, ev, 0.527)]) == (
        "最接近的候选是「触摸屏尺寸」（技术规格书 表行1），综合分 0.53 低于阈值 0.62"
    )
    assert C._no_match_reason(r, [C.Candidate(r, ev, 0.90)]) == (
        "最贴合的候选是「触摸屏尺寸」（技术规格书 表行1），综合分 0.90 已过阈值，"
        "但该证据已被同标的内更贴合的另一条要求独占认领（一条证据不服务两条要求）"
    )


# --------------------------------------------------------------------------- #
# 8. match_all 的离线契约
# --------------------------------------------------------------------------- #


@pytest.mark.ec("E4")
@pytest.mark.ec("C9")
@pytest.mark.ec("J1")
def test_match_all_emits_exactly_one_match_per_requirement_even_offline(mk_req, mk_ev):
    """每条要求恰好一个 Match，且 ``client=None`` 与「client 未启用」逐条等价。

    下游 ``judge_all`` 与 ``build_rows`` 都按索引与要求列表对齐，
    少一个或多一个就会整表错位。离线跑通是硬要求（agent.md 4.3），
    所以 LLM 分支必须能被彻底跳过而不是抛错。
    """
    h = mk_req("一、具体技术参数", no="1", is_heading=True)
    k1 = mk_req("触摸屏尺寸：≥10英寸", no="2")
    k3 = mk_req("工作长度：≥670 mm", no="3", lot_no="9")
    evs = [mk_ev("触摸屏尺寸", "10.4英寸"), mk_ev("工作长度", "670mm")]

    class Disabled:
        enabled = False

    for client in (None, Disabled()):
        ms, ws = C.match_all([h, k1, k3], evs, client=client)
        assert len(ms) == 3
        assert [m.requirement.no for m in ms] == ["1", "2", "3"]
        assert ws == []

        head, hit, miss = ms
        assert head.evidence is None and head.score == 0.0 and head.confidence == 0.0
        assert head.method == "none" and head.reason == "标题/引导句，不参与匹配"

        assert hit.evidence.locator == "技术规格书 表行1"
        assert (hit.score, hit.method, hit.confidence) == (0.90, "exact", 0.9)
        assert hit.reason == "指标名与要求正文一致（名称 1.00 / 数值 0.33）"

        assert miss.evidence is None and miss.method == "none"
        assert miss.reason == "本标的的厂商资料里没有任何可配对的指标"

    # Candidate.flags 必须原样透传到 Match，否则复核报告丢掉全部歧义信息
    # 用两条**指针**造同分歧义：值相同的实值证据会先被 collapse_values 折掉，
    # 根本走不到 mark_ambiguity —— 这两个 pass 的先后次序本身就是被测行为的一部分。
    req = mk_req("工作长度：≥670 mm")
    dup = [
        mk_ev("工作长度", "", doc_type="检测报告", is_pointer=True, row_no=1),
        mk_ev("工作长度", "", doc_type="检测报告", is_pointer=True, row_no=2),
    ]
    flagged = C.match_all([req], dup)[0][0]
    assert flagged.evidence.locator == "检测报告 第1页 表行1"  # 平局取表行号小者
    assert flagged.flags == ("ambiguous_evidence",)
    assert flagged.confidence == 0.62  # 歧义封顶，压过 1.0 的原始分


# --------------------------------------------------------------------------- #
# 9. match_all 的 LLM 兜底接线
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J1")
@pytest.mark.ec("J6")
def test_match_all_calls_the_model_only_in_the_band_where_it_can_help(
    mk_req, mk_ev, fake_llm, clean_env
):
    """语义兜底只接管「过了 ``LLM_FLOOR`` 但没过阈值」的那一段，并且必须留痕。

    低于 ``LLM_FLOOR`` 的候选连问都不该问——那是量纲冲突或词面完全无关，
    问模型只会花配额买一个假阳性。命中之后 ``method`` 改成 ``llm``、
    打上 ``llm_fallback``、置信度封顶 0.70，再配一条告警写进复核报告：
    人工必须能一眼看出哪几行是模型定的。
    """
    text, name, value = "整机尺寸：≥10英寸", "触摸屏尺寸", "详见样机"
    req = mk_req(text)
    client = fake_llm(['{"index": 0}'])
    assert client.enabled is True

    ms, ws = C.match_all([req], [mk_ev(name, value)], client=client)
    m = ms[0]
    assert client.transport.calls == 1
    assert m.evidence.name == name and m.evidence.locator == "技术规格书 表行1"
    assert m.method == "llm" and m.flags == ("llm_fallback",)
    assert m.score == 0.527 and m.confidence == 0.448  # 0.70 封顶，未过 STRONG 再折 0.85
    assert m.reason == "确定性打分 0.29/0.25 未过阈值，由语义模型选定"
    assert ws == [
        "标的1 序号1：确定性打分未过阈值，改用语义模型匹配「触摸屏尺寸」（技术规格书 表行1）"
    ]

    # 对照：同样的输入不给 client，就是未匹配 —— 证明改判确实来自模型分支
    off = C.match_all([mk_req(text)], [mk_ev(name, value)])[0][0]
    assert off.evidence is None and off.method == "none"

    # 模型答非所问（不是 JSON）或索引越界 -> 保持未匹配，绝不崩
    for bad in ('{"index": 9}', "not json at all"):
        c = fake_llm([bad])
        got = C.match_all([mk_req(text)], [mk_ev(name, value)], client=c)[0][0]
        assert got.evidence is None and got.method == "none"

    # 全部候选都低于 LLM_FLOOR -> 一次都不调用（省配额，也省假阳性）
    quiet = fake_llm(['{"index": 0}'])
    none_ms = C.match_all(
        [mk_req("光纤直径：200μm、272μm、400μm 可选")],
        [mk_ev("最大平均输出功率", "84W")],
        client=quiet,
    )[0]
    assert quiet.transport.calls == 0
    assert none_ms[0].evidence is None
