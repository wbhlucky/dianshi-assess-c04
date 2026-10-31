"""确定性打分层单元测试：``bidtable/match/candidates.py`` 的相似度原语与 ``score_candidate``。

每条用例盯一个**互不重叠**的能力，合起来覆盖打分层的全部判据：

1. 指标名的规范形集合（去括号尾巴，但绝不把名字清空）；
2. 剥中心语只能当救济用，过不了 ``MODIFIER_GATE`` 才启用；
3. 分节限定语的加持必须先有「落点」才生效；
4. 计数问句（``泵数量`` 对 ``具有≥4 个泵``）的别名通道；
5. 「逐字回显」的汉字长度门槛（否则 ``40L/min`` 会命中任何含该串的要求）；
6. 量纲冲突判据问的是量纲而不是单位字面量（``9360mA`` vs ``≤10A``）；
7. 拒绝式必须整句全等，子串不算；
8. 定位串归一与表行号提取；
9. 置信度封顶阶梯只降不升；
10. 指针 / 空值 / 拒绝式 / 散文四种值形态各吃一套权重；
11. 前缀保底救的是「名字对上了但值说不出话」的那一类；
12. ``method`` 字段忠实记录是哪一路信号把分数抬上去的。
"""

from __future__ import annotations

import dataclasses

import pytest

from bidtable.config import NEGATIVE_PHRASES
from bidtable.match import candidates as C

pytestmark = [pytest.mark.unit]


# --------------------------------------------------------------------------- #
# 1. 指标名的规范形
# --------------------------------------------------------------------------- #


@pytest.mark.ec("C7")
def test_name_variants_strip_a_paren_tail_but_never_empty_the_name():
    """长配置清单里的括号尾巴要能剥掉，但整名都在括号里时必须原样保留。

    ``3D眼镜数量(眼镜式、夹片式各)`` 是 P5 的真实指标名：括号里是配置口径，
    剥掉之后 ``3d眼镜数量`` 才配得上招标的「3D眼镜数量：≥5副」。
    反过来 ``(选配)`` 整个名字就是括号，剥完为空——那等于把信息全丢了，
    所以必须退回原样。这是同一函数上的两个相反边界。
    """
    assert C.core_name("3D眼镜数量(眼镜式、夹片式各)") == "3D眼镜数量"
    assert C.core_name("支持（水夹层过滤器）") == "支持"
    assert C.core_name("(选配)") == "(选配)"
    assert C.name_variants("(选配)") == ("(选配)",)
    assert C.name_variants("3D眼镜数量(眼镜式、夹片式各)") == (
        "3d眼镜数量(眼镜式,夹片式各)",
        "3d眼镜数量",
        "3d眼镜",
    )
    assert C.name_variants("触摸屏尺寸") == ("触摸屏尺寸", "触摸屏")
    # 双字名不剥中心语，否则「数量」会命中一切含「数量」的正文
    assert C.modifier_of("数量") == "数量"
    assert C.modifier_of("触摸屏尺寸") == "触摸屏"
    # None / 空串安全：抽取层在异常资料包上会给出来
    assert C.name_variants(None) == ()
    assert C.name_variants("") == ()


# --------------------------------------------------------------------------- #
# 2. 剥中心语只是救济，不是常态
# --------------------------------------------------------------------------- #


@pytest.mark.ec("C6")
@pytest.mark.ec("C7")
def test_the_head_noun_is_stripped_only_as_a_rescue():
    """``MODIFIER_GATE`` 之下才允许剥中心语；这是「工作长度/工作直径」的判别对。

    招标 ``4、工作长度≥300mm`` 同时被 ``3D内窥镜工作长度`` 和 ``3D内窥镜工作直径``
    看到。若不剥中心语，两者都只剩 0.2 上下；剥了之后两者的修饰语都是
    ``3d内窥镜工作``，分数一模一样——依旧分不开。真正分开它们的是：
    ``长度`` 那条的全名分（0.4292）本就高于救济分（0.2452），取 max 后仍是全名赢；
    ``直径`` 那条全名分只有 0.1875，救济分 0.2452 反而更高。
    于是同一句正文，长度得 0.4292、直径得 0.2452，方向正确。
    """
    assert C.MODIFIER_GATE == 0.70
    # 名字就是正文前缀 -> 满分，不走救济
    assert round(C.name_similarity("触摸屏尺寸", "触摸屏尺寸：≥10英寸"), 4) == 1.0
    # 复合要求只答了一半：中心语「次数」不在正文，剥出「年技术培训」救回来
    assert round(C.name_similarity("年技术培训次数", "3、每年度提供技术培训不少于3次"), 4) == 0.825
    assert round(C.name_similarity("指示光功率", "护眼指示光：绿光，波长520nm，功率≤5mW。"), 4) == 0.825
    # ★ 判别对：同一句「工作长度」，长度赢、直径输
    assert round(C.name_similarity("3D内窥镜工作长度", "4、工作长度≥300mm"), 4) == 0.4292
    assert round(C.name_similarity("3D内窥镜工作直径", "4、工作长度≥300mm"), 4) == 0.2452
    # 换成「工作直径」正文，胜负立刻翻转 —— 证明它不是在偏袒某一条
    assert round(C.name_similarity("3D内窥镜工作直径", "4、工作直径10mm——12mm"), 4) == 0.3958
    assert round(C.name_similarity("3D内窥镜工作长度", "4、工作直径10mm——12mm"), 4) == 0.2024
    # 空输入不炸，返回 0.0（normalize.squash 对 None 安全）
    assert C.name_similarity("", "工作长度≥300mm") == 0.0
    assert C.name_similarity("泵数量", None) == 0.0


# --------------------------------------------------------------------------- #
# 3. 分节限定语的加持闸
# --------------------------------------------------------------------------- #


@pytest.mark.ec("C6")
def test_the_section_context_may_boost_only_when_the_name_has_a_foothold():
    """分节能加分，但前提是名字与正文已经共享二元组；否则「配套X」会拉高一切。

    P3 的招标表里满是 ``1、数量：1 台`` 这种不带主语的行，设备名只活在分节标题
    （``四、配套LED冷光源``）里。不加Context就分不开五条「数量：1 台」。
    但落点闸（``bigrams(name) & bigrams(body)``）必须存在：正文讲「工作长度」时，
    分节里的「配套3D内窥镜」不该把 ``3D内窥镜数量`` 抬起来——那正是 P3
    「数量/尺寸」串位那一类错误的来源。
    """
    # 闸开（共享「数量」）+ 分节确实讲这台设备 -> 直接抬到满分
    assert round(C.grounded_name_similarity("3D内窥镜数量", "数量：2 条", "配套3D内窥镜"), 4) == 1.0
    # 闸关（正文讲长度，与「3D内窥镜数量」零共享二元组）-> 加持完全无效
    assert round(C.grounded_name_similarity("3D内窥镜数量", "工作长度≥300mm。", "配套3D内窥镜"), 4) == 0.06
    assert round(C.name_similarity("3D内窥镜数量", "工作长度≥300mm。"), 4) == 0.06
    # 闸开但分节讲的是别的设备 -> 加持抬不动，结果与不传分节完全相同
    assert round(C.grounded_name_similarity("电刀数量", "数量：1 台", "四、配套LED冷光源"), 4) == 0.3833
    # 空分节恒等（P1/P2/P4 的分节全是空串，此路径必须退化为原样）
    assert C.grounded_name_similarity(
        "年技术培训次数", "每年度提供技术培训不少于3次", ""
    ) == C.name_similarity("年技术培训次数", "每年度提供技术培训不少于3次")


# --------------------------------------------------------------------------- #
# 4. 计数问句别名
# --------------------------------------------------------------------------- #


@pytest.mark.ec("C7")
def test_a_count_question_is_answered_by_the_counted_noun(mk_req, mk_ev):
    """``X数量`` 这个指标名，问的其实是「X 有几个」，正文里 X 可能不带「数量」二字。

    ``泵数量`` 对 ``具有≥4 个泵`` 词面几乎零重叠，但显然是同一件事。
    三个条件缺一不可：名字形如 ``X+计数中心语``、X 出现在正文、正文带计数量词。
    第三条是防误报的闸：``具有≥4 台设备`` 有量词却没提「泵」，不能算。
    """
    assert C.count_modifier("泵数量") == "泵"
    assert C.count_modifier("电刀数量") == "电刀"
    assert C.count_modifier("3D眼镜数量") == "3d眼镜"
    assert C.count_modifier("触摸屏尺寸") == ""  # 不是计数中心语
    assert C.count_modifier("数量") == ""       # 剥完为空，不算问句
    assert C.count_question("泵数量", "具有≥4 个泵") == "泵"
    assert C.count_question("泵数量", "具有≥4 个泵，且泵头可拆卸") == "泵"
    assert C.count_question("泵数量", "泵数量：3个") == "泵"
    assert C.count_question("电刀数量", "1、数量：1 台") == ""  # 「电刀」不在正文，交给分节
    assert C.count_question("泵数量", "具有≥4 台设备") == ""    # 有量词但没提到「泵」

    # ★ 已知缺口的精确刻画（P5 标的1 序号3 就是这么漏的）：
    # 单字修饰语只吃到 char_containment 那一路——bigrams("泵") 退化成 {"泵"}，
    #  containment 项拿不到分，最终 0.30，配上别名也才 0.48 < THRESHOLD。
    c = C.score_candidate(mk_req("具有≥4 个泵"), mk_ev("泵数量", "3个"))
    assert c.flags == ("count_alias",)
    assert c.name_sim == 0.3
    assert not c.eligible


# --------------------------------------------------------------------------- #
# 5. 「逐字回显」的门槛
# --------------------------------------------------------------------------- #


@pytest.mark.ec("C1")
@pytest.mark.ec("G1")
def test_an_echo_must_be_prose_not_a_bare_number(mk_req, mk_ev):
    """回显判定的三档：够长的汉字串 -> 1.0；短数值 -> 退回指纹；散文 -> 部分分。

    ``MIN_NAME_LEN`` 挡住 ``8小时`` 这类短串（否则 ``连续工作时间≥8小时`` 的
    任何数值型要求都会被判成回显）；``MIN_ECHO_LEN`` + 汉字闸再挡一层：
    ``40L/min`` 长度 6 但一个汉字都没有，若算回显，
    ``设置流量的调节范围：1L/min～40L/min`` 会被整句判满分，
    而厂商其实只答了上限——这正是 X20「复合要求只按被回答的子句判」的反面。
    """
    assert C.MIN_ECHO_LEN == 6 and C.MIN_NAME_LEN == 4
    # 太短 -> 不是回显，但数值指纹仍然一致，所以还是 1.0
    assert C.value_similarity("8小时", "连续工作时间≥8小时") == (1.0, False)
    # Decimal("4") == Decimal("4.0")：尾零不是差异
    assert C.value_similarity("4cm", "≤4.0cm") == (1.0, False)
    # 复合分辨率整体回显
    assert C.value_similarity("3840*2160", "输出分辨率：3840*2160P")[1] is True
    assert C.value_similarity("", "x") == (0.0, False)
    # 散文答非所问：既无回显也无数值
    assert C.value_similarity("详见样机", "工作长度：≥670 mm") == (0.0, False)
    assert C.num_similarity("3840*2160", "输出分辨率：3840*2160P") == 0.5

    # ★ 汉字闸：无汉字的短串不许冒充回显
    c = C.score_candidate(
        mk_req("设置流量的调节范围：1L/min～40L/min"), mk_ev("最大气流量", "40L/min")
    )
    assert c.echo is True          # value_similarity 层面确实回显了
    assert c.method == "fingerprint"  # 但打分层不给 contain 满分
    assert c.score == 0.677
    # 对照：真散文回显 -> 满分
    d = C.score_candidate(
        mk_req("用于泌尿系结石的碎石，泌尿系肿瘤的汽化和凝固。"),
        mk_ev("临床用途", "用于泌尿系结石的碎石"),
    )
    assert d.echo is True and d.method == "contain" and d.score == 1.0


# --------------------------------------------------------------------------- #
# 6. 量纲冲突
# --------------------------------------------------------------------------- #


@pytest.mark.ec("G1")
@pytest.mark.ec("G2")
@pytest.mark.case("X7")
def test_unit_conflict_asks_about_dimension_not_about_the_unit_token(mk_req, mk_ev):
    """``9360mA`` 对 ``≤10A``：单位字面量不同，量纲相同 —— 绝不能判冲突（X7）。

    这是 P4 序号1.4 的原始陷阱：直接比 9360 与 10 会得到荒谬的「正偏离 936 倍」。
    冲突判据必须走 ``units.dimension``，而 ``万`` 这类中文数量词也要归到 count。
    反过来 ``84W`` 对 ``200μm`` 是真冲突（功率 vs 长度），
    这正是 P1 序号4 拒绝配对的原因。
    """
    assert C.unit_conflict("9360mA", "工作电流：≤10A") is False
    assert C.unit_conflict("160000像素", "输出分辨率：16万像素") is False
    assert C.unit_conflict("5mW", "功率≤5mW") is False
    assert C.unit_conflict("84W", "200μm、272μm、400μm 可选") is True
    # 一侧没有单位 -> 无从判冲突，宁可放过（缺单位交给 align_units 的 unit_assumed）
    assert C.unit_conflict("", "工作长度：≥670 mm") is False
    assert C.unit_conflict("详见样机", "工作长度：≥670 mm") is False

    # 冲突落到打分层：直接归零、method="none"、不进候选池
    c = C.score_candidate(
        mk_req("光纤直径：200μm、272μm、400μm 可选"), mk_ev("最大平均输出功率", "84W")
    )
    assert c.unit_clash is True
    assert c.method == "none"
    assert c.score == 0.0
    assert not c.eligible


# --------------------------------------------------------------------------- #
# 7. 拒绝式：整短语，不是子串
# --------------------------------------------------------------------------- #


@pytest.mark.ec("C8")
@pytest.mark.case("X21")
def test_a_refusal_is_a_whole_phrase_verdict():
    """拒绝式判定必须整句全等（X21）：``不含`` 只是 ``标准配置不含…`` 的子串。

    子串匹配会把 ``支持（水夹层过滤器）`` 里的「支持」、
    ``部分支持`` 里的「部分」都误判成拒绝，P1 序号1 那条
    ``临床用途：暂不支持`` 的 alias_rescue 通道会被彻底污染。
    多一个句点也算不上——那是另一句话，宁可漏判也不误判。
    """
    for phrase in NEGATIVE_PHRASES:
        assert C.is_refusal(phrase) is True
        assert C.is_refusal(f" {phrase} ") is True  # squash 吃掉空白
    assert C.is_refusal("支持（水夹层过滤器）") is False
    assert C.is_refusal("不含") is False
    assert C.is_refusal("暂不支持。") is False
    assert C.is_refusal("部分支持") is False
    assert C.is_refusal("") is False
    assert C.is_refusal(None) is False


# --------------------------------------------------------------------------- #
# 8. 定位串与表行号
# --------------------------------------------------------------------------- #


@pytest.mark.ec("D1")
@pytest.mark.ec("D3")
@pytest.mark.case("X26")
def test_the_locator_keeps_its_document_prefix_and_yields_a_stable_row_number(mk_ev):
    """``位置`` 列 = 文档类型 + 定位（X26），且同分打平时的表行号必须稳定。

    前缀只在缺失时补：``检测报告 第1页 表行2`` 本身已带前缀，
    再补一次就成了 ``检测报告 检测报告 第1页``，那是复核报告里最刺眼的 bug。
    ``_row_no`` 取**最后一个**整数而不是第一个——检测报告的行号排在页码后面，
    取第一个会把「第1页 表行7」读成 1，B25 的「取序号更小的那条」就彻底失效。
    """
    spec = mk_ev("工作长度", "670mm", row_no=7)
    assert spec.locator == "技术规格书 表行7"
    assert C._locator_of(spec) == "技术规格书 表行7"
    assert C._row_no(spec) == 7

    svc = dataclasses.replace(mk_ev("巡检", "每月1次", doc_type="服务方案", row_no=3), locator="表行3")
    assert C._locator_of(svc) == "服务方案 表行3"

    rep = mk_ev("脉冲宽度", "", doc_type="检测报告", row_no=2)
    assert rep.locator == "检测报告 第1页 表行2"
    assert C._locator_of(rep) == "检测报告 第1页 表行2"
    assert C._row_no(rep) == 2  # 最后一个整数，不是页码

    cat = mk_ev("整机", "", doc_type="产品综合样本", page=3)
    assert C._row_no(cat) == 3  # 样本没有表行号，退化用页码打平

    assert C._row_no(dataclasses.replace(spec, locator="技术规格书 表行")) == 0
    blank = dataclasses.replace(spec, locator="")
    assert C._row_no(blank) == 0 and C._locator_of(blank) == "技术规格书"

    # doc_type 与 locator 前缀不一致时，仍然补前缀（不猜测、不改写既有串）
    odd = dataclasses.replace(spec, locator="检测报告 第1页 表行2")
    assert C._locator_of(odd) == "技术规格书 检测报告 第1页 表行2"


# --------------------------------------------------------------------------- #
# 9. 置信度封顶阶梯
# --------------------------------------------------------------------------- #


@pytest.mark.ec("D8")
@pytest.mark.ec("D3")
def test_confidence_caps_stack_by_severity_and_only_ever_lower(mk_req, mk_ev):
    """封顶只压不抬，多重折损取最严的那一档；未过 ``STRONG`` 再打 0.85。

    置信度是复核报告的排序键（``CONF_REVIEW=0.85``），所以它的口径必须单调：
    任何「打分靠了非字面证据」的理由都只能把分数往下压。
    ``alias_rescue``（0.55）比 ``ambiguous_evidence``（0.62）更可疑——
    前者是猜出来的挂接，后者只是两条材料难分伯仲。
    """
    r = mk_req("工作长度：≥670 mm")
    plain = mk_ev("工作长度", "670mm")
    ptr = mk_ev("工作长度", "", doc_type="检测报告", is_pointer=True)

    def cand(score, method="exact", flags=(), evidence=None):
        return C.Candidate(
            requirement=r, evidence=evidence or plain,
            score=score, method=method, flags=tuple(flags),
        )

    assert C.THRESHOLD == 0.62 and C.STRONG == 0.85
    assert cand(0.95).confidence == 0.95
    assert cand(0.95, "alias").confidence == 0.65
    assert cand(0.95, "llm").confidence == 0.70
    assert cand(0.95, "exact", (), ptr).confidence == 0.70
    assert cand(0.95, "exact", ("ambiguous_evidence",)).confidence == 0.62
    assert cand(0.95, "alias", ("alias_rescue",)).confidence == 0.55
    assert cand(0.95, "alias", ("ambiguous_evidence", "alias_rescue")).confidence == 0.55
    # 未过 STRONG 一律再折 0.85
    assert cand(0.70).confidence == 0.595
    assert cand(0.85).confidence == 0.85  # 恰好过线不折价
    # 封顶只压不抬：低分候选带上 low_confidence 也不会被抬到 0.55
    assert cand(0.40, "exact", ("low_confidence",)).confidence == round(0.40 * 0.85, 3)
    # 合格线两侧
    assert cand(0.62).eligible is True
    assert cand(0.619).eligible is False


# --------------------------------------------------------------------------- #
# 10. 四种值形态，四套权重
# --------------------------------------------------------------------------- #


@pytest.mark.ec("D8")
@pytest.mark.ec("D2")
def test_four_value_shapes_get_four_different_weights(mk_req, mk_ev):
    """同一个指标名下，指针 / 空值 / 拒绝式 / 散文走四条完全不同的打分口径。

    刻意让要求正文（``整机尺寸``）不是指标名（``触摸屏尺寸``）的前缀，
    把前缀保底关掉，这样四条分支的差异全部来自权重本身。
    四者的 ``name_sim`` 完全相等（0.295），所以分差纯粹是值形态造成的：
    指针 0.295 < 拒绝式 0.5155 < 散文 0.527 < 空值 0.545。
    指针最低是故意的——它没有值，本该给实值证据让路。
    """
    r = mk_req("整机尺寸：≥10英寸")

    def sc(value, *, pointer=False, doc="技术规格书"):
        return C.score_candidate(r, mk_ev("触摸屏尺寸", value, doc_type=doc, is_pointer=pointer))

    ptr = sc("", pointer=True, doc="检测报告")
    empty = sc("")
    refusal = sc("暂不支持")
    prose = sc("详见样机")

    for c in (ptr, empty, refusal, prose):
        assert c.name_sim == 0.295
        assert 0 < c.name_sim < 1

    assert C.W_NAME == 0.60 and C.W_VALUE == 0.40
    assert C.ALIAS_BONUS == 0.25 and C.ALIAS_CAP == 0.75
    assert C.REFUSAL_WEIGHT == 0.90

    # 指针：只吃名称分，且**不**吃别名加分（它连值都没有，别名对齐无从佐证）
    assert ptr.value_sim == 0.0
    assert ptr.score == pytest.approx(ptr.name_sim, abs=5e-5)
    assert ptr.alias == "尺寸"
    assert ptr.score < empty.score
    # 空值：名称分 + 别名加分
    assert empty.score == pytest.approx(empty.name_sim + C.ALIAS_BONUS, abs=5e-5)
    # 拒绝式：名称分先折 0.90，再加别名分
    assert refusal.score == pytest.approx(C.REFUSAL_WEIGHT * refusal.name_sim + C.ALIAS_BONUS, abs=5e-5)
    # 散文：名称/数值加权
    assert prose.value_sim == 0.25
    assert prose.score == pytest.approx(
        C.W_NAME * prose.name_sim + C.W_VALUE * prose.value_sim + C.ALIAS_BONUS, abs=5e-5
    )


# --------------------------------------------------------------------------- #
# 11. 前缀保底
# --------------------------------------------------------------------------- #


@pytest.mark.ec("C7")
@pytest.mark.ec("D2")
def test_the_prefix_rule_rescues_a_name_lead_the_value_cannot_confirm(mk_req, mk_ev):
    """指标名是要求正文的前缀时保底 0.90，但用的是 ``max()`` 而不是覆盖。

    产品综合样本里 ``触摸屏尺寸：详见样机`` 这种行，值根本说不出话，
    可名字与招标原文逐字对齐——那就是同一件事，不该因为值弱而掉出候选池。
    ``max`` 而非覆盖很重要：真回显（1.0）不能被保底压回 0.90。
    ``MIN_NAME_LEN`` 挡住双字词，否则「数量」会命中一切以「数量」开头的行。
    """
    assert C.PREFIX_SCORE == 0.90 and C.MIN_NAME_LEN == 4
    hit = C.score_candidate(mk_req("触摸屏尺寸：≥10英寸"), mk_ev("触摸屏尺寸", "详见样机"))
    assert hit.prefix is True and hit.score == 0.90 and hit.method == "exact"

    miss = C.score_candidate(mk_req("整机尺寸：≥10英寸"), mk_ev("触摸屏尺寸", "详见样机"))
    assert miss.prefix is False and miss.score == 0.527

    strong = C.score_candidate(
        mk_req("触摸屏尺寸：≥10英寸"), mk_ev("触摸屏尺寸", "触摸屏尺寸：≥10英寸")
    )
    assert strong.prefix is True and strong.score > C.PREFIX_SCORE

    short = C.score_candidate(mk_req("尺寸：≥10英寸"), mk_ev("尺寸", "详见样机"))
    assert short.prefix is False


# --------------------------------------------------------------------------- #
# 12. method 字段
# --------------------------------------------------------------------------- #


@pytest.mark.ec("C1")
@pytest.mark.ec("D1")
def test_method_names_the_signal_that_carried_the_score(mk_req, mk_ev):
    """``method`` 与 ``reason`` 是复核报告的溯源凭据，必须忠实记录是哪一路抬的分。

    五种取值的判据有严格次序：回显 contain > 名称 exact > 数值 fingerprint >
    别名 alias > 兜底 exact（仅当分数已过阈值）> none。
    ``reason`` 是写给人看的，所以每种 method 都要能在里面找到对应的关键词；
    指针那一档必须点名 OCR，因为那是唯一需要人工补实测值的情形。
    """

    def m(rt, name, value, **kw):
        return C.score_candidate(mk_req(rt), mk_ev(name, value, **kw))

    rows = {
        "exact": m("整机触摸屏尺寸：≥10英寸", "触摸屏尺寸", "10.4英寸"),
        "fingerprint": m("1、数量：1 台", "电刀数量", "1台"),
        "alias": m("整机尺寸：≥10英寸", "触摸屏尺寸", "详见样机"),
        "contain": m("用于泌尿系结石的碎石，泌尿系肿瘤的汽化和凝固。", "临床用途", "用于泌尿系结石的碎石"),
        "none": m("光纤直径：200μm、272μm、400μm 可选", "最大平均输出功率", "84W"),
    }
    for method, c in rows.items():
        assert c.method == method

    assert (rows["exact"].score, rows["exact"].name_sim) == (0.8333, 1.0)
    assert (rows["fingerprint"].score, rows["fingerprint"].value_sim) == (0.98, 1.0)
    assert rows["fingerprint"].name_sim == 0.3833
    assert rows["alias"].score == 0.527
    assert rows["contain"].score == 1.0
    assert rows["none"].score == 0.0

    assert "OCR" in m("工作长度：≥670 mm", "工作长度", "", doc_type="检测报告", is_pointer=True).reason
    assert "回显" in rows["contain"].reason
    assert "别名组" in rows["alias"].reason
    assert "指纹" in rows["fingerprint"].reason
    assert rows["exact"].reason == "指标名与要求正文一致（名称 1.00 / 数值 0.33）"
