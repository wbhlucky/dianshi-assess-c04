"""bidtable.ingest.scope —— 范围裁剪：哪些条目**不写进**响应偏离表（agent.md 3.7）。

八条用例，各盯一个互不重叠的能力边界：

1. ``body_of`` 要同时吃下 A/B 族的逐字原文（编号还在里面）和 C 族已剥过编号的正文。
2. ``section_context`` 把分节路径还原成匹配层需要的**设备限定语**。
3. 售后服务段的裁剪是「从第一个触发点起到本标的结束」的**顺序性**操作（T6，P5 四个标的）。
4. 正因为是顺序性的，编号跳回前面的孤儿子项也会被一并扫掉（X11）。
5. ``is_service`` 是唯一判据：服务标的的售后服务**就是**正文，不能裁（X15）。
6. 页尾套话按**前缀**匹配，正文里提到「商务要求」的真参数不会被误杀（X25）。
7. 空白表格模板只在 ``is_heading`` 时才裁，同一串字出现在正文行里不裁（X17）。
8. ``Cut`` 同时产出机器可读的 ``counts`` 和人读的 ``reasons``（复核报告第 4 节）。

第 3/4/8 条用 P5 的冻结 tender fixture 驱动，即真实招标文件的抽取结果，不是合成数据。
"""

from __future__ import annotations

import pytest

from bidtable.ingest import scope as sp
from bidtable.models import Requirement, Scope

pytestmark = [pytest.mark.unit]


@pytest.fixture
def p5(frozen):
    """P5 抽取层的真实产物：85 条原始条目 + 4 个标的声明。"""
    td = frozen("P5", "tender")
    sc = frozen("P5", "scope")
    reqs = [Requirement.from_dict(d) for d in td["reqs"]]
    scopes = [Scope.from_dict(d) for d in sc["scopes"]]
    return reqs, scopes


# --------------------------------------------------------------------------- #
# 1. 正文提取
# --------------------------------------------------------------------------- #


@pytest.mark.ec("F3")
@pytest.mark.ec("H6")
@pytest.mark.ec("H8")
@pytest.mark.case("X9")
def test_body_of_strips_numbering_from_verbatim_rows_and_passes_pre_stripped_rows_through(mk_req):
    """判定要看的是剥掉编号与标记之后的正文，而两种版面的 ``text`` 形态并不一样。

    A/B 族的 ``text`` 是 PDF 单元格里的逐字原文，编号还在里面（``▲2.最大单脉冲能量5J。``、
    ``*16、售后服务：…``）；B 族还把 ``★`` 追加进序号单元格（``1.7★``）。C 族（P5）的
    ``text`` 在抽取层就已经剥过编号。所以 ``body_of`` 必须**两种都能工作**：正则匹配不上
    就原样返回，绝不因为「已经是干净正文」而报错或改字。
    """
    verbatim = [
        ("▲2.最大单脉冲能量5J。", "最大单脉冲能量5J。"),
        ("*1.2 输出功率：≥750w", "输出功率：≥750w"),
        ("*16、售后服务：省内有常驻售后机构", "售后服务：省内有常驻售后机构"),
        ("*21.6 提供使用和维修技术人员培训", "提供使用和维修技术人员培训"),
        ("21、售后服务要求", "售后服务要求"),
        ("13.1、维修响应速度：4 小时内响应", "维修响应速度：4 小时内响应"),
        ("十一、妇科电动检查床：", "妇科电动检查床："),
        ("三、商务要求", "商务要求"),
        ("4.2 技术规格偏离表", "技术规格偏离表"),
        ("具体技术参数：", "具体技术参数："),
    ]
    for text, want in verbatim:
        assert sp.body_of(mk_req(text)) == want, text

    # ★ 追加在序号列：正文只剩语义；正文本来就空时退化到 strip_all，保留裸号
    assert sp.body_of(mk_req("1.7★售后服务")) == "售后服务"
    assert sp.body_of(mk_req("1.7★")) == "1.7"

    # C 族已剥过编号 ⇒ 原样穿过（含空串）
    for text in ("可进行成人及儿童治疗", "商务要求", "输出功率：≥750w", ""):
        assert sp.body_of(mk_req(text)) == text

    # head_text 优先于 text：分节路径与正文可能不同源
    assert sp.body_of(mk_req("正文", head_text="1、标题")) == "标题"

    # 已知锋利边：已剥净的正文若以裸数字开头，会被 _LEAD_NUM 的可选编号组吃掉一个字符。
    # 实际不可达（A/B 族的 head_text 仍带 ``1、``），但契约要钉住，别让它悄悄变宽。
    assert sp.body_of(mk_req("4K 采集摄像头")) == "K 采集摄像头"
    assert sp.body_of(mk_req("1、4K 采集摄像头")) == "4K 采集摄像头"


@pytest.mark.ec("I3")
@pytest.mark.ec("I5")
def test_section_context_rebuilds_the_device_qualifier_the_body_text_dropped(mk_req):
    """P3 有大量 ``1、数量：1 台`` 这种正文完全不带设备名的行，设备名只在分节标题里。

    三条规则：**保留完整路径**（末段有时比父级更窄）、**剥掉每段的序号前缀与结尾冒号**
    （否则会和正文自己的冒号粘成假二元组）、**去重且保序**。分节为空时返回 ``""``，
    调用方据此退化成「只用正文」——P1/P2/P4 全走这条路。
    """
    def ctx(section):
        return sp.section_context(mk_req("数量：1 台", section=section))

    assert ctx("") == ""
    assert ctx("四、配套LED冷光源") == "配套LED冷光源"
    assert ctx("十一、妇科电动检查床：") == "妇科电动检查床"
    # 保留完整路径：只取末段会漏掉「4K 腹腔镜」这个关键限定语
    assert ctx("二、配套4K 采集摄像头，4K 腹腔镜内窥镜。 / 1、4K 采集摄像头") == (
        "配套4K 采集摄像头，4K 腹腔镜内窥镜 4K 采集摄像头"
    )
    # 去重且保序
    assert ctx("一、4K3D 摄像系统 / 1、4K 采集摄像头 / 2、4K 采集摄像头") == (
        "4K3D 摄像系统 4K 采集摄像头"
    )
    # 引导句 + 整数二级（I-5）：结尾的「如下」保留，冒号剥掉
    assert ctx("设备运维技术服务要求如下： / 2、设备运行维护：") == (
        "设备运维技术服务要求如下 设备运行维护"
    )
    assert ctx("1、治疗源 / 1.1 输出功率") == "治疗源 1 输出功率"

    # 两处已知缺口：（N）没被剥掉（正则要求后面还有分隔符，而 ） 已被自己吃掉）；
    # 点分序号只剥第一层。都是「多送几个二元组」的方向，不会误伤，钉住防止无声变宽。
    assert ctx("（3）巡检") == "（3）巡检"
    assert ctx("2.1 数量") == "1 数量"


# --------------------------------------------------------------------------- #
# 2. 售后服务段的顺序性裁剪
# --------------------------------------------------------------------------- #


@pytest.mark.ec("H11")
@pytest.mark.case("T6")
def test_the_service_tail_cut_starts_at_the_first_trigger_and_runs_to_the_end_of_the_lot(p5):
    """P5 四个标的都有售后服务段，全部裁掉之后剩下的行数正好等于标准答案的 56 行。

    裁剪是**顺序性**的：从第一个 ``售后服务`` 触发点起，一直裁到本标的结束，中间不做
    任何逐条判断。两个后果都在这里钉住：标的12 有两个触发点（``16、售后服务：…`` 与
    ``17、售后服务要求``），**从第一个裁**；触发点**不必是标题行**（16 是带 ``*`` 的正文行）。
    """
    reqs, scopes = p5
    cut = sp.apply(reqs, scopes)

    assert cut.counts == {sp.R_SERVICE_TAIL: 26}
    assert cut.warnings == []
    assert [(r.lot_no, r.no) for r, _ in cut.dropped] == [
        ("1", "21"), ("1", "21.1"), ("1", "21.2"), ("1", "21.3"),
        ("1", "21.4"), ("1", "21.6"), ("1", "21.7"),
        ("5", "6"), ("5", "6.1"), ("5", "6.2"),
        ("8", "13"), ("8", "13.1"), ("8", "13.2"), ("8", "13.3"),
        ("8", "13.4"), ("8", "13.5"), ("8", "13.6"),
        ("12", "16"), ("12", "17"), ("12", "17.1"), ("12", "17.2"), ("12", "17.3"),
        ("12", "17.4"), ("12", "17.5"), ("12", "17.6"), ("12", "17.7"),
    ]

    twelve = [r for r, _ in cut.dropped if r.lot_no == "12"]
    assert twelve[0].no == "16" and not twelve[0].is_heading   # 第一个触发点，且是正文行
    assert twelve[1].no == "17" and twelve[1].is_heading       # 第二个触发点被顺带扫掉

    per: dict[str, list[str]] = {}
    for r in cut.kept:
        if not r.is_heading:
            per.setdefault(r.lot_no, []).append(r.no)
    assert per == {
        "1": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14",
              "15", "17", "19", "20"],
        "5": ["1.1", "1.2", "1.3", "1.4", "1.5", "2.1", "2.2", "3.1", "3.2", "4", "5"],
        "8": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
        "12": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14",
               "15"],
    }
    assert sum(len(v) for v in per.values()) == 56            # == P5 标准答案行数
    assert "16" not in per["1"] and "18" not in per["1"]      # 招标文件本来就跳号，不补
    # 标题行留在 kept 里（分节阶段还要用），只是不进最终输出
    assert [(r.lot_no, r.no) for r in cut.kept if r.is_heading] == [("5", "1"), ("5", "2"), ("5", "3")]


@pytest.mark.ec("H10")
@pytest.mark.case("X11")
def test_the_cut_reads_document_order_not_item_numbers_so_an_orphan_is_swept_too(p5, mk_req, mk_scope):
    """孤儿子项 P5 标的8 的 ``14.7 提供操作手册和维修手册`` 必须一起消失。

    真实数据里它其实没能独立成行——C 族抽取层把它粘在了 ``13.6`` 的尾巴上，所以是被
    「整行裁掉」顺带带走的。能力本体是 :func:`scope.apply` **从不读 ``req.no``**：
    一个真的独立成行、编号还跳回前面的孤儿，同样会被顺序扫描扫掉。
    """
    reqs, scopes = p5
    cut = sp.apply(reqs, scopes)
    assert not any("14.7" in r.text for r in cut.kept)
    assert [r.no for r, _ in cut.dropped if "14.7" in r.text] == ["13.6"]

    rows = [
        mk_req("12、主机重量：≤5kg", no="12", lot_no="1"),
        mk_req("13、售后服务要求", no="13", lot_no="1", is_heading=True),
        mk_req("13.1 维修响应速度", no="13.1", lot_no="1"),
        mk_req("14.7 提供操作手册和维修手册", no="14.7", lot_no="1"),   # 编号跳到 14
        mk_req("9、备用电池：≥30 分钟", no="9", lot_no="1"),             # 又跳回 9
    ]
    c = sp.apply(rows, [mk_scope("1", "设备")])
    assert [r.no for r in c.kept] == ["12"]
    assert [r.no for r, _ in c.dropped] == ["13", "13.1", "14.7", "9"]


@pytest.mark.ec("B2")
@pytest.mark.case("X15")
def test_only_the_scope_file_decides_whether_after_sales_text_is_the_deliverable(mk_req, mk_scope):
    """P1/P2/P3 的标的2 是 ``设备运维技术服务``：售后服务要求**就是**正文，裁了覆盖率直接掉。

    判据只认 ``00_投标范围.txt`` 的 ``is_service``，不看设备名、不看措辞。诚实记录：五个真实
    资料包里服务标的的正文用的是 ``设备运维技术服务要求如下：`` / ``技术培训`` / ``设备运行维护``，
    ``is_service_trigger`` 对它们全部返回 False——这条守卫在训练集上从未被触发，它是防止下一个
    资料包换措辞时静默掉覆盖率的保险。
    """
    rows = [
        mk_req("1、技术培训：每年 3 次", no="1", lot_no="2"),
        mk_req("2、售后服务要求", no="2", lot_no="2", is_heading=True),
        mk_req("2.1 维修响应速度：4 小时", no="2.1", lot_no="2"),
    ]
    svc = mk_scope("2", "设备运维技术服务", "5年")
    assert svc.is_service                                     # Scope.make 认设备名常量
    cut = sp.apply(rows, [mk_scope("1", "钬激光治疗机"), svc])
    assert [r.no for r in cut.kept] == ["1", "2", "2.1"] and cut.dropped == []

    # 同一个标的，只是范围文件里没声明成服务标的 ⇒ 售后段照裁
    dev = Scope(lot_no="2", device_name="设备运维技术服务", quantity="5年", is_service=False)
    cut = sp.apply(rows, [dev])
    assert [r.no for r in cut.kept] == ["1"]
    assert [r.no for r, _ in cut.dropped] == ["2", "2.1"]

    # 范围文件整体缺失（N3 的降级路径）⇒ service_lots 为空 ⇒ 服务标的也会被裁。
    # 这正是 scopes_from_pack 拿不到 00_投标范围.txt 时必须发告警的原因。
    assert sp.apply(rows, []).counts == {sp.R_SERVICE_TAIL: 2}

    # 触发词只认剥掉编号后的**前缀**：引导句里含「服务」二字不算触发
    assert sp.is_service_trigger(mk_req("设备运维技术服务要求如下：", no="", is_heading=True)) is False
    assert sp.is_service_trigger(mk_req("*16、售后服务：省内有常驻售后机构", no="16")) is True
    assert sp.is_service_trigger(mk_req("1.7★售后服务", no="1.7")) is True


# --------------------------------------------------------------------------- #
# 3. 逐条裁剪：套话与空白模板
# --------------------------------------------------------------------------- #


@pytest.mark.ec("C9")
@pytest.mark.case("X25")
def test_boilerplate_is_cut_on_a_prefix_so_a_real_parameter_that_mentions_it_survives(mk_req):
    """页尾套话与商务条款永远不进响应偏离表（agent.md 3.7 第 3 条）。

    用 ``startswith`` 而不是 ``in``：过度裁剪比漏裁危险得多——一条真参数只要正文里恰好
    提到「商务要求」就会被误杀，而覆盖率是直接扣分项，多余条目却不计分。
    """
    for marker in sp.BOILERPLATE_STARTS:
        assert sp.single_reason(mk_req(marker + "：略")) == sp.R_BOILERPLATE, marker

    hit = [
        "本项目核心产品为：采购清单序号1",          # P1 p77 页尾
        "商务要求",
        "三、商务要求",
        "合同履行期限：30天",
        "付款方式和条件：验收后付款",
        "本采购清单中所列技术规格及主要参数为最低要求",
    ]
    assert [sp.single_reason(mk_req(t)) for t in hit] == [sp.R_BOILERPLATE] * len(hit)

    survive = [
        "设备需满足商务要求中的条款",              # 「商务要求」在中间，不是前缀
        "售后服务中心须具备商务要求所列资质",       # 同上；也不触发售后裁剪（不是前缀）
        "交货期：合同签订后30天内",                # 语义相近但不在词表里
    ]
    assert [sp.single_reason(mk_req(t)) for t in survive] == [None] * len(survive)

    # 词表同时收了裸「商务要求」和带序号的「三、商务要求」：body_of 已经把序号剥掉，
    # 所以两个入口命中的是同一串正文，冗余是故意的（UNKNOWN 版面的 text 可能没剥序号）。
    assert sp.body_of(mk_req("三、商务要求")) == "商务要求"


@pytest.mark.ec("C9")
@pytest.mark.case("X17")
def test_a_blank_template_chapter_heading_is_cut_but_the_same_words_in_a_body_row_are_not(mk_req, mk_scope):
    """P1 第 77 页的 ``4.2 技术规格偏离表`` 长得和参数表一模一样，里面却全是空的 1/2/… 行。

    判据是 ``is_heading`` **加** TEMPLATE_HEADS 子串。注意两条逐条规则的匹配方式**故意不对称**：
    套话用 ``startswith``（宁漏勿误杀），模板标题用 ``in``（标题行本来就没有参数语义）。
    """
    for head in sp.TEMPLATE_HEADS:
        assert sp.single_reason(mk_req(head, is_heading=True)) == sp.R_TEMPLATE, head
    assert sp.single_reason(mk_req("4.2 技术规格偏离表", no="4.2", is_heading=True)) == sp.R_TEMPLATE

    # 不是标题 ⇒ 不裁：抽取层既然把它标成参数行，就当参数行处理
    assert sp.single_reason(mk_req("4.2 技术规格偏离表", no="4.2")) is None
    # ``in`` 的锋利边：标题行只要**提到**模板名就会被裁
    assert sp.single_reason(mk_req("须提交技术规格偏离表", is_heading=True)) == sp.R_TEMPLATE
    # 词表外的说法不裁
    assert sp.single_reason(mk_req("偏离表填写要求见附表", is_heading=True)) is None

    # 第四种原因码：正文为空。★ 单独一行会被 _LEAD_NUM 整串吃掉，退化到 strip_all 仍为空。
    assert sp.single_reason(mk_req("★")) == sp.R_EMPTY
    assert sp.single_reason(mk_req("   ")) == sp.R_EMPTY
    cut = sp.apply([mk_req("", no="9"), mk_req("正常参数：≥5mm", no="10")], [mk_scope()])
    assert cut.counts == {sp.R_EMPTY: 1}
    assert [r.no for r in cut.kept] == ["10"]


# --------------------------------------------------------------------------- #
# 4. 产出契约
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J8")
def test_the_cut_result_carries_machine_counts_and_a_span_style_skip_list(p5, mk_req, mk_scope):
    """``dropped`` 给测试与统计，``reasons`` 给复核报告第 4 节（agent.md 11 节要求写明
    「裁掉了什么、为什么」）。两条契约：

    * ``counts`` 按原因码聚合；逐条裁剪一条一个 reason，**售后服务段整段只出一个 reason**。
    * ``_describe`` 写的是 ``首 + 第二~最后（共 N 条）``——它是**跨度**不是枚举。P5 标的1
      的招标文件里根本没有 ``21.5``，跳过清单照样写 ``21.1~21.7（共 7 条）``。复核人不能
      拿它当编号连续性的证明。
    """
    reqs, scopes = p5
    cut = sp.apply(reqs, scopes)
    assert len(cut.reasons) == 4                              # 四个标的各一条，不是一条一行
    spans = [r.split(" 的售后服务段 ")[1].split(" 已裁剪")[0] for r in cut.reasons]
    assert spans == [
        "21 + 21.1~21.7（共 7 条）",                          # 21.5 在原文里缺失
        "6 + 6.1~6.2（共 3 条）",
        "13 + 13.1~13.6（共 7 条）",
        "16 + 17~17.7（共 9 条）",                            # 两个触发点合成一段
    ]
    assert cut.reasons[0].startswith("标的1 的售后服务段 ")
    assert cut.reasons[0].endswith("设备标的的售后服务要求不在标准答案范围内（agent.md 3.7 第 1 条）。")

    # 逐条裁剪：一条一个 reason，措辞走 _WHY 词表
    rows = [
        mk_req("", no="9"),
        mk_req("4.2 技术规格偏离表", no="4.2", is_heading=True),
        mk_req("合同履行期限：30天", no="10"),
        mk_req("正常参数：≥5mm", no="11"),
    ]
    c = sp.apply(rows, [mk_scope()])
    assert c.counts == {sp.R_EMPTY: 1, sp.R_TEMPLATE: 1, sp.R_BOILERPLATE: 1}
    assert c.reasons == [
        "标的1 序号9 已裁剪：正文为空。",
        "标的1 序号4.2 已裁剪：投标文件格式章节里的空白表格模板。",
        "标的1 序号10 已裁剪：页尾套话 / 商务条款。",
    ]
    assert [r.no for r in c.kept] == ["11"]

    # 退化形态：单条成段、无编号
    solo = sp.apply([mk_req("正常", no="1"), mk_req("2、售后服务要求", no="2")], [mk_scope()])
    assert "售后服务段 2（共 1 条）" in solo.reasons[0]
    anon = sp.apply([mk_req("售后服务要求", no=""), mk_req("响应速度", no="")], [mk_scope()])
    assert "售后服务段 售后服务要求 + ~（共 2 条）" in anon.reasons[0]
    nonum = sp.apply([mk_req("", no="")], [mk_scope()])
    assert nonum.reasons == ["标的1 序号(无编号) 已裁剪：正文为空。"]
