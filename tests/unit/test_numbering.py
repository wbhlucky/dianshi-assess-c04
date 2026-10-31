"""numbering.py：编号形态识别、层级、标题判别（agent.md 3.5 / EC-H）。

十二条用例各盯一个**互不重叠**的能力边界：

1.  ``normalize_no`` 与 grade.py 的 ``norm()`` 对齐，对齐键不因标点写法漂移（B20）。
2.  ``chinese_to_int`` 与 ``int_parts`` 是两套编号词汇，互不认识对方的写法。
3.  ``parse_item`` 认出训练集里**实际出现过**的全部编号形态（X10）。
4.  三类伪编号分别死在两道闸中的**不同**一道，所以两道闸缺一不可。
5.  ``parse_line`` 的消歧优先级：中文序数 > 阿拉伯 > 括号 > 引导句。
6.  ``Sequencer`` 接受三级阶梯、真实跳号，但块首不能开在子项上（X12）。
7.  中文序数 / 引导句**重置游标**：P3 从 26 行恢复到 71 行的根因。
8.  ``has_children`` 的确定性判别式，以及点号对前缀的守卫（X14 之一）。
9.  ``mark_headings`` 扫的是**窗口**而非整块（X14 之二）。
10. ``guard_bare`` 第三道闸挡下的两类续行形态。
11. ``（N）`` 三策略决定子项折叠 / 拆分 / 消失（X13）。
12. 块首孤儿续行自带 Item，并保留它**开始**的那一页（B12）。
"""

from __future__ import annotations

import pytest

from bidtable.ingest import numbering as nb

pytestmark = [pytest.mark.unit]


def rows(pairs, **kw):
    """``[(文本, 页码)]`` -> ``[(序号, 类型, 层级, 页码, 正文)]``。"""
    return [(i.no, i.kind, i.level, i.page, i.text) for i in nb.split_items(pairs, **kw)]


def numbered(raw):
    return nb.Item(numbered=nb.parse_line(raw), lines=[raw], page=1)


@pytest.mark.ec("H12")
@pytest.mark.case("B20")
def test_normalize_no_matches_grade_py_norm_so_alignment_keys_never_drift():
    """全角点转半角点，吃掉空白与残留分隔符。

    grade.py 用 ``norm()`` 造对齐键 ``(标的, 分节, 序号)``。P5 标的5 写的是
    全角 ``1．2 输出功率``，模板侧是半角；两边归一规则一旦不一致，这一行会
    同时被记成「多余条目」和「漏抽」，覆盖率与偏离准确率各扣一次。
    """
    assert nb.normalize_no("1．2") == "1.2"
    assert nb.normalize_no("1、") == "1"
    assert nb.normalize_no("1.") == "1"
    assert nb.normalize_no(" 1 . 2 ") == "1.2"
    assert nb.normalize_no("、1.2、") == "1.2"
    assert nb.normalize_no("1.9.1") == "1.9.1"
    # 边界：空串 / None / 全是标点，一律归到 ""（不抛异常）
    assert nb.normalize_no("") == ""
    assert nb.normalize_no(None) == ""
    assert nb.normalize_no("．．") == ""


@pytest.mark.ec("I2")
def test_chinese_to_int_and_int_parts_are_two_separate_number_vocabularies():
    """中文序数走十进制解析，阿拉伯编号走点分段；两者互不认识对方的写法。

    P3 标的1 有 ``一、``~``十一、`` 十一个中文序数节，每节下又从 ``1、`` 重新
    编号。分节路径两种词汇都要：``十一``->11 用来排节的顺序，``1.9.1``->[1,9,1]
    用来判层级前缀。失败值统一是 ``0`` / ``[]``，调用方靠它判失败。
    """
    assert nb.chinese_to_int("一") == 1
    assert nb.chinese_to_int("十") == 10
    assert nb.chinese_to_int("十一") == 11
    assert nb.chinese_to_int("二十") == 20
    assert nb.chinese_to_int("二十一") == 21
    assert nb.chinese_to_int("") == 0
    assert nb.chinese_to_int(None) == 0
    assert nb.chinese_to_int("百") == 0  # 词表外的字，整串作废
    # 已知怪癖：没有「十」时按位累加，``一二`` -> 12（既不是 3 也不是 [1,2]）
    assert nb.chinese_to_int("一二") == 12

    assert nb.int_parts("1.9.1") == [1, 9, 1]
    assert nb.int_parts("1") == [1]
    assert nb.int_parts("2.10") == [2, 10]  # 逐段 int，不当十进制串
    # 边界：任一段不是纯数字，整串作废（不做部分解析）
    assert nb.int_parts("1.a") == []
    assert nb.int_parts("一") == []
    assert nb.int_parts("1.") == []
    assert nb.int_parts("") == []


@pytest.mark.ec("H1")
@pytest.mark.ec("H2")
@pytest.mark.ec("H3")
@pytest.mark.ec("H4")
@pytest.mark.ec("H5")
@pytest.mark.ec("H6")
@pytest.mark.ec("H7")
@pytest.mark.ec("H8")
@pytest.mark.ec("H12")
@pytest.mark.case("X10")
def test_parse_item_recognizes_every_numbering_form_the_packs_actually_use():
    """训练集里出现过的每一种编号形态都必须被认出，且**只被认出一次**。

    每一行都取自真实资料包：``19.`` 是 P5 标的5、``1.9.3`` 是 P4 独占序号列的
    三级号、``▲2.`` 是 P1 的标记前缀、``*1.2`` / ``1.7★`` 是半角星号在编号前
    与编号后的两种位置（B 族把 ``★`` 追加在序号单元格里）、``13.1、`` 是 P5
    标的8、``1．2`` 是全角点。漏认一种，该条会被静默并进上一条，覆盖率直接塌。
    """
    forms = [
        # 原文, 裸序号, 标记, 顶层号, 层级, 实际分隔符, 正文
        ("1、可进行成人及儿童治疗", "1", "", 1, 1, "、", "可进行成人及儿童治疗"),
        ("19.提供医疗器械注册证", "19", "", 19, 1, ".", "提供医疗器械注册证"),
        ("1.1、4K 采集摄像头", "1.1", "", 1, 2, "、", "4K 采集摄像头"),
        ("1.1 基础率", "1.1", "", 1, 2, " ", "基础率"),
        ("1.9.3", "1.9.3", "", 1, 3, "", ""),
        ("▲2.最大单脉冲能量5J。", "2", "▲", 2, 1, ".", "最大单脉冲能量5J。"),
        ("*1.2 输出功率：≥750w", "1.2", "★", 1, 2, " ", "输出功率：≥750w"),
        ("1.7★", "1.7", "★", 1, 2, "", ""),
        ("1．2 全角点", "1.2", "", 1, 2, " ", "全角点"),
        ("13.1、维修响应速度", "13.1", "", 13, 2, "、", "维修响应速度"),
        ("*21.1 售后服务", "21.1", "★", 21, 2, " ", "售后服务"),
        ("  2、缩进", "2", "", 2, 1, "、", "缩进"),
        ("1.", "1", "", 1, 1, ".", ""),
    ]
    for raw, no, mark, top, level, sep, text in forms:
        n = nb.parse_item(raw)
        assert n is not None, raw
        got = (n.no, n.mark, n.top, n.level, n.sep, n.text)
        assert got == (no, mark, top, level, sep, text), raw
        assert n.raw == raw  # 原文列要逐字一致，缩进也保留


@pytest.mark.ec("F2")
@pytest.mark.ec("H10")
def test_the_pseudo_numbers_die_at_one_of_two_gates():
    """三类伪编号：前两类死在正则闸，第三类死在 Sequencer 闸。

    ``3840*2160`` 里的 ``*`` 是乘号不是标记，``10ml/min-450ml/min`` 编号后没有
    分隔符，两者在 :func:`parse_item` 就返回 ``None``。
    ``0.l-5ml／每次最大给药量`` 是 P5 标的1 的真实续行：``0`` **能过正则**
    （``top=0``、``level=1``、``sep="."``），只有 :class:`Sequencer` 的
    ``top < 1`` 拦得住它。这就是为什么必须有第二道闸。
    """
    for bad in ("3840*2160", "10ml/min-450ml/min", "（1）分辨率", "一、标题", "具体技术参数：", "", "   "):
        assert nb.parse_item(bad) is None, bad
    assert nb.parse_item(None) is None

    n = nb.parse_item("0.l-5ml／每次最大给药量")
    assert n is not None
    assert (n.no, n.top, n.level, n.sep) == ("0", 0, 1, ".")
    assert n.text == "l-5ml／每次最大给药量"
    assert nb.Sequencer().plausible(n) is False  # 第二道闸在这里生效


@pytest.mark.ec("I2")
@pytest.mark.ec("H9")
def test_parse_line_prefers_chinese_over_item_over_paren_over_lead():
    """四种形态按固定优先级消歧，歧义行的归属由优先级而非出现顺序决定。

    ``十一、妇科电动检查床：`` 同时满足中文序数与引导句（以 ``：`` 结尾），必须
    走中文序数：它要进 ``分节`` 路径，而引导句只重置游标。``（1）分辨率≥16 万像素``
    是 P1 续页的项目符号，不能被阿拉伯编号吞掉。``工作电流：≤10A`` 的冒号不在
    行尾，既不是引导句也不是编号，整行判 ``None`` 后并入上一条。
    """
    assert nb.parse_line("十一、妇科电动检查床：").kind == nb.KIND_CHINESE
    assert nb.parse_line("1、aaa").kind == nb.KIND_ITEM
    assert nb.parse_line("（1）分辨率≥16 万像素").kind == nb.KIND_PAREN
    lead = nb.parse_line("设备运维技术服务要求如下：")
    assert (lead.kind, lead.no, lead.level) == (nb.KIND_LEAD, "", 0)
    assert lead.text == "设备运维技术服务要求如下："  # 引导句的正文是整行
    # 半角冒号同样算引导句
    assert nb.parse_line("设备配置要求及用途:").kind == nb.KIND_LEAD
    # 括号编号只认 1~2 位数
    assert nb.parse_paren("（12）xx").no == "12"
    assert nb.parse_paren("（123）xx") is None
    assert nb.parse_paren("(CVVHDF)") is None  # P5 标的1 的真实陷阱：治疗模式缩写
    assert nb.parse_paren("(3) 半角").no == "3"
    assert nb.parse_paren("（0）xx").top == 0  # 认得出，但 Sequencer 拒收
    assert nb.parse_line("工作电流：≤10A") is None
    assert nb.parse_line("3840*2160") is None
    # starts_with_paren 只看行首（允许缩进），是 fold.py 判「排版符号父子」的入口
    assert nb.starts_with_paren("（1）分辨率") is True
    assert nb.starts_with_paren(" （2） x") is True
    assert nb.starts_with_paren("1.1 分辨率") is False
    assert nb.starts_with_paren("(CVVHDF)") is False
    assert nb.starts_with_paren("") is False
    assert nb.starts_with_paren(None) is False


@pytest.mark.ec("H5")
@pytest.mark.ec("H11")
@pytest.mark.case("X12")
def test_the_sequencer_accepts_a_three_level_ladder_and_refuses_to_open_a_block_on_a_child():
    """层级闸：同父前缀下严格递增，深度不限；顶层号落在 ``(last, last+max_gap]``。

    P4 是三层 ``1.9`` -> ``1.9.1`` -> ... -> ``1.9.6`` -> ``1.10``。只记「二级号
    最大值」会把 ``1.9.1`` 判成 ``9 > 9`` 的倒退而整块丢掉。
    P5 标的1 是 15 -> 17 -> 19 -> 20：16 与 18 在原文里**根本不存在**，闸门放行
    真实跳号，但绝不补造缺号（X12）。
    块首不能开在子项上：``1.9 输出功率`` 的父号 ``1`` 还没有出现过。
    """
    seq = nb.Sequencer()
    for raw in ("1.", "1.1", "1.8", "1.9", "1.9.1", "1.9.2"):
        assert seq.offer(nb.parse_item(raw)), raw
    assert seq.last_no == "1.9.2"
    assert seq.plausible(nb.parse_item("1.9.6")) is True  # 同父下继续递增
    assert seq.plausible(nb.parse_item("1.9.2")) is False  # 重复
    assert seq.plausible(nb.parse_item("1.9.1")) is False  # 倒退
    for raw in ("1.9.6", "1.10", "2.", "2.1"):
        assert seq.offer(nb.parse_item(raw)), raw
    assert seq.plausible(nb.parse_item("1.1")) is False  # 跨父前缀

    # 空游标的起点约束，与「已经走到 1」之后不一样
    fresh = nb.Sequencer()
    assert (fresh.last_top, fresh.last_no) == (0, "")
    assert fresh.plausible(nb.parse_item("1、x")) is True
    assert fresh.plausible(nb.parse_item("3、x")) is True  # <= max_gap(5)
    assert fresh.plausible(nb.parse_item("6、x")) is False
    assert fresh.plausible(nb.parse_item("0、x")) is False
    assert fresh.plausible(nb.parse_item("1.9 输出功率")) is False  # 块首不能开在子项
    after_one = nb.Sequencer()
    after_one.advance(nb.parse_item("1、x"))
    assert after_one.plausible(nb.parse_item("6、x")) is True  # 1 < 6 <= 1+5
    assert after_one.plausible(nb.parse_item("7、x")) is False

    gap15 = nb.Sequencer()
    gap15.advance(nb.parse_item("15、x"))
    assert gap15.plausible(nb.parse_item("17、x")) is True
    assert gap15.plausible(nb.parse_item("21、x")) is False
    assert nb.Sequencer(max_gap=2).plausible(nb.parse_item("3、x")) is False
    assert nb.Sequencer(max_gap=20).plausible(nb.parse_item("19、x")) is True


@pytest.mark.ec("I2")
def test_chinese_ordinals_and_lead_ins_reset_the_cursor_so_each_section_renumbers_from_one():
    """中文序数标题与引导句之下的编号是**重新开始**的。

    P3 标的1 的 ``一、4K3D 摄像系统`` 下有 ``1、``~``6、``，``二、配套摄像头`` 下
    又从 ``1、4K 采集摄像头`` 起。不重置游标的话，``二、`` 之后的 ``1`` 会被判成
    「1 <= 6」的倒退而整节丢弃 —— 这正是 P3 从 71 行掉到 26 行的根因。
    引导句（``具体技术参数：``）同理：它只是「下面开始列参数」的过渡句。
    """
    seq = nb.Sequencer()
    for raw in ("1、a", "2、b", "3、c", "4、d", "5、e", "6、f"):
        assert seq.offer(nb.parse_item(raw)), raw
    assert seq.plausible(nb.parse_item("1、g")) is False
    seq.advance(nb.parse_chinese("七、标题"))
    assert (seq.last_top, seq.last_no) == (0, "")
    assert seq.plausible(nb.parse_item("1、g")) is True

    seq2 = nb.Sequencer()
    for raw in ("1、a", "2、b", "3、c"):
        seq2.offer(nb.parse_item(raw))
    seq2.advance(nb.parse_lead("具体技术参数："))
    assert (seq2.last_top, seq2.last_no) == (0, "")
    assert seq2.plausible(nb.parse_item("1、g")) is True

    # 中文序数 / 引导句永远 plausible（它们自己就是节的边界）
    p = nb.Sequencer()
    assert p.plausible(nb.parse_chinese("一、x")) is True
    assert p.plausible(nb.parse_lead("引导句：")) is True
    # 括号编号走独立的上下界（1~30），不看游标
    assert p.plausible(nb.parse_paren("（0）x")) is False
    assert p.plausible(nb.parse_paren("（1）x")) is True
    assert p.plausible(nb.parse_paren("（30）x")) is True
    assert p.plausible(nb.parse_paren("（31）x")) is False
    # offer 失败时游标**不动**，否则一次误判会污染后面整块
    q = nb.Sequencer()
    q.advance(nb.parse_item("3、x"))
    assert q.offer(nb.parse_item("9、x")) is False and q.last_top == 3
    assert q.offer(nb.parse_item("4、x")) is True and q.last_top == 4


@pytest.mark.ec("H5")
@pytest.mark.case("X14")
def test_has_children_is_the_deterministic_heading_test_and_its_dot_guards_the_prefix():
    """``N、Title`` 是标题 **当且仅当** 存在编号为 ``N.k`` 的子项。

    已验证：P3 ``1、4K 采集摄像头`` 是标题（有 1.1~1.6）；P3 ``三、`` 下的
    ``1、数量：2 条`` 是参数项（下一条直接是 ``2、``）；P5 标的5 ``1、治疗源``
    是标题；P1 ``1、技术培训：…`` 是参数项。
    前缀必须带点号：``1`` 不能被 ``11.1 q`` 认作父项。
    ``（N）`` 撑起来的父子关系不算 —— 那是排版符号，不是文档自己的层级编号。
    """
    assert numbered("1、4K 采集摄像头").numbered.is_heading_like is True
    assert numbered("一、标题").numbered.is_heading_like is True
    assert numbered("1.1 子项").numbered.is_heading_like is False
    assert numbered("（1）x").numbered.is_heading_like is False
    assert numbered("引导句：").numbered.is_heading_like is False

    def P(raw):
        return nb.parse_line(raw)

    kids = [P("1、4K 采集摄像头"), P("1.1 xxx")]
    assert nb.has_children(kids, kids[0]) is True
    flat = [P("1、4K 采集摄像头"), P("2、yyy")]
    assert nb.has_children(flat, flat[0]) is False
    trap = [P("1、a"), P("11.1 q")]  # 点号守卫：11.1 不是 1 的子项
    assert nb.has_children(trap, trap[0]) is False
    assert nb.has_children([], P("一、标题")) is True  # 中文序数恒为 True（S4 规则 2）
    deep = [P("1.1 a"), P("1.1.1 q")]  # 二级项自己不当父，三级号不进分节路径
    assert nb.has_children(deep, deep[0]) is False
    paren_kids = [P("1、a"), P("（1）b")]
    assert nb.has_children(paren_kids, paren_kids[0]) is False
    paren_parent = [P("（1）a"), P("（2）b")]
    assert nb.has_children(paren_parent, paren_parent[0]) is False


@pytest.mark.ec("I2")
@pytest.mark.ec("H10")
@pytest.mark.case("X14")
def test_mark_headings_scans_a_window_not_the_whole_block():
    """窗口化扫描：遇到同级兄弟或中文序数 / 引导句即停，窗口内出现 ``N.k`` 才算标题。

    这是 P3 的生死线。P3 标的1 有 ``一、``~``十一、`` 十一节，每节都从 ``1、``
    重新编号。全块扫描时，``一、`` 下的 ``1、3D 输出方式`` 会因为 ``二、`` 下
    存在 ``1.1`` 而被误判成标题 —— P3 因此从 71 行掉到 57 行，并产生 35 个
    重复对齐键。下面 case (f) 就是这个形状的最小复现。
    """
    # (a) 块首孤儿续行 numbered=None：None 检查必须在访问 n.kind 之前
    orphan = nb.Item(numbered=None, lines=["跨页尾巴"], page=1)
    assert nb.mark_headings([orphan, numbered("1、a")]) == [False, False]
    # (b) 中文序数与引导句恒为 True（S4 规则 2 / 规则 4）
    assert nb.mark_headings([numbered("一、标题"), numbered("具体技术参数：")]) == [True, True]
    # (c) 括号项与子项恒为 False
    c = [numbered(x) for x in ("（1）x", "1.1 y", "1.9.3 z")]
    assert nb.mark_headings(c) == [False, False, False]
    # (d) 真标题：1、 后面跟着 1.1
    d = [numbered(x) for x in ("1、4K 采集摄像头", "1.1 xxx", "2、yyy")]
    assert nb.mark_headings(d) == [True, False, False]
    # (e) 参数项：1、 后面直接是 2、
    e = [numbered(x) for x in ("1、数量：2 条", "2、内窥镜类型")]
    assert nb.mark_headings(e) == [False, False]
    # (f) P3 的窗口形状：跨节的 1.1 不许污染上一节
    f = [numbered(x) for x in ("一、4K3D 摄像系统", "1、3D 输出方式", "2、xxx",
                               "二、配套摄像头", "1、4K 采集摄像头", "1.1 子项", "2、yyy")]
    assert nb.mark_headings(f) == [True, False, False, True, True, False, False]
    # (g) 孤儿续行在窗口里是**跳过**，不是中断
    g = [numbered("1、a"), orphan, numbered("1.1 b")]
    assert nb.mark_headings(g) == [True, False, False]
    # (h) 引导句则会**中断**窗口
    h = [numbered(x) for x in ("1、a", "引导句：", "1.1 b")]
    assert nb.mark_headings(h) == [False, True, False]
    # (i) 点号守卫：11.1 不是 1 的子项
    i = [numbered(x) for x in ("1、a", "11.1 b", "2、c")]
    assert nb.mark_headings(i) == [False, False, False]
    # (j)(k) 退化输入
    assert nb.mark_headings([numbered("1、a")]) == [False]
    assert nb.mark_headings([]) == []


@pytest.mark.ec("H2")
@pytest.mark.ec("H10")
def test_guard_bare_demotes_the_two_continuation_shapes_that_fool_the_regex():
    """第三道闸：空白分隔符的假条目，与被折行的标题尾巴。

    两类续行都能骗过正则**和** Sequencer 的跳号检查::

        P3 标的2   …每年提供≥            <- 上一行不以终止标点收尾
                   4 次的设备维护保养服务…   -> 误判成序号 4（2 < 4 <= 7 合理）
        P3 标的1   九、配置适合腹腔镜及开腹手术
                   使用电刀：              -> 误判成引导句，分节被截断

    判据是「分隔符为空白 + 顶层 + 正文非空」再加「上一被接受行以终止标点收尾」。
    对真条目无副作用：``4.`` 独占一行时正文为空，``1、``/``19.``/``13.1、``
    用的是显式标点分隔符，``2.1``/``1.9.3`` 是子层。
    """
    wrapped = [("2、设备性能维护：每年提供≥", 1), ("4 次的设备维护保养服务。", 1)]
    assert rows(wrapped) == [
        ("2", "item", 1, 1, "2、设备性能维护：每年提供≥4 次的设备维护保养服务。")
    ]
    off = rows(wrapped, guard_bare=False)  # 关掉闸门就会裂成两条
    assert [(r[0], r[4]) for r in off] == [
        ("2", "2、设备性能维护：每年提供≥"),
        ("4", "4 次的设备维护保养服务。"),
    ]
    # 上一行已收尾 -> 闸门让路，``4 次的…`` 是真条目
    terminated = [("2、设备性能维护：每年提供≥4 次。", 1), ("4 次的设备维护保养服务。", 1)]
    assert [r[0] for r in rows(terminated)] == ["2", "4"]
    # 正文为空（编号独占一行）-> 闸门让路，P5 标的8 就是这种排版
    bare_no = [("2、设备性能维护：每年提供≥", 1), ("4.", 1), ("基础率时段：≥48 段。", 1)]
    assert rows(bare_no) == [
        ("2", "item", 1, 1, "2、设备性能维护：每年提供≥"),
        ("4", "item", 1, 1, "4.基础率时段：≥48 段。"),
    ]
    # 被折行的标题尾巴降级为续行，分节路径因此保持完整
    tail = [("九、配置适合腹腔镜及开腹手术", 1), ("使用电刀：", 1), ("1、aaa", 1)]
    assert rows(tail) == [
        ("九", "chinese", 1, 1, "九、配置适合腹腔镜及开腹手术使用电刀："),
        ("1", "item", 1, 1, "1、aaa"),
    ]
    # 真引导句出现在块首（prev_terminal 初值为 True），不受影响
    assert rows([("设备运维技术服务要求如下：", 1), ("1、aaa", 1)]) == [
        ("", "lead", 0, 1, "设备运维技术服务要求如下："),
        ("1", "item", 1, 1, "1、aaa"),
    ]


@pytest.mark.ec("H9")
@pytest.mark.case("X13")
def test_the_paren_policy_decides_whether_children_fold_split_or_vanish():
    """``（N）`` 三策略：折叠进父项 / 各自成条 / 整行丢弃。

    策略必须可配，因为 gold **跨包不一致**：P1 标的2 的 ``2、设备运行维护：``
    下面 ``（1）（2）（3）`` 在 gold 里就是一整条（fold）；P2/P3 标的2 的同类文本
    在 gold 里是 ``2.1``/``2.2``/``2.3``（split）。到底走哪条由**厂商证据**决定
    （见 fold.py），这里只钉住三种机械结果。
    """
    lines = [("2、设备运行维护：", 1), ("（1）技术培训", 1), ("（2）设备运行维护", 1), ("（3）巡检", 1)]
    assert rows(lines) == [
        ("2", "item", 1, 1, "2、设备运行维护：（1）技术培训（2）设备运行维护（3）巡检")
    ]
    split = rows(lines, paren=nb.PAREN_SPLIT)
    assert [(r[0], r[1], r[2]) for r in split] == [
        ("2", "item", 1), ("1", "paren", 1), ("2", "paren", 1), ("3", "paren", 1)
    ]
    assert rows(lines, paren=nb.PAREN_DROP) == [("2", "item", 1, 1, "2、设备运行维护：")]


@pytest.mark.ec("H10")
@pytest.mark.case("B12")
def test_an_orphan_keeps_its_own_item_and_the_page_where_it_started():
    """跨页 / 块首的续行不丢：它单独成一个无编号 Item，页码取它**开始**的那一页。

    P3 的参数单元格跨 p8~p14，切分点会落在参数中间（B12）。孤儿 Item 的
    ``no/mark/kind/level`` 全是空值而非 ``None``，调用方可以无脑取属性。
    正文按 agent.md 3.6 **直接拼接不补空格**，这样才能与 gold 逐字一致。
    """
    # 跨页续行并入上一条，页码保留首行那一页
    assert rows([("1、aaa", 8), ("续行", 9), ("2、bbb", 9)]) == [
        ("1", "item", 1, 8, "1、aaa续行"),
        ("2", "item", 1, 9, "2、bbb"),
    ]
    # 拼接时先 strip 每一行，且**不补分隔符**
    items = nb.split_items([("  1、aaa  ", 1), ("  续行  ", 1)])
    assert items[0].text == "1、aaa续行"
    # 空行整行跳过，页码取真正存活的那一行
    assert rows([("", 1), ("   ", 2), ("1、aaa", 3)]) == [("1", "item", 1, 3, "1、aaa")]
    # 块首孤儿：属性是空值而非 None
    orphans = nb.split_items([("跨页尾巴", 4), ("1、aaa", 4)])
    head = orphans[0]
    assert (head.no, head.mark, head.kind, head.level, head.page) == ("", "", "", 0, 4)
    assert head.text == "跨页尾巴"
    # 关掉形态开关时，被关掉的那种也降级成孤儿
    for src, kw in (
        ([("一、标题", 1), ("1、aaa", 1)], {"allow_chinese": False}),
        ([("具体技术参数：", 1), ("1、aaa", 1)], {"allow_lead": False}),
    ):
        assert rows(src, **kw) == [
            ("", "", 0, 1, src[0][0]),
            ("1", "item", 1, 1, "1、aaa"),
        ], kw
    # 外部注入的游标已经走到 5：整块降级成**一个**孤儿，而不是各自成条
    seq = nb.Sequencer()
    seq.advance(nb.parse_item("5、x"))
    injected = nb.split_items([("1、aaa", 1), ("2、bbb", 1)], seq=seq)
    assert [(i.no, i.text) for i in injected] == [("", "1、aaa2、bbb")]
    # iter_numbered 只剔无编号孤儿。已知锋利边：块首孤儿不以终止标点收尾时，
    # 紧跟的真引导句会被 _is_bare_continuation 一并吸进孤儿（README 已登记）。
    swept = nb.split_items([("跨页尾巴", 1), ("具体技术参数：", 1), ("1、aaa", 1)])
    assert swept[0].text == "跨页尾巴具体技术参数："
    assert [(i.no, i.kind) for i in nb.iter_numbered(swept)] == [("1", "item")]
