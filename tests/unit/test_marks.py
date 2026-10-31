"""``bidtable/ingest/marks.py``：``★`` / 半角 ``*`` / ``▲`` 的抽取与剥离。

本文件盯住五个互不重叠的能力边界：

1. ``find_marks`` 的优先级仲裁——``★`` 压过 ``▲``，无标记返回 ``""``，
   且半角 ``*`` 在这一层**不算**标记（它需要位置判定）。
2. ``edge_mark`` 把行首标记归一并剥掉，同时**保留缩进**。
3. ``edge_mark`` 的混淆面——乘号 ``3840*2160`` 与行中间的 ``★`` 都不能被剥。
4. ``strip_trailing_mark`` 只在「标记之前全是编号字符」时才剥，
   否则会把正文吃掉；``None`` / 空串 / 裸 ``*`` 三个退化输入不崩。
5. ``strip_all`` 是编号解析专用的清道夫——删贴边 ``*`` 与全部 ``★``/``▲``，
   但**保留内部乘号**。
"""

from __future__ import annotations

import pytest

from bidtable.ingest import marks

pytestmark = [pytest.mark.unit]


@pytest.mark.ec("F1")
@pytest.mark.ec("F4")
@pytest.mark.ec("F5")
def test_find_marks_arbitrates_by_priority_and_will_not_guess_from_an_asterisk():
    """输出的「标记」列只有一个格子，两个标记同时出现时必须仲裁出一个。

    ``★`` 是实质性要求、风险等级更高，所以压过 ``▲``（P4 的 ``1.7★`` 与
    P1 的 ``▲2.`` 同属这一族）。半角 ``*`` 在这里被**故意忽略**：
    ``0.1~20m1／h``、``3840*2160`` 里的 ``*`` 是乘号，只有贴边才是标记，
    位置判定属于 :func:`marks.edge_mark`，``find_marks`` 不能越权。
    """
    assert marks.find_marks("★输出功率：≥750w") == "★"
    assert marks.find_marks("▲2.最大单脉冲能量5J。") == "▲"
    # 两个都在时 ★ 胜出，且与出现顺序无关
    assert marks.find_marks("▲配置★要求") == "★"
    assert marks.find_marks("★配置▲要求") == "★"
    # 无标记 -> 空串：206 条 gold 行里有 194 条的标记列就是空的
    assert marks.find_marks("工作电流：≤10A") == ""
    assert marks.find_marks("") == ""
    assert marks.find_marks(None) == ""
    # 半角 * 不是本函数的业务：乘号必须被无视，不能因为"像标记"就报 ★
    assert marks.find_marks("3840*2160") == ""
    assert marks.find_marks("*1.2 输出功率") == ""


@pytest.mark.ec("F2")
@pytest.mark.case("X8")
def test_edge_mark_normalizes_a_half_width_asterisk_and_keeps_the_indent():
    """每个 ``00_投标范围.txt`` 都写明「部分文件用半角 ``*`` 表示同样含义」。

    所以 ``*1.2 输出功率：≥750w``（P5 标的5 真实行）必须归一成 ``★``，
    否则它进不了关键项风险统计。缩进属于版面信息：A 族单元格里常带前导
    空格，剥标记时顺手吃掉会让后续按 x 坐标分流的逻辑对不上。
    """
    assert marks.normalize_token("*") == "★"
    assert marks.normalize_token("★") == "★"
    assert marks.normalize_token("▲") == "▲"
    assert marks.normalize_token("x") == "x"

    assert marks.edge_mark("*1.2 输出功率：≥750w") == ("★", "1.2 输出功率：≥750w")
    assert marks.edge_mark("▲2.最大单脉冲能量5J。") == ("▲", "2.最大单脉冲能量5J。")
    assert marks.edge_mark("★治疗源") == ("★", "治疗源")
    # 缩进原样留在正文前面
    assert marks.edge_mark("   ▲1.4 、屈光补偿") == ("▲", "   1.4 、屈光补偿")
    assert marks.edge_mark("\t*16、售后服务：") == ("★", "\t16、售后服务：")
    # 退化输入：只剥行首那一个字符，剩下的即使是标记也留给正文
    assert marks.edge_mark("*") == ("★", "")
    assert marks.edge_mark("★★双标记") == ("★", "★双标记")
    assert marks.edge_mark("") == ("", "")
    assert marks.edge_mark(None) == ("", "")
    assert marks.edge_mark("   ") == ("", "   ")


@pytest.mark.ec("F2")
def test_edge_mark_leaves_a_multiplication_sign_and_a_mid_line_star_alone():
    """混淆面：``*`` 只在整行首字符位置才是标记，``★`` 只在贴边才是标记。

    ``3840*2160`` 是分辨率（B4 用例的原文），``0.l-5ml／每次最大给药量`` 是
    P5 标的1 里一条必须被拒绝的伪编号——两处的 ``*``/``l`` 都在正文中间。
    ``1.7★``（P4 真实序号格）是**尾缀**标记，``edge_mark`` 管不着，
    那是 :func:`marks.strip_trailing_mark` 的活。
    """
    assert marks.edge_mark("3840*2160") == ("", "3840*2160")
    assert marks.edge_mark("输出分辨率：3840*2160P") == ("", "输出分辨率：3840*2160P")
    assert marks.edge_mark("0.l-5ml／每次最大给药量") == ("", "0.l-5ml／每次最大给药量")
    # 行中间的 ★ 属于正文（原文列要逐字保留，方便复核的人对回 PDF）
    assert marks.edge_mark("配置★要求：双探测器") == ("", "配置★要求：双探测器")
    assert marks.edge_mark("1.7★") == ("", "1.7★")


@pytest.mark.ec("F3")
@pytest.mark.case("X9")
def test_strip_trailing_mark_only_fires_when_everything_before_it_is_numbering():
    """B 族把 ``★`` 追加在**序号列**里（P4 只有 ``1.7★ 1.9.1★ 1.9.2★ 1.16★`` 四处）。

    输出的序号必须是裸号，所以尾缀标记要剥；但剥的闸门是「标记之前全是
    数字 / 半角点 / 全角点 / 顿号」。少了这道闸，
    ``2.最大单脉冲能量5J。（大于5J 为优于）★`` 会被削成正文残缺。
    注意本函数**不做全角归一**——``1．7`` 原样返回，那是 ``numbering.normalize_no`` 的职责。
    """
    assert marks.strip_trailing_mark("1.7★") == ("★", "1.7")
    assert marks.strip_trailing_mark("1.9.1★") == ("★", "1.9.1")
    assert marks.strip_trailing_mark("1.16★") == ("★", "1.16")
    assert marks.strip_trailing_mark("21.7*") == ("★", "21.7")
    # 顿号也是合法编号字符
    assert marks.strip_trailing_mark("13.1、★") == ("★", "13.1、")
    # 全角点原样保留，不在这里归一
    assert marks.strip_trailing_mark("1．7★") == ("★", "1．7")
    # 闸门：正文以 ★ 结尾时一个字都不能动
    prose = "2.最大单脉冲能量5J。（大于5J 为优于）★"
    assert marks.strip_trailing_mark(prose) == ("", prose)
    assert marks.strip_trailing_mark("临床用途★") == ("", "临床用途★")
    # 尾部空白先 rstrip 再判定；**前导**空白则会让闸门失配（head 里混进空格）。
    # 这是一处潜在的锋利边：三个调用方（layout_b 的 ``.strip()``、``split_items``
    # 的逐行 strip、layout_a/c 的单元格清洗）都先 strip 过，所以实际不可达。
    assert marks.strip_trailing_mark("1.7★  ") == ("★", "1.7")
    assert marks.strip_trailing_mark("  1.7★") == ("", "  1.7★")
    # 退化输入：空串回空串、None 回 None（原对象，不是 ""）、裸 * 没有可剥的头
    assert marks.strip_trailing_mark("") == ("", "")
    assert marks.strip_trailing_mark(None) == ("", None)
    assert marks.strip_trailing_mark("*") == ("", "*")
    assert marks.strip_trailing_mark("★") == ("", "★")
    # 兼容旧名
    assert marks.trailing_mark is marks.strip_trailing_mark


@pytest.mark.ec("F1")
@pytest.mark.ec("F2")
@pytest.mark.ec("F3")
def test_strip_all_clears_marks_for_the_numbering_parser_but_keeps_interior_asterisks():
    """``strip_all`` 只服务于编号解析，永远不能用来洗「招标要求（原文）」列。

    它先用 ``edge_mark`` 吃掉行首标记，再删掉所有 ``★``/``▲``，
    以及**位置 0 或 len-1** 的 ``*``。位置判定是在 edge_mark 之后的串上做的，
    所以行首 ``*`` 永远走的是 edge_mark 那条路，``i == 0`` 分支只是兜底。
    内部乘号必须活着，否则 ``3840*2160`` 会变成 ``38402160``，
    后面 ``parse_resolution`` 就拿不到复合分辨率了（G8）。
    """
    assert marks.strip_all("*1.2 输出功率：≥750w") == "1.2 输出功率：≥750w"
    assert marks.strip_all("▲2.最大单脉冲能量5J。") == "2.最大单脉冲能量5J。"
    assert marks.strip_all("*16、售后服务：") == "16、售后服务："
    assert marks.strip_all("1.7★") == "1.7"
    assert marks.strip_all("★▲1.1 治疗源★") == "1.1 治疗源"
    # 内部乘号一律保留
    assert marks.strip_all("3840*2160") == "3840*2160"
    assert marks.strip_all("输出分辨率：3840*2160P") == "输出分辨率：3840*2160P"
    # 行首 * 由 edge_mark 吃掉，内部 * 不受影响
    assert marks.strip_all("*3840*2160") == "3840*2160"
    # 行末 * 命中 len-1 分支
    assert marks.strip_all("分辨率：1536*") == "分辨率：1536"
    assert marks.strip_all("") == ""
    assert marks.strip_all(None) == ""
