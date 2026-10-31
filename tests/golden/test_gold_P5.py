# -*- coding: utf-8 -*-
"""P5 的金标准对照：C 族版面、四个不连续标的、半角 ``*``、以及离线能力的那道天花板。

P5（新郑市公立人民医院设备采购，56 行）是五个包里信息最杂的一个，
它独自承担四件别处测不到的事：

1. **C 族版面**：招标 PDF 里没有 A 族的六列采购清单，也没有 B 族的两列表，
   而是四段 ``N-科室-设备名技术参数`` 的标题块。标的号因此是
   ``{1, 5, 8, 12}`` —— 一个**不连续的子集**（B3）。
   把 ``{1,2,3,4}`` 硬编码进去的实现会在 P5 上立刻露馅。
2. **元数据瑕疵**：``00_投标范围.txt`` 里标的12 的数量写的是 ``1None``（B4）。
   解析器必须照常绑定这个标的，而不是因为数量字段脏了就把它整段丢掉。
3. **半角星号**：P5 用 ``*1.2 输出功率：≥750w`` 表示实质性要求，
   按投标范围末尾的说明要当 ``★`` 处理（F2）。成品里因此一个 ``*`` 都不该剩。
4. **离线天花板**：P5 的关键项风险识别是 ``0/1``，五个包里唯一没拿满的，
   也正是总分 ``4/5`` 那道门禁失败的**全部**来源。原因不是逻辑错，
   而是那一行 ★ 的实测值只存在于一张检测报告 JPEG 里，
   离线（无可用 Key，且 O3 禁止装本地 OCR）读不出来。
   本文件把"我们没能识别"与"我们把它标出来交给人工"两件事一起钉住。

与 agent.md 的偏差：无。10.1 的"关键项风险识别 5/5"在离线条件下上限就是 4/5，
这条限制同时记录在 README 的「已知问题」里。
"""

from __future__ import annotations

import pytest

from bidtable import config
from bidtable.output import review as rv

pytestmark = [pytest.mark.golden]

PACK = "P5"
N_ROWS = 56
_LOTS = {"1", "5", "8", "12"}
_PER_LOT = {"1": 18, "5": 11, "8": 12, "12": 15}
_DEVICE_OF_LOT = {
    "1": "床旁血液净化系统",
    "5": "红外辐照治疗装置",
    "8": "胰岛素泵",
    "12": "激光治疗仪",
}
_SECTIONS = {"", "1、治疗源", "2、照射光斑面积", "3、过滤器"}
_RISK_KEY = ("5", "1、治疗源", "1.2")


@pytest.mark.ec("A3")
@pytest.mark.ec("B3")
@pytest.mark.ec("B4")
@pytest.mark.ec("F2")
def test_p5_binds_four_non_contiguous_lots_and_never_leaks_a_half_width_star(offline_run):
    """C 族：56 行落在 {1,5,8,12} 四个不连续标的上，设备名逐个对，星号全部规范化。

    边界在哪：标的号是**从招标标题块里读出来的**，不是数出来的。
    一个"按出现顺序编号 1/2/3/4"的实现能过 P1~P4，唯独过不了 P5；
    一个"只认连续标的"的实现会在 ``12`` 上崩。所以这里断言的是集合**相等**
    （既不能少也不能多），并逐标的钉住行数与设备名。

    设备名那一列还藏着一个真实的脏数据边界：标的5 的标题块写的是
    ``5-耳鼻喉-红外辅照治疗仪技术参数``（**辅**，错别字），
    而块内表头写的是 ``红外辐照治疗装置``（**辐**，正确），
    投标范围里也是 **辐**。``tender_layout_c.device_name`` 的优先级是
    投标范围 > 表头 > 标题，因此成品必须一次都不出现"辅照"。
    这条断言的代价很低、收益很实：它抓住了"标题里有错别字"这类真实招标文件通病。

    标的12 的数量字段在投标范围里是 ``1None``（脏元数据），
    它照样绑上了 ``激光治疗仪`` 并输出 15 行 —— 数量字段不参与绑定，
    解析失败也不该连累整个标的。

    真实资料包实例：``训练集/P5``，13 个文件、四个标的、四份检测报告。

    与 agent.md 的偏差：无。
    """
    result = offline_run(PACK)
    rows = result.rows
    assert result.stats["layout"] == config.LAYOUT_C
    assert len(rows) == N_ROWS

    assert {r.lot_no for r in rows} == _LOTS
    assert _LOTS != {"1", "2", "3", "4"}, "P5 的标的号是不连续子集，不是顺序编号"
    for lot, n in _PER_LOT.items():
        assert sum(1 for r in rows if r.lot_no == lot) == n
    for lot, device in _DEVICE_OF_LOT.items():
        assert {r.device_name for r in rows if r.lot_no == lot} == {device}
    # 标题块里的错别字"辅照"不许渗到成品里（投标范围与表头都是"辐照"）。
    assert not any("辅照" in r.device_name for r in rows)

    # 半角 * 一律按 ★ 处理：标记列只有 ★ 和空，序号列与正文列都不许剩 *。
    assert {r.mark for r in rows} == {"", config.MARK_STAR}
    assert not any("*" in r.mark for r in rows)
    assert not any("*" in r.no for r in rows)
    assert not any("*" in r.req_text for r in rows)
    # 整包只有一个 ★：标的5 序号1.2（原文 ``*1.2 输出功率：≥750w``）。
    assert [(r.lot_no, r.no) for r in rows if r.mark] == [("5", "1.2")]


@pytest.mark.ec("I6")
@pytest.mark.ec("I7")
@pytest.mark.ec("H11")
def test_p5_fills_sections_only_where_headings_exist_and_keeps_every_numbering_gap(offline_run):
    """分节只在标的5 出现（整数混合层 + 同标的内部分填部分不填），序号的空号一律照抄。

    边界在哪：P5 的分节是 **I-6（整数混合层）** 的唯一样本——
    一级标题不是"一、二、"而是"1、治疗源 / 2、照射光斑面积 / 3、过滤器"，
    与 A 族的中文序数完全不同一套。同时它又是 **I-7（部分填部分不填）**：
    同一个标的5 里，序号 1.1~3.2 挂在三个分节下，而序号 4 和 5 的分节**为空**。
    把空的那两行也硬填一个分节（例如沿用上一个），是最容易犯、也最难发现的错——
    它不会报错，只会让这两行的对齐键与标准答案错开。

    另一层边界是**空号必须保留**：标的1 的序号是 1~15、17、19、20，
    真的缺 16 和 18（招标原文就没有），而 21 及以后被售后服务裁剪掉了。
    重排成连续序号会让整包的键对不上；把裁剪掉的尾巴放回来会多出 26 行。
    所以这里同时钉住"缺口在哪"和"尾巴被裁在哪"。

    裁剪本身是 O4 的直接落地：设备标的的售后服务要求不在标准答案范围内
    （agent.md 3.7 第 1 条），四个标的各裁一段，共 26 条，
    而且**每一条都写进跳过清单**——不是悄悄少行。

    真实资料包实例：``训练集/P5``，85 条抽取 → 裁掉 26 → 去掉 3 条标题 → 56 行。

    与 agent.md 的偏差：无。
    """
    result = offline_run(PACK)
    rows = result.rows

    assert {r.section for r in rows} == _SECTIONS
    # 只有标的5 有分节；其余三个标的整列为空（它们的招标块里没有小节标题）。
    assert {r.lot_no for r in rows if r.section} == {"5"}
    lot5 = [(r.no, r.section) for r in rows if r.lot_no == "5"]
    assert lot5 == [
        ("1.1", "1、治疗源"), ("1.2", "1、治疗源"), ("1.3", "1、治疗源"),
        ("1.4", "1、治疗源"), ("1.5", "1、治疗源"),
        ("2.1", "2、照射光斑面积"), ("2.2", "2、照射光斑面积"),
        ("3.1", "3、过滤器"), ("3.2", "3、过滤器"),
        ("4", ""), ("5", ""),  # I-7：同标的内这两行就是不填
    ]
    # I-6：分节以整数序号开头，不是中文序数。
    assert all(s[0].isdigit() for s in _SECTIONS if s)

    # 空号照抄：标的1 缺 16、18，尾巴从 21 起被裁掉。
    lot1 = [r.no for r in rows if r.lot_no == "1"]
    assert lot1 == [str(i) for i in range(1, 16)] + ["17", "19", "20"]
    assert [r.no for r in rows if r.lot_no == "8"] == [str(i) for i in range(1, 13)]
    assert [r.no for r in rows if r.lot_no == "12"] == [str(i) for i in range(1, 16)]

    # 售后服务裁剪：四个标的一段，共 26 条，每段都在跳过清单里留痕。
    assert result.stats["cut_counts"] == {
        "抽取条目": 85,
        "折叠后": 85,
        "裁剪后": 59,
        "分节后": 59,
        "标题/引导句": 3,
        "参与判定": N_ROWS,
    }
    assert len(result.skipped) == 4
    assert all("售后服务段" in s and "已裁剪" in s for s in result.skipped)
    assert {s.split(" 的")[0] for s in result.skipped} == {"标的1", "标的5", "标的8", "标的12"}
    # 被裁掉的售后服务条目一个字都不许出现在成品里。
    assert not any("操作手册" in r.req_text or "售后" in r.req_text for r in rows)


@pytest.mark.ec("F1")
@pytest.mark.ec("E3")
@pytest.mark.ec("J1")
def test_p5_single_key_item_miss_is_the_documented_offline_ceiling_and_it_is_flagged(
    score_pack, gold_map
):
    """P5：覆盖率 1.000、偏离 0.946，但关键项风险 0/1 —— 离线条件下读不出那张检测报告。

    边界在哪：这是整个项目**唯一一处明知做不到**的地方，所以它的边界不是
    "分数够不够高"，而是"失败得是否诚实"。三件事必须同时成立：

    1. 那一行 ★（标的5 / 1、治疗源 / 序号1.2）我们判了 ``无偏离``，
       标准答案是 ``负偏离`` —— 于是关键项风险识别得 0/1，
       总分 ``4/5``，10.1 的满分门禁失败。**失败的来源就是这一行**，不是散布各处。
    2. 这一行的置信度被压到 0.70（指针型证据的上限），带上
       ``ocr_failed`` 与 ``pointer_assumed`` 两个旗标，
       因而低于 ``review.CONF_REVIEW`` 0.85，**必然进优先复核清单**。
       换句话说：机器读不出来，但它明确说了"我读不出来，请人来看"。
    3. 告警里既有那一张 JPEG 的点名告警，也有"11 条要求指向检测报告"的汇总告警。

    为什么读不出来：这张报告是纯图像，没有 XMP/EXIF、尾部也没附带数据；
    账号欠费导致视觉模型不可用，而 O3 明确禁止安装本地 OCR 引擎。
    所以实测值在离线条件下**物理上不可得**。其余四个包的 ★/▲ 行不依赖读图，
    因此都拿到了 1/1。

    真实资料包实例：``训练集/P5``，56 行全部命中标准答案，多余 0。

    与 agent.md 的偏差：10.1 要求关键项风险识别 5/5；离线上限是 4/5。
    这条限制在 README「已知问题」与本测试里同时留档，不做静默降级。
    """
    result, score = score_pack(PACK)

    assert score["条目总数"] == N_ROWS
    assert score["提交匹配条目"] == N_ROWS
    assert score["多余条目"] == 0
    assert score["覆盖率"] == pytest.approx(1.0)
    assert score["偏离类型准确率"] == pytest.approx(53 / N_ROWS, abs=1e-3)
    assert score["证明材料命中率"] == pytest.approx(0.979, abs=1e-3)
    assert score["位置命中率"] == pytest.approx(0.979, abs=1e-3)
    assert score["关键项风险识别"].startswith("0/1")

    matrix = score["混淆矩阵"]
    assert sum(matrix.values()) == N_ROWS
    off = {k: v for k, v in matrix.items() if k.split("->")[0] != k.split("->")[1]}
    # 三格失误全部从"负偏离"出发：与 P2 同一个结构性短板（拒答词表是闭集）。
    assert off == {"负偏离->无偏离": 2, "负偏离->未响应": 1}
    assert {k.split("->")[0] for k in off} == {"负偏离"}

    # 风险行的位置在标准答案里是存在的（覆盖率 1.0），错的只是偏离类型。
    assert _RISK_KEY in set(gold_map(PACK))
    risk = [r for r in result.rows if (r.lot_no, r.section, r.no) == _RISK_KEY]
    assert len(risk) == 1
    row = risk[0]
    assert row.mark == config.MARK_STAR
    assert row.deviation == config.DEV_NONE          # 我们说无偏离；答案是负偏离
    assert row.evidence_file.endswith("检测报告.jpg")
    assert row.location == "检测报告 第1页 表行3"
    assert row.confidence == pytest.approx(0.70)      # 指针型证据的置信上限
    assert {"ocr_failed", "pointer_assumed"} <= set(row.flags)

    # 失败必须是响的：进优先复核清单，且告警点名到具体文件。
    assert row.confidence < rv.CONF_REVIEW
    assert rv.needs_review(row)
    assert any(row.evidence_file in w and "模型不可用" in w for w in result.warnings)
    assert any("11 条要求指向检测报告" in w for w in result.warnings)
    # 我们自报的关键项风险行数是 0 —— 这正是 0/1 的由来，而不是统计口径不同。
    assert result.stats["stage_counts"]["关键项风险（★/▲ 且负偏离或未响应）"] == 0
    assert rv.risk_rows(result.rows) == []
