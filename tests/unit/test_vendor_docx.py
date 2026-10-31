"""bidtable.ingest.vendor_docx —— 厂商 docx 摄取：**表行号就是证据坐标**。

技术规格书与服务方案形状一致（11 段 + 恰好 1 张 ``序号|项目|规格/说明`` 三列表），
所以 ``位置`` 可以写成 ``技术规格书 表行N`` 而不必记页码——但 N 必须和 gold 对得上，
否则整包的位置命中率归零。九条用例各盯一条判别力，1/2/3 打真实资料包，其余打
测试期现场构造的 docx（agent.md 6.2：``fixtures/corrupt`` 不落二进制）：

1. T7：真实 P2 规格书 13 行，``表行N`` 恒等于「序号」列，``投标响应`` 逐字 ``项目：规格``。
2. X29：服务方案没有「型号：」行，型号只能从「一、产品概述」正文里取。
3. B10：只有一行数据的服务方案照样产出 ``表行1``——它是 P2 展开判据的唯一来源。
4. N7：无表格 / 表头列数不对 / 表头名不认，三种都只告警不抛，其余资料照常。
5. 读不出来的文件抛 ``VendorDocError``（退出码 1）：结构缺陷告警，文件损坏才报错。
6. B8：整行空白静默跳过，半行空白要告警——区别是排版填充行 vs 投标人漏答。
7. B9：只有表头的表格产出 0 条证据，不崩也不告警。
8. X27：``表行7`` 中间没有空格；「序号」非纯数字时退回行序，这会撞号。
9. 文档类型只由文件名决定，优先级只由 ``DOC_PRIORITY`` 常量表决定。
"""

from __future__ import annotations

import docx
import pytest

from bidtable import errors
from bidtable.config import DOC_PRIORITY
from bidtable.ingest import vendor_docx as vd

pytestmark = [pytest.mark.unit]

SPEC_HEAD = ["a 技术规格书", "型号：SYN-1"]
TBL = [["序号", "项目", "规格/说明"], ["1", "甲", "1"]]


@pytest.fixture
def mk_docx(tmp_path):
    """``mk_docx(文件名, 段落列表, 表格=None)`` -> 现场造一份 docx 并返回路径。"""

    def _build(name: str, paras: list[str], table: list[list[str]] | None = None):
        d = docx.Document()
        for p in paras:
            d.add_paragraph(p)
        if table:
            t = d.add_table(rows=len(table), cols=len(table[0]))
            for i, row in enumerate(table):
                for j, cell in enumerate(row):
                    t.rows[i].cells[j].text = cell
        path = tmp_path / name
        d.save(str(path))
        return path

    return _build


def spec_doc(pack_path, pack: str, name: str) -> vd.VendorDoc:
    return vd.parse_docx(pack_path(pack) / "02_厂商资料" / name)


@pytest.mark.ec("D1")
@pytest.mark.ec("J3")
@pytest.mark.case("T7")
def test_the_locator_number_is_the_serial_column_on_the_real_p2_spec(pack_path):
    """T7：``BT_BT-6869S_技术规格书.docx`` 13 行数据，``表行N`` 恒等于「序号」列。

    gold 的位置串就是按「序号」列写的，这两个数一旦脱钩，整包 13 行的位置命中率
    直接归零——这也是 ``to_evidence`` 宁可退回行序也不肯留空的原因。``投标响应``
    必须逐字是 ``项目：规格``，因为判偏离拿它当响应原文用（agent.md 3.10）。
    """
    doc = spec_doc(pack_path, "P2", "BT_BT-6869S_技术规格书.docx")
    assert (doc.doc_type, doc.product_name, doc.model) == (
        "技术规格书", "视网膜脉络膜造影仪", "BT-6869S")
    assert doc.maker == "博泰光电设备有限公司" and doc.warnings == []
    assert len(doc.rows) == 13
    assert [no for no, _, _ in doc.rows] == [str(i) for i in range(1, 14)]

    ev = vd.to_evidence(doc, "1")
    assert [e.locator for e in ev] == [f"技术规格书 表行{n}" for n, _, _ in doc.rows]
    assert all(e.response_text == f"{item}：{spec}"
               for e, (_, item, spec) in zip(ev, doc.rows))
    assert {(e.lot_no, e.model, e.filename, e.doc_type, e.priority, e.is_pointer, e.confidence)
            for e in ev} == {("1", "BT-6869S", "BT_BT-6869S_技术规格书.docx",
                              "技术规格书", 1, False, 1.0)}
    # 第 2 行正是 T15 的「暂不支持」：负偏离词表在真实数据里的落点
    assert doc.rows[1] == ("2", "独立成像模式", "暂不支持")


@pytest.mark.ec("D4")
@pytest.mark.ec("B2")
@pytest.mark.case("X29")
def test_a_service_plan_hides_its_model_inside_the_overview_prose(pack_path):
    """X29：服务方案第 2 段只有「投标人：…」，型号藏在「一、产品概述」正文里。

    ``_MODEL_LINE`` 找不到「型号：」就必须退到 ``_OVERVIEW_MODEL``，否则 model 为空、
    lotmap 只能靠文件名猜——而 P3 的服务方案叫 ``TH_服务方案.docx``，文件名里根本没有
    型号，猜不出来就会把这条证据绑到错误的标的上。顺带钉住 X2 的由来：同一个包里
    服务方案的型号（TH-8394S）和规格书的型号（TH-4520）**不同**。
    """
    svc = spec_doc(pack_path, "P3", "TH_服务方案.docx")
    assert (svc.doc_type, svc.model, svc.maker) == (
        "服务方案", "TH-8394S", "泰和生物医学工程有限公司")
    assert svc.product_name == "设备运维技术服务" and svc.warnings == []
    # 项目名逐字是「常巡检维护」（掉了「日」字）：fold._hit 用互相包含就是为它
    assert [item for _, item, _ in svc.rows] == ["常巡检维护"]

    assert spec_doc(pack_path, "P3", "TH_TH-4520_技术规格书.docx").model == "TH-4520"
    # 型号缺失时必须告警而不是静默留空，否则绑定错误无从追查
    assert svc.model != "TH-4520"


@pytest.mark.ec("D4")
@pytest.mark.case("B10")
def test_a_single_row_service_plan_still_yields_row_one(pack_path):
    """B10：``BT_服务方案.docx`` 只有一行数据，仍要产出 ``服务方案 表行1``。

    退化到 n=1 最容易写出 ``表行0`` 或干脆不产出证据。而这一行恰恰是 P2 标的2
    保持展开的唯一判据——``fold.decide`` 看见子项名「日常巡检维护」被逐项应答，
    丢掉它 P2 就会折成 27 行，覆盖率当场掉一档。
    """
    doc = spec_doc(pack_path, "P2", "BT_服务方案.docx")
    assert len(doc.rows) == 1 and doc.model == "BT-6259X" and doc.warnings == []
    ev = vd.to_evidence(doc, "2")
    assert [(e.locator, e.name, e.doc_type, e.priority) for e in ev] == [
        ("服务方案 表行1", "日常巡检维护", "服务方案", 1)]
    assert ev[0].response_text.startswith("日常巡检维护：提供设备运行巡检服务")


@pytest.mark.ec("D1")
@pytest.mark.ec("J1")
@pytest.mark.case("N7")
def test_three_kinds_of_unusable_table_warn_and_skip_instead_of_raising(mk_docx):
    """N7：无表格 / 表头列数不对 / 表头名不认——三种都只告警，返回空 rows。

    一份资料坏掉不该拖垮整包（agent.md 4.5 第 2 条）。告警里必须点名文件、并把
    **实际看到的表头**打出来，否则复核的人无从下手。表头判定用交集而非全等，
    所以「规格」「说明」两种写法都放行，英文表头不放行。
    """
    no_table = vd.parse_docx(mk_docx("A_技术规格书.docx", SPEC_HEAD + ["二、技术指标"]))
    assert no_table.rows == [] and "文档内没有表格" in no_table.warnings[0]
    assert no_table.model == "SYN-1"                 # 表格坏了不影响段落解析

    two_col = vd.parse_docx(mk_docx("B_技术规格书.docx", SPEC_HEAD,
                                    [["序号", "项目"], ["1", "x"]]))
    assert two_col.rows == [] and "['序号', '项目']" in two_col.warnings[0]

    alien = vd.parse_docx(mk_docx("C_技术规格书.docx", SPEC_HEAD,
                                  [["甲", "乙", "丙"], ["1", "x", "y"]]))
    assert alien.rows == [] and alien.warnings[0].startswith("C_技术规格书.docx：表头是")
    assert "已跳过该文件" in alien.warnings[0]

    for hdr in (["序号", "项目", "规格"], ["序号", "项目", "说明"]):
        ok = vd.parse_docx(mk_docx("D_技术规格书.docx", SPEC_HEAD, [hdr, ["1", "x", "y"]]))
        assert ok.rows == [("1", "x", "y")] and ok.warnings == [], hdr
    assert vd.parse_docx(mk_docx("E_技术规格书.docx", SPEC_HEAD,
                                 [["No", "Name", "Value"], ["1", "x", "y"]])).rows == []


@pytest.mark.ec("D1")
@pytest.mark.ec("J7")
def test_an_unreadable_file_raises_with_a_named_file_and_exit_code_one(tmp_path):
    """结构性缺陷只告警，**文件读不出来**才抛——两条路的退出码不一样。

    ``VendorDocError`` 是 ``InputError`` 的子类，退出码 1（输入问题）而不是 2
    （解析失败）。消息必须点名是哪个文件：一个包里有 9 份资料时，不点名就等于
    没报错。文件不存在走同一条路，因为 docx 库对两者抛的是同一种异常。
    """
    bad = tmp_path / "SYN_技术规格书.docx"
    bad.write_bytes(b"this is not a zip container")
    with pytest.raises(errors.VendorDocError) as ei:
        vd.parse_docx(bad)
    assert ei.value.exit_code == 1 and bad.name in str(ei.value)
    assert isinstance(ei.value, errors.InputError)

    with pytest.raises(errors.VendorDocError) as ei2:
        vd.parse_docx(tmp_path / "不存在的_技术规格书.docx")
    assert ei2.value.exit_code == 1 and "不存在的_技术规格书.docx" in str(ei2.value)


@pytest.mark.ec("D1")
@pytest.mark.case("B8")
def test_a_fully_blank_row_is_padding_but_a_half_blank_row_is_missing_data(mk_docx):
    """B8：整行空白**静默**跳过，半行空白要告警——排版填充行 vs 投标人漏答。

    Word 表格末尾常带一整行空白，为它告警会把复核报告淹掉；但只有「项目」没有
    「规格」意味着投标人漏答了一项，那是必须让人看见的信号。两种空的处理**故意不同**。
    另外 ``_clean`` 把单元格里的换行和连续空白收成单个空格，与 gold 的写法对齐。
    """
    doc = vd.parse_docx(mk_docx("S_技术规格书.docx", SPEC_HEAD, [
        ["序号", "项目", "规格/说明"],
        ["1", "", ""], ["2", "有名无值", ""], ["3", "", "有值无名"], ["4", "正常", "值"],
    ]))
    assert doc.rows == [("4", "正常", "值")]
    assert doc.warnings == [
        "S_技术规格书.docx：表行 2 的项目或规格为空，已跳过该行。",
        "S_技术规格书.docx：表行 3 的项目或规格为空，已跳过该行。",
    ]

    clean = vd.parse_docx(mk_docx("T_技术规格书.docx", SPEC_HEAD,
                                  [["序号", "项目", "规格/说明"], ["  1 ", "  甲\n乙 ", "  a   b  "]]))
    assert clean.rows == [("1", "甲 乙", "a b")] and clean.warnings == []


@pytest.mark.ec("D1")
@pytest.mark.case("B9")
def test_a_header_only_table_yields_zero_evidence_without_warning(mk_docx):
    """B9：只有表头的表格 -> 0 条证据，不崩，也**不告警**。

    ``table`` 非空（有一行表头）所以走不到「表格为空」那条分支，``table[1:]`` 是空
    列表所以一行也不产出。这是「投标人交了张空表」的形状：没有数据可判，但也不是
    文件损坏——告警只会制造噪音，真正的问题会在匹配阶段以「未响应」显现出来。
    """
    doc = vd.parse_docx(mk_docx("H_技术规格书.docx", SPEC_HEAD, [["序号", "项目", "规格/说明"]]))
    assert doc.rows == [] and doc.warnings == [] and doc.model == "SYN-1"
    assert vd.to_evidence(doc, "1") == []


@pytest.mark.ec("D1")
@pytest.mark.ec("H12")
@pytest.mark.case("X27")
def test_the_locator_has_no_space_before_the_number_and_falls_back_to_row_order(mk_docx):
    """X27：``表行7`` 中间没有空格；「序号」非纯数字时退回**行序**。

    gold 写的是 ``技术规格书 表行7``，多一个空格整列就对不上。退回行序有个真实的
    锋利边：空「序号」和真「序号 3」会撞成同一个 ``表行3``。训练集踩不到（规格书的
    序号列永远是 1..N 连续整数），但全角 ``３`` 会被 ``isdigit()`` 放行、原样进 locator，
    而 ``numbering`` 层是转半角的——两层对全角的处理不一致，这里如实钉住。
    """
    doc = vd.parse_docx(mk_docx("L_技术规格书.docx", SPEC_HEAD, [
        ["序号", "项目", "规格/说明"],
        ["一", "甲", "1"], ["1.1", "乙", "2"], ["", "丙", "3"], ["3", "丁", "4"],
    ]))
    assert [no for no, _, _ in doc.rows] == ["一", "1.1", "", "3"]
    loc = [e.locator for e in vd.to_evidence(doc, "1")]
    assert loc == ["技术规格书 表行1", "技术规格书 表行2", "技术规格书 表行3", "技术规格书 表行3"]
    assert "技术规格书 表行 3" not in loc                     # ★ 表行与数字之间无空格

    wide = vd.parse_docx(mk_docx("W_技术规格书.docx", SPEC_HEAD,
                                 [["序号", "项目", "规格/说明"], ["３", "甲", "1"]]))
    assert [e.locator for e in vd.to_evidence(wide, "1")] == ["技术规格书 表行３"]


@pytest.mark.ec("D1")
@pytest.mark.ec("D4")
@pytest.mark.ec("J3")
def test_doc_type_comes_from_the_filename_and_priority_from_the_constant_table(mk_docx):
    """文档类型**只**由文件名决定，优先级**只**由 ``DOC_PRIORITY`` 常量表决定。

    ``TITLE_SPECS`` 是有序的，先匹配到的赢，所以 ``服务方案技术规格书.docx`` 判成
    技术规格书；两个都不匹配时默认技术规格书。这件事必须钉住：规格书与服务方案的
    优先级同为 1，看着没差别，但 ``collapse_values`` 在同分时按 doc_type 判是否冲突，
    类型认错就会把「多源一致」误判成「多源冲突」（X3/X4 整条链的起点）。
    """
    cases = {
        "X_技术规格书.docx": "技术规格书",
        "X_规格书.docx": "技术规格书",
        "X_服务方案.docx": "服务方案",
        "服务方案技术规格书.docx": "技术规格书",
        "技术规格书服务方案.docx": "技术规格书",
        "随便一个.docx": "技术规格书",
    }
    for name, want in cases.items():
        doc = vd.parse_docx(mk_docx(name, SPEC_HEAD, TBL))
        ev = vd.to_evidence(doc, "1")
        assert doc.doc_type == want, name
        assert ev[0].priority == DOC_PRIORITY[want] == 1, name
        assert ev[0].locator == f"{want} 表行1", name

    assert DOC_PRIORITY["技术规格书"] == DOC_PRIORITY["服务方案"] == 1
    assert DOC_PRIORITY["检测报告"] == 0 < DOC_PRIORITY["产品综合样本"] == 2
