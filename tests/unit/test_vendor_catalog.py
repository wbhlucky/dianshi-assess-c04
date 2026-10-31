"""bidtable.ingest.vendor_catalog —— 产品综合样本：**页码与指针序号就是证据坐标**。

样本一页一个产品，``■`` 是要点（值），``●`` 是「详见检测报告」指针（没有值）。
两类东西产出的证据契约完全不同，而它们各自决定了 ``位置`` 列的两个不同分支：
``产品综合样本 第N页`` 的 N 是产品所在页，``检测报告 第1页 表行N`` 的 N 是该页内
指针的出现次序 —— 后者正是「不做 OCR 也能填出报告行号」的全部依据（X1）。
八条用例各盯一条判别力，1/2/3/5 打真实资料包，其余打测试期现场构造的 PDF
（agent.md 6.2：``fixtures/corrupt`` 不落二进制）：

1. T8：真实 P1 两页两产品，型号只能从「订购信息」行取，11 条证据的坐标逐条对得上。
2. X1：指针的 ``doc_type`` 是检测报告不是产品综合样本，且 ``表行N`` 按产品重新计数。
3. X30：P3 的服务产品在第 2 页，它的证据写 ``第2页``——页码跟着产品走，不是一律第 1 页。
4. N6：整页 ``get_text()`` 为空（扫描页）时跳过并告警，且**不挪动**后面产品的页码。
5. 折行拼接：源 PDF 行尾那个空格被 ``strip()`` 吃掉，但打分侧看不出来（``squash`` 同值）。
6. 两遍扫描的退出条件：结构行终止折行吸收，「订购信息」行终止整页。
7. 三种畸形页各自降级并给出指名道姓的告警，其余内容照常解析。
8. 打不开的文件只告警返回空 Catalog，绝不抛；空型号不参与 ``product_for_model`` 绑定。
"""

from __future__ import annotations

from collections import Counter

import fitz
import pytest

from bidtable.config import DOC_CATALOG, DOC_PRIORITY, DOC_REPORT
from bidtable.ingest import vendor_catalog as vc
from bidtable.match.normalize import squash

pytestmark = [pytest.mark.unit]

ORDER_LINES = ["订购信息", "订货号：SYN-9    产品名称：合成设备    型号：SYN-1"]


def mk_pdf(tmp_path, name, pages):
    """按行写一份最小样本 PDF；``pages`` 是「每页的行列表」。"""
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        y = 50
        for line in lines:
            page.insert_text((50, y), line, fontname="china-s", fontsize=10)
            y += 16
    path = tmp_path / name
    doc.save(str(path))
    doc.close()
    return path


def catalog_of(pack_path, pack):
    return vc.parse_catalog(pack_path(pack) / "02_厂商资料" / "产品综合样本.pdf")


@pytest.mark.ec("D2")
@pytest.mark.ec("D8")
@pytest.mark.ec("D3")
@pytest.mark.case("T8")
def test_a_real_catalog_gives_one_product_per_page_and_model_from_the_order_line(pack_path):
    """真实 P1：两页 = 两个产品，型号来自「订购信息」行而不是页首大标题。

    页首标题 ``钬激光治疗机YF-9834Pro`` 是产品名与型号直接粘连的，正则切不开；
    真正可靠的型号来源只有页尾那一行 ``订货号：… 产品名称：… 型号：…``。
    """
    cat = catalog_of(pack_path, "P1")
    assert cat.pages == 2 and cat.warnings == []
    assert [(p.page, p.product_name, p.model, p.order_no) for p in cat.products] == [
        (1, "钬激光治疗机", "YF-9834Pro", "YF162472"),
        (2, "设备运维技术服务", "YF-1509Pro", "YF294111"),
    ]
    device, service = cat.products
    assert len(device.bullets) == 7 and device.pointers == [
        "脉冲宽度可调范围", "光纤终端最大平均输出功率", "激光耦合效率"]
    values = dict(device.bullets)
    assert values["指示光功率"] == "5mW"
    assert values["272μm 光纤最大传输功率"] == "84W"
    assert values["200μm 软光纤最小弯曲半径"] == "4cm"
    assert len(service.bullets) == 1 and service.pointers == []

    ev = vc.to_evidence(cat)
    assert len(ev) == 11
    assert Counter(
        (e.doc_type, e.is_pointer, e.priority, e.confidence) for e in ev
    ) == Counter({
        (DOC_CATALOG, False, DOC_PRIORITY[DOC_CATALOG], 0.9): 8,
        (DOC_REPORT, True, DOC_PRIORITY[DOC_REPORT], 0.5): 3,
    })
    assert [e.locator for e in ev] == ["产品综合样本 第1页"] * 7 + [
        "检测报告 第1页 表行1",
        "检测报告 第1页 表行2",
        "检测报告 第1页 表行3",
        "产品综合样本 第2页",
    ]
    first = ev[0]
    assert first.lot_no == "" and first.model == "YF-9834Pro"
    assert first.filename == "产品综合样本.pdf"
    assert first.response_text == f"{first.name}：{first.value}"
    assert first.name == "钬激光治疗系统配置要求"


@pytest.mark.ec("D8")
@pytest.mark.ec("D3")
@pytest.mark.case("X1")
def test_a_pointer_routes_to_the_report_and_its_row_number_restarts_per_product(pack_path):
    """真实 P3：``● …：详见检测报告`` 是路由指令，不是答案。

    它的 ``doc_type`` 必须是检测报告（否则 ``位置`` 会写成产品综合样本，整条错），
    ``value`` 必须是空（否则会被当成实测值去比大小）。``表行N`` 的 N 按**产品**
    重新计数：P3 第 1 页 4 个指针给 1..4，第 2 页那 1 个又从 1 开始 —— 因为每份
    检测报告只对应一个产品，页内出现次序就等于那张报告表格里的「序号」。
    """
    ev = vc.to_evidence(catalog_of(pack_path, "P3"))
    pointers = [e for e in ev if e.is_pointer]
    assert [e.locator for e in pointers] == [
        f"检测报告 第1页 表行{i}" for i in (1, 2, 3, 4, 1)
    ]
    head = pointers[0]
    assert head.doc_type == DOC_REPORT and head.doc_type != DOC_CATALOG
    assert head.name == "LED 冷光源数量" and head.value == ""
    assert head.filename == ""          # 由 lotmap / vendor_report 回填
    assert head.model == "TH-4520"
    assert head.response_text == f"{head.name}：详见检测报告"
    tail = pointers[-1]
    assert tail.model == "TH-8394S" and tail.locator == "检测报告 第1页 表行1"


@pytest.mark.ec("D2")
@pytest.mark.ec("D4")
@pytest.mark.case("X30")
def test_the_catalog_locator_carries_the_product_page_not_a_hardcoded_one(pack_path):
    """真实 P3：服务产品排在第 2 页，它的证据就得写 ``第2页``。

    P3 标的2 序号 2.2「每年提供≥4 次的日常巡检维护」的 gold 位置是
    ``产品综合样本 第2页``，值 ``4.22 次/年``；页码写死成 1 会让整包的服务标的
    位置命中率归零，而这在只看 P1 的时候发现不了（P1 的服务产品也确实在第 2 页，
    但 P1 的服务证据来自 docx，不走样本）。
    """
    cat = catalog_of(pack_path, "P3")
    ev = vc.to_evidence(cat)
    assert [e.locator for e in ev if e.locator == "产品综合样本 第1页"] == [
        "产品综合样本 第1页"] * 13
    tail = [e for e in ev if e.locator == "产品综合样本 第2页"]
    assert [e.name for e in tail] == ["每年维护保养次数", "常巡检维护"]
    assert dict((e.name, e.value) for e in tail)["每年维护保养次数"] == "4.22 次/年"
    assert all(e.model == "TH-8394S" for e in tail)


@pytest.mark.ec("D2")
@pytest.mark.ec("J8")
@pytest.mark.ec("J1")
@pytest.mark.case("N6")
def test_a_blank_scanned_page_is_skipped_without_shifting_later_page_numbers(tmp_path):
    """中间夹一张扫描页：跳过、告警，但**第 3 页仍然叫第 3 页**。

    页码是 ``位置`` 列的一部分，一旦按「已解析产品数」而不是 PDF 物理页号来编号，
    后面每个产品的坐标都会整体前移一格，且这种错在单页样本上永远看不出来。
    """
    path = mk_pdf(tmp_path, "样本.pdf", [
        ["核心亮点", "■ 甲：1", "第三方实测数据摘要（详见检测报告）",
         "● 乙：详见检测报告"] + ORDER_LINES,
        [],
        ["核心亮点", "■ 丙：3"] + ORDER_LINES,
    ])
    cat = vc.parse_catalog(path)
    assert cat.pages == 3
    assert [p.page for p in cat.products] == [1, 3]
    assert cat.warnings == ["样本.pdf 第2页 get_text() 为空（可能是扫描页），已跳过。"]
    assert [e.locator for e in vc.to_evidence(cat)] == [
        "产品综合样本 第1页", "检测报告 第1页 表行1", "产品综合样本 第3页"]


@pytest.mark.ec("D2")
@pytest.mark.ec("C7")
def test_a_folded_bullet_joins_without_a_space_so_the_line_break_is_invisible(pack_path):
    """真实 P1 第 1 页：9 项配置清单被 PDF 折成 3 行，拼回来必须是一个完整的值。

    源文本层第 8 行确实以空格结尾（``…（3）可复用光纤4 ``），但每行先 ``strip()``
    再用 ``""`` 拼接，所以那个空格不会留下来。这不是缺陷：``normalize.squash``
    与 ``grade.norm`` 都吃掉空白，带空格与不带空格在匹配和打分上完全等价。
    """
    cat = catalog_of(pack_path, "P1")
    name, value = cat.products[0].bullets[0]
    assert name == "钬激光治疗系统配置要求"
    assert len(value) == 118 and value.count("（") == 9
    assert value.startswith("（1）Ho:YAG 激光治疗机主机1 台")
    assert value.endswith("（9）电子内窥镜图像处理系统1 套")
    assert "可复用光纤4根（4）" in value and "可复用光纤4 根" not in value
    assert squash("（3）可复用光纤4 根（4）") == squash("（3）可复用光纤4根（4）")


@pytest.mark.ec("D2")
@pytest.mark.ec("J8")
def test_a_structural_line_ends_a_fold_and_the_order_line_ends_the_page(tmp_path):
    """两遍扫描的两个退出条件。

    结构行（``核心亮点`` 等）会把 ``in_body`` 复位，所以它后面那句营销散文不会被
    当成上一个要点的折行吸进去；``订购信息`` 直接 ``break``，页尾再出现 ``■`` 也不收。
    两者都失效的话，脏文本会静默污染 ``投标响应`` 列。
    """
    path = mk_pdf(tmp_path, "样本.pdf", [[
        "核心亮点", "■ 甲：长值开头", "核心亮点", "不该被吸收的散文", "■ 乙：2",
        "订购信息", "订货号：S1    产品名称：合成设备    型号：SYN-1", "■ 丙：3",
    ]])
    cat = vc.parse_catalog(path)
    assert cat.warnings == []
    assert cat.products[0].bullets == [("甲", "长值开头"), ("乙", "2")]


@pytest.mark.ec("D2")
@pytest.mark.ec("J8")
@pytest.mark.ec("J1")
def test_three_malformed_pages_each_degrade_with_a_named_warning(tmp_path):
    """缺型号 / 整页无要点 / 要点缺「名：值」，三种都只降级不中断。

    告警必须能让人一眼看出是哪一页哪一句出的问题，否则复核报告等于没写；
    而缺「订购信息」只让型号留空，页面其余内容照样解析（型号绑定会退回文件名推断）。
    """
    no_order = vc.parse_catalog(mk_pdf(tmp_path, "a.pdf", [["核心亮点", "■ 甲：1"]]))
    assert no_order.products[0].model == "" and no_order.products[0].bullets == [("甲", "1")]
    assert no_order.warnings == ["产品综合样本第1页未找到「订购信息」行，型号留空。"]

    empty = vc.parse_catalog(mk_pdf(
        tmp_path, "b.pdf", [["以临床需求为导向，广泛适用于各级医疗机构。"] + ORDER_LINES]))
    assert empty.products[0].bullets == []
    assert empty.warnings == ["产品综合样本第1页没有解析到任何核心亮点或实测指针。"]

    no_colon = vc.parse_catalog(mk_pdf(
        tmp_path, "c.pdf", [["核心亮点", "■ 缺值要点", "■ 甲：1"] + ORDER_LINES]))
    assert no_colon.warnings == [
        "产品综合样本第1页要点「缺值要点…」缺少「名：值」结构，已跳过。"]
    assert no_colon.products[0].bullets == [("甲", "1")]


@pytest.mark.ec("D2")
@pytest.mark.ec("J1")
@pytest.mark.ec("J7")
def test_an_unopenable_catalog_warns_and_an_empty_model_never_binds(tmp_path, pack_path):
    """样本读不出来时返回空 Catalog（4.5 第 2 条：单个资料损坏不拖垮整包）。

    型号→标的的绑定靠 ``product_for_model``，它的 ``if p.model and …`` 守卫决定了
    「解析不到型号」的产品不会被同样为空的查询串误命中 —— 否则所有缺型号的样本页
    会一起绑到第一个空型号上，把证据挂错标的。
    """
    bad = tmp_path / "坏样本.pdf"
    bad.write_bytes(b"this is not a pdf")
    cat = vc.parse_catalog(bad)
    assert cat.pages == 0 and cat.products == [] and vc.to_evidence(cat) == []
    assert len(cat.warnings) == 1
    assert cat.warnings[0].startswith("坏样本.pdf 无法打开：")

    real = catalog_of(pack_path, "P3")
    assert real.product_for_model("TH-8394S").page == 2
    assert real.product_for_model("NOPE") is None

    blank = vc.parse_catalog(mk_pdf(tmp_path, "无型号.pdf", [["核心亮点", "■ 甲：1"]]))
    assert blank.products[0].model == ""
    assert blank.product_for_model("") is None
