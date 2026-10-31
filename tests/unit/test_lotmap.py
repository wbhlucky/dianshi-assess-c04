"""``ingest.lotmap`` 单元测试：型号 <-> 标的绑定的 9 项边界能力。

1. 真实 P2：两个型号一一绑到两个标的，且与冻结 fixture 完全一致（T10）。
2. 真实 P5：4 个型号绑到**非连续**的 4 个标的且零告警——多出来的型号不是干扰项（X2）。
3. 近似字设备名解析不到任何标的，所以结构化字段是唯一可信来源（X5）。
4. 真实 P1：服务方案里的型号绑到服务标的，绝不串到设备标的（X6）。
5. 两源冲突：先出现的样本/规格书抬头赢，输的那一侧退化成「标的没绑到型号」告警。
6. ``attach_lots`` 的指针回填矩阵：OCR 成功 / 序号非数字 / OCR 失败 / 没有报告，且不改输入。
7. 文件名兜底只补标的不改值，以及它真实存在的前缀碰撞。
8. 绑不上时出**有名有姓**的告警，而不是静默丢证据。
9. ``doc_type_of`` 的取值域，以及它与 ``vendor_docx`` 在混合文件名上的分歧。

1、2、3、4 吃真实资料包；其余用测试时现造的数据类实例（不写二进制）。
"""

from __future__ import annotations

from pathlib import Path

import docx
import pytest

from bidtable.config import DOC_PRIORITY, DOC_REPORT, DOC_SERVICE, DOC_SPEC
from bidtable.ingest import lotmap as lm
from bidtable.ingest import pack as pk
from bidtable.ingest import vendor_catalog as vc
from bidtable.ingest import vendor_docx as vd
from bidtable.ingest import vendor_report as vr
from bidtable.models import Evidence, Scope

pytestmark = [pytest.mark.unit]

UNBOUND_TAIL = "没有匹配到任何厂商型号，该标的的证据将只能来自无法定位型号的资料，相关行可能判为未响应。"


def pack_scopes(pack_path, name: str) -> list[Scope]:
    scopes, _ = pk.scopes_from_pack(pk.discover(pack_path(name)))
    return scopes


def build_pack(pack_path, name: str) -> tuple[list[Scope], lm.LotMap]:
    """按 ``pipeline._stage_vendor`` 的离线口径重建一份真实资料包的绑定结果。"""
    p = pk.discover(pack_path(name))
    scopes, _ = pk.scopes_from_pack(p)
    docs = [vd.parse_docx(f) for f in (*p.spec_docx, *p.service_docx)]
    cat = vc.parse_catalog(p.catalog_pdf[0]) if p.catalog_pdf else None
    reps = [vr.ocr_report(f, None) for f in p.report_images]
    return scopes, lm.build_lotmap(scopes, docs, cat, reps)


def D(model: str, name: str, dt: str = DOC_SPEC) -> vd.VendorDoc:
    return vd.VendorDoc(path=Path(f"X_{model}_技术规格书.docx"), doc_type=dt,
                        product_name=name, model=model, maker="某公司", rows=[], warnings=[])


def C(*prods) -> vc.Catalog:
    return vc.Catalog(path=Path("产品综合样本.pdf"), pages=len(prods),
                      products=list(prods), warnings=[])


def P(page: int, name: str, model: str) -> vc.CatalogProduct:
    return vc.CatalogProduct(page=page, product_name=name, model=model)


def R(model: str, name: str, rows, ok: bool = True) -> vr.Report:
    return vr.Report(path=Path(f"X_{model}_检测报告.jpg"), model=model, sample_name=name,
                     report_no="N-1", rows=[vr.ReportRow(*r) for r in rows], ocr_ok=ok)


def PTR(model: str, idx: int, name: str) -> Evidence:
    return Evidence(lot_no="", model=model, doc_type=DOC_REPORT, filename="",
                    locator=f"检测报告 第1页 表行{idx}", name=name, value="",
                    response_text=f"{name}：详见检测报告", priority=DOC_PRIORITY[DOC_REPORT],
                    is_pointer=True, confidence=0.5)


@pytest.mark.ec("B2")
@pytest.mark.ec("D3")
@pytest.mark.ec("D4")
@pytest.mark.case("T10")
def test_real_p2_binds_one_model_per_lot_and_matches_the_frozen_fixture(pack_path, frozen):
    """P2 的两份报告分属两个标的：``BT-6869S``->标的1、``BT-6259X``->标的2。

    三个来源（样本订购信息、docx 抬头、报告文件名）在真实资料里互相印证，
    所以告警必须是空的——任何一条告警都说明某个来源被读歪了。
    顺带把结果与 ``--freeze`` 出来的 fixture 对齐，防止摄取层静默漂移。
    """
    scopes, M = build_pack(pack_path, "P2")
    assert [sc.lot_no for sc in scopes] == ["1", "2"]
    assert M.model_to_lot == {"BT-6869S": "1", "BT-6259X": "2"}
    assert M.lot_to_model == {"1": "BT-6869S", "2": "BT-6259X"}
    assert sorted(M.report_by_model) == ["BT-6259X", "BT-6869S"]
    assert M.sources == {
        "BT-6869S": ["产品综合样本·订购信息", "技术规格书·抬头"],
        "BT-6259X": ["产品综合样本·订购信息", "服务方案·抬头"],
    }
    assert M.warnings == []
    assert frozen("P2", "vendor")["lotmap"] == {
        "model_to_lot": M.model_to_lot,
        "lot_to_model": M.lot_to_model,
        "sources": M.sources,
    }


@pytest.mark.ec("B3")
@pytest.mark.ec("D1")
@pytest.mark.case("X2")
def test_real_p5_binds_a_non_contiguous_lot_subset_with_zero_warnings(pack_path, frozen):
    """P5 声明 4 个标的（1/5/8/12，编号非连续），厂商恰好给了 4 个型号。

    「多出来的型号」不是需要滤掉的噪声，它们正是其余标的的产品；
    任何按主投标型号过滤证据的写法都会一次丢掉 3 个标的。
    零告警 + 4/4 全绑上，是这条边界的唯一合格答案。
    """
    scopes, M = build_pack(pack_path, "P5")
    assert [sc.lot_no for sc in scopes] == ["1", "5", "8", "12"]
    assert M.model_to_lot == {
        "QM-2776": "1", "QM-1220X": "5", "QM-5448S": "8", "QM-9932II": "12"}
    assert M.lot_to_model == {
        "1": "QM-2776", "5": "QM-1220X", "8": "QM-5448S", "12": "QM-9932II"}
    assert M.warnings == []
    assert all(v == ["产品综合样本·订购信息", "技术规格书·抬头"] for v in M.sources.values())
    assert sorted(M.report_by_model) == sorted(M.model_to_lot)
    assert frozen("P5", "vendor")["lotmap"]["model_to_lot"] == M.model_to_lot


@pytest.mark.ec("B1")
@pytest.mark.ec("D1")
@pytest.mark.ec("J5")
@pytest.mark.case("X5")
def test_a_near_miss_device_name_resolves_to_nothing_so_the_structured_field_is_load_bearing(
    pack_path,
):
    """近似字设备名（``红外辅照治疗仪`` vs ``红外辐照治疗装置``）解析不到任何标的。

    P5 招标文件 PDF 的块标题带错字，``00_投标范围.txt`` 的结构化字段才是对的。
    ``normalize_name`` 只去空白和标点，**不做**形近字映射——这是有意的：
    宁可绑不上并出告警，也不能把两个真实不同的设备名合并。
    三级解析次序（精确 -> 互相包含 -> 唯一服务标的兜底）也在这里钉住。
    """
    scopes = pack_scopes(pack_path, "P5")
    assert lm._best_scope("红外辐照治疗装置", scopes).lot_no == "5"
    assert lm._best_scope("红外辅照治疗仪", scopes) is None
    assert lm.normalize_name("红外辅照治疗仪") == "红外辅照治疗仪"
    assert lm.normalize_name("热蒸汽治疗设备（进口）") == "热蒸汽治疗设备进口"
    assert lm.normalize_name(None) == ""

    pair = [Scope.make("1", "红外辐照治疗装置", "1台"),
            Scope.make("2", "设备运维技术服务", "5年")]
    assert lm._best_scope("辐照治疗", pair).lot_no == "1"          # 包含
    assert lm._best_scope("运维服务", pair).lot_no == "2"          # 唯一服务标的兜底
    assert lm._best_scope("运维服务", pair[:1]) is None            # 没有服务标的就不兜底
    twins = [Scope.make("1", "设备运维技术服务"), Scope.make("2", "设备运维技术服务")]
    assert lm._best_scope("运维服务", twins) is None               # 两个服务标的：拒绝猜
    assert lm._best_scope("", pair) is None
    assert lm._best_scope("完全不相干的东西", pair) is None


@pytest.mark.ec("B2")
@pytest.mark.ec("D4")
@pytest.mark.case("X6")
def test_the_service_plan_model_binds_to_the_service_lot_not_the_device_lot(pack_path, frozen):
    """服务方案的型号必须落在服务标的上，否则标的2 的证据会被标的过滤全数丢掉。

    P1 的 ``YF_服务方案.docx`` 抬头写的是 ``YF-1509Pro``，与设备标的的
    ``YF-9834Pro`` 只差几位数字；样本第 2 页的订购信息独立给出了同一对
    (产品名, 型号)，两源互证才让这条绑定可信。
    """
    scopes, M = build_pack(pack_path, "P1")
    assert [sc.lot_no for sc in scopes] == ["1", "2"]
    assert M.model_to_lot == {"YF-9834Pro": "1", "YF-1509Pro": "2"}
    assert M.lot_to_model["2"] == "YF-1509Pro"
    assert M.sources["YF-1509Pro"] == ["产品综合样本·订购信息", "服务方案·抬头"]
    assert M.sources["YF-9834Pro"] == ["产品综合样本·订购信息", "技术规格书·抬头"]
    assert sorted(M.report_by_model) == ["YF-9834Pro"]   # P1 只出了一份报告，属标的1
    assert M.warnings == []
    assert frozen("P1", "vendor")["lotmap"]["sources"] == M.sources


@pytest.mark.ec("J8")
@pytest.mark.ec("D1")
def test_a_two_source_conflict_keeps_the_first_binding_and_names_the_loser():
    """同一个型号被两个来源指向不同标的：先绑的赢，后来的只出告警不改写。

    告警里带型号、两个标的号和**触发冲突的那个来源**，人工复核才有的查。
    输掉的一侧随后会变成「标的没绑到型号」告警——两条告警是因果关系，不是一条。
    把两个来源的产品名对调，赢家与告警方向必须一起翻过来。
    """
    scopes = [Scope.make("1", "设备甲", "1台"), Scope.make("2", "设备运维技术服务", "5年")]

    M = lm.build_lotmap(scopes, [D("M-1", "设备运维技术服务")], C(P(1, "设备甲", "M-1")), [])
    assert M.model_to_lot == {"M-1": "1"}
    assert M.lot_to_model == {"1": "M-1"}
    assert M.sources == {"M-1": ["产品综合样本·订购信息", "技术规格书·抬头"]}
    assert M.warnings == [
        "型号 M-1 的标的绑定冲突：1 vs 2（来源 技术规格书·抬头）；以先出现的样本/规格书抬头为准。",
        f"标的 2（设备运维技术服务）{UNBOUND_TAIL}",
    ]

    swapped = lm.build_lotmap(
        scopes, [D("M-1", "设备甲")], C(P(1, "设备运维技术服务", "M-1")), [])
    assert swapped.model_to_lot == {"M-1": "2"}
    assert "冲突：2 vs 1（来源 技术规格书·抬头）" in swapped.warnings[0]
    assert swapped.warnings[1] == f"标的 1（设备甲）{UNBOUND_TAIL}"


@pytest.mark.ec("D3")
@pytest.mark.ec("D8")
@pytest.mark.ec("J1")
def test_attach_lots_backfills_pointer_evidence_across_four_report_states():
    """指针证据（``详见检测报告``）的四种报告状态，回填结果各不相同。

    OCR 成功且 ``表行N`` 命中 -> 实测值 + ``项目：值`` + 置信 1.0；
    序号被 OCR 认成汉字 -> 退回按位置取第 N 行；越界 -> 只补文件名，值留给推定；
    OCR 失败 / 根本没有报告 -> 同样只补文件名或什么都不补。
    ``Evidence`` 是 frozen 的，所以这里必须是 ``replace`` 出新对象，输入列表一个字段都不能变。
    """
    rows = [("1", "屈光补偿范围", "-25.5~21D"), ("2", "最小瞳孔直径", "3.25mm")]
    M = lm.LotMap(model_to_lot={"M-1": "1"}, lot_to_model={"1": "M-1"},
                  report_by_model={"M-1": R("M-1", "合成设备", rows)})
    src = [PTR("M-1", 1, "屈光补偿范围"), PTR("M-1", 2, "最小瞳孔直径"),
           PTR("M-1", 9, "越界指标")]
    out = lm.attach_lots(src, M)
    assert [(e.lot_no, e.filename, e.value, e.response_text, e.confidence) for e in out] == [
        ("1", "X_M-1_检测报告.jpg", "-25.5~21D", "屈光补偿范围：-25.5~21D", 1.0),
        ("1", "X_M-1_检测报告.jpg", "3.25mm", "最小瞳孔直径：3.25mm", 1.0),
        ("1", "X_M-1_检测报告.jpg", "", "越界指标：详见检测报告", 0.5),
    ]
    assert all(e.doc_type == DOC_REPORT and e.is_pointer for e in out)
    assert all(e.priority == DOC_PRIORITY[DOC_REPORT] for e in out)   # 回填不动优先级
    assert out[0] is not src[0]
    assert [(e.lot_no, e.filename, e.value) for e in src] == [("", "", "")] * 3

    odd = lm.LotMap(model_to_lot={"M-1": "1"}, report_by_model={
        "M-1": R("M-1", "合成设备", [("甲", "项目一", "1"), ("乙", "项目二", "2")])})
    got = lm.attach_lots([PTR("M-1", 2, "指标")], odd)[0]
    assert (got.value, got.response_text, got.confidence) == ("2", "项目二：2", 1.0)

    assert lm._pointer_index("检测报告 第1页 表行3") == 3
    assert lm._pointer_index("产品综合样本 第1页") == 0
    loose = Evidence(lot_no="1", model="M-1", doc_type=DOC_REPORT, filename="",
                     locator="检测报告 第1页", name="指标", value="3",
                     response_text="指标：详见检测报告", priority=DOC_PRIORITY[DOC_REPORT],
                     is_pointer=True, confidence=0.5)
    got = lm.attach_lots([loose], M)[0]   # 定位串里没有表行N：索引 0，值与置信度原样
    assert (got.value, got.confidence) == ("3", 0.5)
    assert got.filename == "X_M-1_检测报告.jpg"

    dark = lm.LotMap(model_to_lot={"M-1": "1"},
                     report_by_model={"M-1": R("M-1", "合成设备", rows, ok=False)})
    got = lm.attach_lots([PTR("M-1", 1, "屈光补偿范围")], dark)[0]
    assert (got.filename, got.value, got.confidence) == ("X_M-1_检测报告.jpg", "", 0.5)

    bare = lm.LotMap(model_to_lot={"M-1": "1"})
    got = lm.attach_lots([PTR("M-1", 1, "屈光补偿范围")], bare)[0]
    assert (got.filename, got.value, got.confidence) == ("", "", 0.5)


@pytest.mark.ec("D1")
@pytest.mark.ec("J3")
@pytest.mark.ec("J5")
def test_the_filename_fallback_fills_only_the_lot_and_really_can_collide_on_a_prefix(mk_ev):
    """型号绑不上标的时，用文件名里的厂商前缀再试一次；只补 ``lot_no``，不动值和置信度。

    前缀碰撞是这条兜底的真实缺陷：它按 dict 插入序返回**第一个**出现在文件名里的型号，
    所以 ``{"QM-1":…, "QM-12":…}`` 会把 ``X_QM-12_…`` 判给 QM-1 的标的，
    把字典顺序反过来结论就翻。训练集里没有互为前缀的型号，但缺陷必须写在明面上。
    """
    M = lm.LotMap(model_to_lot={"M-9": "2"})
    ev = mk_ev("指标", "3", lot_no="", model="ZZ-未知", filename="XX_M-9_技术规格书.docx")
    got = lm.attach_lots([ev], M)[0]
    assert got.lot_no == "2"
    assert (got.value, got.response_text, got.confidence) == ("3", "指标：3", 1.0)
    assert got.filename == ev.filename

    assert lm._lot_from_filename("XX_NOPE_技术规格书.docx", M) == ""
    assert lm._lot_from_filename("", M) == ""

    fn = "X_QM-12_技术规格书.docx"
    first = lm.LotMap(model_to_lot={"QM-1": "1", "QM-12": "2"})
    second = lm.LotMap(model_to_lot={"QM-12": "2", "QM-1": "1"})
    assert lm._lot_from_filename(fn, first) == "1"
    assert lm._lot_from_filename(fn, second) == "2"


@pytest.mark.ec("B1")
@pytest.mark.ec("J8")
@pytest.mark.ec("J3")
def test_unbindable_inputs_produce_named_warnings_instead_of_silent_loss():
    """三种绑不上的形态都要出告警：文件名没型号 token、有型号但没人认领、型号为空。

    证据绑不上标的不会崩，但会在匹配阶段被标的过滤挡掉，最终表现成整片「未响应」。
    所以告警必须点名到文件名和标的号，否则复核的人只能看到一堆未响应却查不出原因。
    空型号连 ``sources`` 都不该留痕——它是解析失败的产物，不是一个真实型号。
    """
    scopes = [Scope.make("1", "设备甲", "1台"), Scope.make("2", "设备运维技术服务", "5年")]

    M = lm.build_lotmap(scopes, [], None, [vr.Report(path=Path("检测报告.jpg"))])
    assert M.model_to_lot == {} and M.report_by_model == {}
    assert M.warnings == [
        "检测报告.jpg：文件名里没有型号 token，无法绑定到标的。",
        f"标的 1（设备甲）{UNBOUND_TAIL}",
        f"标的 2（设备运维技术服务）{UNBOUND_TAIL}",
    ]

    M = lm.build_lotmap(scopes, [], None, [R("M-1", "", [], ok=False)])
    assert M.model_to_lot == {} and sorted(M.report_by_model) == ["M-1"]
    assert M.warnings == [f"标的 1（设备甲）{UNBOUND_TAIL}",
                          f"标的 2（设备运维技术服务）{UNBOUND_TAIL}"]

    M = lm.build_lotmap(scopes, [D("", "设备甲")], C(P(1, "设备甲", "")), [])
    assert M.model_to_lot == {} and M.sources == {}
    assert M.warnings == [f"标的 1（设备甲）{UNBOUND_TAIL}",
                          f"标的 2（设备运维技术服务）{UNBOUND_TAIL}"]


@pytest.mark.ec("D1")
@pytest.mark.ec("J3")
def test_doc_type_of_reads_the_filename_and_disagrees_with_vendor_docx_on_a_hybrid_name(
    tmp_path,
):
    """``doc_type_of`` 的取值域，以及它与 ``vendor_docx`` 在混合文件名上的**相反**结论。

    ``lotmap`` 先看「服务方案」，``vendor_docx.TITLE_SPECS`` 是有序表且「技术规格书」在前，
    于是 ``服务方案技术规格书.docx`` 一个判服务方案、一个判技术规格书。
    真实资料里不存在这种文件名，但两个模块的分歧必须被钉住，否则以后谁改一边都会静默错位。
    """
    assert lm.doc_type_of("A_服务方案.docx") == DOC_SERVICE
    assert lm.doc_type_of("A_技术规格书.docx") == DOC_SPEC
    assert lm.doc_type_of("A_规格书.docx") == DOC_SPEC
    assert lm.doc_type_of("A_检测报告.jpg") == DOC_REPORT
    assert lm.doc_type_of("产品综合样本.pdf") == ""

    hybrid = "服务方案技术规格书.docx"
    assert lm.doc_type_of(hybrid) == DOC_SERVICE
    d = docx.Document()
    d.add_paragraph("合成设备 技术规格书")
    d.save(tmp_path / hybrid)
    assert vd.parse_docx(tmp_path / hybrid).doc_type == DOC_SPEC
