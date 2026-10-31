"""pack.py：资料包发现、厂商文件归类、``00_投标范围.txt`` 解析。

六条用例各盯一个能力边界：

1. ``discover`` 的前缀路由，以及 ``04_标准答案`` **绝不**进入任何清单（O10）。
2. 目录不存在 / 不是目录时，错误消息必须带上它**实际尝试过**的绝对路径（N1）。
3. 完整性检查区分「唯一致命项」与「两项告警」（N2）。
4. 厂商文件归类只认文件名与扩展名，不支持的格式落 unknown（O6）。
5. 投标范围声明决定哪些标的参投、哪个是服务标的（T1）。
6. 缺 ``00_投标范围.txt`` 时降级为空清单 + 告警，而不是崩（N3）。
"""

from __future__ import annotations

import pytest

from bidtable import errors
from bidtable.config import SERVICE_LOT_NAME
from bidtable.ingest import pack as pk

pytestmark = [pytest.mark.unit]

VENDOR = {
    "02_厂商资料/YF_YF-9834Pro_技术规格书.docx": "x",
    "02_厂商资料/BT_BT-6869S_服务方案.docx": "x",
    "02_厂商资料/产品综合样本.pdf": "x",
    "02_厂商资料/QM_QM-2776_检测报告.jpg": "x",
    "02_厂商资料/~$YF-9834Pro_技术规格书.docx": "x",  # Office 锁文件
    "02_厂商资料/读我.bin": "x",
}


@pytest.mark.ec("D1")
@pytest.mark.ec("D2")
@pytest.mark.ec("D3")
@pytest.mark.ec("D4")
@pytest.mark.ec("J5")
def test_discover_routes_files_by_prefix_and_never_reads_the_answer_key(make_pack):
    """四类厂商文件各走各的清单，``04_`` 前缀的标准答案在扫描时被直接跳过。

    这是 O10 的第一道防线：流水线连 ``04_标准答案_*.xlsx`` 的**路径**都不该持有，
    否则任何一次「顺手读一下」都构成作弊。``~$`` 锁文件与 ``.`` 开头的隐藏文件
    同样不进清单，免得 Windows 上打开过 Excel 的目录跑不出结果。
    """
    root = make_pack(
        "SYN",
        {
            "00_投标范围.txt": "标的 1：合成设备  数量 1台\n",
            "01_招标文件_合成.pdf": "x",
            "03_响应偏离表_模板.xlsx": "x",
            "04_标准答案_响应偏离表.xlsx": "x",
            ".hidden": "x",
            **VENDOR,
        },
    )
    p = pk.discover(root)
    assert p.name == "SYN"
    assert p.scope_file.name == "00_投标范围.txt"
    assert p.tender_pdf.name == "01_招标文件_合成.pdf"
    assert p.template_xlsx.name == "03_响应偏离表_模板.xlsx"
    assert p.vendor_dir.name == "02_厂商资料"
    assert [x.name for x in p.spec_docx] == ["YF_YF-9834Pro_技术规格书.docx"]
    assert [x.name for x in p.service_docx] == ["BT_BT-6869S_服务方案.docx"]
    assert [x.name for x in p.catalog_pdf] == ["产品综合样本.pdf"]
    assert [x.name for x in p.report_images] == ["QM_QM-2776_检测报告.jpg"]
    assert [x.name for x in p.unknown_files] == ["读我.bin"]
    # vendor_files() 的顺序就是证据优先级的物理来源，不能乱
    assert [x.name for x in p.vendor_files()] == [
        "YF_YF-9834Pro_技术规格书.docx",
        "BT_BT-6869S_服务方案.docx",
        "产品综合样本.pdf",
        "QM_QM-2776_检测报告.jpg",
    ]
    assert p.has_vendor_file("产品综合样本.pdf") is True
    assert p.has_vendor_file("不存在.docx") is False
    # O10：标准答案的路径不曾被任何清单持有
    every = [p.scope_file, p.tender_pdf, p.template_xlsx, *p.vendor_files(), *p.unknown_files]
    assert not any(x is not None and x.name.startswith("04_") for x in every)
    assert p.has_vendor_file("04_标准答案_响应偏离表.xlsx") is False


@pytest.mark.ec("J5")
@pytest.mark.ec("J7")
@pytest.mark.case("N1")
def test_a_missing_pack_directory_names_the_absolute_path_it_tried(tmp_path):
    """``--pack`` 指错时，消息里必须是**解析后的绝对路径**，且退出码 1。

    N1 的判据是「人读消息 + 正确退出码 + 不打 traceback」。相对路径在 CI 与
    本地会解析成不同的串，只有 ``resolve()`` 之后的路径能让人一眼看出打错了哪里。
    ``--pack`` 指到一个**文件**上时是另一种失败，消息要说清「需要是目录」。
    """
    gone = tmp_path / "不存在的包"
    with pytest.raises(errors.PackNotFoundError) as ei:
        pk.discover(gone)
    assert str(gone.resolve()) in str(ei.value)
    assert pk.SCOPE_FILE in str(ei.value)  # 顺带告诉用户正确的目录长什么样
    assert ei.value.exit_code == 1

    a_file = tmp_path / "我是文件.txt"
    a_file.write_text("x", encoding="utf-8")
    with pytest.raises(errors.PackNotFoundError) as ei2:
        pk.discover(a_file)
    assert "需要是目录" in str(ei2.value)
    assert str(a_file.resolve()) in str(ei2.value)


@pytest.mark.ec("J7")
@pytest.mark.case("N2")
def test_completeness_separates_the_one_fatal_gap_from_the_two_warnings(make_pack):
    """只有招标 PDF 缺失是致命的；缺范围文件与缺厂商资料只进告警清单。

    理由：没有招标 PDF 就抽不出任何要求，产物必然为空，继续跑只是浪费时间；
    而缺 ``00_投标范围.txt`` 还能从招标文件推断标的（N3），缺厂商资料则整表
    输出「未响应」——都是有意义的结果，不该退出。
    ``.missing`` 必须是**完整**清单，不能只留致命项，否则复核报告说不清缺了什么。
    """
    empty = pk.discover(make_pack("EMPTY", {}, vendor=False))
    assert pk.check_completeness(empty) == [
        "01_招标文件_*.pdf（招标 PDF）",
        pk.SCOPE_FILE,
        "02_厂商资料/ 下的任意厂商文件",
    ]
    with pytest.raises(errors.PackIncompleteError) as ei:
        pk.require_pack(empty)
    assert len(ei.value.missing) == 3
    assert "01_招标文件" in str(ei.value)
    assert ei.value.exit_code == 1

    # 只有招标 PDF：仍缺两项，但 require_pack 放行
    partial = pk.discover(make_pack("PART", {"01_招标文件.pdf": "x"}, vendor=False))
    assert len(pk.check_completeness(partial)) == 2
    pk.require_pack(partial)  # 不抛

    # 反过来：有范围文件与厂商资料、独缺招标 PDF -> 致命
    no_pdf = pk.discover(make_pack("NOPDF", {"00_投标范围.txt": "标的 1：X  数量 1台\n", **VENDOR}))
    assert pk.check_completeness(no_pdf) == ["01_招标文件_*.pdf（招标 PDF）"]
    with pytest.raises(errors.PackIncompleteError):
        pk.require_pack(no_pdf)

    # 齐全的资料包：零缺失
    full = pk.discover(
        make_pack("FULL", {"00_投标范围.txt": "标的 1：X\n", "01_招标文件.pdf": "x", **VENDOR})
    )
    assert pk.check_completeness(full) == []


@pytest.mark.ec("D1")
@pytest.mark.ec("D3")
def test_classify_vendor_file_uses_the_filename_and_refuses_formats_it_cannot_read(tmp_path):
    """归类只看文件名关键词 + 扩展名；不支持的格式落 unknown 而不是猜（O6）。

    分支有**顺序**：``服务方案`` 先于 ``技术规格书``，所以一个同时含两个词的文件
    归服务方案。``规格书`` / ``报告`` / ``catalogue`` 是别名分支。
    ``.doc``（老格式）、``.pdf.txt``（双扩展名）、没有关键词的 ``照片.jpg`` 都必须
    落 unknown —— 静默猜错类型会让证据优先级（报告 0 < 规格书 1 < 样本 2）整体错位。
    """
    def cls(name: str):
        p = pk.Pack(root=tmp_path, name="X")
        pk.classify_vendor_file(p, tmp_path / name)
        for attr in ("spec_docx", "service_docx", "catalog_pdf", "report_images", "unknown_files"):
            if getattr(p, attr):
                return attr
        return None

    assert cls("A_技术规格书.docx") == "spec_docx"
    assert cls("A_规格书.docx") == "spec_docx"  # 别名分支
    assert cls("A_服务方案.docx") == "service_docx"
    assert cls("A_服务方案技术规格书.docx") == "service_docx"  # 顺序：服务方案优先
    assert cls("产品综合样本.pdf") == "catalog_pdf"
    assert cls("Catalogue.pdf") == "catalog_pdf"  # 大小写不敏感
    assert cls("A_检测报告.jpg") == "report_images"
    assert cls("A_检测报告.jpeg") == "report_images"
    assert cls("A_检测报告.png") == "report_images"
    assert cls("A_报告.jpg") == "report_images"  # 别名分支
    # 不支持 / 不可判定的，一律 unknown
    assert cls("A_技术规格书.doc") == "unknown_files"  # O6：不做 docx 以外的格式
    assert cls("产品综合样本.pdf.txt") == "unknown_files"
    assert cls("照片.jpg") == "unknown_files"  # 缺关键词，不猜
    assert cls("读我.md") == "unknown_files"


@pytest.mark.ec("B1")
@pytest.mark.ec("B3")
@pytest.mark.case("T1")
def test_the_scope_file_declares_which_lots_are_in_and_how_many(pack_path):
    """P5 的真实范围文件：13 项设备里只投 4 项，标的号是**非连续子集** 1/5/8/12。

    非连续是关键：C 族版面按 ``N-科室-设备名`` 的块标题映射标的号，若把块序号
    当成标的号，标的5 会被认成标的2，整包对齐键全错（B3）。
    标的12 的数量在源文件里就是残缺的 ``1None``，必须**原样保留**：它进的是
    xlsx 的展示字段，任何「修正」都等于篡改招标口径（B4）。
    """
    doc = pk.parse_scope_file(pack_path("P5") / pk.SCOPE_FILE)
    assert [(s.lot_no, s.device_name, s.quantity) for s in doc.scopes] == [
        ("1", "床旁血液净化系统", "1台"),
        ("5", "红外辐照治疗装置", "1台"),
        ("8", "胰岛素泵", "10台"),
        ("12", "激光治疗仪", "1None"),
    ]
    assert all(s.is_service is False for s in doc.scopes)
    assert doc.purchaser == "新郑市公立人民医院"
    assert doc.bidder == "启明视讯科技有限公司"
    assert doc.project.startswith("新郑市公立人民医院设备采购技术参数")
    assert len(doc.notes) == 1 and doc.notes[0].startswith("说明：")


@pytest.mark.ec("B2")
@pytest.mark.ec("B4")
@pytest.mark.case("N3")
def test_a_missing_or_unparsable_scope_file_degrades_to_a_warning_not_a_crash(tmp_path):
    """``is_service`` 只认设备名**全等**；范围文件缺失 / 解析不出标的时降级 + 告警。

    服务标的的「售后服务」就是正文，裁掉它覆盖率直接塌（X15），所以这个判据
    必须严格：``设备运维技术服务（含培训）`` 不是服务标的。
    ``项目名称：`` 这一写法**不匹配**元数据正则（正则只认 ``项目：``），该行被
    静默忽略 —— 已登记为已知锋利边，因为它不影响标的清单。
    N3：缺范围文件时返回 ``([], [告警])``，让 C 族从招标文件自行推断。
    """
    doc = pk.parse_scope_text(
        "项目名称：会被忽略的一行\n"
        "项目：合成项目\n"
        "采购人：某医院\n"
        "投标人：某公司\n"
        "\n"
        "本次投标响应以下标的（其余标的不投，无需响应）：\n"
        "  标的 1：合成设备  数量 1台\n"
        f"  标的 2：{SERVICE_LOT_NAME}  数量 5年\n"
        f"  标的 3：{SERVICE_LOT_NAME}（含培训）  数量 1项\n"
        "  标的4:无空格无全角\n"
        "\n"
        "说明：★ 为实质性要求。\n"
    )
    assert doc.project == "合成项目"  # ``项目名称：`` 不匹配，被忽略
    assert [(s.lot_no, s.is_service) for s in doc.scopes] == [
        ("1", False), ("2", True), ("3", False), ("4", False)
    ]
    assert doc.scopes[1].quantity == "5年"
    assert doc.scopes[2].device_name == f"{SERVICE_LOT_NAME}（含培训）"
    assert doc.scopes[3].device_name == "无空格无全角"  # 半角冒号 + 无数量也能解析
    assert doc.scopes[3].quantity == ""
    assert len(doc.notes) == 1

    # N3：没有范围文件
    bare = pk.Pack(root=tmp_path, name="BARE")
    scopes, warns = pk.scopes_from_pack(bare)
    assert scopes == []
    assert len(warns) == 1 and pk.SCOPE_FILE in warns[0] and "推断" in warns[0]

    # 有文件但一个标的都解析不出来
    blank = tmp_path / pk.SCOPE_FILE
    blank.write_text("只有说明：没有标的声明。\n", encoding="utf-8")
    scopes2, warns2 = pk.scopes_from_pack(pk.Pack(root=tmp_path, name="B", scope_file=blank))
    assert scopes2 == [] and "未解析到任何标的声明" in warns2[0]

    # 正常路径：零告警
    good = tmp_path / "good" / pk.SCOPE_FILE
    good.parent.mkdir()
    good.write_text("标的 1：合成设备  数量 1台\n", encoding="utf-8")
    scopes3, warns3 = pk.scopes_from_pack(pk.Pack(root=good.parent, name="G", scope_file=good))
    assert [s.lot_no for s in scopes3] == ["1"] and warns3 == []
