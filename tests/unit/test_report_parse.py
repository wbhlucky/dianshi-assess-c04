"""``ingest.vendor_report`` 单元测试：检测报告摄取这一层的 7 项边界能力。

1. 真实报告版式的 OCR 纯文本 -> 编号/样品名/型号 + 恰好 3 行数据，备注只落在末行（T9）。
2. base64 长度闸门：未超限逐字节原样、真实超限图降采样一次即达标、
   极限上限走完 4 级阶梯、非图片文件降级为「降采样失败」并把超限的 b64 照样交回上层（B17）。
3. 表格窗口状态机：表头两个必要条件、``_END_HINTS`` 关窗、非数字行首跳过、
   备注列要两个以上空格才切得出来。
4. 结构化 JSON 兜底：夹在散文里能抠出来、根本不是 JSON 时报出原文异常、缺 rows 键静默返回 0 行。
5. ``ocr_report`` 的五条降级路径全都只出告警不抛异常（N8）。
6. ``model_from_filename`` 的 token 语法：字母前缀 1..6 个且必须带连字符。
7. ``Report`` / ``ReportRow`` 的字段契约（下游 ``lotmap.attach_lots`` 与 ``judge`` 都依赖它）。

2、6 直接吃真实资料包里的报告图片；1、3、4、5、7 用测试时现造的 JPEG 与文本
（agent.md 6.2：``fixtures/corrupt`` 不提交二进制）。
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import pytest
from PIL import Image

from bidtable.ingest import vendor_report as vr

pytestmark = [pytest.mark.unit]

# P2 报告的真实版式（字段名、列间距、签名段都照抄），OCR 出来就是这个形状。
TRANSCRIPT = """检测报告
报告编号：BT-2024-0613
委托单位：许昌市中心医院
样品名称：视网膜脉络膜造影仪    样品型号：BT-6869S
检测类别：委托检验
检测日期：2024-06-13
检测结果：
序号 检测项目 实测值 备注
1 屈光补偿范围 -24.5~20D
2 最小瞳孔直径 3.21mm
3 液晶显示器尺寸 15英寸   符合
以下空白
检测：张三   审核：李四   批准：王五
"""

_MODEL_RE = re.compile(r"[A-Za-z]{1,6}-[A-Za-z0-9._\-]+")
_PERCENT_RE = re.compile(r"(\d+%)")


def mk_jpeg(tmp_path: Path, name: str, size: tuple[int, int] = (40, 30)) -> Path:
    p = tmp_path / name
    Image.new("RGB", size, (200, 30, 30)).save(p, format="JPEG")
    return p


def raw_len(p: Path) -> int:
    return len(base64.b64encode(p.read_bytes()))


@pytest.fixture(scope="module")
def real_reports(repo_root) -> list[Path]:
    hits = sorted((repo_root / "训练集").glob("*/02_厂商资料/*_检测报告.jpg"))
    if not hits:
        pytest.skip("训练集里没有检测报告图片")
    return hits


@pytest.fixture(scope="module")
def oversized(real_reports) -> Path:
    for p in real_reports:
        if raw_len(p) > vr.MAX_B64_CHARS:
            return p
    pytest.skip("这批资料里没有超过 base64 上限的报告图片")


@pytest.mark.ec("D3")
@pytest.mark.ec("J1")
@pytest.mark.case("T9")
def test_a_real_layout_transcript_yields_three_rows_with_the_note_only_on_the_last():
    """真实报告版式一次解析到位：3 行数据、备注只在末行、三个抬头字段全拿到。

    这是 L1 窄口子的主干能力——OCR 只要把版面拍平成文本，剩下全靠确定性正则。
    ``备注`` 用 ``\\s{2,}`` 切开，所以列间距本身就是结构信息。
    """
    no, name, model, rows, warns = vr.parse_report_text(TRANSCRIPT)
    assert (no, name, model) == ("BT-2024-0613", "视网膜脉络膜造影仪", "BT-6869S")
    assert warns == []
    assert [(r.no, r.item, r.value, r.note) for r in rows] == [
        ("1", "屈光补偿范围", "-24.5~20D", ""),
        ("2", "最小瞳孔直径", "3.21mm", ""),
        ("3", "液晶显示器尺寸", "15英寸", "符合"),
    ]


@pytest.mark.ec("D3")
@pytest.mark.ec("J1")
@pytest.mark.ec("J8")
@pytest.mark.case("B17")
def test_the_base64_gate_is_byte_exact_under_the_cap_and_steps_down_over_it(
    tmp_path, real_reports, oversized
):
    """长度闸门的四段行为：原样、降一次就够、阶梯走完、降不动。

    真实资料里只有一张报告图（P3 的 TH-4520）超上限，其余 9 张都在 174k~189k 之间
    原样通过——所以「降采样」这条分支在线上几乎不触发，必须靠 ``max_chars`` 参数逼出来。
    阶梯走完仍超限时**不抛异常**：把超限的 b64 交回上层，由上层决定降级。
    """
    small = mk_jpeg(tmp_path, "BT_BT-9999Z_检测报告.jpg")
    b64, warns = vr.encode_image(small)
    assert warns == []
    assert b64 == base64.b64encode(small.read_bytes()).decode("ascii")

    normals = [p for p in real_reports if raw_len(p) <= vr.MAX_B64_CHARS]
    assert normals, "至少要有一张报告图在闸门之内"
    b64, warns = vr.encode_image(normals[0])
    assert warns == [] and len(b64) <= vr.MAX_B64_CHARS

    assert raw_len(oversized) > vr.MAX_B64_CHARS
    b64, warns = vr.encode_image(oversized)
    assert len(warns) == 1 and "已降采样到 80% 后重传" in warns[0]
    assert len(b64) <= vr.MAX_B64_CHARS

    b64, warns = vr.encode_image(oversized, max_chars=10)
    assert [_PERCENT_RE.search(w).group(1) for w in warns] == ["80%", "60%", "45%", "34%"]
    assert len(b64) > 10

    bogus = tmp_path / "notimage.jpg"
    bogus.write_bytes(b"x" * 600)
    b64, warns = vr.encode_image(bogus, max_chars=10)
    assert len(warns) == 1 and "降采样失败：cannot identify image file" in warns[0]
    assert b64 == base64.b64encode(bogus.read_bytes()).decode("ascii")


@pytest.mark.ec("D3")
@pytest.mark.ec("J8")
def test_the_table_window_needs_both_header_words_and_closes_on_the_first_end_hint():
    """表格窗口的开与关，以及备注列的空格门槛。

    开窗条件是 ``序号`` **且** (``检测项目`` 或 ``实测值``)：只写「序号 备注」的
    签名表、目录表都不能把散文吸成数据行。关窗靠 ``_END_HINTS``，
    所以「以下空白」后面再冒出来的合法数据行必须被丢掉——真实报告常在空白页后接附注。
    """
    assert vr.parse_report_text("序号 备注\n1 甲 2")[3] == []
    assert vr.parse_report_text("序号 实测值\n1 甲 2")[3] == [vr.ReportRow("1", "甲", "2", "")]

    rows = vr.parse_report_text("序号 检测项目 实测值\n1 甲 2\n以下空白\n2 乙 3")[3]
    assert [r.no for r in rows] == ["1"]

    rows = vr.parse_report_text("序号 检测项目 实测值\n甲 1 2\n2 乙 3")[3]
    assert [(r.no, r.item, r.value) for r in rows] == [("2", "乙", "3")]

    one = vr.parse_report_text("序号 检测项目 实测值\n1 甲 2 单备注")[3][0]
    two = vr.parse_report_text("序号 检测项目 实测值\n1 乙 3  双备注")[3][0]
    assert (one.value, one.note) == ("2 单备注", "")
    assert (two.value, two.note) == ("3", "双备注")


@pytest.mark.ec("D3")
@pytest.mark.ec("J1")
@pytest.mark.ec("J7")
def test_the_json_fallback_tolerates_prose_but_reports_a_named_failure_on_garbage():
    """结构化兜底解析的三种输入形态：夹在散文里、根本不是 JSON、有 JSON 但没 rows。

    第三种最阴：返回 0 行**且不告警**，所以 ``ocr_report`` 必须自己补一条
    「两种 OCR 解析都没拿到数据行」，否则一次静默失败就没有任何痕迹。
    """
    embedded = (
        '前言文字 {"report_no":"J-1","sample_name":"合成设备","sample_model":"SYN-1",'
        '"rows":[{"no":"1","item":"甲","value":"2","note":"n"}]} 后记'
    )
    assert vr.parse_report_json(embedded) == (
        "J-1", "合成设备", "SYN-1", [vr.ReportRow("1", "甲", "2", "n")], []
    )
    assert vr.parse_report_json("not json at all") == (
        "", "", "", [], ["检测报告结构化解析失败：substring not found"]
    )
    assert vr.parse_report_json('{"report_no":"J-2"}') == ("J-2", "", "", [], [])


@pytest.mark.ec("D3")
@pytest.mark.ec("J1")
@pytest.mark.ec("J7")
@pytest.mark.case("N8")
def test_ocr_report_degrades_in_five_distinct_ways_and_never_raises(tmp_path, fake_llm):
    """``ocr_report`` 的五条出口：无 client、读不到文件、空文本、成功、两次都失败。

    agent.md 4.5 要求「单个资料损坏不拖垮整包」，所以这个函数**任何**输入都不能抛。
    第 2 条是本次修的洞：``encode_image`` 里 ``read_bytes()`` 没兜，
    图片缺失时 ``FileNotFoundError`` 会一路窜出 ``process_pack``。
    第 4 条顺带钉住「OCR 认出的样品型号覆盖文件名推定的型号」。
    """
    img = mk_jpeg(tmp_path, "BT_BT-9999Z_检测报告.jpg")
    missing = tmp_path / "BT_BT-0000Z_检测报告.jpg"

    rep = vr.ocr_report(img, None)
    assert rep.ocr_ok is False and rep.model == "BT-9999Z"
    assert len(rep.warnings) == 1
    assert "模型不可用" in rep.warnings[0] and "ocr_failed" in rep.warnings[0]

    client = fake_llm([TRANSCRIPT])
    rep = vr.ocr_report(missing, client)
    assert rep.ocr_ok is False and client.transport.calls == 0
    assert len(rep.warnings) == 1
    assert rep.warnings[0].startswith(f"{missing.name} 无法读取：")
    assert rep.warnings[0].endswith("；降级为指针推定。")

    client = fake_llm([""])
    rep = vr.ocr_report(img, client)
    assert rep.ocr_ok is False and client.transport.calls == 1
    assert rep.warnings == [f"{img.name}：OCR 调用失败或返回空文本，降级为指针推定。"]

    client = fake_llm([TRANSCRIPT])
    rep = vr.ocr_report(img, client)
    assert rep.ocr_ok is True and client.transport.calls == 1 and rep.warnings == []
    assert rep.model == "BT-6869S"
    assert rep.report_no == "BT-2024-0613" and rep.sample_name == "视网膜脉络膜造影仪"
    assert len(rep.rows) == 3 and rep.raw_text == TRANSCRIPT

    client = fake_llm(["完全不是检测报告的散文", "still not json"])
    rep = vr.ocr_report(img, client)
    assert rep.ocr_ok is False and client.transport.calls == 2
    assert rep.warnings == [
        "检测报告 OCR 文本里没有解析出任何数据行（预期 3~4 行）。",
        "检测报告结构化解析失败：substring not found",
        f"{img.name}：两种 OCR 解析都没拿到数据行，降级为指针推定。",
    ]


@pytest.mark.ec("D3")
@pytest.mark.ec("J5")
def test_the_model_token_grammar_needs_a_short_letter_prefix_and_a_dash(real_reports):
    """型号 token 的语法边界：字母前缀 1..6 个，且必须带连字符。

    型号是标的绑定的唯一抓手（``lotmap``），所以宁可取不到（退回文件名/样本推断）
    也不能取错——``BT6259X`` 这种少个横线的写法就故意不认，避免把厂商简称当成型号。
    """
    for p in real_reports:
        assert _MODEL_RE.fullmatch(vr.model_from_filename(p)), p.name

    assert vr.model_from_filename(Path("YF_YF-9834Pro_技术规格书.docx")) == "YF-9834Pro"
    assert vr.model_from_filename(Path("a_QM-1220X.jpg")) == "QM-1220X"
    for name in ("检测报告.jpg", "产品综合样本.pdf", "BT_BT6259X_检测报告.jpg",
                 "XX_ABCDEFG-1_检测报告.jpg"):
        assert vr.model_from_filename(Path(name)) == "", name


@pytest.mark.ec("D3")
@pytest.mark.ec("J3")
def test_the_report_dataclasses_give_downstream_modules_a_stable_contract():
    """``Report`` 可变、``ReportRow`` 冻结、``no`` 是字符串、可变默认值不共享。

    ``lotmap.attach_lots`` 用 ``str(idx) == row.no`` 定位数据行，
    ``vendor_catalog`` 那侧的指针序号也是字符串，两边必须同型才对得上；
    ``ocr_report`` 就地往 ``Report`` 里填字段，所以它不能是 frozen。
    """
    row = vr.ReportRow("1", "甲", "2")
    assert row.note == "" and isinstance(row.no, str)
    with pytest.raises(AttributeError):
        row.no = "9"

    a, b = vr.Report(path=Path("x.jpg")), vr.Report(path=Path("y.jpg"))
    assert a.rows is not b.rows and a.warnings is not b.warnings
    assert (a.model, a.sample_name, a.report_no, a.rows, a.ocr_ok) == ("", "", "", [], False)
    a.ocr_ok = True
    assert a.ocr_ok is True

    assert vr._END_HINTS == ("以下空白", "检测：", "审核：", "批准：", "检测专用章")
