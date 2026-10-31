# -*- coding: utf-8 -*-
"""错误契约的**集成层**：异常形状、进程退出码、未知版面降级。

单元测试已经逐条覆盖了 ``errors.py`` 的类层次和各解析器的告警文案
（``tests/unit/test_errors.py``）。本文件只盯三件**跨进程/跨模块**才看得见的
能力，刻意不与单元测试重复：

1. ``tender_pdf.open_pdf`` 的**七种**打不开形态 —— 每种都必须变成
   ``TenderParseError``，且消息里同时有「哪个文件」「底层原因」「下一步怎么办」。
   其中一种是**反例**：改了扩展名的 JPEG 不会报错，因为 MuPDF 会把它自动包成
   一页 PDF。把这条钉住，是为了防止后人"顺手补一个图片检测"却误伤正常扫描件。
2. ``run.py`` 的**退出码阶梯** —— 同一个错误在库内是一个异常，在进程外必须是
   一个可读的 ``× 错误：`` 行 + 正确的退出码 + **绝不吐 traceback**。
   这是单元测试用 ``cli.main()`` 打不到的部分：它验证的是解释器启动、
   ``PYTHONPATH``、子进程编码这些真实交付面。
3. 未知版面（``layout == "?"``）的**降级路径**（N13）—— 不抛异常，改走
   「全文捞参数行」，并且把置信度压到 :data:`decide.UNKNOWN_LAYOUT_CAP` 以下。
   关键不是那个数字本身，而是它**必须低于** :data:`review.CONF_REVIEW`，
   这样每一条降级行都自动落进复核报告的「优先复核清单」。

坏 PDF 一律在测试现场用 ``fitz`` / ``PIL`` / 裸字节生成，仓库里不存二进制。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import fitz
import pytest
from PIL import Image

from bidtable.errors import TenderParseError
from bidtable.ingest import tender_pdf as tp

pytestmark = [pytest.mark.integration]

REPO = Path(__file__).resolve().parents[2]

# fitz 存不下 0 页的文档（``cannot save with zero pages``），只能手写裸字节。
_ZERO_PAGE_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
    b"xref\n0 3\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n111\n%%EOF\n"
)

_SCOPE_TEXT = (
    "项目名称：合成测试项目\n采购人：某医院\n投标人：某公司\n"
    "标的 1：合成设备  数量 1台\n"
)


def _one_page_pdf(path: Path) -> bytes:
    """造一份合法的 1 页 PDF，返回它的原始字节（供截断用例切片）。"""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 80), "合成参数页", fontname="china-s", fontsize=10)
    doc.save(path)
    doc.close()
    return path.read_bytes()


# 截断点定在前 128 字节：实测这个长度对**任何**正文都必然致命。
# 「切一半」不行——MuPDF 会把 CJK 正文的半截文件修复成一页**空白**文档，
# 却对 ASCII 正文的半截文件直接报错；同一种损坏给出两种结果，钉不住契约。
_TRUNCATE_AT = 128


def _bad_pdfs(root: Path) -> tuple[dict[str, Path], bytes]:
    """六种「打不开」+ 两种「其实打得开」的样本；第二个返回值是好 PDF 的字节。"""
    raw = _one_page_pdf(root / "_good.pdf")
    out: dict[str, Path] = {}

    out["truncated"] = root / "truncated.pdf"
    out["truncated"].write_bytes(raw[:_TRUNCATE_AT])

    out["empty"] = root / "empty.pdf"
    out["empty"].write_bytes(b"")

    out["not_pdf_text"] = root / "not_pdf_text.pdf"
    out["not_pdf_text"].write_text("这根本不是 PDF，只是一段纯文本", encoding="utf-8")

    out["zero_pages"] = root / "zero_pages.pdf"
    out["zero_pages"].write_bytes(_ZERO_PAGE_PDF)

    doc = fitz.open()
    doc.new_page().insert_text((50, 80), "加密", fontname="china-s", fontsize=10)
    doc.save(
        root / "encrypted.pdf",
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="hunter2",
        owner_pw="hunter2",
    )
    doc.close()
    out["encrypted"] = root / "encrypted.pdf"

    # PIL 显式指定 format，所以扩展名叫 .pdf 也照样写成 JPEG。
    Image.new("RGB", (40, 30), (200, 30, 30)).save(root / "not_pdf_jpeg.pdf", format="JPEG")
    out["not_pdf_jpeg"] = root / "not_pdf_jpeg.pdf"

    out["missing"] = root / "does_not_exist.pdf"
    return out, raw


@pytest.mark.ec("J7")
@pytest.mark.case("N5")
def test_open_pdf_turns_every_unreadable_tender_into_one_actionable_message(tmp_path):
    """六种坏 PDF 都必须变成 ``TenderParseError``，且消息可执行；两种不该报错。

    边界在哪：``open_pdf`` 只有四个分支（不存在 / fitz 抛错 / 需要密码 /
    0 页），但要覆盖七种现实形态。这里逐个钉住**消息行数与首行文案**——
    首行是给用户看的分类，中间行是底层原因，末行是下一步动作。
    行数不同正说明四个分支各自给了不同粒度的解释，而不是一句"打不开"了事。

    另一半边界是**它故意不管的事**：``open_pdf`` 不是损坏检测器。
    只丢了 ``startxref`` 指针的 PDF 会被 MuPDF 完整修复（正文一字不少），
    切掉一半的 CJK 正文 PDF 会修复成一页**空白**文档并照常返回。
    两者都不报错，因为判据只能是"打不开"，不能是"我觉得它可疑"。
    钉住这一点是为了防止后人加一道激进校验，把结构不规整但内容完好的
    扫描件（招标文件里非常常见）误杀在门口。

    真实资料包实例：训练集里没有坏 PDF，所以样本在现场生成；
    ``not_pdf_jpeg`` 是唯一**不该报错**的一条，它记录了 MuPDF 的实际行为
    （自动把图片包成 1 页 PDF），也解释了为什么 ``open_pdf`` 那句
    「不是改了扩展名的图片」的提示对图片永远不可达。

    与 agent.md 的偏差：N5 原文假设改扩展名的图片会被识别并报错，实测不会。
    """
    cases, good_raw = _bad_pdfs(tmp_path)

    # 分支 1：文件不存在 —— 只有一行，且必须给出绝对路径（用户要能直接找到它）。
    with pytest.raises(TenderParseError) as ei:
        tp.open_pdf(cases["missing"])
    lines = str(ei.value).splitlines()
    assert lines == [f"招标 PDF 不存在：{cases['missing']}"]
    assert Path(lines[0].split("：", 1)[1]).is_absolute()

    # 分支 2：fitz 打不开（截断 / 0 字节 / 纯文本）—— 三行：分类 + 原因 + 下一步。
    for name in ("truncated", "empty", "not_pdf_text"):
        with pytest.raises(TenderParseError) as ei:
            tp.open_pdf(cases[name])
        lines = str(ei.value).splitlines()
        assert len(lines) == 3, name
        assert lines[0] == f"招标 PDF 打不开：{cases[name].name}", name
        assert lines[1].startswith("原因："), name
        assert "请确认该文件未损坏、未加密" in lines[2], name
        # 原因里必须带上底层库的原话，否则排查时无从下手。
        assert len(lines[1]) > len("原因："), name

    # 三种"打不开"的底层原因必须**互不相同**，否则等于没解释。
    reasons = set()
    for name in ("truncated", "empty", "not_pdf_text"):
        with pytest.raises(TenderParseError) as ei:
            tp.open_pdf(cases[name])
        reasons.add(str(ei.value).splitlines()[1])
    assert len(reasons) == 3

    # 分支 3：加密 —— 两行，且必须明说"解密后重跑"，不能只说"打不开"。
    with pytest.raises(TenderParseError) as ei:
        tp.open_pdf(cases["encrypted"])
    lines = str(ei.value).splitlines()
    assert lines[0] == f"招标 PDF 已加密，无法读取：{cases['encrypted'].name}"
    assert lines[1] == "请提供解密后的版本，或去掉打开密码后重跑。"

    # 分支 4：0 页 —— 打得开但没正文，是**独立**的一类，不能混进"打不开"。
    with pytest.raises(TenderParseError) as ei:
        tp.open_pdf(cases["zero_pages"])
    lines = str(ei.value).splitlines()
    assert lines[0] == f"招标 PDF 是 0 页的空文件：{cases['zero_pages'].name}"
    assert lines[1] == "没有正文就无法抽取招标要求。"

    # 反例：改了扩展名的 JPEG 不报错，MuPDF 把它包成 1 页 PDF。
    doc = tp.open_pdf(cases["not_pdf_jpeg"])
    try:
        assert doc.page_count == 1
        assert doc.needs_pass == 0
        assert doc.is_encrypted is False
    finally:
        doc.close()

    # 修复容忍度 A：只去掉 startxref 指针 —— 完整修复，正文一字不少。
    repaired = tmp_path / "repaired.pdf"
    repaired.write_bytes(good_raw[: good_raw.rfind(b"startxref")])
    doc = tp.open_pdf(repaired)
    try:
        assert doc.page_count == 1
        assert "合成参数页" in doc.load_page(0).get_text()
    finally:
        doc.close()

    # 修复容忍度 B：切一半 —— 打得开，但正文被吞空。
    # 后果是下游拿到"空表"而不是"报错"，这条差异正是本段存在的理由。
    halfcut = tmp_path / "halfcut.pdf"
    halfcut.write_bytes(good_raw[: len(good_raw) // 2])
    doc = tp.open_pdf(halfcut)
    try:
        assert doc.page_count == 1
        assert doc.load_page(0).get_text().strip() == ""
    finally:
        doc.close()


def _cli(args: list[str], out: Path) -> subprocess.CompletedProcess:
    """在子进程里跑真正的 ``run.py``，环境剥掉两个 Key 以免走到线上通道。"""
    env = {k: v for k, v in os.environ.items() if k not in ("LLM_API_KEY", "DASHSCOPE_API_KEY")}
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "run.py", "--out", str(out), "--offline", *args],
        cwd=str(REPO), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


def _make_bad_pack(root: Path) -> Path:
    """一个"除了招标 PDF 是垃圾、其余都合法"的资料包，用来打到阶段 3。"""
    import docx

    root.mkdir(parents=True)
    (root / "00_投标范围.txt").write_text(_SCOPE_TEXT, encoding="utf-8")
    (root / "01_招标文件_坏.pdf").write_bytes(b"%PDF-1.4\n this is truncated garbage")
    vendor = root / "02_厂商资料"
    vendor.mkdir()
    d = docx.Document()
    table = d.add_table(rows=2, cols=3)
    for i, head in enumerate(["序号", "项目", "规格/说明"]):
        table.rows[0].cells[i].text = head
    for i, val in enumerate(["1", "输出功率", "100W"]):
        table.rows[1].cells[i].text = val
    d.save(vendor / "SYN_技术规格书.docx")
    return root


@pytest.mark.ec("J7")
@pytest.mark.case("N5")
def test_cli_exit_code_ladder_is_layered_and_never_leaks_a_traceback(tmp_path):
    """三类失败在进程外必须是三个**不同**的退出码语义，且都不吐 traceback。

    边界在哪：退出码不是随便定的，它分层——
    ``2`` 留给"输入本身没法处理"（招标 PDF 坏掉，换台机器也一样坏），
    ``1`` 留给"用户指错了/资料不全"（改个参数就能重跑）。
    同一个 ``TenderParseError`` 在库里是异常，在 CLI 必须变成一行
    ``× 错误：`` 加两行可执行建议，**并且不能写出半个 xlsx**。

    真实资料包实例：训练集五个包都是完好的，打不出这条路径，
    所以坏包在现场合成；但它复用的是真实的 ``run.py`` 子进程，
    因此连解释器启动、``PYTHONPATH``、子进程 UTF-8 编码一起验了。

    与 agent.md 的偏差：无。这是 N1~N15 共同要求的
    「可读消息 + 正确退出码 + 无 traceback」在进程边界上的那一次验证。
    """
    bad = _make_bad_pack(tmp_path / "BAD")
    empty = tmp_path / "空包"
    empty.mkdir()
    missing = tmp_path / "不存在"

    # 阶段 3 才失败：进度必须打到 [3/7]，且不能出现 [4/7]。
    out = tmp_path / "bad.xlsx"
    pr = _cli(["--pack", str(bad)], out)
    assert pr.returncode == 2
    assert "Traceback" not in pr.stderr
    assert pr.stdout == ""
    assert not out.exists()
    err = pr.stderr.splitlines()
    assert "[1/7] 解析投标范围…" in err
    assert "[3/7] 抽取招标要求…" in err
    assert not any(line.startswith("[4/7]") for line in err)
    assert err[-3] == "  × 错误：招标 PDF 打不开：01_招标文件_坏.pdf"
    assert err[-1].startswith("请确认该文件未损坏")

    # 目录不存在：一行错误 + 一行"该指向什么"，退出码 1。
    out2 = tmp_path / "missing.xlsx"
    pr = _cli(["--pack", str(missing)], out2)
    assert pr.returncode == 1
    assert "Traceback" not in pr.stderr
    assert not out2.exists()
    err = pr.stderr.splitlines()
    assert len(err) == 2
    assert err[0] == f"  × 错误：资料包目录不存在：{missing}"
    assert err[1] == "请用 --pack 指向一个包含 00_投标范围.txt 与 01_招标文件_*.pdf 的目录。"

    # 目录存在但空：包名（=目录名）要出现在消息里，退出码同为 1 但文案不同。
    out3 = tmp_path / "empty.xlsx"
    pr = _cli(["--pack", str(empty)], out3)
    assert pr.returncode == 1
    assert "Traceback" not in pr.stderr
    assert not out3.exists()
    err = pr.stderr.splitlines()
    assert len(err) == 2
    assert err[0] == "  × 错误：资料包 空包 缺少必需文件：01_招标文件_*.pdf（招标 PDF）"
    assert err[1] == "没有招标 PDF 就无法抽取招标要求，无法产出结果。"


@pytest.mark.ec("A4")
@pytest.mark.case("N13")
def test_unknown_layout_degrades_loudly_and_caps_confidence_below_the_review_line(tmp_path, repo_root):
    """版面认不出来时：不抛异常、响亮告警、置信度压到复核线以下（N13）。

    边界在哪：这条路径要同时守住两端。
    一端是**不能崩**——考核现场换一份没见过的招标 PDF，宁可交覆盖率低的表；
    另一端是**不能装懂**——降级行的置信度必须低于
    :data:`review.CONF_REVIEW`，否则它会混进"看起来可信"的行里被放过。
    :data:`decide.UNKNOWN_LAYOUT_CAP` 就是这两端之间的那道闸，
    本用例钉住的正是「闸门值 < 复核线」这个**关系**，而不是某个具体数字。

    真实资料包实例：把 P3 的招标 PDF（93 页，真实版面 A）的 ``layout``
    强改成 ``"?"``，让同一份文件走降级路径；这样对比的是
    「同一份输入、认得 vs 认不得」两种行为，而不是两份不同的文件。

    与 agent.md 的偏差：4.5 只写了「置信度打到很低」，没给数字；
    实现落在 ``decide.UNKNOWN_LAYOUT_CAP = 0.40``，且**刻意不加新旗标**——
    靠 0.40 < 0.85 让它自动进复核清单，避免旗标词汇表膨胀。
    """
    import dataclasses

    from bidtable.ingest import pack as pk
    from bidtable.ingest import tender as td
    from bidtable.judge import decide as jd
    from bidtable.models import Evidence, Match, Requirement, Row
    from bidtable.output import review as rv

    pdf = repo_root / "训练集" / "P3" / "01_招标文件_许昌中心医院_4K3D腹腔镜.pdf"
    if not pdf.is_file():
        pytest.skip(f"缺少招标 PDF：{pdf}")
    scopes = pk.parse_scope_file(repo_root / "训练集" / "P3" / "00_投标范围.txt").scopes

    real = tp.locate(pdf)
    try:
        assert real.layout == "A"  # 前提：这份文件本来是认得的
        unknown = dataclasses.replace(
            real, layout="?", chapter_page=0, param_pages=(1, 2), blocks=[], warnings=[]
        )
        reqs, warns = td._extract_unknown(unknown, scopes)
    finally:
        real.close()

    # 不抛异常，且交出一份结构合法的表。
    assert reqs, "降级路径不该交空表——P3 有 93 页正文"
    assert len(reqs) == 95
    # 全部挂到第一个**设备**标的（标的2 是服务标的，被 devices 过滤掉了）。
    assert {r.lot_no for r in reqs} == {"1"}
    assert {r.device_name for r in reqs} == {"4K3D 腹腔镜摄像系统"}
    assert {r.layout for r in reqs} == {"?"}
    # 没有版面信息就不该编造分节，也不该把任何行判成标题。
    assert all(r.section == "" for r in reqs)
    assert not any(r.is_heading for r in reqs)

    # 恰好一条告警，且必须点名文件、说清降级到什么模式、要求人工核对。
    assert len(warns) == 1
    assert warns[0] == (
        "【版面告警】招标 PDF「01_招标文件_许昌中心医院_4K3D腹腔镜.pdf」"
        "不属于已支持的 A/B/C 三种版面族，已降级为「全文捞参数行」模式："
        "全部条目都挂到标的1（4K3D 腹腔镜摄像系统）下，分节留空，"
        "置信度按最低档处理。覆盖率会明显下降，请人工核对招标文件的参数表结构。"
    )

    # 空结果分支：合成一份没有任何编号行的 PDF，必须多给一条"输出将为空表"。
    doc = fitz.open()
    doc.new_page().insert_text((50, 80), "本页没有任何参数", fontname="china-s", fontsize=10)
    blank = tmp_path / "blank.pdf"
    doc.save(blank)
    doc.close()
    t_blank = tp.locate(blank)
    try:
        empty_reqs, empty_warns = td._extract_unknown(
            dataclasses.replace(t_blank, layout="?"), scopes
        )
    finally:
        t_blank.close()
    assert empty_reqs == []
    assert len(empty_warns) == 2
    assert empty_warns[1] == "降级模式也没有捞到任何像参数的行，输出将为空表。"

    # --- 置信度闸门：同一条配对，认得版面 vs 认不得版面 --- #
    def _req(layout: str) -> Requirement:
        return Requirement(
            lot_no="1", device_name="合成设备", section="", no="1", mark="",
            text="输出功率：≥100W", page=8, is_heading=False, layout=layout,
            head_text="输出功率：≥100W",
        )

    ev = Evidence(
        lot_no="1", model="SYN-1", doc_type="技术规格书", filename="SYN_技术规格书.docx",
        locator="技术规格书 表行1", name="输出功率", value="150W",
        response_text="输出功率：150W", priority=0, is_pointer=False, confidence=1.0,
    )

    def _match(layout: str) -> Match:
        return Match(
            requirement=_req(layout), evidence=ev, score=0.9, method="exact",
            confidence=0.9, reason="", flags=(),
        )

    known = jd.decide(_match("A"))
    degraded = jd.decide(_match("?"))
    # 判定结论本身不变（150W > 100W 仍是正偏离），变的只是"我们有多确定"。
    assert known.deviation == degraded.deviation == "正偏离"
    assert known.confidence == pytest.approx(0.9)
    assert degraded.confidence == pytest.approx(jd.UNKNOWN_LAYOUT_CAP)
    assert degraded.note == (
        "招标版面未识别，本行按最低档置信度输出，请人工核对原文与列的对应关系。"
    )
    assert known.note == ""

    # 闸门必须**低于**复核线，降级行才会自动进「优先复核清单」。
    assert jd.UNKNOWN_LAYOUT_CAP < rv.CONF_REVIEW
    degraded_row = Row(
        lot_no="1", device_name="合成设备", section="", no="1", mark="",
        req_text="输出功率：≥100W", resp_text=degraded.resp_text,
        deviation=degraded.deviation, evidence_file=degraded.evidence_file,
        location=degraded.location, confidence=degraded.confidence,
        flags=degraded.flags, note=degraded.note,
    )
    assert rv.needs_review(degraded_row)
    # 对照：同样这条配对在认得版面时置信度 0.9，本身**不该**只因为低分进清单。
    assert known.confidence > rv.CONF_REVIEW

    # 已有的 note 不能被覆盖，只能追加；未响应行（置信度本来就是 0）也要带上提示。
    none_match = Match(
        requirement=_req("?"), evidence=None, score=0.0, method="none",
        confidence=0.0, reason="本标的的厂商资料里找不到可配对的指标", flags=(),
    )
    dec_none = jd.decide(none_match)
    assert dec_none.deviation == "未响应"
    assert dec_none.confidence == 0.0
    assert dec_none.note == degraded.note
