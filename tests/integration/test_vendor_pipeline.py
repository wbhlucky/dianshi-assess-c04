# -*- coding: utf-8 -*-
"""集成层：厂商资料的三级流水线（docx/PDF 解析 -> 标的绑定 -> 输出行的 ``证明材料文件`` 与 ``位置``）。

单元层已经分别钉住了「服务方案的型号藏在概述正文里」（X29）、「样本定位写产品所在页」
（X30）、「base64 超限降采样」（B17）。但这三件事真正的风险都在**跨阶段传递**上：
型号解析对了却绑到错的标的、页码算对了却没写进输出、降采样触发了却没告诉复核人。
所以本文件的三条用例都从真资料包一路走到 ``PackResult``，各自盯一个传递环节：

1. X29 服务方案型号 -> 服务标的 -> 输出行的 ``证明材料文件``（P1/P2/P3 三包）。
2. X30 样本型号所在页 -> 输出行的 ``位置``（P5 的 4 页样本必须 4 页都用到）。
3. B17 真图集里唯一超限的那张 -> ``PackResult.warnings`` -> 复核报告 ``## 5. 告警``。
"""
from __future__ import annotations

import base64
import json

import pytest

from bidtable.config import DOC_SERVICE, DOC_SPEC
from bidtable.ingest import pack as pk
from bidtable.ingest import vendor_catalog as vc
from bidtable.ingest import vendor_docx as vd
from bidtable.ingest import vendor_report as vr
from bidtable.output import review as rv

pytestmark = [pytest.mark.integration]

SERVICE_PACKS = ("P1", "P2", "P3")
SPEC_MODEL = {"P1": "YF-9834Pro", "P2": "BT-6869S", "P3": "TH-4520"}
SERVICE_MODEL = {"P1": "YF-1509Pro", "P2": "BT-6259X", "P3": "TH-8394S"}


def _vendor_fixture(fixtures_dir, pack):
    return json.loads((fixtures_dir / "packs" / pack / "vendor.json").read_text(encoding="utf-8"))


@pytest.mark.ec("D4")
@pytest.mark.ec("B2")
@pytest.mark.case("X29")
def test_x29_the_overview_model_travels_all_the_way_to_the_service_lot_rows(
    repo_root, fixtures_dir, offline_rows
):
    """服务方案的型号只在「一、产品概述」正文里，且必须一路走到服务标的的输出行。

    边界在哪：``*_服务方案.docx`` 的段落形状是 ``para[1]`` 标题、``para[2]`` 投标人、
    ``para[3]`` 「一、产品概述」，**型号既不在标题也不在投标人行**，而是概述正文的
    主语（``YF-1509Pro 是 …面向…``）。按固定下标取 ``para[1]``/``para[2]`` 的实现
    会拿到空型号，此时文件名兜底也救不了——``YF_服务方案.docx`` 里根本没有型号 token。
    型号一空，服务证据就绑不到标的2，标的2 整段退化成 ``未响应``。

    本用例把三个环节串起来证明这条链没断：文档里确实没有「型号：」行 ->
    概述正则捞到型号 -> 型号绑到**服务标的**（不是设备标的）-> 输出行真的引用了这份 docx。
    并且服务型号与规格书型号在三个包里**全都不同**，绑错标的会立刻被发现。
    """
    for pack in SERVICE_PACKS:
        vendor_dir = repo_root / "训练集" / pack / "02_厂商资料"
        if not vendor_dir.is_dir():
            pytest.skip(f"缺少 {pack} 的厂商资料目录")
        spec = next(vendor_dir.glob("*技术规格书.docx"))
        plan = next(vendor_dir.glob("*服务方案.docx"))
        d_spec, d_plan = vd.parse_docx(spec), vd.parse_docx(plan)

        assert (d_spec.doc_type, d_plan.doc_type) == (DOC_SPEC, DOC_SERVICE)
        assert d_spec.warnings == [] and d_plan.warnings == []
        assert (d_spec.model, d_plan.model) == (SPEC_MODEL[pack], SERVICE_MODEL[pack])
        assert d_spec.model != d_plan.model, "两个型号必须不同，否则绑错标的也看不出来"

        paras, _ = vd.read_docx(plan)
        assert not any(vd._MODEL_LINE.search(p) for p in paras), "服务方案里没有「型号：」行"
        hits = [vd._OVERVIEW_MODEL.search(p) for p in paras]
        hits = [m for m in hits if m]
        assert len(hits) == 1 and hits[0].group(1) == d_plan.model
        # 文件名兜底是空的：型号一旦漏解析，就没有第二条路能把它找回来。
        assert vr.model_from_filename(plan) == ""

        binding = _vendor_fixture(fixtures_dir, pack)["lotmap"]["model_to_lot"]
        assert binding[d_spec.model] == "1", "规格书型号绑设备标的"
        assert binding[d_plan.model] == "2", "服务方案型号绑服务标的"

        svc_ev = [e for e in _vendor_fixture(fixtures_dir, pack)["evidence"]
                  if e["doc_type"] == DOC_SERVICE]
        assert svc_ev and {e["model"] for e in svc_ev} == {d_plan.model}
        assert {e["lot_no"] for e in svc_ev} == {"2"}

        cited = {r.evidence_file for r in offline_rows(pack) if r.lot_no == "2" and r.evidence_file}
        assert plan.name in cited, "服务标的的输出行必须真的引用到这份服务方案"
        assert spec.name not in cited, "设备规格书不许串到服务标的上"


@pytest.mark.ec("D2")
@pytest.mark.case("X30")
def test_x30_the_catalog_page_in_the_output_follows_the_bound_model(
    repo_root, fixtures_dir, offline_rows
):
    """``产品综合样本 第N页`` 的 N 由「该行所属标的绑定的型号在样本里的页码」决定。

    边界在哪：写死 ``第1页`` 是最省事的实现，而且在只有一个产品的包上**完全看不出来**。
    P5 是唯一能证伪它的包：样本 4 页、4 个型号（QM-2776/1220X/5448S/9932II）、
    4 个非连续标的（1/5/8/12），输出里 4 个页码全部出现。
    本用例不只断言「4 页都用到」，还逐行反查 ``型号 -> 页码``，
    这样即便页码全都被平移一位（1->2、2->3…）也照样能抓住。

    与 agent.md 的偏差：X30 原文的锚点是「P3 服务 2.2 -> 第2页」。实测我们的 P3 输出
    只会出现 ``产品综合样本 第1页``（6 行），而 P3 标的2 序号2.2 是 ``未响应``、
    ``位置`` 为空——原锚点在本实现里不成立，故改锚 P5。
    """
    by_pack = {}
    for pack in ("P1", "P2", "P3", "P4", "P5"):
        rows = by_pack[pack] = offline_rows(pack)
        model_of_lot = _vendor_fixture(fixtures_dir, pack)["lotmap"]["lot_to_model"]
        catalogs = pk.discover(repo_root / "训练集" / pack).catalog_pdf
        assert len(catalogs) == 1, f"{pack} 应当恰好一份产品综合样本"
        cat = vc.parse_catalog(catalogs[0])
        page_of_model = {p.model: p.page for p in cat.products}

        used = set()
        for r in rows:
            if "产品综合样本" not in r.location:
                continue
            page = int(r.location.rsplit("第", 1)[1].rstrip("页"))
            model = model_of_lot.get(r.lot_no)
            assert model in page_of_model, f"{pack} 标的{r.lot_no} 的型号 {model} 不在样本里"
            assert page == page_of_model[model], (
                f"{pack} 标的{r.lot_no}（{model}）应当写第 {page_of_model[model]} 页，实际写了第 {page} 页"
            )
            used.add(page)
        assert used, f"{pack} 应当至少有一行引用产品综合样本"

    # P5 是唯一四页齐用的包；写死 第1页 的实现在这里必然翻车。
    p5 = [r.location for r in by_pack["P5"] if "产品综合样本" in r.location]
    assert sorted(set(p5)) == [f"产品综合样本 第{i}页" for i in (1, 2, 3, 4)]
    # P3 恰好反证 agent.md 的原锚点：它只会用到第 1 页。
    assert {r.location for r in by_pack["P3"] if "产品综合样本" in r.location} == {"产品综合样本 第1页"}


@pytest.mark.ec("J8")
@pytest.mark.case("B17")
def test_b17_the_one_oversized_real_image_warns_and_the_warning_reaches_the_reviewer(
    repo_root, offline_run
):
    """真图集里只有一张超过 base64 上限，它的告警必须一路走到复核报告的 ``## 5. 告警``。

    边界在哪：单元层已经证明了「上限之下逐字节不变、之上按 0.8 降采样」这条闸门本身。
    集成层要证明的是另外两件事：
    其一，训练集 10 张检测报告里**只有一张**真的越线（``P3/TH_TH-4520``，
    原始 base64 193348 > 190000，降采样一次到 174548），其余 9 张原样通过、零告警——
    也就是说这条分支在真数据上只会被触发一次，写死「总是降采样」会让另外 9 张白白损失清晰度；
    其二，这条告警不能只留在函数返回值里，它必须出现在 ``PackResult.warnings``
    并被 ``review.render`` 原样打进 ``## 5. 告警``——降采样是**有损**操作，
    OCR 实测值的可信度因此下降，复核人必须被告知。

    与 agent.md 的偏差：B17 写的「实测约 188,416 字符」其实是
    ``P2/BT_BT-6869S_检测报告.jpg``，它在上限**之下**、不会触发降采样。
    """
    images = sorted(
        (repo_root / "训练集").glob("P*/02_厂商资料/*检测报告*.jpg"),
        key=lambda p: str(p),
    )
    if not images:
        pytest.skip("训练集里没有检测报告图片")
    assert len(images) == 10

    over = []
    for img in images:
        raw = len(base64.b64encode(img.read_bytes()))
        b64, warns = vr.encode_image(img)
        if raw <= vr.MAX_B64_CHARS:
            assert (len(b64), warns) == (raw, []), f"{img.name} 在上限之下应当原样通过"
        else:
            over.append((img, raw, len(b64), warns))

    assert len(over) == 1, "整个训练集只有一张图真的越线"
    img, raw, out_len, warns = over[0]
    assert img.parent.parent.name == "P3" and img.name == "TH_TH-4520_检测报告.jpg"
    assert (raw, out_len) == (193348, 174548)
    assert out_len <= vr.MAX_B64_CHARS
    assert warns == [f"{img.name} base64 长度超过 {vr.MAX_B64_CHARS}，已降采样到 80% 后重传。"]

    # 跨产物：同一条告警必须出现在 PackResult 与渲染出的复核报告里。
    result = offline_run("P3")
    assert result.warnings[0] == warns[0]
    md = rv.render(result)
    tail = md.split("## 5. 告警", 1)[1]
    assert f"- {warns[0]}" in tail
    assert md.count(warns[0]) == 1, "告警只该在告警节出现一次，不要污染别的章节"
