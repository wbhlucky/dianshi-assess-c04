"""``output/review.py``：旁挂 Markdown 复核报告（agent.md 10.3「产品感验收」）。

复核信息**一律不进 xlsx**（I2 严格 10 列 + ``grade.py`` 用 ``wb.active``），
所以这份报告的文本结构本身就是交付物。十条用例各盯一个互不重叠的能力：

1. ``deviation_counts`` 五个桶恒在，未知偏离类型会**新造**一个桶。
2. ``risk_rows`` 的 2x4 真值表（★/▲ x 四类偏离）。
3. ``needs_review`` 的阈值是**严格小于** 0.85，且任何旗标都能一票否决。
4. ``flag_label`` 覆盖流水线实际会吐出的全部 13 个旗标，未登记的原样返回。
5. ``_md`` 消毒器：竖线、换行、连续空白、``None``、恰好等于 limit 的截断边界。
6. ``render`` 骨架稳定 + 三行配置项只在 ``cfg`` 非空时出现（离线/在线两种措辞）。
7. 四个「无。」占位分支各自的文案，以及第 3 节空清单后**少一个空行**的实测排版。
8. 复核清单按置信度升序、同分按 ``(标的,分节,序号)`` **字符串**排序；
   ``stats["decisions"]`` 与行是**位置 zip**，短了补 ``{}``、长了截断。
9. 单条复核项的正文：空列渲染成「（空）」、旗标清单一个不丢、``note`` 走引用块。
10. ``summary_line`` 的确切形状（零计数桶省略）+ ``write_report`` 建父目录、
   UTF-8 落盘、转发 ``cfg`` 但**不转发** ``generated``（所以文件不是逐字节可复现的）。
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from bidtable.config import (
    DEV_NEGATIVE,
    DEV_NONE,
    DEV_NO_RESPONSE,
    DEV_POSITIVE,
    MARK_STAR,
    MARK_TRIANGLE,
    Config,
)
from bidtable.models import PackResult
from bidtable.output import review

pytestmark = [pytest.mark.unit]

#: 流水线三个产旗标的模块（candidates / semantic / decide）实际会吐出的全部旗标。
EMITTED_FLAGS = frozenset({
    "low_margin", "low_confidence", "unit_assumed", "unit_converted", "ocr_failed",
    "pointer_assumed", "ambiguous_evidence", "close_call", "conflict_resolved",
    "alias_rescue", "count_alias", "llm_fallback", "llm_arbitrated",
})


@pytest.mark.ec("J8")
@pytest.mark.ec("E1")
@pytest.mark.ec("E2")
@pytest.mark.ec("E3")
@pytest.mark.ec("E4")
def test_deviation_counts_always_seeds_five_buckets(mk_row):
    """空输入也必须给出四类偏离 + 空串五个桶（分布行靠它取值，不能 KeyError）。

    边界在最后一句：``out[r.deviation] = out.get(...) + 1`` 对**没登记过**的偏离类型
    会新造一个桶。``render`` 的分布行只遍历 ``(*DEV_ORDER, "")``，
    所以这种桶会被算出来却**不显示**——P4 那条「不判分」行如果混进来就会静默消失。
    """
    assert review.deviation_counts([]) == {
        DEV_NONE: 0, DEV_POSITIVE: 0, DEV_NEGATIVE: 0, DEV_NO_RESPONSE: 0, "": 0,
    }
    rows = [
        mk_row(no="1", deviation=DEV_NONE),
        mk_row(no="2", deviation=DEV_NONE),
        mk_row(no="3", deviation=DEV_NEGATIVE),
        mk_row(no="4", deviation=""),
        mk_row(no="5", deviation="不判分"),
    ]
    counts = review.deviation_counts(rows)
    assert counts[DEV_NONE] == 2
    assert counts[DEV_NEGATIVE] == 1
    assert counts[""] == 1                       # 标题行单列，不混进四类
    assert counts["不判分"] == 1                  # ← 新造的桶
    assert counts[DEV_POSITIVE] == 0 and counts[DEV_NO_RESPONSE] == 0
    assert len(counts) == 6


@pytest.mark.ec("F1")
@pytest.mark.ec("F4")
@pytest.mark.ec("E3")
@pytest.mark.ec("E4")
def test_risk_rows_truth_table_over_marks_and_deviations(mk_row):
    """关键项风险 = (★ 或 ▲) 且 (负偏离 或 未响应)。与 ``grade.py`` 的口径一致。

    六行覆盖真值表里会翻转结论的全部组合，外加「有偏离但没标记」这条最容易误报的。
    """
    rows = [
        mk_row(no="1", mark=MARK_STAR, deviation=DEV_NEGATIVE),
        mk_row(no="2", mark=MARK_TRIANGLE, deviation=DEV_NO_RESPONSE),
        mk_row(no="3", mark=MARK_STAR, deviation=DEV_NONE),
        mk_row(no="4", mark=MARK_STAR, deviation=DEV_POSITIVE),
        mk_row(no="5", mark=MARK_TRIANGLE, deviation=DEV_NEGATIVE),
        mk_row(no="6", mark="", deviation=DEV_NEGATIVE),
    ]
    assert [r.no for r in review.risk_rows(rows)] == ["1", "2", "5"]
    assert review.risk_rows([rows[2], rows[3], rows[5]]) == []


@pytest.mark.ec("J3")
@pytest.mark.ec("J8")
def test_needs_review_threshold_is_strict_and_any_flag_overrides_it(mk_row):
    """阈值取 0.85 是为了和匹配层的 ``STRONG`` 对齐，所以**恰好等于**不进清单。

    两条独立通道：置信度严格小于 0.85，或者带任何旗标。
    满置信 + 有旗标仍然要复核（``low_margin`` 就是典型：分数很高但比值落在 τ 断层里）。
    """
    assert review.CONF_REVIEW == 0.85
    assert review.needs_review(mk_row(confidence=1.0, flags=("low_margin",))) is True
    assert review.needs_review(mk_row(confidence=review.CONF_REVIEW)) is False
    assert review.needs_review(mk_row(confidence=0.849)) is True
    assert review.needs_review(mk_row(confidence=0.9)) is False


@pytest.mark.ec("J8")
@pytest.mark.ec("J3")
def test_flag_label_covers_every_flag_the_pipeline_emits():
    """报告里的「旗标」一栏必须是人话：流水线会吐的旗标一个都不能漏登记。

    反向也钉住——表里不允许有死条目，否则说明某个旗标被改名而标签没跟上。
    未登记的旗标**原样返回**而不是丢掉，保证新增旗标最多是英文裸词，不会丢信息。
    """
    assert len(review.FLAG_LABELS) == 13
    assert set(review.FLAG_LABELS) == set(EMITTED_FLAGS)
    for flag, label in review.FLAG_LABELS.items():
        assert review.flag_label(flag) == label
        assert label.strip() and label != flag
    assert review.flag_label("brand_new_flag") == "brand_new_flag"
    assert review.flag_label("") == ""


@pytest.mark.ec("J8")
@pytest.mark.ec("J5")
def test_md_sanitizer_keeps_the_markdown_table_intact():
    """塞进表格单元格前的消毒：竖线换成同形字 ``丨``（不是转义），换行压成空格。

    截断边界最容易写错：``len(s) <= limit`` 时原样返回，超长时取 ``s[:limit-1] + "…"``，
    所以结果长度**恰好等于** limit。``None`` 走 ``text or ""`` 变成空串。
    """
    assert review._md(None) == ""
    assert review._md("a|b") == "a丨b"
    assert review._md("第一行\r\n第二行\n第三行") == "第一行 第二行 第三行"
    assert review._md("  多   空格\t制表 ") == "多 空格 制表"
    assert review._md("123456", limit=6) == "123456"        # 恰好等于 limit，不截断
    assert review._md("1234567", limit=6) == "12345…"       # 超一个字符就截
    assert len(review._md("1234567", limit=6)) == 6
    assert review._md("招标要求（原文）", 60) == "招标要求（原文）"   # limit 是位置参数


@pytest.mark.ec("J8")
@pytest.mark.ec("J1")
def test_render_skeleton_is_stable_and_cfg_bullets_are_conditional(mk_row, tmp_path):
    """报告骨架（标题 + 5 个 ``## N.`` 小节）固定，三行配置项只在给了 ``cfg`` 时出现。

    ``generated=`` 必须原样落进「生成时间」（测试要可复现，不能读时钟）。
    模型通道有两种措辞：``offline=True`` → 「离线（未配置可用 Key）」，否则「在线」。
    """
    res = PackResult(pack="P1", rows=[mk_row(no="1")], warnings=["w1"], skipped=["s1"])
    md = review.render(res, generated="2026-10-30 00:00:00")
    lines = md.splitlines()
    assert lines[0] == "# 响应偏离表复核报告 · P1"
    assert "- 生成时间：2026-10-30 00:00:00" in lines
    assert "- 输出行数：**1**" in lines
    assert [line for line in lines if line.startswith("## ")] == [
        "## 1. 汇总",
        "## 2. 关键项风险（★ / ▲）",
        "## 3. 优先复核清单（按置信度升序）",
        "## 4. 主动裁剪 / 跳过清单",
        "## 5. 告警",
    ]
    assert "- w1" in lines and "- s1" in lines
    assert "四类偏离分布：" in md and "无偏离 1" in md and "正偏离 0" in md
    assert "模型通道" not in md                 # cfg=None → 三行配置全都不出现
    assert "正偏离阈值 τ" not in md and "分节策略" not in md

    cfg = Config(pack=Path("P1"), out=tmp_path / "o.xlsx", offline=True)
    md2 = review.render(res, cfg=cfg, generated="STAMP")
    assert "- 生成时间：STAMP" in md2.splitlines()
    assert f"- 正偏离阈值 τ：`{cfg.positive_margin}`" in md2
    assert f"（低置信带 `{cfg.low_margin_band}`）" in md2
    assert f"- 分节策略：`{cfg.section_policy}`" in md2
    assert f"``（N）`` 策略：`{cfg.paren_policy}`" in md2
    assert f"OCR 降级：`{cfg.ocr_fallback}`" in md2
    assert "- 模型通道：离线（未配置可用 Key）" in md2.splitlines()

    md3 = review.render(res, cfg=replace(cfg, offline=False), generated="STAMP")
    assert "- 模型通道：在线" in md3.splitlines()
    assert "离线（未配置可用 Key）" not in md3


@pytest.mark.ec("J8")
@pytest.mark.ec("E5")
def test_render_empty_branches_have_their_own_placeholder_copy(mk_row, monkeypatch):
    """四个空分支各有各的文案，第 4、5 节共用一个「无。」（靠小节标题区分）。

    顺带钉两处实测排版：占位文案里的阈值是从 ``CONF_REVIEW`` 格式化出来的（会跟着常量走），
    以及第 3 节空清单后面**少一个空行**，「## 4.」直接贴在占位文案下面。
    """
    res = PackResult(pack="P9", rows=[mk_row(no="1", confidence=1.0)])
    md = review.render(res, generated="S")
    lines = md.splitlines()
    assert "无。所有 ★/▲ 项均判为无偏离或正偏离。" in lines
    placeholder = "无。全部行的置信度均 ≥ 0.85 且无降信旗标。"
    assert placeholder in lines
    i4 = lines.index("## 4. 主动裁剪 / 跳过清单")
    assert lines[i4 - 1] == placeholder          # ← 中间没有空行，如实钉住

    tail = md.split("## 4. 主动裁剪 / 跳过清单", 1)[1]
    assert tail.splitlines() == ["", "", "无。", "", "## 5. 告警", "", "无。"]

    monkeypatch.setattr(review, "CONF_REVIEW", 0.9)
    assert "无。全部行的置信度均 ≥ 0.90 且无降信旗标。" in review.render(res, generated="S")


@pytest.mark.ec("J8")
@pytest.mark.ec("E5")
def test_review_list_is_confidence_ascending_and_decisions_are_positional(mk_row):
    """清单排序键是 ``(置信度, 标的, 分节, 序号)``，全是**字符串**比较。

    所以同置信度下 ``序号10`` 会排在 ``序号3`` 前面——这是实测行为，不是笔误。
    ``stats["decisions"]`` 与 ``rows`` 是**位置 zip**（不按 key 关联）：
    短了补 ``{}``（渲染成「—」），长了被 ``zip`` 静默截断。
    """
    rows = [
        mk_row(lot_no="1", no="3", confidence=0.40),
        mk_row(lot_no="1", no="1", confidence=0.40),
        mk_row(lot_no="2", no="1", confidence=0.20),
        mk_row(lot_no="1", no="10", confidence=0.40),
    ]
    res = PackResult(
        pack="P1", rows=rows, stats={"decisions": [{"reason": "R1", "method": "compare"}]}
    )
    md = review.render(res, generated="S")
    heads = [line for line in md.splitlines() if line.startswith("### ")]
    # 用 _key_of 拼期望值，避免在测试里手写 U+3000 分隔符
    assert [h.split("`")[0].rstrip() for h in heads] == [
        f"### {review._key_of(r)}" for r in (rows[2], rows[1], rows[3], rows[0])
    ]
    assert "| 判定依据 | R1 |" in md
    assert "| 判定路径 | compare |" in md
    assert md.count("| 判定依据 | — |") == 3      # 其余 3 行被 {} 补位
    last = md.split("### ")[-1]                  # rows[0]（序号3）拿到 decisions[0]
    assert "| 判定依据 | R1 |" in last and "| 判定依据 | — |" not in last

    md2 = review.render(
        PackResult(pack="P1", rows=rows[:1],
                   stats={"decisions": [{"reason": "A", "method": "m"}, {"reason": "B"}]}),
        generated="S",
    )
    assert "| 判定依据 | A |" in md2 and "B" not in md2      # 多出来的被 zip 丢掉


@pytest.mark.ec("J8")
@pytest.mark.ec("D7")
def test_review_entry_body_renders_evidence_flags_and_note(mk_row):
    """单条复核项的正文：七行键值表 + 旗标清单 + ``note`` 引用块。

    未响出行的三列是空的，表里要显示成「（空）」而不是留白（否则复核人以为漏渲染）。
    旗标一个都不能丢：没登记过 ``FLAG_LABELS`` 的原样打印成英文。
    """
    row = mk_row(
        no="1", mark=MARK_TRIANGLE, deviation=DEV_NO_RESPONSE,
        req_text="工作长度：≥670 mm", resp_text="", evidence_file="", location="",
        confidence=0.42, flags=("alias_rescue", "brand_new_flag"), note="找不到配对",
    )
    md = review.render(PackResult(pack="P1", rows=[row]), generated="S")
    lines = md.splitlines()
    assert "| 招标要求 | 工作长度：≥670 mm |" in lines
    assert "| 投标响应 | （空） |" in lines
    assert "| 偏离类型 | 未响应 |" in lines
    assert "| 置信度 | 0.42 |" in lines
    assert "| 判定依据 | — |" in lines           # 没有 decisions
    assert any(line.startswith("| 证明材料 | （空）") for line in lines)
    assert "- `alias_rescue`：靠别名表挂接的拒答式证据，名称是唯一信号" in lines
    assert "- `brand_new_flag`：brand_new_flag" in lines
    assert "> 找不到配对" in lines
    assert lines.index("旗标：") > lines.index("| 判定路径 | — |")   # 旗标块在表之后


@pytest.mark.ec("J8")
@pytest.mark.ec("J5")
def test_summary_line_shape_and_write_report_round_trip(mk_row, tmp_path):
    """收尾摘要的确切形状：零计数的偏离类型**省略**（``render`` 的分布行是全列的）。

    ``write_report`` 自己建父目录、UTF-8 落盘、**转发** ``cfg``，
    但**不转发** ``generated`` ——所以报告文件里的时间戳永远是当时钟点，
    两次运行不可能逐字节相同（记进已知问题）。
    """
    rows = [
        mk_row(no="1", deviation=DEV_NONE, confidence=1.0),
        mk_row(no="2", deviation=DEV_NEGATIVE, mark=MARK_STAR, confidence=0.60),
        mk_row(no="3", deviation="", confidence=1.0),
    ]
    res = PackResult(pack="P1", rows=rows, warnings=["w1", "w2"])
    assert review.summary_line(res) == (
        "P1：3 行｜无偏离=1 负偏离=1｜关键项风险 1 行｜待复核 1 行｜告警 2 条"
    )
    # 未登记的偏离类型在摘要里彻底隐身，分布段塌成两个相邻的竖线
    odd = PackResult(pack="P4", rows=[mk_row(no="1", deviation="不判分")])
    assert review.summary_line(odd) == "P4：1 行｜｜关键项风险 0 行｜待复核 0 行｜告警 0 条"

    cfg = Config(pack=Path("P1"), out=tmp_path / "o.xlsx", offline=True)
    out = tmp_path / "报告" / "子 目录" / "P1.review.md"
    assert not out.parent.exists()
    p = review.write_report(res, out, cfg=cfg)
    assert p == out and p.is_file()
    text = p.read_text(encoding="utf-8")
    assert text.startswith("# 响应偏离表复核报告 · P1")
    assert "- 模型通道：离线（未配置可用 Key）" in text.splitlines()   # cfg 被转发
    stamp = next(l for l in text.splitlines() if l.startswith("- 生成时间："))
    assert re.fullmatch(r"- 生成时间：\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", stamp)

    def drop_stamp(text: str) -> list[str]:
        return [l for l in text.splitlines() if not l.startswith("- 生成时间：")]

    assert drop_stamp(text) == drop_stamp(review.render(res, cfg=cfg))
