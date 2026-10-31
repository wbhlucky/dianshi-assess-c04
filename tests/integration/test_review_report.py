# -*- coding: utf-8 -*-
"""集成层：``分节`` 列的分隔符契约，以及复核报告与 xlsx 的跨产物一致性。

两条用例盯的是同一件事的两端——**同一个 ``PackResult`` 被投影成两种产物时不许走样**：

1. B21 ``分节`` 列。我们输出 ``" / "``（两侧各一个空格），标准答案里存的是**裸 ``/``**，
   且连正文内部的空格也一并吃掉（``一、4K3D摄像系统``）。两者逐字节不同，靠
   ``grade.norm`` 才等价。本用例证明这条归一化是**承重**的而非装饰：直接比字节 P3 会
   71 行全错，归一后 17 个分节值与标准答案精确集合相等。
2. 复核报告。``review.render`` 的每个数字都必须能从 ``PackResult`` 与 xlsx 反推出来，
   不许出现「报告说 17 行、表里 18 行」这种跨产物漂移。

B21 放在本文件而非单独建文件，是因为 ``分节`` 除 xlsx 外只在复核报告的 ``_key_of`` 里
再次露面——而那里恰好用同一个 ``" / "`` 当键分隔符，两级分节会让一个键里出现三个
``" / "``。这个副作用只有在渲染层才看得见，故与报告一致性同处一个文件。
"""
from __future__ import annotations

import re

import pytest
from openpyxl import load_workbook

import grade
from bidtable.ingest import section as sec
from bidtable.output import review as rv
from bidtable.output import writer

pytestmark = [pytest.mark.integration]

STAMP = re.compile(r"^- 生成时间：\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$")
SECTIONS = [
    "## 1. 汇总",
    "## 2. 关键项风险（★ / ▲）",
    "## 3. 优先复核清单（按置信度升序）",
    "## 4. 主动裁剪 / 跳过清单",
    "## 5. 告警",
]


def _block(md: str, heading: str) -> str:
    """取出某个 ``## `` 小节的正文，不含标题行本身，也不含后面的小节。"""
    return md.split(heading + "\n", 1)[1].split("\n## ", 1)[0]


@pytest.mark.ec("I3")
@pytest.mark.ec("I5")
@pytest.mark.case("B21")
def test_b21_the_section_separator_is_uniform_and_only_grade_norm_bridges_it_to_gold(
    offline_rows, gold_map
):
    """输出的 ``分节`` 分隔符必须统一，且与标准答案的等价只能由 ``grade.norm`` 建立。

    边界在哪：``分节`` 是全表唯一一列由我们自己拼装、而标准答案又用了不同书写习惯的列。
    P3 有 17 个取值，其中 10 个是两级（``七、专用屏幕台车 / 1、专用屏幕台车``），
    标准答案写成 ``七、专用屏幕台车/1、专用屏幕台车``。任何一侧改了空格习惯，
    逐字节比较就会把 P3 的 71 行全判错——而 ``grade.norm`` 吃掉所有空白，
    于是这个 bug 在自测分数上**完全隐形**。所以两头都钉：既要求我们的输出严格统一为
    ``sec.SEP``（人读的一致性是产品要求），又要求归一后与标准答案集合精确相等
    （评分等价是硬要求），并显式断言「未归一时两者确实不相等」，
    防止有人把归一化当死代码删掉。P5 的分节不含空格，正好证明这条断言没有误伤。
    """
    for pack in ("P1", "P2", "P3", "P4", "P5"):
        rows = offline_rows(pack)
        ours = {r.section for r in rows}
        gold_secs = {v["sec"] for v in gold_map(pack).values()}

        # 输出侧：分隔符只允许 sec.SEP 这一种写法，且拼接干净。
        for s in ours:
            if s == "":
                continue  # EC I-1：整列不填是合法形态，不参与拼接检查
            parts = s.split(sec.SEP)
            assert 1 <= len(parts) <= 2, f"{pack} 分节层数越界：{s!r}"
            for part in parts:
                assert part == part.strip() and part, f"{pack} 分节片段带空白：{s!r}"
            assert not s.startswith(sec.SEP) and not s.endswith(sec.SEP)
            assert sec.SEP + sec.SEP not in s and "//" not in s
            # 归一后分隔符两侧的空格无关紧要 —— 这正是 B21 要的「归一后等价」。
            assert grade.norm(s) == grade.norm(s.replace(sec.SEP, "/"))

        # 评分侧：归一后与标准答案精确集合相等。
        assert {grade.norm(s) for s in ours} == gold_secs, f"{pack} 分节集合与标准答案不符"
        # 含空格/分隔符的包，未归一时必须真的不相等，否则 grade.norm 就是死代码。
        if any(sec.SEP in s or " " in s for s in ours):
            assert ours != gold_secs, f"{pack} 未归一即相等，说明 grade.norm 没起作用"

    # P3 是分节最丰富的包：17 个取值、7 个两级、10 个单级、0 个空分节。
    p3 = [r.section for r in offline_rows("P3")]
    assert len(set(p3)) == 17 and "" not in set(p3)
    assert sum(1 for s in set(p3) if sec.SEP in s) == 7
    # 三种填充形态都得真实存在，否则上面的 continue 会把断言掏空：
    # P1/P2/P4 全不填（I-1）、P3 全填（I-2/I-3）、P5 同一个包里混着来（I-7）。
    p5 = {r.section for r in offline_rows("P5")}
    assert p5 == {"", "1、治疗源", "2、照射光斑面积", "3、过滤器"}
    # 两级分节嵌进 _key_of 后一个键里会出现三个 " / "，但键仍须两两不同。
    keys = [rv._key_of(r) for r in offline_rows("P3")]
    assert len(keys) == len(set(keys)) == 71
    assert max(k.count(sec.SEP) for k in keys) == 3


@pytest.mark.ec("J8")
def test_the_review_report_is_a_faithful_projection_of_the_pack_result(
    repo_root, offline_run, tmp_path
):
    """复核报告的每个数字、旗标、告警都必须能在 ``PackResult`` 与 xlsx 里找到出处。

    边界在哪：报告是人决定「信不信这张表」的唯一入口，一旦与 xlsx 漂移就会误导复核。
    最典型的三种漂移：行数对不上（``render`` 会用 ``zip`` 静默截断过长的 ``decisions``）、
    优先复核清单漏行（``needs_review`` 的判据是「有旗标 **或** 置信度 <0.85」，
    写成 ``and`` 就会漏掉所有「低置信但无旗标」的行，P1 的 3 条未响应全属此类）、
    告警只留在返回值里没进报告（B17 的降采样告警即属此类）。
    本用例逐条对账，并要求 §3 的顺序真的是「按置信度升序」——标题这么写了，
    顺序错了就是欺骗读者。P1（17 行 / 2 告警 / 1 裁剪）与 P3（71 行 / 8 告警 / 0 裁剪）
    形状差异足够大，很难同时蒙对两边。
    """
    for pack in ("P1", "P3"):
        result = offline_run(pack)
        rows = result.rows
        template = repo_root / result.stats["template_xlsx"]
        xlsx, warns = writer.write(
            rows, tmp_path / f"{pack}.xlsx", template=template if template.is_file() else None
        )
        assert warns == []

        md = rv.render(result)
        lines = md.splitlines()
        assert lines[0] == f"# 响应偏离表复核报告 · {pack}"
        assert STAMP.fullmatch(lines[2]), lines[2]
        assert lines[3] == f"- 输出行数：**{len(rows)}**"
        assert [line for line in lines if line.startswith("## ")] == SECTIONS

        # 对账 1：报告行数 == xlsx 数据行数 == PackResult 行数。
        ws = load_workbook(xlsx).active
        assert ws.title == "响应偏离表"
        assert ws.max_row - 1 == len(rows) == len(result.stats["decisions"])

        # 对账 2：§3 的条目与顺序都由 needs_review 决定，且按置信度升序。
        want = sorted(
            (i for i, r in enumerate(rows) if rv.needs_review(r)),
            key=lambda i: (rows[i].confidence, rows[i].lot_no, rows[i].section, rows[i].no),
        )
        got = [line for line in lines if line.startswith("### ")]
        assert len(got) == len(want) > 0
        assert [g[4:].split("`")[0].strip() for g in got] == [rv._key_of(rows[i]) for i in want]
        assert all("`" in g for g in got), "每条优先复核条目都要带标记后缀"
        confs = [rows[i].confidence for i in want]
        assert confs == sorted(confs)
        # needs_review 是「或」不是「与」：低置信但无旗标的行也必须进清单。
        assert {i for i, r in enumerate(rows) if r.confidence < rv.CONF_REVIEW} <= set(want)
        assert any(not rows[i].flags for i in want), "本包应存在仅靠低置信入列的行"

        # 对账 3：§1 的三个数字与汇总行全部可反推。
        risk = rv.risk_rows(rows)
        counts = rv.deviation_counts(rows)
        dist = " ".join(f"{d}={counts[d]}" for d in rv.DEV_ORDER if counts[d])
        assert rv.summary_line(result) == (
            f"{pack}：{len(rows)} 行｜{dist}｜关键项风险 {len(risk)} 行"
            f"｜待复核 {len(want)} 行｜告警 {len(result.warnings)} 条"
        )
        assert f"- 关键项风险行（★/▲ 且负偏离或未响应）：**{len(risk)}**" in md
        assert f"- 需优先复核行：**{len(want)}** / {len(rows)}" in md
        assert f"- 告警：**{len(result.warnings)}** 条；主动裁剪：**{len(result.skipped)}** 条" in md

        # 对账 4：§2 逐行列出关键项风险，位置列就是 _key_of，行数不多不少。
        head = _block(md, SECTIONS[1])
        assert risk and len(risk) == {"P1": 1, "P3": 2}[pack]
        table = [x for x in head.splitlines() if x.startswith("|")]
        assert len(table) == len(risk) + 2, "§2 只该有表头、分隔行与风险行"
        for r in risk:
            assert f"| {rv._key_of(r)} |" in head
            assert r.mark in ("★", "▲") and r.deviation in ("负偏离", "未响应")

        # 对账 5：出现过的旗标必须带中文标签出现在 §3，不许只剩裸英文名。
        body = _block(md, SECTIONS[2])
        fired = {f for r in rows for f in r.flags}
        assert fired, "真资料包上必然有旗标，否则本断言是空的"
        for flag in fired:
            assert rv.flag_label(flag) != flag, f"{flag} 缺少中文标签"
            assert f"- `{flag}`：{rv.flag_label(flag)}" in body

        # 对账 6：裁剪清单与告警清单原样落地，且告警在全文只出现一次。
        clip = _block(md, SECTIONS[3])
        assert (clip.strip() == "无。") == (not result.skipped)
        for s in result.skipped:
            assert f"- {s}" in clip
        tail = _block(md, SECTIONS[4])
        assert len(result.warnings) == len([x for x in tail.splitlines() if x.startswith("- ")])
        for w in result.warnings:
            assert f"- {w}" in tail and md.count(w) == 1
