# -*- coding: utf-8 -*-
"""复核报告（agent.md 10.3「产品感验收」）。

**复核信息一律不写进结果 xlsx。** I2 要求严格 10 列，而 ``grade.py`` 用 ``wb.active``
取表，多挂一个工作表就有被读错的风险；几列批注的收益不值得冒这个险。
所以低置信度清单、跳过清单、告警全部走 ``--report <path>.md`` 的旁挂 Markdown。

报告的读者是**投标复核人**，不是评委。他要的三件事按顺序给：
先看哪些行必须人工确认（置信度升序），再看我们主动裁掉了什么（跳过清单），
最后才是汇总数字。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..config import (
    DEV_NEGATIVE,
    DEV_NONE,
    DEV_NO_RESPONSE,
    DEV_POSITIVE,
    MARK_STAR,
    MARK_TRIANGLE,
)
from ..models import PackResult, Row

#: 置信度低于此值的行进「优先复核」清单。取 0.85：匹配层的 ``STRONG`` 也是 0.85，
#: 两者对齐后「强匹配 + 无降信旗标」的行不会进清单，噪声最小。
CONF_REVIEW = 0.85

#: 旗标 -> 复核人看得懂的话。报告里的「为什么」一栏由它翻译。
FLAG_LABELS: dict[str, str] = {
    "low_margin": "比值落在 τ 断层内，正/无偏离之间属于经验判断",
    "low_confidence": "要求含数值门槛，但响应侧找不到同量纲的量",
    "unit_assumed": "厂商漏写单位，按要求侧单位继承（B24）",
    "unit_converted": "做过单位换算（如 mA→A、寸→英寸）",
    "ocr_failed": "检测报告 OCR 不可用，实测值缺失",
    "pointer_assumed": "响应为「详见检测报告」，按无偏离（推定）输出",
    "ambiguous_evidence": "存在同分证据，取舍带有任意性",
    "close_call": "次优证据分数很接近，建议人工确认取材",
    "conflict_resolved": "多份文档给出同值，已按 3.8 优先级取一份",
    "alias_rescue": "靠别名表挂接的拒答式证据，名称是唯一信号",
    "count_alias": "「X数量」类问题，按量词对齐而非名称对齐",
    "llm_fallback": "确定性打分未过阈值，改用语义模型匹配",
    "llm_arbitrated": "判定被模型仲裁改判",
}

#: 汇总摘要里四类偏离的固定次序。
DEV_ORDER = (DEV_NONE, DEV_POSITIVE, DEV_NEGATIVE, DEV_NO_RESPONSE)


def flag_label(flag: str) -> str:
    """旗标的人话版；表里没有的原样返回，绝不因为漏登记而丢信息。"""
    return FLAG_LABELS.get(flag, flag)


def deviation_counts(rows: Sequence[Row]) -> dict[str, int]:
    """四类偏离的分布。空串（标题行）单列，不混进四类里。"""
    out: dict[str, int] = {d: 0 for d in DEV_ORDER}
    out[""] = 0
    for r in rows:
        out[r.deviation] = out.get(r.deviation, 0) + 1
    return out


def risk_rows(rows: Sequence[Row]) -> list[Row]:
    """关键项风险行：★/▲ 且判为负偏离或未响应。与 ``grade.py`` 的口径一致。"""
    return [
        r
        for r in rows
        if r.mark in (MARK_STAR, MARK_TRIANGLE)
        and r.deviation in (DEV_NEGATIVE, DEV_NO_RESPONSE)
    ]


def needs_review(row: Row) -> bool:
    """这行该不该进优先复核清单。"""
    return bool(row.flags) or row.confidence < CONF_REVIEW


def _md(text: Any, limit: int = 60) -> str:
    """塞进 Markdown 表格单元格：去掉竖线与换行，过长截断。"""
    s = str(text or "").replace("|", "丨").replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _table(header: Sequence[str], body: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return lines


def _key_of(row: Row) -> str:
    return f"标的{row.lot_no} / {row.section or '（无分节）'} / 序号{row.no}"


def render(
    result: PackResult,
    *,
    cfg: Any = None,
    generated: str | None = None,
) -> str:
    """渲染整份 Markdown 复核报告。纯函数，便于测试直接断言文本。"""
    rows = result.rows
    decs: list[dict[str, Any]] = list(result.stats.get("decisions") or [])
    while len(decs) < len(rows):
        decs.append({})
    counts = deviation_counts(rows)
    risks = risk_rows(rows)
    review = sorted(
        ((r, d) for r, d in zip(rows, decs) if needs_review(r)),
        key=lambda t: (t[0].confidence, t[0].lot_no, t[0].section, t[0].no),
    )
    stamp = generated or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out: list[str] = [f"# 响应偏离表复核报告 · {result.pack}", ""]
    out += [f"- 生成时间：{stamp}", f"- 输出行数：**{len(rows)}**"]
    if cfg is not None:
        out += [
            f"- 正偏离阈值 τ：`{getattr(cfg, 'positive_margin', 1.10)}`"
            f"（低置信带 `{getattr(cfg, 'low_margin_band', (1.05, 1.15))}`）",
            f"- 分节策略：`{getattr(cfg, 'section_policy', 's4')}`"
            f"　``（N）`` 策略：`{getattr(cfg, 'paren_policy', 'auto')}`"
            f"　OCR 降级：`{getattr(cfg, 'ocr_fallback', 'assume')}`",
        ]
        mode = "离线（未配置可用 Key）" if getattr(cfg, "offline", True) else "在线"
        out.append(f"- 模型通道：{mode}")
    out.append("")

    # --- 1. 汇总 -------------------------------------------------------------
    out += ["## 1. 汇总", ""]
    dist = "　".join(f"{d or '（标题行）'} {counts.get(d, 0)}" for d in (*DEV_ORDER, ""))
    out += [f"四类偏离分布：{dist}", ""]
    out += [
        f"- 关键项风险行（★/▲ 且负偏离或未响应）：**{len(risks)}**",
        f"- 需优先复核行：**{len(review)}** / {len(rows)}",
        f"- 告警：**{len(result.warnings)}** 条；主动裁剪：**{len(result.skipped)}** 条",
    ]
    stage_counts = result.stats.get("stage_counts") or {}
    if stage_counts:
        out += ["", "阶段计数：", ""]
        out += _table(
            ["阶段", "数量"],
            [[_md(k, 30), str(v)] for k, v in stage_counts.items()],
        )
    out.append("")

    # --- 2. 关键项风险 -------------------------------------------------------
    out += ["## 2. 关键项风险（★ / ▲）", ""]
    if risks:
        out += _table(
            ["标记", "位置", "招标要求", "投标响应", "偏离类型", "置信度"],
            [
                [r.mark, _md(_key_of(r), 46), _md(r.req_text, 50), _md(r.resp_text, 40),
                 r.deviation or "—", f"{r.confidence:.2f}"]
                for r in risks
            ],
        )
    else:
        out.append("无。所有 ★/▲ 项均判为无偏离或正偏离。")
    out.append("")

    # --- 3. 优先复核清单 -----------------------------------------------------
    out += ["## 3. 优先复核清单（按置信度升序）", ""]
    if not review:
        out.append("无。全部行的置信度均 ≥ %.2f 且无降信旗标。" % CONF_REVIEW)
    for r, d in review:
        out += [f"### {_key_of(r)}　`{r.mark or '无标记'}`", ""]
        out += _table(
            ["项", "内容"],
            [
                ["招标要求", _md(r.req_text, 160)],
                ["投标响应", _md(r.resp_text, 160) or "（空）"],
                ["偏离类型", r.deviation or "（标题行）"],
                ["证明材料", f"{r.evidence_file or '（空）'}　{r.location or ''}"],
                ["置信度", f"{r.confidence:.2f}"],
                ["判定依据", _md(d.get("reason", ""), 200) or "—"],
                ["判定路径", _md(d.get("method", ""), 40) or "—"],
            ],
        )
        if r.flags:
            out += ["", "旗标：", ""]
            out += [f"- `{f}`：{flag_label(f)}" for f in r.flags]
        if r.note:
            out += ["", f"> {r.note}"]
        out.append("")

    # --- 4. 跳过清单 ---------------------------------------------------------
    out += ["## 4. 主动裁剪 / 跳过清单", ""]
    if result.skipped:
        out += [f"- {s}" for s in result.skipped]
    else:
        out.append("无。")
    out.append("")

    # --- 5. 告警 -------------------------------------------------------------
    out += ["## 5. 告警", ""]
    if result.warnings:
        out += [f"- {w}" for w in result.warnings]
    else:
        out.append("无。")
    out.append("")
    return "\n".join(out)


def write_report(result: PackResult, path: str | Path, *, cfg: Any = None) -> Path:
    """把报告写到 ``path``，自动建父目录。返回实际路径。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(result, cfg=cfg), encoding="utf-8")
    return p


def summary_line(result: PackResult) -> str:
    """收尾摘要（agent.md 10.3 最后一条），打到 stdout 供管道消费。"""
    counts = deviation_counts(result.rows)
    dist = " ".join(f"{d}={counts.get(d, 0)}" for d in DEV_ORDER if counts.get(d, 0))
    n_review = sum(1 for r in result.rows if needs_review(r))
    return (
        f"{result.pack}：{len(result.rows)} 行｜{dist}｜"
        f"关键项风险 {len(risk_rows(result.rows))} 行｜"
        f"待复核 {n_review} 行｜告警 {len(result.warnings)} 条"
    )
