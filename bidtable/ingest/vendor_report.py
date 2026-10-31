"""``*_检测报告.jpg`` 摄取（窄口子 L1）。

报告版面是固定的：``检测报告 / 报告编号：… / 委托单位 / 样品名称：… 样品型号：… /
检测类别 / 检测日期 / 检测结果： / 序号 检测项目 实测值 备注 / N 个数据行 / 以下空白 / …``。

实测取舍：``qwen-vl-ocr`` 出纯文本 2.8s/181 token，``qwen3.7-plus`` 出 JSON
12.8s/773 token。所以默认要纯文本 + 本地正则解析，解析失败才退回结构化 JSON。

**OCR 不可用时的降级**（当前 Key 实测欠费，见 agent.md 9.2）：
``产品综合样本`` 里的 ``● 指标名：详见检测报告`` 指针在该页内的出现次序
== 报告表格的 ``序号``，所以 ``检测报告 第1页 表行N`` 与证明材料文件名
都能纯确定性推出；只有实测值拿不到。此时按
「厂商已出具第三方检测报告」推定 ``无偏离``，``投标响应`` 逐字写样本里的
``指标名：详见检测报告``，打 ``ocr_failed`` 旗标并置顶进复核报告，提示人工补录。
这个推定是可关掉的：``--ocr-fallback no-response``。
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..llm import prompts
from ..llm.client import LLMClient

MAX_B64_CHARS = 190_000

_REPORT_NO = re.compile(r"报告编号\s*[：:]\s*([A-Za-z0-9\-]+)")
_SAMPLE_NAME = re.compile(r"样品名称\s*[：:]\s*([^\s　]+)")
_SAMPLE_MODEL = re.compile(r"样品型号\s*[：:]\s*([A-Za-z0-9][A-Za-z0-9._\-]*)")
_DATA_ROW = re.compile(r"^\s*(\d+)\s+(\S.*?)\s+(\S.*?)(?:\s{2,}(\S.*?))?\s*$")
_END_HINTS = ("以下空白", "检测：", "审核：", "批准：", "检测专用章")


@dataclass(frozen=True)
class ReportRow:
    no: str
    item: str
    value: str
    note: str = ""


@dataclass
class Report:
    path: Path
    model: str = ""
    sample_name: str = ""
    report_no: str = ""
    test_date: str = ""
    rows: list[ReportRow] = field(default_factory=list)
    ocr_ok: bool = False
    raw_text: str = ""
    warnings: list[str] = field(default_factory=list)


def model_from_filename(path: Path) -> str:
    """``XX_XX-0000X_检测报告.jpg`` -> ``XX-0000X``。"""
    stem = path.stem
    parts = stem.split("_")
    for token in parts:
        if re.fullmatch(r"[A-Za-z]{1,6}-[A-Za-z0-9._\-]+", token):
            return token
    return ""


def encode_image(path: Path, *, max_chars: int = MAX_B64_CHARS) -> tuple[str, list[str]]:
    """base64 编码；超限则用 Pillow 降采样后再编（N8 / B17）。"""
    warnings: list[str] = []
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    if len(b64) <= max_chars:
        return b64, warnings
    try:
        from PIL import Image

        scale = 0.8
        for _ in range(4):
            with Image.open(io.BytesIO(data)) as im:
                im = im.convert("RGB")
                new_size = (max(1, int(im.width * scale)), max(1, int(im.height * scale)))
                im = im.resize(new_size)
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=82)
            data = buf.getvalue()
            b64 = base64.b64encode(data).decode("ascii")
            warnings.append(
                f"{path.name} base64 长度超过 {max_chars}，已降采样到 {scale:.0%} 后重传。"
            )
            if len(b64) <= max_chars:
                return b64, warnings
            scale *= 0.75
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"{path.name} 降采样失败：{exc}")
    return b64, warnings


def parse_report_text(text: str) -> tuple[str, str, str, list[ReportRow], list[str]]:
    """从 OCR 纯文本确定性解析报告。返回 (编号, 样品名, 样品型号, 数据行, 告警)。"""
    warnings: list[str] = []
    m = _REPORT_NO.search(text)
    report_no = m.group(1) if m else ""
    sample_name = ""
    sample_model = ""
    m = _SAMPLE_NAME.search(text)
    if m:
        sample_name = m.group(1)
    m = _SAMPLE_MODEL.search(text)
    if m:
        sample_model = m.group(1)

    rows: list[ReportRow] = []
    in_table = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "序号" in line and ("检测项目" in line or "实测值" in line):
            in_table = True
            continue
        if any(line.startswith(h) for h in _END_HINTS):
            in_table = False
            continue
        if not in_table:
            continue
        m = _DATA_ROW.match(line)
        if not m:
            continue
        rows.append(ReportRow(no=m.group(1), item=m.group(2).strip(),
                              value=m.group(3).strip(), note=(m.group(4) or "").strip()))
    if not rows:
        warnings.append("检测报告 OCR 文本里没有解析出任何数据行（预期 3~4 行）。")
    return report_no, sample_name, sample_model, rows, warnings


def parse_report_json(text: str) -> tuple[str, str, str, list[ReportRow], list[str]]:
    """结构化 JSON 兜底解析（``qwen3.7-plus``）。"""
    warnings: list[str] = []
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError) as exc:
        warnings.append(f"检测报告结构化解析失败：{exc}")
        return "", "", "", [], warnings
    rows = [
        ReportRow(
            no=str(r.get("no", "")),
            item=str(r.get("item", "")).strip(),
            value=str(r.get("value", "")).strip(),
            note=str(r.get("note", "")).strip(),
        )
        for r in data.get("rows", [])
    ]
    return (
        str(data.get("report_no", "")),
        str(data.get("sample_name", "")),
        str(data.get("sample_model", "")),
        rows,
        warnings,
    )


def ocr_report(path: Path, client: LLMClient | None) -> Report:
    """对一份报告跑 L1 OCR。任何失败都降级为 ``ocr_ok=False``，绝不抛异常。"""
    report = Report(path=path, model=model_from_filename(path))
    try:
        b64, warns = encode_image(path)
    except OSError as exc:
        report.warnings.append(f"{path.name} 无法读取：{exc}；降级为指针推定。")
        return report
    report.warnings.extend(warns)

    if client is None or not client.enabled:
        report.warnings.append(
            f"{path.name}：模型不可用，检测报告实测值无法读取；"
            f"相关行按「已出具第三方检测报告」推定为无偏离并打 ocr_failed 旗标，请人工复核。"
        )
        return report

    resp = client.complete("ocr", prompts.ocr_messages(b64))
    if resp is None or not resp.text.strip():
        report.warnings.append(f"{path.name}：OCR 调用失败或返回空文本，降级为指针推定。")
        return report

    report.raw_text = resp.text
    no, name, model, rows, warns = parse_report_text(resp.text)
    report.warnings.extend(warns)
    if not rows:
        # 纯文本解析失败 -> 退回结构化 JSON（换 vision 模型）
        resp2 = client.complete(
            "vision", prompts.ocr_messages(b64, structured=True),
            params={"response_format": {"type": "json_object"}, "temperature": 0},
        )
        if resp2 is not None and resp2.text.strip():
            no, name, model, rows, warns2 = parse_report_json(resp2.text)
            report.warnings.extend(warns2)
            report.raw_text = resp2.text
    if rows:
        report.ocr_ok = True
        report.report_no = no
        report.sample_name = name
        if model:
            report.model = model
        report.rows = rows
    else:
        report.warnings.append(f"{path.name}：两种 OCR 解析都没拿到数据行，降级为指针推定。")
    return report
