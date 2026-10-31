"""厂商 ``*_技术规格书.docx`` 与 ``*_服务方案.docx`` 摄取。

两类文件形状一致：11 段 + **恰好 1 张** ``序号|项目|规格/说明`` 三列表。
注意 ``para[0]`` 是空段，所以定位靠正则而不是固定下标。

``位置`` 列写 ``技术规格书 表行N`` / ``服务方案 表行N``，
N == 该表 ``序号`` 列的值（训练集里两者恒等，见 T7）。
``投标响应`` 逐字取 ``f"{项目}：{规格/说明}"``（3.10）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .. import errors
from ..config import DOC_PRIORITY, DOC_SERVICE, DOC_SPEC
from ..models import Evidence

TITLE_SPECS = [("技术规格书", DOC_SPEC), ("规格书", DOC_SPEC), ("服务方案", DOC_SERVICE)]
_MODEL_LINE = re.compile(r"型号\s*[：:]\s*([A-Za-z0-9][A-Za-z0-9._\-]*)")
_BIDDER_LINE = re.compile(r"投标人\s*[：:]\s*(\S+)")
_OVERVIEW_MODEL = re.compile(r"([A-Za-z]{1,6}-[A-Za-z0-9._\-]+)\s*是\s*(.+?)\s*面向")
HEADER_CELLS = {"序号", "项目", "规格/说明", "规格", "说明"}


@dataclass
class VendorDoc:
    """一份 docx 的结构化结果。"""

    path: Path
    doc_type: str          # 技术规格书 / 服务方案
    product_name: str      # 段落标题里的设备名 / 服务名
    model: str             # 型号；服务方案从「一、产品概述」正文取（X29）
    maker: str
    rows: list[tuple[str, str, str]]  # (序号, 项目, 规格/说明)
    warnings: list[str]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def read_docx(path: Path) -> tuple[list[str], list[list[str]]]:
    """返回 (段落文本列表, 表格单元格矩阵)。读不出就抛 ``VendorDocError``。"""
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover - 依赖缺失
        raise errors.VendorDocError(
            f"缺少 python-docx，无法读取 {path.name}。请执行 pip install -r requirements.txt。"
        ) from exc
    try:
        document = docx.Document(str(path))
    except Exception as exc:  # noqa: BLE001 - docx 的异常类型不稳定
        raise errors.VendorDocError(f"{path.name} 不是有效的 docx 文件：{exc}") from exc
    paras = [p.text for p in document.paragraphs]
    tables = [[[c.text for c in row.cells] for row in t.rows] for t in document.tables]
    return paras, tables


def parse_docx(path: Path) -> VendorDoc:
    """解析一份厂商 docx。

    失败模式（N7）：无表格 / 表头列数 != 3 -> 记告警并返回空 rows，由调用方跳过。
    """
    paras, tables = read_docx(path)
    warnings: list[str] = []
    name = path.name
    doc_type = DOC_SERVICE if "服务方案" in name else DOC_SPEC
    for token, dtype in TITLE_SPECS:
        if token in name:
            doc_type = dtype
            break

    product_name, model, maker = "", "", ""
    for raw in paras:
        line = raw.strip()
        if not line:
            continue
        if not product_name and (line.endswith("技术规格书") or line.endswith("服务方案")):
            product_name = re.sub(r"(技术规格书|服务方案)$", "", line).strip()
            continue
        if not model:
            m = _MODEL_LINE.search(line)
            if m:
                model = m.group(1)
        if not maker:
            mm = re.search(r"(?:制造商|投标人)\s*[：:]\s*(\S+)", line)
            if mm:
                maker = mm.group(1)
        if product_name and model:
            break

    # X29：服务方案 para[2] 只有「投标人：…」，没有型号行，型号藏在「一、产品概述」正文里
    if not model:
        for raw in paras:
            m = _OVERVIEW_MODEL.search(raw)
            if m:
                model = m.group(1)
                if not maker:
                    maker = m.group(2).strip()
                break
    if not model:
        warnings.append(f"{name}：未能从文档中解析出型号，标的绑定将退回文件名推断。")

    rows: list[tuple[str, str, str]] = []
    if not tables:
        warnings.append(f"{name}：文档内没有表格，已跳过（不影响其余资料）。")
        return VendorDoc(path, doc_type, product_name, model, maker, rows, warnings)
    table = tables[0]
    if not table:
        warnings.append(f"{name}：表格为空，已跳过。")
        return VendorDoc(path, doc_type, product_name, model, maker, rows, warnings)

    header = [_clean(c) for c in table[0]]
    if len(header) != 3 or not (set(header) & HEADER_CELLS):
        warnings.append(
            f"{name}：表头是 {header}，不是预期的 3 列「序号|项目|规格/说明」，已跳过该文件。"
        )
        return VendorDoc(path, doc_type, product_name, model, maker, rows, warnings)

    for cells in table[1:]:
        vals = [_clean(c) for c in cells]
        if len(vals) < 3:
            vals += [""] * (3 - len(vals))
        no, item, spec = vals[0], vals[1], vals[2]
        if not item and not spec:
            continue  # B8：空行跳过并告警
        if not item or not spec:
            warnings.append(f"{name}：表行 {no or '?'} 的项目或规格为空，已跳过该行。")
            continue
        rows.append((no, item, spec))
    return VendorDoc(path, doc_type, product_name, model, maker, rows, warnings)


def to_evidence(vd: VendorDoc, lot_no: str) -> list[Evidence]:
    """把 docx 数据行转成 :class:`Evidence`。``表行N`` 优先用「序号」列的值。"""
    priority = DOC_PRIORITY[vd.doc_type]
    out: list[Evidence] = []
    for index, (no, item, spec) in enumerate(vd.rows, start=1):
        row_no = no if no.isdigit() else str(index)
        out.append(
            Evidence(
                lot_no=lot_no,
                model=vd.model,
                doc_type=vd.doc_type,
                filename=vd.path.name,
                locator=f"{vd.doc_type} 表行{row_no}",
                name=item,
                value=spec,
                response_text=f"{item}：{spec}",
                priority=priority,
                is_pointer=False,
                confidence=1.0,
            )
        )
    return out
