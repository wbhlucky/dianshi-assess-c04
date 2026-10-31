"""资料包发现、文件分类、``00_投标范围.txt`` 解析。

硬约束：流水线**绝不**读取 ``04_标准答案_*.xlsx``。
那是 ``selfcheck`` / ``grade.py`` 的参照物，碰它就是作弊（O10）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import errors
from ..config import SERVICE_LOT_NAME
from ..models import Scope

SCOPE_FILE = "00_投标范围.txt"
TEMPLATE_PREFIX = "03_"
GOLD_PREFIX = "04_"
VENDOR_DIR_HINTS = ("02_", "厂商资料")

_LOT_RE = re.compile(r"标的\s*(\d+)\s*[：:]\s*(.+?)\s*(?:数量\s*(.*))?$")
_META_RE = re.compile(r"^\s*(项目|采购人|投标人)\s*[：:]\s*(.+?)\s*$")


@dataclass
class Pack:
    """一个资料包的全部输入文件。"""

    root: Path
    name: str
    scope_file: Path | None = None
    tender_pdf: Path | None = None
    template_xlsx: Path | None = None
    vendor_dir: Path | None = None
    spec_docx: list[Path] = field(default_factory=list)
    service_docx: list[Path] = field(default_factory=list)
    catalog_pdf: list[Path] = field(default_factory=list)
    report_images: list[Path] = field(default_factory=list)
    unknown_files: list[Path] = field(default_factory=list)

    def vendor_files(self) -> list[Path]:
        return [*self.spec_docx, *self.service_docx, *self.catalog_pdf, *self.report_images]

    def has_vendor_file(self, filename: str) -> bool:
        return any(p.name == filename for p in self.vendor_files())


def discover(root: str | Path) -> Pack:
    """扫描资料包目录并分类文件。

    目录不存在 -> ``PackNotFoundError``（退出码 1，消息含实际路径，N1）。
    空目录 -> ``PackIncompleteError``，消息列出缺失项（N2）。
    """
    path = Path(root)
    if not path.exists():
        raise errors.PackNotFoundError(
            f"资料包目录不存在：{path.resolve()}\n"
            f"请用 --pack 指向一个包含 {SCOPE_FILE} 与 01_招标文件_*.pdf 的目录。"
        )
    if not path.is_dir():
        raise errors.PackNotFoundError(f"--pack 需要是目录，实际是文件：{path.resolve()}")

    pack = Pack(root=path, name=path.name)
    entries = [p for p in path.iterdir() if not p.name.startswith(".")]

    for p in sorted(entries):
        if p.is_dir():
            if any(h in p.name for h in VENDOR_DIR_HINTS):
                pack.vendor_dir = p
            continue
        name = p.name
        if name == SCOPE_FILE:
            pack.scope_file = p
        elif name.startswith(GOLD_PREFIX):
            continue  # 标准答案：流水线不读
        elif name.startswith(TEMPLATE_PREFIX) and name.endswith(".xlsx"):
            pack.template_xlsx = p
        elif name.startswith("01_") and name.lower().endswith(".pdf"):
            pack.tender_pdf = p
        else:
            pack.unknown_files.append(p)

    if pack.vendor_dir is not None:
        for p in sorted(pack.vendor_dir.iterdir()):
            if not p.is_file() or p.name.startswith("~$"):
                continue
            classify_vendor_file(pack, p)

    return pack


def classify_vendor_file(pack: Pack, path: Path) -> None:
    """按文件名 + 扩展名归类厂商资料（S2 / O6）。"""
    name = path.name
    low = name.lower()
    if low.endswith(".docx") and "服务方案" in name:
        pack.service_docx.append(path)
    elif low.endswith(".docx") and ("技术规格书" in name or "规格书" in name):
        pack.spec_docx.append(path)
    elif low.endswith(".pdf") and ("样本" in name or "catalogue" in low):
        pack.catalog_pdf.append(path)
    elif low.endswith((".jpg", ".jpeg", ".png")) and ("检测报告" in name or "报告" in name):
        pack.report_images.append(path)
    else:
        pack.unknown_files.append(path)


def check_completeness(pack: Pack) -> list[str]:
    """返回缺失项清单。招标 PDF 缺失是致命的；其余只告警。"""
    missing: list[str] = []
    if pack.tender_pdf is None:
        missing.append("01_招标文件_*.pdf（招标 PDF）")
    if pack.scope_file is None:
        missing.append(SCOPE_FILE)
    if not pack.vendor_files():
        missing.append("02_厂商资料/ 下的任意厂商文件")
    return missing


def require_pack(pack: Pack) -> None:
    missing = check_completeness(pack)
    fatal = [m for m in missing if m.startswith("01_")]
    if fatal:
        raise errors.PackIncompleteError(
            f"资料包 {pack.name} 缺少必需文件：{'；'.join(fatal)}\n"
            f"没有招标 PDF 就无法抽取招标要求，无法产出结果。",
            missing=tuple(missing),
        )


@dataclass
class ScopeDoc:
    """``00_投标范围.txt`` 的解析结果。"""

    project: str = ""
    purchaser: str = ""
    bidder: str = ""
    scopes: list[Scope] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def parse_scope_text(text: str) -> ScopeDoc:
    """解析投标范围说明。

    典型行：``  标的 1：床旁血液净化系统  数量 1台``
    数量允许瑕疵（P5 标的12 是 ``1None``），**原样保留**（EC-B / B4）。
    """
    doc = ScopeDoc()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        meta = _META_RE.match(line)
        if meta:
            key, value = meta.group(1), meta.group(2)
            if key == "项目":
                doc.project = value
            elif key == "采购人":
                doc.purchaser = value
            else:
                doc.bidder = value
            continue
        if line.startswith("说明"):
            doc.notes.append(line)
            continue
        m = _LOT_RE.search(line)
        if m:
            lot_no, device, qty = m.group(1), m.group(2).strip(), (m.group(3) or "").strip()
            doc.scopes.append(
                Scope(
                    lot_no=lot_no,
                    device_name=device,
                    quantity=qty,
                    is_service=device == SERVICE_LOT_NAME,
                )
            )
    return doc


def parse_scope_file(path: Path) -> ScopeDoc:
    return parse_scope_text(path.read_text(encoding="utf-8"))


def scopes_from_pack(pack: Pack) -> tuple[list[Scope], list[str]]:
    """取标的清单；缺 ``00_投标范围.txt`` 时返回空表 + 告警（N3）。

    真正的「从招标文件推断」在 ``tender_layout_c`` 里完成，
    因为只有 C 族版面的块标题带 ``N-科室-设备名`` 这种可直接映射标的号的结构。
    """
    warnings: list[str] = []
    if pack.scope_file is None:
        warnings.append(
            f"资料包缺少 {SCOPE_FILE}，改为从招标文件推断标的（结果可能包含未投标的标的，请人工复核）。"
        )
        return [], warnings
    doc = parse_scope_file(pack.scope_file)
    if not doc.scopes:
        warnings.append(f"{SCOPE_FILE} 中未解析到任何标的声明，改为从招标文件推断。")
    return doc.scopes, warnings
