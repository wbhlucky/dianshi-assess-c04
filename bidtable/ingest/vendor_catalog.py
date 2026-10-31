"""``产品综合样本`` PDF 摄取。

版面（实测）：一页一个产品 ——
公司抬头 -> ``产品综合样本 · PRODUCT CATALOGUE`` -> 免责声明 -> ``<产品名><型号>`` 标题
-> 营销语 -> ``核心亮点`` -> ``■ 指标名：值`` 若干（**会跨行折行，按 "" 拼接**）
-> ``第三方实测数据摘要（详见检测报告）`` -> ``● 指标名：详见检测报告`` 若干
-> ``订购信息`` -> ``订货号：X    产品名称：Y    型号：Z``。

两个关键结论：
1. ``● …：详见检测报告`` 是**指针**而不是值（X1）。它在该页内的出现次序
   == 对应检测报告表格里的 ``序号``，因此 ``检测报告 第1页 表行N`` 的 N
   可以纯确定性地推出，无需 OCR。
2. ``产品综合样本 第N页`` 的 N 是产品所在页（X30），不是一律第 1 页。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DOC_CATALOG, DOC_PRIORITY, DOC_REPORT
from ..models import Evidence

BULLET = "■"
POINTER = "●"
POINTER_VALUE = "详见检测报告"

_ORDER_RE = re.compile(r"订货号\s*[：:]\s*(\S+)\s+产品名称\s*[：:]\s*(.+?)\s+型号\s*[：:]\s*(\S+)")
_STRUCTURAL = (
    "核心亮点",
    "第三方实测数据摘要",
    "订购信息",
    "产品综合样本",
    "PRODUCT CATALOGUE",
    "订货号",
)


@dataclass
class CatalogProduct:
    """样本里的一页 = 一个产品。"""

    page: int
    product_name: str
    model: str
    order_no: str = ""
    bullets: list[tuple[str, str]] = field(default_factory=list)   # (名, 值)
    pointers: list[str] = field(default_factory=list)              # 名，按出现次序


@dataclass
class Catalog:
    path: Path
    pages: int
    products: list[CatalogProduct] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def product_for_model(self, model: str) -> CatalogProduct | None:
        for p in self.products:
            if p.model and p.model == model:
                return p
        return None


def _is_structural(line: str) -> bool:
    return any(line.startswith(s) for s in _STRUCTURAL)


def parse_page_text(text: str, page: int) -> tuple[CatalogProduct, list[str]]:
    """把一页纯文本解析成 :class:`CatalogProduct`。

    两遍扫描：先按 ``■`` / ``●`` / 结构行切块并吸收折行，再逐块拆「名：值」。
    折行拼接用 ``""``（不插空格）。但每行先 ``strip()``，所以原文行尾自带的空格
    **不会**保留：``可复用光纤4 `` + ``根（4）…`` -> ``可复用光纤4根（4）…``。
    这不影响打分 —— ``grade.norm`` 与 :func:`match.normalize.squash` 都吃掉空白，
    带空格与不带空格两种写法在匹配和评分上不可区分。
    """
    warnings: list[str] = []
    product = CatalogProduct(page=page, product_name="", model="")

    flat = re.sub(r"[ \t]+", " ", text)
    order_m = _ORDER_RE.search(flat)
    if order_m:
        product.order_no = order_m.group(1)
        product.product_name = order_m.group(2).strip()
        product.model = order_m.group(3).strip()
    else:
        warnings.append(f"产品综合样本第{page}页未找到「订购信息」行，型号留空。")

    lines = [ln.strip() for ln in text.splitlines()]
    blocks: list[tuple[str, list[str]]] = []   # (kind, 行片段)
    kind = ""
    buf: list[str] = []
    in_body = False
    for stripped in lines:
        if not stripped:
            continue
        if stripped.startswith("订购信息"):
            break
        if stripped.startswith(BULLET):
            if kind:
                blocks.append((kind, buf))
            kind, buf, in_body = BULLET, [stripped.lstrip(BULLET).strip()], True
            continue
        if stripped.startswith(POINTER):
            if kind:
                blocks.append((kind, buf))
            kind, buf, in_body = POINTER, [stripped.lstrip(POINTER).strip()], True
            continue
        if _is_structural(stripped):
            if kind:
                blocks.append((kind, buf))
            kind, buf, in_body = "", [], False
            continue
        if in_body:
            buf.append(stripped)   # 折行：直接拼接，不加空格（每行已 strip，行尾空格不保留）
    if kind:
        blocks.append((kind, buf))

    for k, parts in blocks:
        joined = "".join(parts).strip()
        if not joined:
            continue
        if k == POINTER:
            pname = re.split(r"[：:]", joined, maxsplit=1)[0].strip()
            product.pointers.append(pname)
        else:
            pieces = re.split(r"[：:]", joined, maxsplit=1)
            if len(pieces) != 2 or not pieces[0].strip():
                warnings.append(f"产品综合样本第{page}页要点「{joined[:24]}…」缺少「名：值」结构，已跳过。")
                continue
            product.bullets.append((pieces[0].strip(), pieces[1].strip()))

    if not product.bullets and not product.pointers:
        warnings.append(f"产品综合样本第{page}页没有解析到任何核心亮点或实测指针。")
    return product, warnings


def parse_catalog(path: Path) -> Catalog:
    """读整份样本 PDF。单页解析失败只告警，不中断（4.5 第 2 条）。"""
    import fitz

    cat = Catalog(path=path, pages=0)
    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # noqa: BLE001
        cat.warnings.append(f"{path.name} 无法打开：{exc}")
        return cat
    with doc:
        cat.pages = doc.page_count
        for i in range(doc.page_count):
            text = doc[i].get_text("text")
            if not text.strip():
                cat.warnings.append(f"{path.name} 第{i + 1}页 get_text() 为空（可能是扫描页），已跳过。")
                continue
            product, warns = parse_page_text(text, i + 1)
            cat.warnings.extend(warns)
            cat.products.append(product)
    return cat


def to_evidence(cat: Catalog) -> list[Evidence]:
    """把样本转成证据。

    ``■`` 要点 -> ``产品综合样本 第N页``（priority 2）；
    ``●`` 指针 -> 预生成 ``检测报告 第1页 表行i`` 的槽位（priority 0，
    ``is_pointer=True``），文件名与实测值由 :mod:`vendor_report` 回填。
    """
    out: list[Evidence] = []
    for product in cat.products:
        for name, value in product.bullets:
            out.append(
                Evidence(
                    lot_no="",
                    model=product.model,
                    doc_type=DOC_CATALOG,
                    filename=cat.path.name,
                    locator=f"{DOC_CATALOG} 第{product.page}页",
                    name=name,
                    value=value,
                    response_text=f"{name}：{value}",
                    priority=DOC_PRIORITY[DOC_CATALOG],
                    is_pointer=False,
                    confidence=0.9,
                )
            )
        for idx, name in enumerate(product.pointers, start=1):
            out.append(
                Evidence(
                    lot_no="",
                    model=product.model,
                    doc_type=DOC_REPORT,
                    filename="",  # 由 lotmap / vendor_report 回填
                    locator=f"{DOC_REPORT} 第1页 表行{idx}",
                    name=name,
                    value="",
                    response_text=f"{name}：{POINTER_VALUE}",
                    priority=DOC_PRIORITY[DOC_REPORT],
                    is_pointer=True,
                    confidence=0.5,
                )
            )
    return out
