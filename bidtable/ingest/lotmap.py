"""标的 <-> 型号绑定（S7），三源交叉校验。

三个独立来源：
1. ``产品综合样本`` 每页的 ``订购信息``（产品名称 + 型号）；
2. ``技术规格书`` / ``服务方案`` docx 抬头（产品名 + 型号）；
3. ``检测报告`` 文件名里的型号 token（``XX_XX-0000X_检测报告.jpg``），
   OCR 成功时再用 ``样品型号`` 复核。

X2 / X6 的坑：「多出来的那个型号」不是干扰项，它就是**服务标的**的产品。
绝不能按主投标型号过滤证据，否则标的2 的证据会全丢，或者被错分给标的1。
X5 的坑：设备名近似字（``红外辅照治疗仪`` vs ``红外辐照治疗装置``）——
以 ``00_投标范围.txt`` 的结构化字段为准，不信 PDF 块标题。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import DOC_REPORT, DOC_SERVICE, DOC_SPEC
from ..models import Evidence, Scope
from .vendor_docx import VendorDoc
from .vendor_catalog import Catalog
from .vendor_report import Report


def normalize_name(name: str) -> str:
    """设备名归一：去空白、去标点、统一近似写法。仅用于绑定，不改输出。"""
    s = re.sub(r"[\s　]+", "", name or "")
    s = re.sub(r"[（）()【】\[\]、，,。．.：:；;／/\\\-—–]", "", s)
    return s


@dataclass
class LotMap:
    """型号 -> 标的号 的绑定结果。"""

    model_to_lot: dict[str, str] = field(default_factory=dict)
    lot_to_model: dict[str, str] = field(default_factory=dict)
    report_by_model: dict[str, Report] = field(default_factory=dict)
    sources: dict[str, list[str]] = field(default_factory=dict)  # model -> [来源…]
    warnings: list[str] = field(default_factory=list)

    def lot_of(self, model: str) -> str:
        return self.model_to_lot.get(model or "", "")

    def report_for(self, model: str) -> Report | None:
        return self.report_by_model.get(model)


def _best_scope(device_name: str, scopes: list[Scope]) -> Scope | None:
    """把厂商文档里的产品名对到 scope。先精确，再包含，最后编辑距离式兜底。"""
    if not device_name:
        return None
    target = normalize_name(device_name)
    for sc in scopes:
        if normalize_name(sc.device_name) == target:
            return sc
    for sc in scopes:
        n = normalize_name(sc.device_name)
        if n and (n in target or target in n):
            return sc
    # 服务类文档的产品名有时写作「运维服务」之类，退回 is_service 唯一项
    if "服务" in device_name:
        services = [sc for sc in scopes if sc.is_service]
        if len(services) == 1:
            return services[0]
    return None


def build_lotmap(
    scopes: list[Scope],
    docs: list[VendorDoc],
    catalog: Catalog | None,
    reports: list[Report],
) -> LotMap:
    """三源交叉校验。冲突时以样本 ``订购信息`` 为准（它带明确的产品名+型号对）。"""
    lm = LotMap()

    def bind(model: str, device_name: str, source: str) -> None:
        if not model:
            return
        lm.sources.setdefault(model, [])
        if source not in lm.sources[model]:
            lm.sources[model].append(source)
        sc = _best_scope(device_name, scopes)
        if sc is None:
            return
        prev = lm.model_to_lot.get(model)
        if prev and prev != sc.lot_no:
            lm.warnings.append(
                f"型号 {model} 的标的绑定冲突：{prev} vs {sc.lot_no}（来源 {source}）；"
                f"以先出现的样本/规格书抬头为准。"
            )
            return
        lm.model_to_lot[model] = sc.lot_no
        lm.lot_to_model.setdefault(sc.lot_no, model)

    # 源 1：样本订购信息
    if catalog is not None:
        for product in catalog.products:
            bind(product.model, product.product_name, "产品综合样本·订购信息")

    # 源 2：docx 抬头
    for vd in docs:
        bind(vd.model, vd.product_name, f"{vd.doc_type}·抬头")

    # 源 3：报告文件名 + OCR 样品型号
    for rep in reports:
        model = rep.model
        if not model:
            lm.warnings.append(f"{rep.path.name}：文件名里没有型号 token，无法绑定到标的。")
            continue
        lm.report_by_model[model] = rep
        if rep.ocr_ok and rep.sample_name:
            bind(model, rep.sample_name, "检测报告·样品名称")
        elif model not in lm.model_to_lot:
            # 只知道型号不知道产品名：借同型号的规格书/样本产品名回填
            for vd in docs:
                if vd.model == model:
                    bind(model, vd.product_name, "检测报告·文件名")
                    break
            else:
                if catalog is not None:
                    product = catalog.product_for_model(model)
                    if product:
                        bind(model, product.product_name, "检测报告·文件名")

    # 兜底告警：某个标的没绑到任何型号
    for sc in scopes:
        if sc.lot_no not in lm.lot_to_model:
            lm.warnings.append(
                f"标的 {sc.lot_no}（{sc.device_name}）没有匹配到任何厂商型号，"
                f"该标的的证据将只能来自无法定位型号的资料，相关行可能判为未响应。"
            )
    return lm


def attach_lots(evidence: list[Evidence], lm: LotMap) -> list[Evidence]:
    """把 ``lot_no`` 填进证据，并回填报告文件名。

    返回新列表（Evidence 是 frozen 的，用 ``replace``）。
    """
    from dataclasses import replace

    out: list[Evidence] = []
    for ev in evidence:
        lot = lm.lot_of(ev.model)
        filename = ev.filename
        value = ev.value
        response = ev.response_text
        confidence = ev.confidence
        if ev.is_pointer:
            rep = lm.report_for(ev.model)
            if rep is not None:
                filename = rep.path.name
                if rep.ocr_ok:
                    idx = _pointer_index(ev.locator)
                    row = next((r for r in rep.rows if r.no == str(idx)), None)
                    if row is None and 1 <= idx <= len(rep.rows):
                        row = rep.rows[idx - 1]
                    if row is not None:
                        value = row.value
                        response = f"{row.item}：{row.value}"
                        confidence = 1.0
        if not lot:
            # 型号没绑上标的：docx 证据可以用文件名的厂商前缀再试一次
            lot = _lot_from_filename(ev.filename, lm)
        out.append(
            replace(ev, lot_no=lot, filename=filename, value=value,
                    response_text=response, confidence=confidence)
        )
    return out


def _pointer_index(locator: str) -> int:
    m = re.search(r"表行(\d+)", locator)
    return int(m.group(1)) if m else 0


def _lot_from_filename(filename: str, lm: LotMap) -> str:
    if not filename:
        return ""
    for model in lm.model_to_lot:
        if model and model in filename:
            return lm.model_to_lot[model]
    return ""


def doc_type_of(filename: str) -> str:
    if "服务方案" in filename:
        return DOC_SERVICE
    if "技术规格书" in filename or "规格书" in filename:
        return DOC_SPEC
    if "检测报告" in filename:
        return DOC_REPORT
    return ""
