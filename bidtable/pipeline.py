"""端到端流水线：资料包 -> 响应偏离表 xlsx + 人工复核报告。

七个阶段与 :data:`logging_.STAGE_NAMES` 一一对应（agent.md 4.1 / 8.2 / 12）：

====== ========== =====================================================
阶段    名字        做什么
====== ========== =====================================================
scope   解析投标范围  ``00_投标范围.txt`` -> ``[Scope]``
vendor  解析厂商资料  docx / 样本 pdf / 报告 jpg -> ``[Evidence]``
tender  抽取招标要求  ``01_*.pdf`` -> ``[Requirement]``（原始，含标题行）
cut     范围裁剪与分节 ``（N）`` 折叠 -> 售后服务裁剪 -> S4 分节
match   要求-证据匹配  ``[Match]``（一条要求至多一条证据，见 candidates.claim）
judge   偏离判定      ``[Decision]``
write   写出结果      xlsx + Markdown 复核报告
====== ========== =====================================================

**缓存策略**：前四个阶段落盘到 ``.cache/stages/<pack_hash>/<stage>.json``，
因为它们要么读 PDF/docx，要么可能调模型；后三个阶段是纯 Python、毫秒级，
**每次重算**——这样调 τ / 匹配阈值不必清缓存。``--from-stage X`` 的含义是
「X 之前的阶段全部走缓存，从 X 起重算」。缓存坏了或缺了就现算，绝不报错。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from . import errors
from .config import COLUMNS, Config
from .ingest import fold as fo
from .ingest import lotmap as lmp
from .ingest import pack as pk
from .ingest import scope as scp
from .ingest import section as sec
from .ingest import tender as td
from .ingest import tender_pdf as tp
from .ingest import vendor_catalog as vcat
from .ingest import vendor_docx as vdx
from .ingest import vendor_report as vrep
from .judge.decide import Decision, judge_all
from .llm.client import LLMClient, build_transport
from .logging_ import Progress
from .match.candidates import match_all
from .models import Evidence, Match, PackResult, Requirement, Row, Scope
from .output import review as rv
from .output import writer as wr

# 与 logging_.STAGE_NAMES 同序；名字同时充当 --from-stage 的取值与缓存文件名。
STAGES: tuple[str, ...] = ("scope", "vendor", "tender", "cut", "match", "judge", "write")
# 只有这四个落盘缓存。match/judge 是纯函数且极快，重算比读盘还便宜。
CACHED_STAGES: tuple[str, ...] = ("scope", "vendor", "tender", "cut")

_WS_RE = re.compile(r"\s+")


def one_line(text: Any) -> str:
    """I13：``招标要求（原文）`` 列不许出现换行。

    只在**确实含换行**时折叠空白，其余逐字返回。跨页拼接出来的原文本来就带
    ``\\n``（cellstitch 用换行连接同一单元格的多行），而单行原文必须保持
    字节级一致——gold 的原文列虽然不计分，但逐字保留是复核者的底线。
    """
    s = "" if text is None else str(text)
    if "\n" not in s and "\r" not in s:
        return s
    return _WS_RE.sub(" ", s).strip()


def pack_hash(pack_dir: Path) -> str:
    """资料包指纹：相对路径 + 大小 + mtime。

    刻意**不读文件内容**——整包 PDF 几十 MB，读一遍比重新解析还慢。
    代价是 ``git checkout`` 之类的操作会刷 mtime 导致缓存失效，这是安全的方向。
    """
    h = hashlib.sha256()
    for p in sorted(pack_dir.rglob("*")):
        if not p.is_file():
            continue
        st = p.stat()
        h.update(
            f"{p.relative_to(pack_dir).as_posix()}|{st.st_size}|{int(st.st_mtime_ns)}\n".encode()
        )
    return h.hexdigest()[:16]


def stage_index(name: str | int | None) -> int:
    """把 ``--from-stage`` 归一化成 0 基下标。阶段名或 1..7 都接受。"""
    if name is None:
        return 0
    if isinstance(name, bool):  # bool 是 int 的子类，明确拒掉
        raise errors.InputError(f"--from-stage 不接受 {name!r}。")
    if isinstance(name, int):
        i = name - 1
    else:
        s = str(name).strip()
        if not s:
            return 0
        if s.isdigit():
            i = int(s) - 1
        elif s in STAGES:
            i = STAGES.index(s)
        else:
            raise errors.InputError(
                f"--from-stage 不认识 {s!r}。可选阶段名：{', '.join(STAGES)}；"
                f"也可以写 1..{len(STAGES)}。"
            )
    if not 0 <= i < len(STAGES):
        raise errors.InputError(
            f"--from-stage 超出范围（1..{len(STAGES)} 或 {', '.join(STAGES)}）：{name!r}"
        )
    return i


class StageCache:
    """``<root>/<stage>.json`` 的读写。``root=None`` 表示整体关闭。"""

    def __init__(self, root: Path | None) -> None:
        self.root = root
        self.hits = 0
        self.misses = 0

    def path(self, stage: str) -> Path | None:
        return None if self.root is None else self.root / f"{stage}.json"

    def get(self, stage: str) -> dict[str, Any] | None:
        p = self.path(stage)
        if p is None or not p.is_file():
            self.misses += 1
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.misses += 1  # 缓存损坏：当没有，重算
            return None
        if not isinstance(data, dict):
            self.misses += 1
            return None
        self.hits += 1
        return data

    def put(self, stage: str, payload: dict[str, Any]) -> None:
        p = self.path(stage)
        if p is None:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        except (OSError, TypeError):
            pass  # 缓存写不进去不影响主流程


def _take(
    prog: Progress,
    cache: StageCache,
    stage: str,
    start: int,
    compute: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """阶段缓存的统一取值。返回 ``(payload, 是否命中缓存)``。"""
    if start > STAGES.index(stage):
        data = cache.get(stage)
        if data is not None:
            prog.info(f"（{stage} 阶段命中缓存，跳过重算）")
            return data, True
        prog.info(f"（{stage} 阶段缓存缺失或已损坏，改为重新计算）")
    data = compute()
    if stage in CACHED_STAGES:
        cache.put(stage, data)
    return data, False


# --------------------------------------------------------------------------- #
# 各阶段的纯计算：返回**可 JSON 序列化**的 payload，缓存与 fixture 共用一套形状
# --------------------------------------------------------------------------- #


def _stage_scope(pack: pk.Pack) -> dict[str, Any]:
    """阶段 1：读 ``00_投标范围.txt``（N3：缺了就空表 + 告警，交给版面推断）。"""
    scopes, warnings = pk.scopes_from_pack(pack)
    missing = pk.check_completeness(pack)
    for item in missing:
        if not item.startswith("01_"):  # 招标 PDF 缺失是致命的，require_pack 已经抛了
            warnings.append(f"资料包缺少：{item}")
    if pack.template_xlsx is None:
        warnings.append(
            "资料包没有 03_响应偏离表_模板.xlsx，已按内置的 10 列常量与列宽重建表头（N4）。"
        )
    if pack.unknown_files:
        warnings.append(
            "以下文件不属于已知的四类厂商资料，未参与解析："
            + "、".join(sorted(p.name for p in pack.unknown_files))
        )
    return {
        "scopes": [s.to_dict() for s in scopes],
        "warnings": warnings,
        "missing": missing,
        "template_xlsx": str(pack.template_xlsx) if pack.template_xlsx else "",
    }


def _stage_vendor(
    pack: pk.Pack,
    scopes: list[Scope],
    client: LLMClient | None,
) -> dict[str, Any]:
    """阶段 2：docx / 样本 pdf / 报告 jpg -> ``[Evidence]``。

    检测报告走 L1 OCR（``client`` 可用时）。离线时 :func:`vendor_report.ocr_report`
    自己降级成 ``ocr_ok=False``，:func:`lotmap.attach_lots` 于是保留
    ``指标名：详见检测报告`` 指针，由判定层按 ``--ocr-fallback`` 处理。

    标的绑定需要 ``scopes``，所以本阶段依赖阶段 1 的产物；缓存键是整个资料包的
    指纹，范围文件一改缓存自然失效，不会读到错配的旧证据。
    """
    docs = [vdx.parse_docx(f) for f in (*pack.spec_docx, *pack.service_docx)]
    cat = vcat.parse_catalog(pack.catalog_pdf[0]) if pack.catalog_pdf else None
    reports = [vrep.ocr_report(f, client) for f in pack.report_images]

    warnings: list[str] = []
    for vd in docs:
        warnings.extend(vd.warnings)
    if cat is not None:
        warnings.extend(cat.warnings)
    for rep in reports:
        warnings.extend(rep.warnings)

    lm = lmp.build_lotmap(scopes, docs, cat, reports)
    warnings.extend(lm.warnings)

    raw: list[Evidence] = []
    for vd in docs:
        raw.extend(vdx.to_evidence(vd, lm.lot_of(vd.model)))
    if cat is not None:
        raw.extend(vcat.to_evidence(cat))
    evidence = lmp.attach_lots(raw, lm)

    unbound = sorted({e.filename for e in evidence if not e.lot_no})
    if unbound:
        warnings.append(
            f"以下资料的型号没能绑定到任何标的，其证据不参与匹配：{'、'.join(unbound)}"
        )

    per_doc: dict[str, int] = {}
    for e in evidence:
        per_doc[e.doc_type or "(未分类)"] = per_doc.get(e.doc_type or "(未分类)", 0) + 1
    return {
        "evidence": [e.to_dict() for e in evidence],
        "warnings": warnings,
        "lotmap": {
            "model_to_lot": lm.model_to_lot,
            "lot_to_model": lm.lot_to_model,
            "sources": lm.sources,
        },
        "counts": {
            "厂商证据": len(evidence),
            "检测报告指针": sum(1 for e in evidence if e.is_pointer),
            "OCR 成功": sum(1 for r in reports if r.ocr_ok),
            "报告张数": len(reports),
            **per_doc,
        },
    }


def _stage_tender(
    pack: pk.Pack,
    scopes: list[Scope],
    paren_policy: str,
) -> dict[str, Any]:
    """阶段 3：按版面族 A/B/C 抽参数表；未识别版面走降级路径（N13）。"""
    t = tp.locate(pack.tender_pdf)
    try:
        reqs, warnings = td.extract(t, scopes, paren_policy=paren_policy)
        meta = {
            "layout": t.layout,
            "npages": t.npages,
            "chapter_page": t.chapter_page,
            "param_pages": list(t.param_pages),
        }
    finally:
        t.close()
    return {"reqs": [r.to_dict() for r in reqs], "warnings": warnings, **meta}


def _stage_cut(
    reqs: list[Requirement],
    evidence: list[Evidence],
    scopes: list[Scope],
    *,
    paren_policy: str,
    section_policy: str,
) -> dict[str, Any]:
    """阶段 4：``（N）`` 折叠 -> 售后服务裁剪 -> 分节。三步都有人读的说明。"""
    fr = fo.resolve(reqs, evidence, scopes, policy=paren_policy)
    cut = scp.apply(fr.reqs, scopes)
    final, wsec = sec.apply(cut.kept, scopes, policy=section_policy)
    skipped = [*fr.notes, *cut.reasons]
    return {
        "reqs": [r.to_dict() for r in final],
        "skipped": skipped,
        "warnings": [*fr.warnings, *cut.warnings, *wsec],
        "counts": {
            "抽取条目": len(reqs),
            "折叠后": len(fr.reqs),
            "裁剪后": len(cut.kept),
            "分节后": len(final),
            "标题/引导句": sum(1 for r in final if r.is_heading),
            "参与判定": sum(1 for r in final if not r.is_heading),
        },
        "cut_counts": cut.counts,
    }


# --------------------------------------------------------------------------- #
# 组装输出行
# --------------------------------------------------------------------------- #


def build_rows(
    reqs: list[Requirement],
    matches: list[Match],
    decisions: list[Decision],
) -> tuple[list[Row], list[dict[str, Any]]]:
    """把 ``(要求, 匹配, 判定)`` 三元组压成输出行 + 复核用的判定说明。

    三个列表等长同序：``match_all`` 保证一条要求一条 Match，
    ``judge_all`` 保证一条 Match 一条 Decision。``reqs`` 里**不含标题行**
    ——标题行的 ``偏离类型`` 只能是空串（I4），而空串行写进 xlsx 会污染
    grade.py 的行字典，所以标题行在进匹配前就被摘掉了（I14 的「条目数」
    指的是参与判定的正文条目数）。
    """
    rows: list[Row] = []
    decs: list[dict[str, Any]] = []
    for req, m, d in zip(reqs, matches, decisions):
        rows.append(
            Row(
                lot_no=req.lot_no,
                device_name=req.device_name,
                section=req.section,
                no=req.no,
                mark=req.mark,
                req_text=one_line(req.text),
                resp_text=d.resp_text,
                deviation=d.deviation,
                evidence_file=d.evidence_file,
                location=d.location,
                confidence=d.confidence,
                flags=tuple(d.flags),
                note=d.note,
            )
        )
        decs.append(
            {
                "reason": d.reason,
                "method": d.method,
                "match_method": m.method,
                "match_score": round(m.score, 3),
                "match_reason": m.reason,
            }
        )
    return rows, decs


def _dedupe_aligned(
    rows: list[Row],
    extras: list[Any],
    result: PackResult,
) -> tuple[list[Row], list[Any]]:
    """按 I1 去重，并让与 ``rows`` 平行的 ``extras`` 同步收缩。

    去重规则完全交给 :func:`output.writer.dedupe`（保留第一条），这里只负责
    把判定说明按对象身份对齐——所以 ``writer.write`` 内部再 dedupe 一次是
    无害的空操作，告警也只在这里出一次。
    """
    kept, warns = wr.dedupe(rows)
    for w in warns:
        result.add_warning(w)
    if len(kept) == len(rows):
        return rows, extras
    keep = {id(r) for r in kept}
    return kept, [x for r, x in zip(rows, extras) if id(r) in keep]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def make_client(cfg: Config, *, replay_path: str | None = None) -> LLMClient:
    """按 ``cfg.transport`` 建客户端。

    离线或无 Key 时 :func:`build_transport` 返回 ``None``，``client.enabled``
    为 False，三个 LLM 口子（L1 OCR / L2 语义匹配 / L3 裁决）全部自动降级，
    链路不需要任何分支判断。
    """
    return LLMClient(cfg, build_transport(cfg, replay_path=replay_path))


def process_pack(
    pack_dir: str | Path,
    cfg: Config,
    *,
    progress: Progress | None = None,
    client: LLMClient | None = None,
    from_stage: str | int | None = None,
    use_cache: bool = True,
    cache_root: Path | None = None,
) -> PackResult:
    """跑完阶段 1..6，返回 :class:`PackResult`。**不写盘**——写盘在 :func:`run`。

    可抛出的异常都是 :class:`errors.BidTableError` 子类，``str(exc)`` 即可直接
    打印给用户（agent.md 4.5：绝不裸露 traceback）。

    ``cache_root`` 显式指定阶段缓存目录，覆盖 ``cfg.stage_cache_dir/<pack_hash>``。
    ``bidtable.selfcheck --freeze`` 用它把四个可缓存阶段的产物直接写成冻结 fixture。
    """
    root = Path(pack_dir)
    prog = progress if progress is not None else Progress(quiet=True)
    start = stage_index(from_stage)

    pack = pk.discover(root)  # N1 路径不存在 / N2 空目录：消息就在这里成形
    pk.require_pack(pack)

    result = PackResult(pack=pack.name)
    result.stats["pack_dir"] = str(root)
    if not use_cache:
        cache = StageCache(None)
    else:
        cache = StageCache(cache_root if cache_root is not None else cfg.stage_cache_dir / pack_hash(root))
    if client is None:
        client = make_client(cfg)

    # ---- 1 解析投标范围 ---------------------------------------------------
    prog.stage(1)
    pay, _ = _take(prog, cache, "scope", start, lambda: _stage_scope(pack))
    for w in pay.get("warnings", []):
        result.add_warning(w)
    scopes = [Scope.from_dict(d) for d in pay["scopes"]]
    result.scopes = scopes
    result.stats["template_xlsx"] = pay.get("template_xlsx", "")
    prog.info(
        f"{len(scopes)} 个标的："
        + ("；".join(f"标的{s.lot_no} {s.device_name}" for s in scopes) or "（未声明，改由招标文件推断）")
    )

    # ---- 2 解析厂商资料 ---------------------------------------------------
    prog.stage(2)
    pay, _ = _take(prog, cache, "vendor", start, lambda: _stage_vendor(pack, scopes, client))
    for w in pay.get("warnings", []):
        result.add_warning(w)
    evidence = [Evidence.from_dict(d) for d in pay["evidence"]]
    result.stats["vendor_counts"] = pay.get("counts", {})
    prog.info(
        f"{len(evidence)} 条厂商证据："
        + "，".join(f"{k} {v}" for k, v in pay.get("counts", {}).items() if k != "厂商证据")
    )

    # ---- 3 抽取招标要求 ---------------------------------------------------
    prog.stage(3)
    pay, _ = _take(
        prog, cache, "tender", start,
        lambda: _stage_tender(pack, scopes, cfg.paren_policy),
    )
    for w in pay.get("warnings", []):
        result.add_warning(w)
    reqs = [Requirement.from_dict(d) for d in pay["reqs"]]
    result.stats["layout"] = pay.get("layout", "")
    prog.info(
        f"版面族 {pay.get('layout', '?')}，PDF 共 {pay.get('npages', 0)} 页，"
        f"需求章节第 {pay.get('chapter_page', 0)} 页，参数页 {len(pay.get('param_pages', []))} 张；"
        f"抽到 {len(reqs)} 条原始条目"
    )

    # ---- 4 范围裁剪与分节 -------------------------------------------------
    prog.stage(4)
    pay, _ = _take(
        prog, cache, "cut", start,
        lambda: _stage_cut(
            reqs, evidence, scopes,
            paren_policy=cfg.paren_policy, section_policy=cfg.section_policy,
        ),
    )
    for w in pay.get("warnings", []):
        result.add_warning(w)
    for s in pay.get("skipped", []):
        result.add_skipped(s)
    final = [Requirement.from_dict(d) for d in pay["reqs"]]
    counts = pay.get("counts", {})
    result.stats["cut_counts"] = counts
    prog.info(
        f"{counts.get('抽取条目', len(reqs))} → 折叠 {counts.get('折叠后', 0)} → "
        f"裁剪 {counts.get('裁剪后', 0)} → 分节 {counts.get('分节后', len(final))}；"
        f"其中标题/引导句 {counts.get('标题/引导句', 0)} 条不输出，"
        f"主动裁剪 {len(result.skipped)} 处"
    )

    # ---- 5 要求-证据匹配 --------------------------------------------------
    prog.stage(5)
    body = [r for r in final if not r.is_heading]
    matches, wm = match_all(body, evidence, client=client)
    for w in wm:
        result.add_warning(w)
    hit = sum(1 for m in matches if m.evidence is not None)
    prog.info(f"{len(body)} 条参与匹配，配上证据 {hit} 条，未响应 {len(body) - hit} 条")

    # ---- 6 偏离判定 -------------------------------------------------------
    prog.stage(6)
    decisions, wj = judge_all(matches, cfg, client)
    for w in wj:
        result.add_warning(w)
    rows, decs = build_rows(body, matches, decisions)
    rows, decs = _dedupe_aligned(rows, decs, result)
    result.rows = rows
    result.stats["decisions"] = decs

    dist = rv.deviation_counts(rows)
    stage_counts: dict[str, int] = {
        "标的": len(scopes),
        "厂商证据": len(evidence),
        "抽取条目": len(reqs),
        "标题/引导句（不输出）": len(final) - len(body),
        "输出行": len(rows),
    }
    stage_counts.update({f"· {d}": dist.get(d, 0) for d in rv.DEV_ORDER if dist.get(d, 0)})
    stage_counts["关键项风险（★/▲ 且负偏离或未响应）"] = len(rv.risk_rows(rows))
    stage_counts["待复核"] = sum(1 for r in rows if rv.needs_review(r))
    result.stats["stage_counts"] = stage_counts

    if client is not None:
        result.stats["llm"] = {
            "enabled": bool(client.enabled),
            "calls": getattr(client, "calls", 0),
            "cache_hits": getattr(client, "cache_hits", 0),
        }
        for note in getattr(client, "degraded", None) or []:
            result.add_warning(f"模型通道降级：{note}")
    result.stats["cache"] = {"hits": cache.hits, "misses": cache.misses}
    prog.info(
        "　".join(f"{d} {dist.get(d, 0)}" for d in rv.DEV_ORDER)
        + f"｜关键项风险 {stage_counts['关键项风险（★/▲ 且负偏离或未响应）']} 行"
        + f"｜待复核 {stage_counts['待复核']} 行"
    )
    return result


def run(
    cfg: Config,
    *,
    progress: Progress | None = None,
    client: LLMClient | None = None,
    from_stage: str | int | None = None,
    use_cache: bool = True,
) -> PackResult:
    """阶段 1..7 全跑并落盘。``cfg.out`` 是 xlsx，``cfg.report`` 可选。"""
    prog = progress if progress is not None else Progress()
    result = process_pack(
        cfg.pack, cfg,
        progress=prog, client=client, from_stage=from_stage, use_cache=use_cache,
    )

    # ---- 7 写出结果 -------------------------------------------------------
    prog.stage(7)
    tpl = result.stats.get("template_xlsx") or None
    path, wwrite = wr.write(result.rows, cfg.out, template=tpl)  # N11 建父目录 / N12 占用
    for w in wwrite:
        result.add_warning(w)
    result.stats["xlsx"] = str(path)
    prog.info(f"已写出 {path}（{len(result.rows)} 行 × {len(COLUMNS)} 列，工作表「{wr.SHEET_NAME}」）")

    if cfg.report:
        rp = rv.write_report(result, cfg.report, cfg=cfg)
        result.stats["report"] = str(rp)
        prog.info(f"复核报告：{rp}")

    if result.skipped:
        prog.info(f"主动裁剪 {len(result.skipped)} 处（明细见复核报告第 4 节）")
    for w in result.warnings:
        prog.warn(w)
    return result
