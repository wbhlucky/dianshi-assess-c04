"""离线自测：用**仓库里那份真 ``grade.py``** 给自己打分（agent.md 6.8 / 10.2）。

::

    python -m bidtable.selfcheck --packs 训练集
    python -m bidtable.selfcheck --packs 训练集 --json out/selfscore.json
    python -m bidtable.selfcheck --packs 训练集/P1 --freeze

两条硬规矩：

1. **绝不复制 grade.py 的逻辑。** 用 :mod:`importlib` 把仓库根目录那份原文件当
   模块加载，指标由它自己的 ``main()`` 计算，我们只解析它打印出来的 JSON。
   这样考核口径一旦改动，自测分数立刻跟着变——不会出现「自测满分、真评分崩」。
2. **绝不在流水线里读 ``04_标准答案``。** 打分是流水线之外独立的一步；
   :func:`bidtable.ingest.pack.discover` 本身也会主动跳过 ``04_`` 前缀的文件，
   所以 ``run.py`` 的产物与标准答案在数据流上是隔离的（O10）。
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import unicodedata
from pathlib import Path
from types import ModuleType
from typing import Any

from . import errors
from .config import load_config
from .logging_ import Progress
from .output import review as rv
from .pipeline import make_client, process_pack, run as pipeline_run

GOLD_PREFIX = "04_"
SCOPE_FILE = "00_投标范围.txt"

# agent.md 10.1 的验收门槛。(指标名, 阈值, 比较方向, 是否冲刺线)
GATES: tuple[tuple[str, float, str, bool], ...] = (
    ("覆盖率", 0.95, ">=", False),
    ("偏离类型准确率", 0.85, ">=", False),
    ("证明材料命中率", 0.90, ">=", False),
    ("位置命中率", 0.85, ">=", False),
)
SPRINT: dict[str, float] = {"偏离类型准确率": 0.92, "证明材料命中率": 0.95, "位置命中率": 0.92}
MAX_EXTRA = 2  # 每包多余条目上限（不计分，但要打印）


# --------------------------------------------------------------------------- #
# grade.py 的动态加载
# --------------------------------------------------------------------------- #


def repo_root(start: Path | None = None) -> Path:
    """从本文件往上找到含 ``grade.py`` 的目录。"""
    cur = (start or Path(__file__)).resolve()
    for cand in (cur, *cur.parents):
        if (cand / "grade.py").is_file():
            return cand
    raise errors.InputError(
        f"在 {cur} 及其父目录里找不到 grade.py，无法自测。"
        "请在仓库根目录下运行，或用 --repo 指定。"
    )


_GRADE_MOD: ModuleType | None = None


def load_grade(root: Path | None = None) -> ModuleType:
    """把 ``grade.py`` 当模块加载（**不 import、不复制**）。结果缓存一份。"""
    global _GRADE_MOD
    if _GRADE_MOD is not None:
        return _GRADE_MOD
    path = (root or repo_root()) / "grade.py"
    spec = importlib.util.spec_from_file_location("bidtable_grade", path)
    if spec is None or spec.loader is None:  # pragma: no cover - 防御
        raise errors.InputError(f"无法把 {path} 作为模块加载。")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _GRADE_MOD = mod
    return mod


def score_pair(gold_xlsx: Path, sub_xlsx: Path, grade_mod: ModuleType | None = None) -> dict[str, Any]:
    """让 ``grade.main()`` 自己算分，我们只收它打印的 JSON。"""
    mod = grade_mod or load_grade()
    buf = io.StringIO()
    argv = sys.argv
    try:
        sys.argv = ["grade.py", str(gold_xlsx), str(sub_xlsx)]
        with contextlib.redirect_stdout(buf):
            mod.main()
    finally:
        sys.argv = argv
    return json.loads(buf.getvalue())


# --------------------------------------------------------------------------- #
# 资料包发现 / 标准答案
# --------------------------------------------------------------------------- #


def find_gold(pack_dir: Path) -> Path | None:
    """``04_标准答案_*.xlsx``。找不到返回 None（只告警，不致命）。"""
    hits = sorted(p for p in pack_dir.glob(f"{GOLD_PREFIX}*.xlsx") if not p.name.startswith("~$"))
    return hits[0] if hits else None


def pack_dirs(base: Path) -> list[Path]:
    """``base`` 自己是资料包就返回它；否则返回含 ``00_投标范围.txt`` 的子目录。"""
    if (base / SCOPE_FILE).is_file():
        return [base]
    return sorted(d for d in base.iterdir() if d.is_dir() and (d / SCOPE_FILE).is_file())


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #


def _parse_star(text: str) -> tuple[int, int]:
    """``"4/5（★/▲且为负偏离或未响应的条目中被正确识别的数量）"`` -> ``(4, 5)``。"""
    head = str(text).split("（")[0]
    a, _, b = head.partition("/")
    try:
        return int(a), int(b)
    except ValueError:
        return 0, 0


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """把逐包分数汇成总分。

    条目数、覆盖数、偏离正确数、``judged`` 分母都能从 grade.py 打印的
    **混淆矩阵**里精确还原（矩阵对每条 gold 行都记了一格，缺行记作 ``X->缺行``），
    所以这三项不做四舍五入的二次加权。材料/位置命中率 grade.py 只给了比率，
    按各包 ``judged`` 加权——单包 3 位小数的舍入误差 < 0.05 行，可忽略。
    """
    n = sum(int(s["条目总数"]) for s in scores)
    hit = sum(int(s["提交匹配条目"]) for s in scores)
    extra = sum(int(s["多余条目"]) for s in scores)
    conf: dict[str, int] = {}
    dev_ok = judged = 0
    star_ok = star_total = 0
    file_w = loc_w = 0.0
    for s in scores:
        matrix = s.get("混淆矩阵") or {}
        per_judged = 0
        for k, v in matrix.items():
            conf[k] = conf.get(k, 0) + int(v)
            a, _, b = k.partition("->")
            if a == b:
                dev_ok += int(v)
            if a != "未响应":
                judged += int(v)
                per_judged += int(v)
        so, st = _parse_star(s["关键项风险识别"])
        star_ok += so
        star_total += st
        file_w += float(s["证明材料命中率"]) * per_judged
        loc_w += float(s["位置命中率"]) * per_judged
    return {
        "条目总数": n,
        "提交匹配条目": hit,
        "多余条目": extra,
        "覆盖率": round(hit / n, 3) if n else 0.0,
        "偏离类型准确率": round(dev_ok / n, 3) if n else 0.0,
        "关键项风险识别": f"{star_ok}/{star_total}（★/▲且为负偏离或未响应的条目中被正确识别的数量）",
        "证明材料命中率": round(file_w / judged, 3) if judged else 0.0,
        "位置命中率": round(loc_w / judged, 3) if judged else 0.0,
        "混淆矩阵": {k: conf[k] for k in sorted(conf)},
    }


def gate_lines(total: dict[str, Any], per_pack: list[dict[str, Any]]) -> list[tuple[str, bool, str]]:
    """逐条门槛判定，返回 ``(描述, 是否通过, 实测)``。"""
    out: list[tuple[str, bool, str]] = []
    for name, floor, _, _ in GATES:
        got = float(total[name])
        ok = got >= floor
        if name == "覆盖率":  # 额外要求：每包 ≥0.90
            worst = min((float(s[name]) for s in per_pack), default=1.0)
            ok = ok and worst >= 0.90
            out.append((f"{name} ≥{floor}（每包 ≥0.90）", ok, f"{got:.3f}／最差包 {worst:.3f}"))
            continue
        sprint = SPRINT.get(name)
        tag = f"（冲刺 ≥{sprint}{'，已达' if sprint and got >= sprint else ''}）" if sprint else ""
        out.append((f"{name} ≥{floor}{tag}", ok, f"{got:.3f}"))
    so, st = _parse_star(total["关键项风险识别"])
    out.append(("关键项风险识别 = 满分", st > 0 and so == st, f"{so}/{st}"))
    worst_extra = max((int(s["多余条目"]) for s in per_pack), default=0)
    out.append((f"每包多余条目 ≤{MAX_EXTRA}（不计分）", worst_extra <= MAX_EXTRA, f"最差包 {worst_extra}"))
    return out


# --------------------------------------------------------------------------- #
# 打印
# --------------------------------------------------------------------------- #

_HEAD = ("资料包", "条目", "覆盖率", "偏离类型", "关键项风险", "证明材料", "位置", "多余")


def _dw(s: str) -> int:
    """显示宽度：全角/宽字符算 2，其余算 1。终端表格对齐靠它。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, width: int, *, right: bool = False) -> str:
    gap = " " * max(0, width - _dw(s))
    return gap + s if right else s + gap


def _cells(s: dict[str, Any], name: str) -> list[str]:
    so, st = _parse_star(s["关键项风险识别"])
    return [
        name,
        str(int(s["条目总数"])),
        f"{float(s['覆盖率']):.3f}",
        f"{float(s['偏离类型准确率']):.3f}",
        f"{so}/{st}",
        f"{float(s['证明材料命中率']):.3f}",
        f"{float(s['位置命中率']):.3f}",
        str(int(s["多余条目"])),
    ]


def render_table(per_pack: list[tuple[str, dict[str, Any]]], total: dict[str, Any]) -> str:
    """五包 + 合计的对齐表格。纯函数，方便测试直接断言文本。"""
    body = [_cells(s, name) for name, s in per_pack]
    tail = _cells(total, "合计")
    widths = [max(_dw(h), *[_dw(r[i]) for r in (*body, tail)]) for i, h in enumerate(_HEAD)]
    rule = "  ".join("-" * w for w in widths)
    lines = [
        "  ".join(_pad(h, widths[i]) for i, h in enumerate(_HEAD)),
        rule,
        *["  ".join(_pad(c, widths[i], right=i > 0) for i, c in enumerate(r)) for r in body],
        rule,
        "  ".join(_pad(c, widths[i], right=i > 0) for i, c in enumerate(tail)),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 冻结 fixture
# --------------------------------------------------------------------------- #


def freeze_pack(pack_dir: Path, dest_root: Path, *, cfg: Any = None) -> list[Path]:
    """把四个可缓存阶段的产物冻结成 ``<dest_root>/<包名>/<stage>.json``。

    实现上就是把 :class:`pipeline.StageCache` 的根目录指到 fixture 目录、并强制
    ``from_stage=0``（永远重算、永远回写）——不额外维护第二套序列化代码，
    所以 fixture 的形状与线上缓存**必然**一致。
    """
    dest = dest_root / pack_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    c = cfg or load_config(pack=pack_dir, out=dest / "_freeze.xlsx", offline=True)
    process_pack(
        pack_dir, c,
        progress=Progress(quiet=True), client=make_client(c),
        from_stage=None, cache_root=dest,
    )
    return sorted(dest.glob("*.json"))


def export_gold(pack_dir: Path, dest_root: Path, grade_mod: ModuleType | None = None) -> Path | None:
    """把 ``04_标准答案`` 导成只读参考 fixture。

    用 grade.py 自己的 ``load()``，键就是它对齐用的 ``(标的, 分节, 序号)``，
    值就是它比较的那七列——同样是「不复制口径」。
    """
    gold = find_gold(pack_dir)
    if gold is None:
        return None
    mod = grade_mod or load_grade()
    rows = mod.load(gold)
    dest = dest_root / f"{pack_dir.name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps({"|".join(k): v for k, v in rows.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return dest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bidtable.selfcheck",
        description="跑完整流水线，再用仓库自带的 grade.py 给自己打分。",
    )
    p.add_argument("--packs", default="训练集", help="资料包目录，或含多个资料包的父目录（默认 训练集）")
    p.add_argument("--out-dir", default="out/selfcheck", help="中间产物 xlsx 的落盘目录")
    p.add_argument("--json", dest="json_out", default=None, help="把成绩导出成 JSON（例如 out/selfscore.json）")
    p.add_argument("--freeze", action="store_true", help="只冻结阶段 fixture，不打分")
    p.add_argument("--fixtures-dir", default="tests/fixtures/packs", help="--freeze 的阶段 fixture 目录")
    p.add_argument("--gold-dir", default="tests/fixtures/gold", help="--freeze 的标准答案参考目录")
    p.add_argument("--repo", default=None, help="仓库根目录（默认自动向上查找 grade.py）")
    p.add_argument("--no-cache", action="store_true", help="禁用阶段缓存，全部重算")
    p.add_argument(
        "--online", action="store_true",
        help="允许真实调用模型 API（默认离线：自测口径必须可复现，"
             "而三个模型口子在离线时都有确定的降级路径）",
    )
    p.add_argument("--strict", action="store_true", help="任一门禁不达标就以退出码 1 结束")
    p.add_argument("-q", "--quiet", action="store_true", help="不打阶段进度")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prog = Progress(quiet=args.quiet)
    try:
        root = repo_root(Path(args.repo)) if args.repo else repo_root()
        grade_mod = load_grade(root)
        base = Path(args.packs)
        if not base.exists():
            raise errors.InputError(f"--packs 指向的路径不存在：{base.resolve()}")
        dirs = pack_dirs(base)
        if not dirs:
            raise errors.InputError(
                f"{base.resolve()} 下没有找到任何资料包（判据：目录里有 {SCOPE_FILE}）。"
            )

        if args.freeze:
            fx, gd = Path(args.fixtures_dir), Path(args.gold_dir)
            for d in dirs:
                files = freeze_pack(d, fx)
                prog.info(f"已冻结 {d.name}：{', '.join(f.name for f in files)} -> {fx / d.name}")
                g = export_gold(d, gd, grade_mod)
                if g is not None:
                    prog.info(f"已导出标准答案参考：{g}")
            return 0

        out_dir = Path(args.out_dir)
        per_pack: list[tuple[str, dict[str, Any]]] = []
        for d in dirs:
            sub = out_dir / f"{d.name}.xlsx"
            cfg = load_config(pack=d, out=sub, offline=not args.online)
            result = pipeline_run(cfg, progress=prog, use_cache=not args.no_cache)
            print(rv.summary_line(result))
            gold = find_gold(d)
            if gold is None:
                prog.warn(f"{d.name}：没有 {GOLD_PREFIX}*.xlsx，跳过打分。")
                continue
            per_pack.append((d.name, score_pair(gold, sub, grade_mod)))

        if not per_pack:
            prog.error("没有任何资料包带标准答案，无法打分。")
            return 1

        scores = [s for _, s in per_pack]
        total = aggregate(scores)
        gates = gate_lines(total, scores)
        print()
        print(render_table(per_pack, total))
        print()
        for desc, ok, got in gates:
            print(f"  [{'PASS' if ok else 'FAIL'}] {desc}　实测 {got}")
        print()
        print("混淆矩阵（标准答案 -> 本项目）：")
        for k, v in total["混淆矩阵"].items():
            print(f"    {k:<16} {v}")

        if args.json_out:
            p = Path(args.json_out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(
                    {
                        "per_pack": {n: s for n, s in per_pack},
                        "total": total,
                        "gates": [{"项": d, "通过": ok, "实测": g} for d, ok, g in gates],
                    },
                    ensure_ascii=False, indent=1,
                ),
                encoding="utf-8",
            )
            prog.info(f"自测成绩已写出：{p}")

        failed = [d for d, ok, _ in gates if not ok]
        if failed and args.strict:
            return 1
        return 0
    except errors.BidTableError as exc:
        prog.error(str(exc))
        return exc.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
