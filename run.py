#!/usr/bin/env python
"""命令行入口。

::

    python run.py --pack 训练集/P1 --out out/P1.xlsx --report out/P1.review.md

这个文件只做四件事：解析参数 -> :func:`bidtable.config.load_config`
-> :func:`bidtable.pipeline.run` -> 打印一行收尾摘要。**没有业务逻辑**，
所有异常都在这里收口成「一行可行动的中文消息 + 正确的退出码」，
绝不裸露 traceback（agent.md 4.5，负面用例 N1..N15）。

退出码：``0`` 成功；``1`` 输入层面的问题（路径、缺文件、参数取值不对）；
``2`` 无法产出结果（招标 PDF 读不出来、xlsx 被 Excel 占用）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bidtable import __version__, errors
from bidtable.config import (
    Config,
    OCR_FALLBACKS,
    PAREN_POLICIES,
    SECTION_POLICIES,
    load_config,
)
from bidtable.logging_ import Progress
from bidtable.models import PackResult
from bidtable.output import review
from bidtable.pipeline import STAGES, make_client, run
from bidtable.selfcheck import find_gold, load_grade, score_pair

TRANSPORTS = ("auto", "live", "record", "replay", "fake")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="从资料包生成《响应偏离表》xlsx 与人工复核报告。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python run.py --pack 训练集/P1 --out out/P1.xlsx --report out/P1.review.md\n"
            "  python run.py --pack 训练集/P3 --out out/P3.xlsx --positive-margin 1.05\n"
            "  python run.py --pack 训练集/P5 --out out/P5.xlsx --from-stage match\n"
            "  python -m bidtable.selfcheck --packs 训练集 --json out/selfscore.json\n"
        ),
    )
    p.add_argument("--pack", required=True, metavar="DIR",
                   help="资料包目录（含 00_投标范围.txt 与 01_招标文件_*.pdf）")
    p.add_argument("--out", required=True, metavar="XLSX",
                   help="输出 xlsx 路径；父目录不存在会自动创建")
    p.add_argument("--report", metavar="MD", default=None,
                   help="可选：Markdown 人工复核报告路径（复核信息只进这里，不进 xlsx）")

    g = p.add_argument_group("判定与策略")
    g.add_argument("--positive-margin", type=float, default=1.10, metavar="TAU",
                   help="正偏离阈值 τ：ratio > τ 才判正偏离（默认 1.10）")
    g.add_argument("--section-policy", default="s4", metavar="P",
                   help=f"分节策略，可选 {'/'.join(SECTION_POLICIES)}（默认 s4）")
    g.add_argument("--paren-policy", default="auto", metavar="P",
                   help=f"``（N）`` 编号策略，可选 {'/'.join(PAREN_POLICIES)}（默认 auto）")
    g.add_argument("--ocr-fallback", default="assume", metavar="M",
                   help=f"检测报告 OCR 不可用时的降级方式，可选 {'/'.join(OCR_FALLBACKS)}（默认 assume）")

    g = p.add_argument_group("模型通道（默认离线：无 Key 或欠费时自动降级，不报错）")
    g.add_argument("--offline", action="store_true", help="强制离线，完全不调用模型")
    g.add_argument("--base-url", default=None, metavar="URL", help="覆盖 LLM_BASE_URL")
    g.add_argument("--model-text", default=None, metavar="NAME", help="覆盖 MODEL_TEXT（L2 语义匹配）")
    g.add_argument("--model-vision", default=None, metavar="NAME", help="覆盖 MODEL_VISION（L1 结构化 OCR）")
    g.add_argument("--model-ocr", default=None, metavar="NAME", help="覆盖 MODEL_OCR（L1 文本 OCR）")
    g.add_argument("--transport", default="auto", metavar="M",
                   help=f"传输层，可选 {'/'.join(TRANSPORTS)}（默认 auto；replay/fake 供测试用）")
    g.add_argument("--replay", default=None, metavar="JSONL", help="--transport replay 的语料路径")

    g = p.add_argument_group("运行控制")
    g.add_argument("--from-stage", default=None, metavar="STAGE",
                   help=f"从第几个阶段起重算，之前的走缓存。可选 {'/'.join(STAGES)} 或 1..{len(STAGES)}")
    g.add_argument("--no-cache", action="store_true", help="完全禁用阶段缓存")
    g.add_argument("--self-check", action="store_true",
                   help="跑完后立刻用仓库自带的 grade.py 给这份产物打分（需要包里有 04_标准答案）")
    g.add_argument("--verbose", action="store_true", help="打印更细的中间信息")
    g.add_argument("--version", action="version", version=f"bidtable {__version__}")
    return p


def _validate(args: argparse.Namespace) -> None:
    """策略类参数取值校验。

    刻意不用 argparse 的 ``choices``：那样会以退出码 2 结束，而「参数写错」属于
    输入问题（退出码 1），跟「招标 PDF 读不出来」不是一回事。
    """
    for name, value, allowed in (
        ("--section-policy", args.section_policy, SECTION_POLICIES),
        ("--paren-policy", args.paren_policy, PAREN_POLICIES),
        ("--ocr-fallback", args.ocr_fallback, OCR_FALLBACKS),
        ("--transport", args.transport, TRANSPORTS),
    ):
        if value not in allowed:
            raise errors.InputError(f"{name} 取值 {value!r} 不支持，可选：{'、'.join(allowed)}。")
    if args.positive_margin <= 0:
        raise errors.InputError(
            f"--positive-margin 必须是正数（它是 ratio 的阈值 τ），当前是 {args.positive_margin}。"
        )


def _self_check(pack_dir: Path, sub: Path, prog: Progress) -> int:
    gold = find_gold(pack_dir)
    if gold is None:
        prog.warn(f"{pack_dir} 里没有 04_标准答案_*.xlsx，--self-check 跳过。")
        return 0
    score = score_pair(gold, sub, load_grade())
    print(json.dumps(score, ensure_ascii=False, indent=1))
    return 0


def _verbose_dump(prog: Progress, cfg: Config, result: PackResult) -> None:
    """``--verbose``：把最终生效的配置、模型通道与逐阶段计数打到 stderr。

    这三样是排查「这个包为什么判成这样」的第一手信息，但它们既不该进 xlsx，
    也不该进 stdout——stdout 只放那一行收尾摘要，方便 ``| tee`` 和脚本采集。
    """
    prog.info(
        f"配置：offline={cfg.offline} transport={cfg.transport} τ={cfg.positive_margin} "
        f"分节={cfg.section_policy} （N）={cfg.paren_policy} "
        f"OCR 降级={cfg.ocr_fallback} 缓存目录={cfg.cache_dir}"
    )
    prog.info(
        f"模型：text={cfg.model_text} strong={cfg.model_strong} "
        f"vision={cfg.model_vision} ocr={cfg.model_ocr} Key 来源={cfg.key_source or '无'}"
    )
    llm = result.stats.get("llm", {})
    stage = result.stats.get("cache", {})
    prog.info(
        f"调用：模型 {llm.get('calls', 0)} 次（命中缓存 {llm.get('cache_hits', 0)} 次），"
        f"阶段缓存 命中 {stage.get('hits', 0)}／未命中 {stage.get('misses', 0)}"
    )
    prog.info("计数：" + json.dumps(result.stats.get("stage_counts", {}), ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prog = Progress()
    try:
        _validate(args)
        cfg = load_config(
            pack=args.pack,
            out=args.out,
            report=args.report,
            offline=args.offline,
            base_url=args.base_url,
            model_text=args.model_text,
            model_vision=args.model_vision,
            model_ocr=args.model_ocr,
            positive_margin=args.positive_margin,
            section_policy=args.section_policy,
            paren_policy=args.paren_policy,
            ocr_fallback=args.ocr_fallback,
            verbose=args.verbose,
            transport=args.transport,
        )
        # --replay 的语料路径只有传输层认识，而 pipeline.run 只接受已构造好的
        # client，所以在这里就把 client 建出来（args.replay 为 None 时等价于
        # process_pack 内部自己建的那一个）。
        result = run(
            cfg,
            progress=prog,
            client=make_client(cfg, replay_path=args.replay),
            from_stage=args.from_stage,
            use_cache=not args.no_cache,
        )
    except errors.BidTableError as exc:
        prog.error(str(exc))
        return exc.exit_code
    except KeyboardInterrupt:  # pragma: no cover
        prog.error("已中断，未写出结果。")
        return 130

    # 收尾摘要走 stdout（进度与告警走 stderr），方便 `python run.py ... | tee`。
    if cfg.verbose:
        _verbose_dump(prog, cfg, result)
    print(review.summary_line(result))
    if args.self_check:
        return _self_check(Path(args.pack), Path(result.stats["xlsx"]), prog)
    return 0


if __name__ == "__main__":
    sys.exit(main())
