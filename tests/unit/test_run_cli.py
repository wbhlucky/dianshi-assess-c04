"""``run.py``：命令行入口的对外契约（参数 -> Config -> pipeline -> 一行摘要）。

这个文件本身**没有业务逻辑**，所以这里盯的全是边界契约，每条各管一件事：

1. 策略类参数取值不合法 -> 退出码 1 + 一行中文 + 合法取值列全
2. argparse 自己那一层：缺必填参数退出码 2、``--version`` 退出码 0
3. ``BidTableError`` 家族按各自的 ``exit_code`` 分层收口，绝不裸露 traceback
4. 非 ``BidTableError`` 的意外异常**不**被吞；Ctrl-C 有专属退出码 130
5. ``--pack`` 不存在时消息给出 resolve 后的绝对路径，且不留半截产物
6. 每一个 CLI 开关都落到 ``Config`` 的对应字段（含 ``--offline`` 压过真 Key）
7. ``--transport`` 决定通道；``--replay`` 的语料路径真的被读进去
8. ``--from-stage`` / ``--no-cache`` 原样透传，CLI 层不做翻译
9. 成功路径：摘要走 stdout、进度与告警走 stderr、xlsx 与复核报告都落盘
10. 输出路径含中文与空格时产物逐格一致（B22）
11. ``--self-check`` 在没有标准答案时只告警，退出码仍是 0
12. ``--self-check`` 有标准答案时把 grade.py 的 JSON 追加到 stdout
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
from pathlib import Path

import pytest
from openpyxl import load_workbook

import run as cli
from bidtable import __version__, errors
from bidtable.config import OCR_FALLBACKS, PAREN_POLICIES, SECTION_POLICIES
from bidtable.models import PackResult

pytestmark = [pytest.mark.unit]


def _main(argv):
    """跑一次 ``cli.main``，把 stdout / stderr 分开收回来。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


def _exit(argv):
    """跑一次 ``cli.main``，但允许 argparse 的 ``SystemExit`` 冒出来。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
    return exc.value.code, out.getvalue(), err.getvalue()


def _argv(tmp_path, *extra):
    return ["--pack", str(tmp_path), "--out", str(tmp_path / "o.xlsx"), *extra]


@pytest.fixture
def stub_run(monkeypatch):
    """把 ``pipeline.run`` 换成记录器：只测 CLI 这一层的接线，不跑真流水线。"""
    seen: dict = {}

    def fake(cfg, *, progress=None, client=None, from_stage=None, use_cache=True):
        seen.update(cfg=cfg, client=client, progress=progress,
                    from_stage=from_stage, use_cache=use_cache)
        return PackResult(pack="P1")

    monkeypatch.setattr(cli, "run", fake)
    return seen


# --------------------------------------------------------------------------- #
# 1~2. 参数层
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J3")
@pytest.mark.ec("J7")
def test_bad_switch_values_exit_one_and_list_the_allowed_values(tmp_path):
    """取值域错误是**输入问题**（退出码 1），不是「产不出结果」（退出码 2）。

    ``_validate`` 刻意绕开 argparse 的 ``choices``——``choices`` 会以 2 结束，
    而 2 在这个 CLI 里被保留给招标 PDF 读不出来、xlsx 被 Excel 占用这类情况。
    消息里必须把合法取值列全，用户不用再去翻 ``--help``。
    ``--from-stage`` 的校验在更深一层（``pipeline.stage_index``），但契约相同。
    """
    for flag, bad, allowed in (
        ("--section-policy", "s9", SECTION_POLICIES),
        ("--paren-policy", "nope", PAREN_POLICIES),
        ("--ocr-fallback", "maybe", OCR_FALLBACKS),
        ("--transport", "carrier-pigeon", cli.TRANSPORTS),
    ):
        rc, out, err = _main(_argv(tmp_path, flag, bad, "--offline"))
        assert rc == 1, flag
        assert out == ""                                   # stdout 必须干净
        assert f"× 错误：{flag} 取值 {bad!r} 不支持，可选：{'、'.join(allowed)}。" in err
        assert "Traceback" not in err

    for bad in ("0", "-1.5"):
        rc, out, err = _main(_argv(tmp_path, "--positive-margin", bad, "--offline"))
        assert rc == 1 and out == ""
        assert "--positive-margin 必须是正数" in err and "τ" in err

    rc, out, err = _main(_argv(tmp_path, "--offline", "--from-stage", "9"))
    assert rc == 1 and out == "" and "超出范围" in err
    rc, out, err = _main(_argv(tmp_path, "--offline", "--from-stage", "bogus"))
    assert rc == 1 and out == ""
    assert "scope, vendor, tender, cut, match, judge, write" in err


@pytest.mark.ec("J7")
@pytest.mark.ec("J8")
def test_argparse_layer_exits_two_and_version_exits_zero():
    """退出码分层的另一半：argparse 自己的 2 与 0。

    缺必填参数属于「不会用」，argparse 直接 ``SystemExit(2)`` 并打 usage 到
    stderr；``--version`` 是正常退出（0），版本串走 **stdout**，脚本能直接采集。
    这两条与 (1) 的 1 一起构成完整的三段式退出码契约。
    """
    code, out, err = _exit([])
    assert code == 2 and out == ""
    assert err.startswith("usage: run.py") and "--pack DIR" in err

    code, out, err = _exit(["--version"])
    assert code == 0 and err == ""
    assert out.strip() == f"bidtable {__version__}"


# --------------------------------------------------------------------------- #
# 3~4. 异常收口
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J7")
@pytest.mark.case("N12")
def test_bidtable_errors_collapse_to_one_line_and_their_own_exit_code(monkeypatch, tmp_path):
    """``BidTableError`` 的 ``exit_code`` 是**分层契约**，不能一律压成 1。

    1 = 输入层面的问题（路径、缺文件、参数取值），改一下命令就能重试；
    2 = 跑起来了但产不出结果（招标 PDF 坏、xlsx 被 Excel 占用、模型通道彻底坏），
    需要人去动环境。N12 是 2 里最高频的一个：产物文件正被打开。
    """
    ladder = [
        (errors.InputError("参数不对"), 1),
        (errors.PackNotFoundError("包不在"), 1),
        (errors.PackIncompleteError("缺文件"), 1),
        (errors.VendorDocError("docx 坏了"), 1),
        (errors.TenderParseError("招标 PDF 读不出来"), 2),
        (errors.UnsupportedLayoutError("版面不认识"), 2),
        (errors.OutputError("P1.xlsx 可能正被 Excel 打开"), 2),
        (errors.LLMAuthError("Key 无效"), 2),
        (errors.LLMQuotaError("账户欠费"), 2),
    ]
    for exc, want in ladder:
        def boom(cfg, **kw):
            raise exc

        monkeypatch.setattr(cli, "run", boom)
        rc, out, err = _main(_argv(tmp_path, "--offline"))
        assert rc == want, type(exc).__name__
        assert out == "" and "Traceback" not in err
        assert err.strip().startswith("× 错误：")           # 一行、可读、带下一步


@pytest.mark.ec("J7")
@pytest.mark.ec("J8")
def test_unexpected_errors_stay_loud_and_ctrl_c_exits_130(monkeypatch, tmp_path):
    """两个反面契约：意外异常**不**被吞，Ctrl-C 有专属退出码。

    只捕获 ``BidTableError`` 是刻意的——把 ``ValueError`` 之类也收口成「一行
    中文」等于把 bug 藏起来，用户会以为是自己的输入有问题。130 是 shell 约定
    （128 + SIGINT），CI 靠它区分「跑挂了」和「被人掐了」。
    """
    def bad(cfg, **kw):
        raise ValueError("内部 bug")

    monkeypatch.setattr(cli, "run", bad)
    with pytest.raises(ValueError):                        # 不吞，原样冒出去
        _main(_argv(tmp_path, "--offline"))

    def interrupted(cfg, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run", interrupted)
    rc, out, err = _main(_argv(tmp_path, "--offline"))
    assert rc == 130 and out == ""
    assert err.strip() == "× 错误：已中断，未写出结果。"


# --------------------------------------------------------------------------- #
# 5. 最高频的误用
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J7")
@pytest.mark.ec("J5")
@pytest.mark.case("N1")
def test_missing_pack_names_the_resolved_absolute_path_and_leaves_no_output(tmp_path):
    """``--pack`` 打错一个字是最高频的误用，消息必须给出**绝对路径**。

    用户传进来的往往是相对路径（``训练集/P1``），而进程 cwd 未必是仓库根，
    只回显相对路径等于没解释。这条走真 ``main``（不 stub），顺带钉住
    「失败时不留半截产物」——半个 xlsx 比没有 xlsx 更危险。
    """
    missing = tmp_path / "训练集" / "P9"
    out_xlsx = tmp_path / "o.xlsx"
    rc, out, err = _main(["--pack", str(missing), "--out", str(out_xlsx)])
    assert rc == 1 and out == "" and "Traceback" not in err
    assert f"资料包目录不存在：{missing.resolve()}" in err
    assert "00_投标范围.txt" in err and "--pack" in err     # 给出下一步动作
    assert not out_xlsx.exists()


# --------------------------------------------------------------------------- #
# 6~8. 接线
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J1")
@pytest.mark.ec("J6")
@pytest.mark.ec("J3")
def test_every_switch_lands_on_the_matching_config_field(stub_run, clean_env, tmp_path):
    """CLI 与 ``Config`` 之间是一一映射，接错一根线就会静默改变判定结果。

    ``--positive-margin`` 最危险：没接上的话 τ 会悄悄退回 1.10，正偏离数量
    变了却没有任何报错。``--offline`` 必须**压过**环境里的真 Key；反过来，
    没有 Key 时必须自动离线而不是报错（agent.md 4.4 的兜底列）。
    ``--model-strong`` 故意没有 CLI 开关：仲裁模型只由 ``MODEL_STRONG`` 决定。
    """
    rc, out, err = _main(_argv(
        tmp_path, "--report", str(tmp_path / "r.md"), "--offline",
        "--positive-margin", "1.2", "--section-policy", "s2", "--paren-policy", "fold",
        "--ocr-fallback", "no-response", "--verbose",
        "--base-url", "http://example.test/v1/", "--model-text", "mt",
        "--model-vision", "mv", "--model-ocr", "mo",
    ))
    assert rc == 0 and out.startswith("P1：0 行｜")
    cfg = stub_run["cfg"]
    assert (cfg.positive_margin, cfg.section_policy) == (1.2, "s2")
    assert (cfg.paren_policy, cfg.ocr_fallback) == ("fold", "no-response")
    assert cfg.verbose is True
    assert cfg.base_url == "http://example.test/v1"        # 尾斜杠被 rstrip
    assert (cfg.model_text, cfg.model_vision, cfg.model_ocr) == ("mt", "mv", "mo")
    assert cfg.report == tmp_path / "r.md" and cfg.out == tmp_path / "o.xlsx"
    assert cfg.offline is True and cfg.llm_available is False
    assert stub_run["progress"].quiet is False             # 进度必须真打到 stderr
    assert "--model-strong" not in cli.build_parser().format_help()

    clean_env.setenv("LLM_API_KEY", "sk-unit-test")
    _main(_argv(tmp_path))                                 # 不给 --offline
    cfg = stub_run["cfg"]
    assert cfg.offline is False and cfg.llm_available is True
    assert cfg.key_source == "LLM_API_KEY"

    clean_env.delenv("LLM_API_KEY")
    _main(_argv(tmp_path))                                 # 无 Key：自动降级
    assert stub_run["cfg"].api_key is None
    assert stub_run["cfg"].offline is True


@pytest.mark.ec("J1")
@pytest.mark.ec("J4")
@pytest.mark.ec("J6")
def test_transport_picks_the_channel_and_replay_really_loads_the_corpus(stub_run, tmp_path):
    """``--transport`` 决定走哪条通道，``--replay`` 的语料必须真被读进去。

    三种语义完全不同：``auto`` 在无 Key/离线时降级成 ``None``（客户端
    ``enabled=False``，三个 LLM 钩子全部走确定性回退）；``fake`` 是给测试用的
    桩，**即使 ``--offline`` 也生效**；``replay`` 从 jsonl 命中、未命中即报错。
    ``--replay`` 曾经是个死参数（解析了却没往下传），这条用例把它钉死。
    """
    _main(_argv(tmp_path, "--offline"))
    assert stub_run["client"].transport is None
    assert stub_run["client"].enabled is False

    _main(_argv(tmp_path, "--offline", "--transport", "fake"))
    assert stub_run["client"].transport.name == "fake"
    assert stub_run["client"].enabled is True              # fake 不受 --offline 影响

    _main(_argv(tmp_path, "--offline", "--transport", "replay"))
    default = stub_run["client"].transport
    assert default.name == "replay"
    assert default.path == Path("tests/fixtures/llm/recorded.jsonl")   # 文档里的默认语料

    corpus = tmp_path / "recorded.jsonl"
    corpus.write_text(json.dumps({"key": "k1", "text": "hi"}) + "\n", encoding="utf-8")
    _main(_argv(tmp_path, "--offline", "--transport", "replay", "--replay", str(corpus)))
    tr = stub_run["client"].transport
    assert tr.path == corpus and list(tr.table) == ["k1"]


@pytest.mark.ec("J4")
@pytest.mark.ec("J3")
def test_from_stage_and_no_cache_are_forwarded_verbatim(stub_run, tmp_path):
    """``--from-stage`` 原样透传：阶段名与 1..7 两种写法都交给 ``stage_index``。

    CLI 这一层**不做翻译**是关键：一旦在这里把 ``judge`` 转成 5，缓存键与
    ``pipeline.STAGES`` 就有了两套真相。``--no-cache`` 是 ``store_true``，
    默认必须用缓存——单包重跑约 1.2 秒，全量自测要跑 5 个包。
    """
    _main(_argv(tmp_path, "--offline"))
    assert stub_run["from_stage"] is None and stub_run["use_cache"] is True

    for token in ("judge", "7", "scope", "1", "write"):
        _main(_argv(tmp_path, "--offline", "--from-stage", token))
        assert stub_run["from_stage"] == token             # 不翻译、不在这里校验

    _main(_argv(tmp_path, "--offline", "--no-cache"))
    assert stub_run["use_cache"] is False


def _clone(src: Path, dest: Path, *, gold: bool = False) -> Path:
    """把真实资料包整体拷到 tmp 下，默认摘掉 ``04_标准答案``。

    摘掉标准答案才能测 ``--self-check`` 的降级分支；留着才能测它真的把
    ``grade.py`` 的 JSON 接在摘要行后面。两条路都必须过，所以做成开关。
    """
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if not gold and f.name.startswith("04_"):
            continue
        if f.is_dir():
            shutil.copytree(f, dest / f.name)
        else:
            shutil.copy2(f, dest / f.name)
    return dest


@pytest.fixture(scope="module")
def twin(tmp_path_factory, repo_root):
    """同一份 P1 分别落在纯 ASCII 路径与「中文 + 空格」路径下，各真跑一遍 CLI。

    真实训练集只有 5 个包，跑一次约 1 秒，所以这里舍得真跑：路径编码与
    空格是本项目的真实风险（Windows + 中文目录名），桩掉 ``run`` 就测不到了。
    ``--out`` 故意留两层不存在的目录，顺带覆盖 N11 在 CLI 这一层的表现。
    """
    src = repo_root / "训练集" / "P1"
    if not src.is_dir():
        pytest.skip(f"训练集资料包缺失：{src}")
    base = tmp_path_factory.mktemp("cli")
    runs = {}
    for tag, name in (("ascii", "plain"), ("cjk", "中文 目录 名字")):
        pack = _clone(src, base / name)
        out = base / tag / "out" / "P1.xlsx"
        rep = base / tag / "out" / "P1.review.md"
        rc, so, se = _main(["--pack", str(pack), "--out", str(out), "--report", str(rep),
                            "--offline", "--no-cache"])
        runs[tag] = {"pack": pack, "out": out, "report": rep,
                     "rc": rc, "stdout": so, "stderr": se}
    return runs


@pytest.mark.integration
@pytest.mark.ec("J8")
@pytest.mark.ec("J1")
def test_success_splits_summary_to_stdout_and_progress_to_stderr(twin):
    """成功路径的出口契约：一行摘要进 stdout，七阶段进度与告警进 stderr。

    这是「摘要可被管道消费、过程可被人看」的分界。反过来就会污染
    ``run.py ... | jq`` 这类用法；而 xlsx / 复核报告的父目录由
    ``writer.write`` 与 ``review.write_report`` 自己建，CLI 不负责。
    """
    r = twin["ascii"]
    assert r["rc"] == 0
    assert r["stdout"].count("\n") == 1                     # stdout 只有摘要这一行
    assert r["stdout"].startswith("plain：17 行｜")          # 包名取自目录名
    assert "无偏离=10 正偏离=1 负偏离=3 未响应=3" in r["stdout"]
    assert "关键项风险 1 行｜待复核 15 行｜告警 2 条" in r["stdout"]

    err = r["stderr"]
    stages = [ln for ln in err.splitlines() if ln.startswith("[")]
    prefixes = [s.split("]")[0] + "]" for s in stages]
    assert prefixes == [f"[{i}/7]" for i in range(1, 8)]    # 七个阶段各报一次，顺序不乱
    assert err.count("  ! 告警：") == 2                      # 离线 OCR 的两条诚实告警
    assert "Traceback" not in err

    assert r["out"].is_file() and r["report"].is_file()      # 父目录被自动创建
    wb = load_workbook(r["out"])
    ws = wb.active
    assert ws.title == "响应偏离表"
    assert ws.dimensions == "A1:J18"                         # 1 表头 + 17 数据行 × 10 列
    assert ws.freeze_panes is None                           # 走模板路径：模板不带冻结窗格
    wb.close()
    assert r["report"].read_text(encoding="utf-8").startswith("# 响应偏离表复核报告 · plain")


@pytest.mark.integration
@pytest.mark.ec("J5")
@pytest.mark.case("B22")
def test_cjk_and_space_paths_produce_identical_cells(twin):
    """B22：``--out`` / ``--pack`` 路径含中文与空格时，产物必须逐格一致。

    边界在于 Windows 上路径会经过 ``Path`` → ``str`` → openpyxl → NTFS 四次
    转手，任何一处按本地码本编码就会静默写错文件。这里不只断言「跑通了」，
    而是把两张工作表的 title / dimensions / 全部单元格值摆在一起比。
    复核报告只允许差标题行（包名不同）。时间戳只精确到秒，两次运行完全可能落在
    同一秒里，所以先归一化再比——否则这条断言会随机器快慢随机变红。
    """
    a, c = twin["ascii"], twin["cjk"]
    assert c["rc"] == 0
    assert c["stdout"].startswith("中文 目录 名字：17 行｜")

    def grid(path):
        wb = load_workbook(path)
        ws = wb.active
        got = (ws.title, ws.dimensions, [[x.value for x in row] for row in ws.iter_rows()])
        wb.close()
        return got

    assert grid(a["out"]) == grid(c["out"])

    ra = a["report"].read_text(encoding="utf-8").splitlines()
    rc = c["report"].read_text(encoding="utf-8").splitlines()
    assert len(ra) == len(rc)
    assert ra[0] == "# 响应偏离表复核报告 · plain"
    assert rc[0] == "# 响应偏离表复核报告 · 中文 目录 名字"
    stamp = re.compile(r"- 生成时间：\d{4}-\d\d-\d\d \d\d:\d\d:\d\d")
    assert stamp.fullmatch(ra[2]) and stamp.fullmatch(rc[2]), (ra[2], rc[2])
    ra[2] = rc[2] = "- 生成时间：<归一化>"
    diff = [i for i, (x, y) in enumerate(zip(ra, rc)) if x != y]
    assert diff == [0], f"除标题外还有内容不一致，行号 {diff}"


@pytest.mark.integration
@pytest.mark.ec("J7")
@pytest.mark.ec("J8")
def test_self_check_warns_and_still_exits_zero_without_gold(twin, tmp_path):
    """``--self-check`` 在没有标准答案的包上只告警，仍然退出 0。

    交付现场跑的包几乎都不带 ``04_标准答案_*.xlsx``。如果这里报错退出，
    ``--self-check`` 就变成了「只有在训练集上才能加」的开关，用户宁可不用。
    顺带钉住降级分支不改摘要行：同一份输入两次跑出的 stdout 必须逐字节相同。
    """
    r = twin["ascii"]
    rc, so, se = _main(["--pack", str(r["pack"]), "--out", str(tmp_path / "sc.xlsx"),
                        "--self-check", "--offline", "--no-cache"])
    assert rc == 0
    assert so == r["stdout"]
    assert se.splitlines()[-1].endswith("里没有 04_标准答案_*.xlsx，--self-check 跳过。")
    assert str(r["pack"]) in se.splitlines()[-1]             # 告警里点名了具体目录


@pytest.mark.integration
@pytest.mark.ec("J8")
@pytest.mark.ec("J1")
def test_self_check_appends_grade_json_when_gold_exists(tmp_path, repo_root):
    """``--self-check`` 在有标准答案时，把 grade.py 的 JSON 接在摘要行之后。

    边界是「stdout 变成两种内容的拼接」：第 1 行仍是给人看的摘要，
    第 2 行起必须是**一整块可被 json.loads 的**打分结果。中间不能混进
    进度行——那些都走 stderr。键集合直接对齐 grade.py，防止有人给
    grade.py 加字段时这边静默漏掉。
    """
    src = repo_root / "训练集" / "P1"
    if not src.is_dir():
        pytest.skip(f"训练集资料包缺失：{src}")
    pack = _clone(src, tmp_path / "P1", gold=True)
    rc, so, se = _main(["--pack", str(pack), "--out", str(tmp_path / "sc.xlsx"),
                        "--self-check", "--offline", "--no-cache"])
    assert rc == 0
    lines = so.splitlines()
    assert lines[0].startswith("P1：17 行｜")                 # 包名取自目录名
    payload = json.loads("\n".join(lines[1:]))
    assert set(payload) == {"条目总数", "提交匹配条目", "多余条目", "覆盖率",
                            "偏离类型准确率", "关键项风险识别", "证明材料命中率",
                            "位置命中率", "混淆矩阵"}
    assert payload["条目总数"] == 17
    assert payload["覆盖率"] == 1.0
    assert "Traceback" not in se
