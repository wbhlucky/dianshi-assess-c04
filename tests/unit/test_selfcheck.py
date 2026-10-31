"""``bidtable/selfcheck.py``：用仓库里那份真 ``grade.py`` 给自己打分。

这里盯的是"考核口径不能自己发明"这条硬规矩（agent.md 6.8 / O5），以及把
grade.py 的逐包输出汇成总分时的**分母**——分母写错，分数就会假高。

1. ``repo_root`` / ``load_grade``：动态加载真 grade.py，找不到就退出码 1
2. ``find_gold`` / ``pack_dirs``：Excel 锁文件与"包目录 vs 父目录"两种入参
3. ``_parse_star`` / ``_dw`` / ``_pad``：CJK 显示宽度算术
4. ``render_table``：每一行的**显示宽度**必须完全相同（终端表格对齐）
5. ``aggregate``：偏离准确率用混淆矩阵对角线，材料/位置用 ``judged`` 加权
6. ``gate_lines``：6 条门禁的顺序、冲刺标签、覆盖率的"每包 ≥0.90"合取
7. ``score_pair``：拿标准答案给自己打分必须满分（真 grade.py，不复制口径）
8. ``freeze_pack`` / ``export_gold``：冻结产物与已提交 fixture 同构
9. ``main`` 的三条失败路径：退出码 1 + stderr 一行中文 + stdout 全空
10. ``main`` 的成功路径：表格、混淆矩阵、``--json``、``--strict`` 退出码
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import unicodedata

import pytest

from bidtable import errors
from bidtable import selfcheck as sc

pytestmark = [pytest.mark.unit]

FIXTURES = sc.repo_root() / "tests" / "fixtures"


def _score(n, hit, extra, star, file_rate, loc_rate, matrix):
    """造一份 grade.py 形状的逐包分数（键名与真 grade.py 完全一致）。"""
    return {
        "条目总数": n, "提交匹配条目": hit, "多余条目": extra,
        "覆盖率": round(hit / n, 3) if n else 0.0,
        "偏离类型准确率": 0.0,
        "关键项风险识别": f"{star[0]}/{star[1]}（★/▲且为负偏离或未响应的条目中被正确识别的数量）",
        "证明材料命中率": file_rate, "位置命中率": loc_rate,
        "混淆矩阵": matrix,
    }


A = _score(10, 9, 1, (1, 2), 0.5, 1.0,
           {"无偏离->无偏离": 5, "负偏离->无偏离": 2,
            "未响应->未响应": 2, "无偏离->缺行": 1})
B = _score(6, 6, 0, (2, 2), 1.0, 0.75,
           {"无偏离->无偏离": 4, "正偏离->正偏离": 1, "未响应->未响应": 1})


# --------------------------------------------------------------------------- #
# 1. grade.py 的动态加载
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J3")
@pytest.mark.ec("J7")
def test_load_grade_imports_the_real_file_and_repo_root_explains_when_it_cannot(tmp_path):
    """O5 的实现证据：我们是**加载**那份 grade.py，不是抄它的逻辑。

    ``load_grade`` 返回的模块必须自带 ``main`` 与 ``load``，且结果被缓存成同
    一个对象（否则每个包都要 exec 一遍 3.7KB 的评分代码）。找不到 grade.py
    是输入问题（退出码 1），消息里要给出 ``--repo`` 这个出路。
    """
    mod = sc.load_grade()
    assert mod.__name__ == "bidtable_grade"
    assert callable(mod.main) and callable(mod.load)
    assert sc.load_grade() is mod                     # 缓存，不重复 exec
    assert mod.__file__.endswith("grade.py")           # 加载的确实是仓库根那一份
    assert sc.repo_root().joinpath("grade.py").is_file()

    with pytest.raises(errors.InputError) as exc:
        sc.repo_root(tmp_path)                        # tmp 及其父目录都没有 grade.py
    assert exc.value.exit_code == 1
    assert "--repo" in str(exc.value)


# --------------------------------------------------------------------------- #
# 2. 资料包发现
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J5")
@pytest.mark.ec("J7")
def test_find_gold_and_pack_dirs_tolerate_the_shapes_users_actually_pass(tmp_path, pack_path):
    """``--packs`` 既能指向单个包，也能指向父目录；``04_`` 缺失只告警不致命。

    ``~$`` 前缀是 Excel 打开文件时留下的锁文件，它匹配 ``04_*.xlsx`` 的 glob
    却读不出内容——不滤掉就会以一个 openpyxl 异常收场。
    """
    real = pack_path("P1")
    assert sc.find_gold(real).name.startswith("04_")

    syn = tmp_path / "pack"
    syn.mkdir()
    assert sc.find_gold(syn) is None                  # 没有标准答案 -> None，不抛
    (syn / "~$04_标准答案.xlsx").write_bytes(b"lock")  # Excel 锁文件必须被跳过
    assert sc.find_gold(syn) is None
    (syn / "04_标准答案_响应偏离表.xlsx").write_bytes(b"x")
    assert sc.find_gold(syn).name == "04_标准答案_响应偏离表.xlsx"

    assert sc.pack_dirs(real) == [real]               # 自己就是包
    assert [d.name for d in sc.pack_dirs(real.parent)] == ["P1", "P2", "P3", "P4", "P5"]
    assert sc.pack_dirs(tmp_path) == []               # 都不是包 -> 空列表，由 main 报错


# --------------------------------------------------------------------------- #
# 3. 宽度算术
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J3")
@pytest.mark.ec("J8")
def test_display_width_helpers_count_cjk_as_two_columns():
    """表格对齐的全部秘密：``east_asian_width`` 落在 W/F 的字符算 2 列。

    ``★``/``▲`` 属于 East Asian **Ambiguous**，按 1 列算。在把 Ambiguous 渲染成
    双宽的终端里这张表会歪一格——这是已知取舍，写进 README 已知问题。
    ``_parse_star`` 则是把 grade.py 那句带中文括号解释的 ``"4/5（…）"``
    拆回两个整数；解析不出来时退回 ``(0, 0)`` 而不是抛——门禁会因此判 FAIL，
    这比崩溃更容易被发现。
    """
    assert sc._parse_star("4/5（★/▲且为负偏离或未响应的条目中被正确识别的数量）") == (4, 5)
    assert sc._parse_star("0/0（x）") == (0, 0)
    assert sc._parse_star("3/3") == (3, 3)
    assert sc._parse_star("garbage") == (0, 0)

    assert sc._dw("资料包") == 6
    assert sc._dw("P1") == 2
    assert sc._dw("★") == 1                            # Ambiguous，不是 W
    assert sc._dw("") == 0
    assert sc._pad("P1", 6) == "P1    "
    assert sc._pad("P1", 6, right=True) == "    P1"
    assert sc._pad("资料包", 4) == "资料包"          # 已经超宽就不再补空格


# --------------------------------------------------------------------------- #
# 4. 表格
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J8")
def test_render_table_pads_every_line_to_the_same_display_width():
    """产品感验收（10.3）要求这张表在终端里是齐的。

    断言"每行显示宽度相同"比断言具体列宽更硬：它同时管住了表头、分隔线、
    数据行、合计行，也管住了包名长短不一（``PP2长名字``）这种真实情况。
    """
    total = sc.aggregate([A, B])
    table = sc.render_table([("P1", A), ("PP2长名字", B)], total)
    lines = table.splitlines()
    assert len(lines) == 6                            # 表头 + 分隔 + 2 包 + 分隔 + 合计
    widths = {sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in ln)
              for ln in lines}
    assert len(widths) == 1, (widths, lines)
    assert lines[0].split()[:2] == ["资料包", "条目"]
    assert lines[1] == lines[4] and set(lines[1]) == {"-", " "}
    assert "合计" in lines[5] and "PP2长名字" in lines[3]


# --------------------------------------------------------------------------- #
# 5. 汇总
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J3")
@pytest.mark.ec("J8")
def test_aggregate_uses_the_right_denominator_for_each_metric():
    """三个分母各不相同，这是自测最容易"假高分"的地方。

    覆盖率 / 偏离准确率的分母是**条目总数**（含未响应），偏离准确率直接数
    混淆矩阵的对角线；材料与位置命中率的分母是 ``judged``（**扣掉 gold 里
    未响应的行**，因为那些行没有证明材料可比）。用 16 条、13 条 judged 的
    合成数据把三个分母同时钉死。
    """
    total = sc.aggregate([A, B])
    assert total["条目总数"] == 16
    assert total["提交匹配条目"] == 15
    assert total["多余条目"] == 1
    assert total["覆盖率"] == 0.938                   # 15/16
    assert total["偏离类型准确率"] == 0.812            # 对角线 13 / 16（含"缺行"格）
    # judged = 16 - 3 条未响应 = 13；(0.5*8 + 1.0*5)/13
    assert total["证明材料命中率"] == 0.692
    assert total["位置命中率"] == 0.904                # (1.0*8 + 0.75*5)/13
    assert total["关键项风险识别"].startswith("3/4（")  # 分子分母分别累加，不是取平均
    assert list(total["混淆矩阵"]) == sorted(total["混淆矩阵"])   # 键排序，输出可复现
    assert total["混淆矩阵"]["无偏离->无偏离"] == 9
    assert total["混淆矩阵"]["无偏离->缺行"] == 1       # 漏抽在矩阵里是显式的一格

    assert sc.aggregate([])["覆盖率"] == 0.0           # 空输入不除零


# --------------------------------------------------------------------------- #
# 6. 门禁
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J3")
@pytest.mark.ec("J7")
def test_gate_lines_reports_all_six_gates_in_a_fixed_order():
    """门禁顺序与措辞是对外的验收契约（agent.md 10.1），不能随手改。

    四个细节：覆盖率那条是**合取**（总分 ≥0.95 且最差包 ≥0.90）；冲刺线只在
    真达到时才追加"已达"；关键项风险在分母为 0 时必须判 FAIL 而不是"0/0 满分"；
    ``per_pack`` 为空时"最差包"退回总分而不是 0（否则单包自测会被自己判死）。
    """
    total = sc.aggregate([A, B])
    gates = sc.gate_lines(total, [A, B])
    assert [d.split()[0] for d, _, _ in gates] == [     # 三条比较符 ≥ = ≤ 各不相同
        "覆盖率", "偏离类型准确率", "证明材料命中率", "位置命中率",
        "关键项风险识别", "每包多余条目",
    ]
    by_name = {d: (ok, got) for d, ok, got in gates}
    cov = next(d for d in by_name if d.startswith("覆盖率"))
    assert by_name[cov] == (False, "0.938／最差包 0.900")
    dev = next(d for d in by_name if d.startswith("偏离类型准确率"))
    assert "冲刺 ≥0.92" in dev and "已达" not in dev    # 0.812 连基础线都没过
    loc = next(d for d in by_name if d.startswith("位置命中率"))
    assert by_name[loc] == (True, "0.904") and "已达" not in loc   # 过基础线、没到冲刺线
    assert by_name["关键项风险识别 = 满分"] == (False, "3/4")
    assert by_name[f"每包多余条目 ≤{sc.MAX_EXTRA}（不计分）"] == (True, "最差包 1")

    # 满分样本：三条冲刺线都追加"已达"，关键项风险仍然 FAIL（分母 0）
    perfect = sc.aggregate([_score(4, 4, 0, (0, 0), 1.0, 1.0, {"无偏离->无偏离": 4})])
    sprint = {d: (ok, got) for d, ok, got in sc.gate_lines(perfect, [])}
    assert all("已达" in d for d in sprint if d.startswith(("偏离类型", "证明材料", "位置")))
    assert sprint["覆盖率 ≥0.95（每包 ≥0.90）"] == (True, "1.000／最差包 1.000")
    assert sprint["关键项风险识别 = 满分"] == (False, "0/0")   # 避免"没有关键项"蒙混过关
    assert sprint[f"每包多余条目 ≤{sc.MAX_EXTRA}（不计分）"] == (True, "最差包 0")


# --------------------------------------------------------------------------- #
# 7. 真 grade.py 打分
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.ec("J1")
@pytest.mark.ec("J8")
def test_score_pair_gives_a_perfect_score_to_the_gold_file_itself(pack_path):
    """自检的"零点校准"：标准答案跟自己比必须每一项都满分。

    这条用例同时证明了两件事——我们没有偷偷改 grade.py 的口径（O5），
    以及 ``score_pair`` 确实是在**执行**它的 ``main()`` 并解析 stdout 的 JSON，
    而不是自己算了一遍。
    """
    gold = sc.find_gold(pack_path("P1"))
    score = sc.score_pair(gold, gold)
    assert score["条目总数"] == score["提交匹配条目"] == 17
    assert score["多余条目"] == 0
    assert score["覆盖率"] == score["偏离类型准确率"] == 1.0
    assert score["证明材料命中率"] == score["位置命中率"] == 1.0
    assert score["关键项风险识别"].startswith("1/1")
    matrix = score["混淆矩阵"]
    assert all(a == b for a, _, b in (k.partition("->") for k in matrix))   # 全在对角线上
    assert sum(matrix.values()) == 17


# --------------------------------------------------------------------------- #
# 8. 冻结 fixture
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.ec("J4")
@pytest.mark.ec("J1")
def test_freeze_pack_reproduces_the_committed_stage_fixtures(pack_path, tmp_path):
    """阶段 fixture 是全部离线测试的地基，它必须可复现且与线上缓存同形。

    ``freeze_pack`` 的实现就是把 ``StageCache`` 的根指到 fixture 目录，所以
    这里比对的正是"今天重跑一遍流水线，前四阶段的产物有没有变"。
    ``template_xlsx`` 存的是**调用方给的那个路径**（相对或绝对取决于 cwd），
    所以只比它的文件名。
    """
    files = sc.freeze_pack(pack_path("P4"), tmp_path / "fx")
    assert [f.name for f in files] == ["cut.json", "scope.json", "tender.json", "vendor.json"]

    committed = FIXTURES / "packs" / "P4"
    for f in files:
        got = json.loads(f.read_text(encoding="utf-8"))
        want = json.loads((committed / f.name).read_text(encoding="utf-8"))
        for d in (got, want):
            if "template_xlsx" in d:
                d["template_xlsx"] = d["template_xlsx"].replace("\\", "/").rsplit("/", 1)[-1]
        assert got == want, f.name

    assert sc.export_gold(tmp_path, tmp_path / "gold") is None      # 没有 04_ 就返回 None
    out = sc.export_gold(pack_path("P4"), tmp_path / "gold")
    assert out.name == "P4.json"
    table = json.loads(out.read_text(encoding="utf-8"))
    assert len(table) == 37                                          # P4 标准答案 37 行
    assert all("|" in k for k in table)                              # 键是 标的|分节|序号
    assert set(next(iter(table.values()))) == {
        "lot", "sec", "no", "star", "req", "resp", "dev", "file", "loc"}


# --------------------------------------------------------------------------- #
# 9~10. CLI
# --------------------------------------------------------------------------- #


def _main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = sc.main(argv)
    return rc, out.getvalue(), err.getvalue()


@pytest.mark.ec("J7")
@pytest.mark.ec("J8")
@pytest.mark.integration
def test_main_fails_with_one_readable_line_and_an_empty_stdout(tmp_path, pack_path):
    """三条失败路径都必须是"退出码 1 + stderr 一行中文 + stdout 干净"。

    stdout 干净很重要：它是给 ``| tee`` 和 JSON 消费方用的，混进错误信息
    就会让下游解析崩掉。第三条路径（包能跑但没有标准答案）尤其容易漏——
    流水线已经成功了，缺的是打分对象。
    """
    rc, out, err = _main(["--packs", str(tmp_path / "没有这个目录"), "-q"])
    assert rc == 1 and out == ""
    assert "× 错误：--packs 指向的路径不存在" in err

    rc, out, err = _main(["--packs", str(tmp_path), "-q"])
    assert rc == 1 and out == ""
    assert "没有找到任何资料包" in err and "00_投标范围.txt" in err

    hollow = tmp_path / "NG"
    hollow.mkdir()
    for f in pack_path("P1").iterdir():                 # 目录也要拷：02_厂商资料/
        if f.name.startswith("04_"):
            continue
        if f.is_dir():
            shutil.copytree(f, hollow / f.name)
        else:
            shutil.copy2(f, hollow / f.name)
    rc, out, err = _main(["--packs", str(hollow), "-q", "--out-dir", str(tmp_path / "o")])
    assert rc == 1
    assert out.startswith("NG：17 行｜")               # 流水线跑完了，摘要照打
    assert "没有任何资料包带标准答案" in err
    assert "Traceback" not in err


@pytest.mark.integration
@pytest.mark.ec("J8")
@pytest.mark.ec("J3")
def test_main_prints_the_score_table_and_honours_strict_and_json(pack_path, tmp_path):
    """成功路径的完整产物：摘要行、对齐表格、门禁、混淆矩阵、``--json``。

    用 P1 单包（全门禁通过）验证 ``--strict`` 在达标时返回 0；
    ``--json`` 的三个顶层键是 CI 要消费的契约。
    """
    js = tmp_path / "nested" / "selfscore.json"
    rc, out, err = _main(["--packs", str(pack_path("P1")), "-q", "--no-cache", "--strict",
                          "--out-dir", str(tmp_path / "o"), "--json", str(js)])
    assert rc == 0, err
    assert (tmp_path / "o" / "P1.xlsx").is_file()
    assert out.startswith("P1：17 行｜")
    assert "资料包" in out and "合计" in out
    assert out.count("[PASS]") == 6 and "[FAIL]" not in out
    assert "混淆矩阵（标准答案 -> 本项目）：" in out
    assert "正偏离->无偏离" in out                      # P1 唯一那处偏离误判要显式可见

    payload = json.loads(js.read_text(encoding="utf-8"))   # --json 的父目录自动创建
    assert sorted(payload) == ["gates", "per_pack", "total"]
    assert list(payload["per_pack"]) == ["P1"]
    assert len(payload["gates"]) == 6
    assert all(g["通过"] for g in payload["gates"])
    assert set(payload["gates"][0]) == {"项", "通过", "实测"}
    assert payload["total"]["覆盖率"] == 1.0
    assert payload["total"]["偏离类型准确率"] == 0.941
