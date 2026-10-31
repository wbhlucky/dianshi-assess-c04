# -*- coding: utf-8 -*-
"""eval 层：真跑完整流水线，再用仓库自带的 grade.py 给自己打分（agent.md 6.8 / 10.1）。

前四层全部读**冻结的阶段 fixture**——不碰原始 PDF、不真跑流水线，因此快且稳，
但也因此看不见 fixture 与现实之间的漂移。本层是唯一的真跑层：起子进程执行
``python -m bidtable.selfcheck``，从 ``训练集`` 的原始 PDF/docx/jpg 一路跑到 xlsx，
再调 grade.py 打分。只有本层能证明四件事：

1. 打包好的自测入口在**干净解释器**里能跑通（不靠 pytest 注入的 sys.path）；
2. 10.1 的六道门禁是**实测值**，且唯一不过的那一道被精确定位；
3. CLI 报出来的分数与测试**独立重算**的分数逐字段相同（排除自己给自己放水）；
4. 前四层赖以生存的 5×4 份阶段 fixture、以及 5 份标准答案参考 fixture，
   **今天重跑仍然一模一样**。

代价是慢（两个子进程约 15~20 秒）且必须读原始资料，所以整层挂在 ``eval`` 标记后：
默认 ``pytest`` 不跑，``pytest -m eval`` 才跑（6.9 的门禁之一）。

三条用例各管一件事，互不重复：

1. :func:`test_the_shipped_selfcheck_cli_prints_the_six_gates_and_exits_zero`
   —— 子进程侧的**输出契约**：退出码、stdout 的四个块（摘要 / 表 / 门禁 / 矩阵）、
      JSON 旁挂文件的三个顶层键。门禁文字不在这里重抄一遍，
      而是断言"CLI 打印的必须逐字等于 ``gate_lines`` 现场算出来的"，
      这样阈值只在 golden 层钉一次，两层不会各说各话。
2. :func:`test_rescoring_the_artifacts_independently_meets_every_ten_one_floor`
   —— 分数本身：测试自己拿 grade.py 把 5 份 xlsx 重打一遍，逐项卡 10.1 的下限，
      任何一项不达标就把**完整混淆矩阵**打进失败信息（6.8 第 3 条）。
3. :func:`test_refreezing_all_five_packs_reproduces_every_committed_fixture`
   —— 漂移守卫。单元测试只查了 P4（五个包里最简单的那个：单标的、B 族版面、无分节）；
      这里查全部 5 个，含 P3（71 行 / 17 个分节 / 二级标题）与 P5（4 个标的 / C 族版面）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from bidtable import selfcheck as sc

pytestmark = [pytest.mark.eval, pytest.mark.slow]

PACKS = ("P1", "P2", "P3", "P4", "P5")
# 摘要行报的是**输出行数**；表格「条目」列报的是**计分行数**。
# P4 两者不同（37 vs 36）：多出来那行是买方引导句，标准答案标了「不判分」。
OUTPUT_ROWS = {"P1": 17, "P2": 25, "P3": 71, "P4": 37, "P5": 56}
SCORED_ROWS = {"P1": 17, "P2": 25, "P3": 71, "P4": 36, "P5": 56}
N_TOTAL = 205
_SUMMARY_RE = r"{}：(\d+) 行｜.*｜关键项风险 (\d+) 行｜待复核 (\d+) 行｜告警 (\d+) 条"
_MATRIX_HEAD = "混淆矩阵（标准答案 -> 本项目）："
_FAILING_GATE = ("关键项风险识别 = 满分", "4/5")


def _explain(tag: str, score: dict[str, Any]) -> str:
    """门禁不达标时把完整混淆矩阵打出来（6.8 第 3 条）。

    光说「0.836 不达标」没法定位：得知道是哪一类被判成了哪一类。
    """
    head = {k: v for k, v in score.items() if k != "混淆矩阵"}
    body = "\n".join(f"    {k:<16} {v}" for k, v in score["混淆矩阵"].items())
    return (f"{tag} 未达 agent.md 10.1 下限：\n"
            f"{json.dumps(head, ensure_ascii=False, indent=1)}\n"
            f"{_MATRIX_HEAD}\n{body}")


@pytest.fixture(scope="module")
def selfscore(tmp_path_factory, repo_root, child_run, training_dir):
    """真跑一次 ``python -m bidtable.selfcheck``：5 个包从原始 PDF 一路到打分。

    ``--no-cache`` 是刻意的。阶段缓存以 ``pack_hash``（文件名/大小/mtime）为键，
    **代码改了而资料没改时它照样命中**，于是"自测成绩"可能来自旧代码的产物。
    自测这条命令是 6.9 的门禁，门禁不许吃缓存。
    ``--out-dir`` 指到 tmp（中间 xlsx 是一次性的），``--json`` 指到仓库的
    ``out/selfscore.json``（README 第 4 节直接贴这一份，所以它必须是真产物）。
    """
    art = tmp_path_factory.mktemp("selfscore")
    json_out = repo_root / "out" / "selfscore.json"
    p = child_run(["-m", "bidtable.selfcheck", "--packs", "训练集",
                   "--out-dir", str(art), "--json", str(json_out), "--no-cache"])
    if p.returncode != 0:
        pytest.fail(f"selfcheck 子进程退出码 {p.returncode}\n{p.stderr[-4000:]}")
    return {"proc": p, "json_path": json_out,
            "payload": json.loads(json_out.read_text(encoding="utf-8")),
            "artifacts": {pack: art / f"{pack}.xlsx" for pack in PACKS}}


@pytest.fixture(scope="module")
def refrozen(tmp_path_factory, child_run, training_dir):
    """真跑一次 ``--freeze``：把 5 个包的阶段产物与标准答案参考重新导出一份到 tmp。"""
    dest = tmp_path_factory.mktemp("refreeze")
    p = child_run(["-m", "bidtable.selfcheck", "--packs", "训练集", "--freeze",
                   "--fixtures-dir", str(dest / "packs"), "--gold-dir", str(dest / "gold")])
    if p.returncode != 0:
        pytest.fail(f"--freeze 子进程退出码 {p.returncode}\n{p.stderr[-4000:]}")
    return dest


@pytest.mark.ec("J7")
@pytest.mark.ec("J8")
@pytest.mark.ec("J1")
def test_the_shipped_selfcheck_cli_prints_the_six_gates_and_exits_zero(selfscore):
    """10.2 第 2 条：自测入口在干净解释器里退出码 0，且 stdout 的四个块各司其职。

    边界在哪：三处。

    第一处是**stdout 与 stderr 的分工**。stdout 只有可被脚本消费的东西
    （5 行摘要 + 表 + 门禁 + 矩阵），阶段进度与告警全在 stderr。
    所以这里断言 ``stderr.count("[1/7]") == 5``——五个包各报一次第一阶段，
    证明真的把五个包都跑了一遍，而不是命中了什么缓存后只打了表。

    第二处是**同一份数据在两个块里的口径不同**。摘要行报输出行数，
    表格「条目」列报计分行数；P4 是 37 vs 36。两个数字都钉住，
    就堵死了"把不判分那行算进覆盖率分母"这类改动。

    第三处是**门禁文字不许在这里重抄**。阈值字符串已经在 golden 层逐字钉过一次
    （``test_gold_all_keys``），再抄一遍就成了两处各自维护的同一个契约。
    这里改成断言"CLI 打印的每一行都等于 ``gate_lines`` 现场算出来的那一行"，
    于是阈值只在 golden 层有一份真值，本层只验渲染与 JSON 旁挂文件是否忠实。

    真实资料包实例：``训练集`` 全部五个包，205 条要求。

    与 agent.md 的偏差：10.1 期望六道门禁全过；离线上限是 5 过 1 不过，
    不过的恒为 ``关键项风险识别 = 满分``（实测 4/5，原因见 test_gold_P5）。
    """
    proc = selfscore["proc"]
    so = proc.stdout.splitlines()
    assert "Traceback" not in proc.stderr
    assert proc.stderr.count("[1/7]") == len(PACKS)          # 五个包都真跑了

    # 块 1：5 行收尾摘要。包名取自目录名，行数 == 输出行数。
    for pack, ln in zip(PACKS, so[:len(PACKS)]):
        m = re.fullmatch(_SUMMARY_RE.format(pack), ln)
        assert m, ln
        assert int(m.group(1)) == OUTPUT_ROWS[pack], pack

    # 块 2：成绩表。用 split() 取字段，不钉列宽（列宽对齐由单元测试管）。
    table = [ln for ln in so if ln.split(" ")[0] in (*PACKS, "合计")]
    assert len(table) == len(PACKS) + 1
    for pack, ln in zip(PACKS, table):
        cells = ln.split()
        assert cells[0] == pack
        assert int(cells[1]) == SCORED_ROWS[pack], pack       # 计分口径，不是 xlsx 行数
    assert table[-1].split()[0] == "合计"
    assert int(table[-1].split()[1]) == N_TOTAL

    # 块 3：六道门禁。逐字等于 gate_lines 现场算出来的结果。
    payload = selfscore["payload"]
    gates = sc.gate_lines(payload["total"], [payload["per_pack"][k] for k in PACKS])
    assert len(gates) == 6
    assert [ln for ln in so if ln.startswith("  [")] == [
        f"  [{'PASS' if ok else 'FAIL'}] {desc}\u3000实测 {got}" for desc, ok, got in gates
    ]
    assert [(g["项"], g["通过"], g["实测"]) for g in payload["gates"]] == list(gates)
    assert [(d, g) for d, ok, g in gates if not ok] == [_FAILING_GATE]

    # 块 4：混淆矩阵。行数与格数跟着实测走，但总和恒为 205。
    assert _MATRIX_HEAD in so
    matrix = so[so.index(_MATRIX_HEAD) + 1:]
    conf = payload["total"]["混淆矩阵"]
    assert [ln.split()[0] for ln in matrix] == list(conf)
    assert sum(int(ln.split()[-1]) for ln in matrix) == N_TOTAL

    # JSON 旁挂文件：三个顶层键，per_pack 五包齐全。
    assert sorted(payload) == ["gates", "per_pack", "total"]
    assert sorted(payload["per_pack"]) == list(PACKS)


@pytest.mark.ec("J3")
@pytest.mark.ec("J9")
def test_rescoring_the_artifacts_independently_meets_every_ten_one_floor(selfscore, training_dir):
    """6.8 第 2~4 条：测试自己拿 grade.py 把 5 份产物重打一遍，逐项卡 10.1 的下限。

    边界在哪：**两条独立路径必须给出同一组数字**。
    CLI 内部算的分写进了 JSON，本测试则绕过 CLI、直接对同一批 xlsx 调
    ``score_pair`` + ``aggregate``。两者逐字段相等，才排除了
    "selfcheck 挑了个好看的口径报给自己"。这也顺带把 O5 落到实处：
    我们是**加载**仓库那份 grade.py（``load_grade``），不是抄它的算法，
    所以 grade.py 一改口径，这条断言立刻跟着变。

    分包下限只卡覆盖率与偏离类型，**不卡材料/位置**。原因是 10.1 的分包门禁
    原文只写了「每个包覆盖率 ≥0.90」，而材料/位置的分母是"标准答案非未响应"的行数，
    单包样本太小（P3 实测 0.866 / 0.836 就低于总分下限）；这两项的下限按 10.1
    定义在总分上，P3 的分包缺口由 ``test_gold_P3`` 单独钉住，这里不重复。
    「P3/P5 不得显著低于均值」同理，已在 golden 层用 ``总分-最差包 ≤0.06`` 表达。

    真实资料包实例：``训练集`` 全部五个包。

    与 agent.md 的偏差：证明材料命中率的冲刺线（≥0.95）未达，实测 0.941；
    本测试只卡基础线，冲刺线用一条独立断言记录"偏离类型已达 0.92"这一事实。
    """
    payload = selfscore["payload"]
    mine: dict[str, dict[str, Any]] = {}
    for pack in PACKS:
        sub = selfscore["artifacts"][pack]
        assert sub.is_file(), f"selfcheck 没有产出 {pack} 的 xlsx：{sub}"
        gold = sc.find_gold(training_dir / pack)
        assert gold is not None, pack
        mine[pack] = sc.score_pair(gold, sub)
        assert mine[pack] == payload["per_pack"][pack], _explain(pack, mine[pack])

    total = sc.aggregate([mine[p] for p in PACKS])
    assert total == payload["total"], _explain("合计", total)

    for pack in PACKS:                                     # 10.1 分包门禁
        s = mine[pack]
        assert float(s["覆盖率"]) >= 0.90, _explain(pack, s)
        assert float(s["偏离类型准确率"]) >= 0.85, _explain(pack, s)
        assert int(s["多余条目"]) <= sc.MAX_EXTRA, _explain(pack, s)

    assert float(total["覆盖率"]) >= 0.95, _explain("合计", total)
    assert float(total["偏离类型准确率"]) >= 0.85, _explain("合计", total)
    assert float(total["偏离类型准确率"]) >= 0.92            # 冲刺线：已达
    assert float(total["证明材料命中率"]) >= 0.90, _explain("合计", total)
    assert float(total["位置命中率"]) >= 0.85, _explain("合计", total)
    assert total["关键项风险识别"].startswith("4/5")          # 离线上限，见 test_gold_P5

    # 6.8 第 4 条：这份 JSON 就是 README 第 4 节要贴的成绩单，必须在仓库里、可解析。
    assert selfscore["json_path"].is_file()
    on_disk = json.loads(selfscore["json_path"].read_text(encoding="utf-8"))
    assert on_disk["total"] == total


@pytest.mark.ec("J4")
@pytest.mark.ec("J9")
def test_refreezing_all_five_packs_reproduces_every_committed_fixture(refrozen, repo_root):
    """漂移守卫：今天重跑一遍，25 份已提交的 fixture 必须逐字段复现。

    边界在哪：前四层测试全部站在 ``tests/fixtures/`` 上，
    而 fixture 是**某个时刻的快照**。代码改了、快照没重导，
    四百多条离线用例就会齐刷刷地绿在昨天的行为上——这是最难发现的一类失效，
    因为没有任何一条用例会红。所以必须有一条用例定期把快照重算一遍比对。

    第二处边界更关键，是**防作弊**（O10 / J9）。
    ``tests/fixtures/gold/P*.json`` 是 golden 层的评分参照。
    如果有人手改这份文件，所有 golden 用例就会对着一份被篡改的标准答案打分，
    而且全绿。这里从 ``04_标准答案_*.xlsx`` 用 grade.py 自己的 ``load()``
    重新导出一份来比，等于给答案本身上了一把锁：
    参照只能来自仓库里那份原始 xlsx，不能来自任何人的手。

    ``template_xlsx`` 只比文件名：它存的是调用方给的路径，
    相对还是绝对取决于 cwd，比全路径会把用例绑死在某台机器上。

    真实资料包实例：``训练集`` 全部五个包 × (4 个阶段 + 1 份标准答案) = 25 份。

    与 agent.md 的偏差：6.2 写的 fixture 文件名是
    ``{scope,tender,vendor,evidence,match,rows}.json``；实际是**四个可缓存阶段**
    ``{scope,vendor,tender,cut}.json``——``freeze_pack`` 直接复用 ``StageCache``，
    不另维护第二套序列化代码，所以 fixture 的形状与线上缓存必然一致。
    另外单元测试只查了 P4 一个包，本条查全部五个。
    """
    committed = repo_root / "tests" / "fixtures"
    checked = 0
    for pack in PACKS:
        fresh = refrozen / "packs" / pack
        assert fresh.is_dir(), f"--freeze 没有产出 {pack}：{fresh}"
        names = sorted(f.name for f in fresh.glob("*.json"))
        assert names == ["cut.json", "scope.json", "tender.json", "vendor.json"], names
        for name in names:
            got = json.loads((fresh / name).read_text(encoding="utf-8"))
            old_path = committed / "packs" / pack / name
            assert old_path.is_file(), old_path
            want = json.loads(old_path.read_text(encoding="utf-8"))
            for d in (got, want):
                if "template_xlsx" in d:
                    d["template_xlsx"] = Path(d["template_xlsx"]).name
            assert got == want, f"{pack}/{name}：阶段产物与已提交 fixture 不一致（漂移）"
            checked += 1

        new_gold = refrozen / "gold" / f"{pack}.json"
        old_gold = committed / "gold" / f"{pack}.json"
        assert new_gold.is_file() and old_gold.is_file(), pack
        assert json.loads(new_gold.read_text(encoding="utf-8")) == \
            json.loads(old_gold.read_text(encoding="utf-8")), \
            f"{pack}：标准答案参考与 04_标准答案 不一致（golden 层的评分参照被动过）"
        checked += 1

    assert checked == 25                                   # 非空转：5 包 × 5 份
