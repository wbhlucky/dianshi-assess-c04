# -*- coding: utf-8 -*-
"""invariants 层：测试套件自己的账本（agent.md 6.9 / 7.5）。

同层的另外两个文件盯的是**产物**：xlsx 的十列语义，以及源码里没有标准答案。
本文件盯的是**测试集本身**，也就是「用例要能反映项目的各个边界能力，
不应是重复的用例叠加」这句话的机器可查形式。

为什么用 AST 静态扫描，而不是在测试里跑 ``pytest --collect-only``：
静态扫描确定、离线、毫秒级，而且它证明的是**标记确实写在源码里**。
被 ``-m`` 反选掉的用例不会因为写在源码里就"这轮跑到了"，但它仍然占着一个编号
——这恰恰是账本要管的事：账本管"这个能力有没有被写下来"，不管这一轮跑没跑到。

四条用例各管一件事，互不重叠：

1. 等价类账本（EC-A..EC-J 共 72 个）必须全部被认领，且不许出现表外编号；
2. 用例账本（T/X/B/N/IV 共 104 个）必须全部被认领，**除逐条说明的白名单**；
3. 标记卫生 + 反重复：每条用例至少一个 ``ec``、至多一个 ``case``，
   且**任一 case 编号被认领不超过 2 次**——这条就是"不许叠加重复用例"的闸门；
4. 分层完整性：每个测试文件都待在自己那一层、都声明了那一层的标记，
   否则 ``-m "not eval and not network and not slow"`` 就会漏选，
   60 秒离线门禁随时可能被一个慢文件打穿。

★ 与 agent.md 的偏差：7.5 写「合计 ≥160 条」，实际是 400 多个测试函数
  （参数化展开后更多）；7.1 的等价类表只到 EC-I，EC-J（J1..J9，工程契约类）
  是施工过程中补出来的诚实扩展，在这里一并作为账本的合法编号。
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import pytest

pytestmark = [pytest.mark.invariants]

TESTS_DIR = Path(__file__).resolve().parents[1]

# agent.md 7.1 的 EC-A..EC-I，加上施工期补出的 EC-J（工程契约类，见模块 docstring）。
_EC_LETTERS = (("A", 4), ("B", 4), ("C", 9), ("D", 8), ("E", 5),
               ("F", 5), ("G", 8), ("H", 12), ("I", 8), ("J", 9))
EC_UNIVERSE = frozenset(f"{letter}{i}" for letter, n in _EC_LETTERS for i in range(1, n + 1))

# agent.md 7.2（典型 T）/ 7.3（混淆 X）/ 7.4（边界 B）/ 6.7（异常 N）/ 6.6（不变量 IV）。
CASE_UNIVERSE = frozenset(
    [f"T{i}" for i in range(1, 19)] + [f"X{i}" for i in range(1, 33)] +
    [f"B{i}" for i in range(1, 26)] + [f"N{i}" for i in range(1, 16)] +
    [f"IV{i}" for i in range(1, 15)]
)

# 「编号在册、但确实不该实现」的例外。必须逐条写清原因，不留空口子。
UNIMPLEMENTABLE = {
    "B11": ("匹配到 0 个标的 → 明确报错 + 退出码 2。实测 lotmap.build_lotmap 一个标的"
            "都没绑上时只出一条告警、照常返回，全流程不会失败。这是刻意的设计：厂商资料"
            "里缺型号是常态，硬报错会让整包做不出来。要满足 B11 就得改产品行为，而那与"
            "§3.8 的容错取向冲突，所以记为例外，而不是硬凑一条假绿的用例把它划掉。"),
}

# 每一层必须在模块级声明的标记；pytest.ini 的 addopts 靠它反选 eval/network/slow。
LAYER_MARKS = {
    "unit": ("unit",),
    "integration": ("integration",),
    "golden": ("golden",),
    "invariants": ("invariants",),
    "eval": ("eval", "slow"),
}

# 非空守卫。glob 或 AST 遍历写错时账本会"全绿"——扫到 0 个文件，
# 于是"没有任何编号缺失"这种断言毫无意义地通过。这两个下限堵掉这个假绿。
_MIN_FILES = 50
_MIN_TESTS = 300
_MAX_CLAIMS_PER_CASE = 2


class Found(NamedTuple):
    rel: str
    name: str
    ecs: tuple[str, ...]
    cases: tuple[str, ...]


def _key(s: str) -> tuple[str, int]:
    """``X10`` -> ("X", 10)，让失败信息按编号而不是按字典序排。"""
    m = re.fullmatch(r"([A-Z]+)(\d+)", s)
    return (m.group(1), int(m.group(2))) if m else (s, 0)


def _attr_chain(node: ast.expr) -> tuple[str, ...] | None:
    """``pytest.mark.ec`` -> ``("pytest", "mark", "ec")``；不是这条链就返回 None。"""
    parts: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return tuple(reversed(parts))
    return None


def _decorator_marks(node: ast.FunctionDef) -> tuple[list[str], list[str]]:
    """收集一条用例的 ``ec`` / ``case`` 标记。

    两种写法都认：一个装饰器带多个参数，以及同类装饰器叠加多层
    （仓库里的约定是「一个装饰器一个参数，多个就叠加」，但账本不该因为
    写法差异而漏记，否则它会反过来限制人怎么写测试）。
    """
    ecs: list[str] = []
    cases: list[str] = []
    for dec in node.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        chain = _attr_chain(dec.func)
        if chain is None or len(chain) != 3 or chain[:2] != ("pytest", "mark"):
            continue
        bucket = ecs if chain[2] == "ec" else cases if chain[2] == "case" else None
        if bucket is None:
            continue
        bucket.extend(a.value for a in dec.args
                      if isinstance(a, ast.Constant) and isinstance(a.value, str))
    return ecs, cases


def _module_marks(tree: ast.Module) -> tuple[str, ...]:
    """模块级 ``pytestmark = ...`` 声明了哪些标记。"""
    out: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            continue
        vals = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
        for v in vals:
            chain = _attr_chain(v)
            if chain is not None and len(chain) == 3 and chain[:2] == ("pytest", "mark"):
                out.append(chain[2])
    return tuple(out)


def _scan() -> tuple[list[Found], dict[str, tuple[str, ...]]]:
    found: list[Found] = []
    marks: dict[str, tuple[str, ...]] = {}
    for path in sorted(TESTS_DIR.rglob("*.py")):
        # conftest 只提供 fixture，不装用例；本文件自己也在扫描范围内（自洽）。
        if path.name == "conftest.py" or not path.name.startswith("test_"):
            continue
        rel = path.relative_to(TESTS_DIR).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        marks[rel] = _module_marks(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                ecs, cases = _decorator_marks(node)
                found.append(Found(rel, node.name, tuple(ecs), tuple(cases)))
    return found, marks


@pytest.fixture(scope="module")
def ledger() -> tuple[list[Found], dict[str, tuple[str, ...]]]:
    found, marks = _scan()
    assert len(marks) >= _MIN_FILES, f"只扫到 {len(marks)} 个测试文件，rglob 可能写错了"
    assert len(found) >= _MIN_TESTS, f"只扫到 {len(found)} 个测试函数，AST 遍历可能写错了"
    return found, marks


def _claims(found: list[Found], which: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for f in found:
        for cid in (f.ecs if which == "ec" else f.cases):
            out[cid].append(f"{f.rel}::{f.name}")
    return dict(out)


@pytest.mark.ec("J9")
def test_every_equivalence_class_in_the_spec_is_claimed_by_at_least_one_test(ledger):
    """EC 账本：7.1 的每个等价类都要有用例认领，且不许出现表外编号。

    边界在哪：等价类是**测试设计的索引**，不是产物属性，所以只能靠标记来追踪。
    缺一个编号意味着某一类输入压根没人验过；多一个表外编号更阴险——
    比如把 ``ec("I12")`` 写进去（EC-I 只到 I8），账本上看着"认领了"，
    实际那一类还是空的。所以缺失和越界都要断言。
    """
    found, _ = ledger
    claimed = _claims(found, "ec")
    missing = sorted(EC_UNIVERSE - set(claimed), key=_key)
    alien = sorted(set(claimed) - EC_UNIVERSE, key=_key)
    assert not missing, f"以下等价类没有任何用例认领：{missing}"
    assert not alien, f"以下 ec 编号不在 7.1／EC-J 的编号表里，写了等于漏掉一类：{alien}"


@pytest.mark.ec("J9")
def test_every_named_case_is_claimed_except_the_documented_allow_list(ledger):
    """用例账本：T/X/B/N/IV 全部认领，只允许白名单里那几条例外。

    边界在哪：白名单是**双向**的。既断言"缺失的编号必须在白名单里"，
    也断言"白名单里的编号必须还真的缺失"——否则哪天 B11 被实现了，
    白名单就变成一张永久免检的死条，掩盖后来新增的缺口。
    """
    found, _ = ledger
    claimed = _claims(found, "case")
    missing = sorted(CASE_UNIVERSE - set(claimed), key=_key)
    alien = sorted(set(claimed) - CASE_UNIVERSE, key=_key)
    assert not alien, f"以下 case 编号不在 7.2／7.3／7.4／6.6／6.7 的表里：{alien}"
    unexpected = [c for c in missing if c not in UNIMPLEMENTABLE]
    assert not unexpected, f"以下用例编号没人认领，也不在 UNIMPLEMENTABLE 白名单里：{unexpected}"
    stale = [c for c in UNIMPLEMENTABLE if c not in missing]
    assert not stale, f"这些例外已经被实现了，请从 UNIMPLEMENTABLE 删掉：{stale}"


@pytest.mark.ec("J9")
def test_marks_are_hygienic_and_no_case_is_claimed_more_than_twice(ledger):
    """标记卫生，以及"同一个编号最多被两条用例认领"的反叠加闸门。

    边界在哪：这条是用户那句「不应是重复的用例叠加」唯一能被机器查的部分。
    一个编号被认领两次是允许的、而且是有意的——比如 B22 在 unit 层比两份克隆、
    在 eval 层比原地与搬家，两条各证明一件对方证不了的事。但第三次认领
    几乎必然是把同一条断言抄了三遍。超过上限时把完整分布打进失败信息，
    好让维护者当场判断是"真的第三种能力"还是"该合并"。

    另外两条卫生规则：没有 ``ec`` 的用例等于没登记等价类，账本就漏了一格；
    一条用例挂两个 ``case`` 则会让"每个编号被认领几次"这个计数失真。
    """
    found, _ = ledger
    bare = [f"{f.rel}::{f.name}" for f in found if not f.ecs]
    assert not bare, f"以下用例没有任何 ec 标记，等价类账本会漏掉它们：{bare}"

    doubled = [f"{f.rel}::{f.name} {f.cases}" for f in found if len(f.cases) > 1]
    assert not doubled, f"一条用例至多挂一个 case 标记，以下违反：{doubled}"

    claimed = _claims(found, "case")
    over = {c: v for c, v in claimed.items() if len(v) > _MAX_CLAIMS_PER_CASE}
    if over:
        detail = "\n".join(
            f"  {c} 被 {len(v)} 条用例认领：\n" + "\n".join(f"    - {x}" for x in v)
            for c, v in sorted(over.items(), key=lambda kv: _key(kv[0]))
        )
        pytest.fail(f"以下编号被重复叠加，请合并或改成不同的能力：\n{detail}")


@pytest.mark.ec("J3")
def test_every_test_file_declares_the_marker_of_the_layer_it_lives_in(ledger):
    """分层完整性：目录决定层级，层级必须在模块级声明出来。

    边界在哪：``pytest.ini`` 的 addopts 是
    ``-m "not eval and not network and not slow"``，60 秒离线门禁完全依赖
    "慢的用例都带了标记"。一个放在 ``tests/eval/`` 却忘了写 ``pytestmark``
    的文件会**静默混进离线套件**——它照样绿，只是把 40 秒的套件拖成几分钟，
    而且没人会在当时注意到。反过来，``tests/unit/`` 下的文件若漏了
    ``unit`` 标记，``-m unit`` 就跑不全，分层报告也就失真。

    ★ 这条不是假想风险：施工期实测 ``tests/unit/test_parse_req.py`` 与
      ``test_parse_val.py``（26 条用例）就没有 ``pytestmark``，是被本用例抓出来补上的。
    """
    _, marks = ledger
    stray: list[str] = []
    silent: list[str] = []
    for rel, declared in marks.items():
        parts = Path(rel).parts
        layer = parts[0] if len(parts) > 1 else ""
        need = LAYER_MARKS.get(layer)
        if need is None:
            stray.append(rel)
            continue
        if not all(m in declared for m in need):
            silent.append(f"{rel} 在 {layer}/ 层，声明了 {declared or '（无）'}，缺 {need}")
    assert not stray, f"以下测试文件不在任何已知分层目录里：{stray}"
    assert not silent, "以下测试文件的层级标记不完整：\n" + "\n".join(silent)

    # 非空守卫：五层都得真的在，且不许冒出来一个账本不认识的新目录。
    seen = {Path(rel).parts[0] for rel in marks if len(Path(rel).parts) > 1}
    assert seen == set(LAYER_MARKS), f"分层目录与账本不一致：{sorted(seen)}"
