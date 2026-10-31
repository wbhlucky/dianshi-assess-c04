"""防作弊与密钥卫生门禁（agent.md O10 / §9 / §10.2 第 5 条）。

这一层不是「再测一遍功能」，它测的是**项目边界本身有没有被突破**。O10 写得很直白：
绝不把训练集的标准答案埋进代码。可这句话只有变成可执行的断言才有约束力，否则它就
只是一句表态。所以这里把「泄题」拆成四种互相独立、都能被机器判定的形态：

1.  答案字面值出现在**产品代码**里（``bidtable/``、``run.py``、``tools/``）——最直接的作弊；
2.  只有 OCR 才拿得到的实测值出现在**任何 .py** 里——离线环境物理上取不到这个值，
    它出现在代码里只可能是抄来的；
3.  运行时**真的去打开**了 ``04_标准答案_*.xlsx``——比 grep 强，因为 grep 只能证明
    代码里没写死路径，证明不了运行时没绕过去读；
4.  API Key 以字面值形式落进了仓库。

一个必须说清楚的边界：**招标要求（原文）不算答案**。它是招标 PDF 里的公开输入，
任何人都能从 ``01_招标文件_*.pdf`` 读到；把它算成泄题，测试就只能用无意义的假字符串，
反而丧失覆盖真实数据形态的能力。所以本文件的比对语料只取标准答案的
``投标响应`` 与 ``证明材料文件`` 两列——那两列才是「做了工作才能得到」的部分。

用例清单：

1.  产品代码里没有任何答案字面值
2.  任何 .py 里都没有「只能靠 OCR 得到」的实测值
3.  离线天花板那个具体实测值没有被写死
4.  仓库里没有任何 API Key 字面值
5.  .gitignore 挡住了密钥与运行产物
6.  只有 selfcheck / run.py 能碰到打分器
7.  discover() 把标准答案直接丢在地上
8.  一次完整离线运行从头到尾没有打开过标准答案
"""

from __future__ import annotations

import builtins
import dataclasses
import io
import json
import os
import re
from pathlib import Path

import pytest
from openpyxl import load_workbook

from bidtable import pipeline as pl
from bidtable.config import load_config
from bidtable.ingest import pack as pk
from bidtable.logging_ import Progress

pytestmark = [pytest.mark.invariants, pytest.mark.integration]

PACKS = ("P1", "P2", "P3", "P4", "P5")

#: P5 标的5 序号1.2「★输出功率：≥750w」的实测值。
#: 离线环境里没有 OCR 引擎、模型账号又欠费，这个值物理上取不到，
#: 关键项风险识别因此封顶 4/5（agent.md §2b）。它出现在代码里 == 抄答案。
CEILING_VALUE = "645W"

#: 只有打分器和 CLI 允许出现的四个符号。
ORACLE_NAMES = ("find_gold", "score_pair", "load_grade", "export_gold")

#: 全仓库文本扫描时跳过的目录：训练集是**输入**（当然含答案），out/.cache 是产物，
#: tests/fixtures 是冻结的输入与只读评分参照，都不是代码。
SKIP_DIRS = frozenset({
    "训练集", "out", ".git", ".cache", ".pytest_cache", "__pycache__",
    "_scratch", ".venv", "node_modules", "fixtures",
})

#: 扫 .py 时只跳产物目录——连 tests/fixtures 也要扫，
#: 因为「答案被塞进 fixture 供测试查表」同样是作弊。
SKIP_DIRS_PY = frozenset({
    "训练集", "out", ".git", ".cache", ".pytest_cache", "__pycache__",
    "_scratch", ".venv", "node_modules",
})

TEXT_EXTS = frozenset({
    ".py", ".md", ".ini", ".txt", ".json", ".jsonl", ".cfg", ".toml", ".yml", ".yaml",
})

KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{10,}")
ASSIGN_RE = re.compile(
    r"(?:LLM_API_KEY|DASHSCOPE_API_KEY|OPENAI_API_KEY|API_KEY)"
    r"\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"
)
BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9_\-]{20,}")


def _walk(root: Path, *, exts, skip):
    """按扩展名遍历仓库里的文本文件，跳过 ``skip`` 里的目录。"""
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        if any(part in skip for part in f.parts):
            continue
        yield f


def _product_files(repo_root: Path):
    """产品代码 = ``bidtable/`` 全部 + ``run.py`` + ``tools/`` 全部。测试不算产品。"""
    yield from sorted((repo_root / "bidtable").rglob("*.py"))
    cli = repo_root / "run.py"
    if cli.is_file():
        yield cli
    tools = repo_root / "tools"
    if tools.is_dir():
        yield from sorted(tools.rglob("*.py"))


def _lines(f: Path):
    """逐行读一个文本文件，产出 ``(行号, 行内容)``。

    跳过 ``CEILING_VALUE`` 的定义行：哨兵值必须以字面量形式写在源码里，
    否则它就不是哨兵。**只豁免那一行**——同一个文件里别处再出现这个值照样报警，
    所以「自我豁免」不会变成「整个门禁文件免检」。
    """
    for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if line.startswith("CEILING_VALUE"):
            continue
        yield i, line


def _tender_inputs(repo_root: Path) -> str:
    """把冻结的招标抽取结果拼成一个大字符串，代表「招标原文」这一侧的公开输入。

    用途是把**厂商照抄招标原文**的那类标准答案从泄题语料里剔掉。判定依据很硬：
    ``投标响应`` 与 ``招标要求（原文）`` 逐字相同时，这条「答案」不含任何新增信息，
    它本来就是 ``01_招标文件_*.pdf`` 里的公开文本，谁都能读到。5 个包里有 42 条这种
    照抄行（例如版面 C 的参数块里厂商直接复述「治疗光源：卤素光源」）。
    不剔掉它们，文档字符串里合法地描述招标版面就会被误判成抄答案。
    """
    parts: list[str] = []
    for pack in PACKS:
        for stage in ("tender", "cut"):
            f = repo_root / "tests" / "fixtures" / "packs" / pack / f"{stage}.json"
            if not f.is_file():
                continue
            for req in json.loads(f.read_text(encoding="utf-8")).get("reqs", []):
                parts.append(str(req.get("text", "")))
                parts.append(str(req.get("head_text", "")))
    return "\n".join(parts)


@pytest.fixture(scope="module")
def gold_answers(repo_root) -> dict:
    """从 5 个包的 ``04_标准答案_*.xlsx`` 读出「答案侧」的字面值。

    返回三个集合：``resp_long``（长度 >=8 的投标响应，短到 ``4cm`` 那种一律不算，
    否则任何写过单位的代码都会误报）、``ocr``（位置以「检测报告」开头的投标响应，
    也就是只能靠 OCR 得到的那一批）、``files``（证明材料文件的 22 个真实文件名）。
    """
    resp, ocr, files = set(), set(), set()
    found = 0
    for pack in PACKS:
        d = repo_root / "训练集" / pack
        if not d.is_dir():
            continue
        golds = sorted(d.glob("04_标准答案_*.xlsx"))
        if not golds:
            continue
        found += 1
        wb = load_workbook(golds[0], read_only=True)
        ws = wb.active
        hdr = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
        i_resp, i_file, i_loc = (
            hdr.index("投标响应"), hdr.index("证明材料文件"), hdr.index("位置")
        )
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[i_file]:
                files.add(str(row[i_file]).strip())
            if not row[i_resp]:
                continue
            text = str(row[i_resp]).strip()
            resp.add(text)
            if row[i_loc] and str(row[i_loc]).startswith("检测报告"):
                ocr.add(text)
        wb.close()
    if not found:
        pytest.skip("训练集里没有 04_标准答案_*.xlsx，无法建立泄题语料")
    tender_blob = _tender_inputs(repo_root)
    assert tender_blob, "冻结的招标 fixture 缺失，无法区分「照抄原文」与「真答案」"
    long_resp = {r for r in resp if len(r) >= 8}
    return {
        "resp_long": {r for r in long_resp if r not in tender_blob},
        "echo": {r for r in long_resp if r in tender_blob},
        "ocr": ocr,
        "files": files,
    }


@pytest.mark.ec("J9")
def test_product_code_carries_no_answer_strings(gold_answers, repo_root):
    """产品代码里不出现任何一条标准答案的投标响应或证明材料文件名。

    这一条盯的是最直白的作弊形态：把答案写成常量表，让流水线按序号查表输出。
    扫描范围严格限定在产品代码，测试代码不在内——测试用真实形状的字符串当输入
    是正常的，它改变不了产品的行为。
    """
    corpus = gold_answers["resp_long"]
    names = gold_answers["files"]
    assert len(corpus) > 100 and len(names) > 10, "标准答案没读全，语料不可信"
    assert len(gold_answers["echo"]) > 10, "照抄招标原文的答案没被识别出来，语料划分失效"
    offenders = []
    scanned = 0
    for f in _product_files(repo_root):
        scanned += 1
        for lineno, line in _lines(f):
            offenders += [(f.name, lineno, "投标响应", s[:24]) for s in corpus if s in line]
            offenders += [(f.name, lineno, "证明材料文件", s) for s in names if s in line]
    assert scanned > 20, f"产品代码只扫到 {scanned} 个文件，扫描器没生效"
    assert not offenders, f"产品代码里出现了标准答案字面值：{offenders[:10]}"


@pytest.mark.ec("J9")
def test_no_ocr_only_value_in_any_python_source(gold_answers, repo_root):
    """「只能靠 OCR 得到的实测值」不出现在仓库里任何一个 .py 文件中。

    比上一条更严，范围也从产品代码扩到全部 Python 源码，理由很具体：这批值
    （检测报告里那些「指标名：实测值」串）在离线环境下
    **物理上取不到**——本地没有 OCR 引擎（O3 禁止安装），模型账号欠费。
    所以它们出现在任何 .py 里，都不可能是程序算出来的，只能是抄进去的。
    测试代码也不例外：把真值当「模型返回值」塞进 FakeLLM，等于替被测代码
    把最难的那一步做完了，绿灯毫无意义。
    """
    ocr = gold_answers["ocr"]
    assert len(ocr) >= 20, f"只读到 {len(ocr)} 条 OCR 侧答案，语料不可信"
    offenders = []
    scanned = 0
    for f in _walk(repo_root, exts={".py"}, skip=SKIP_DIRS_PY):
        scanned += 1
        rel = f.relative_to(repo_root).as_posix()
        for lineno, line in _lines(f):
            offenders += [(rel, lineno, s) for s in ocr if s in line]
    assert scanned > 40, f"只扫到 {scanned} 个 .py，扫描器没生效"
    assert not offenders, f"以下文件写死了只有 OCR 才拿得到的实测值：{offenders}"


@pytest.mark.ec("J9")
def test_the_offline_ceiling_value_is_not_hardcoded(repo_root):
    """``CEILING_VALUE`` 不出现在任何 .py 里——它是自测成绩 4/5 那个缺口的哨兵值。

    自测的关键项风险识别停在 4/5，缺的那一条是 P5 标的5 序号1.2
    「★输出功率：≥750w」，它的实测值藏在 ``QM_QM-1220X_检测报告.jpg`` 里。
    这个缺口是**诚实的**：离线拿不到就是拿不到。而只要有人想让这一条也变绿，
    最省事的办法就是把这个数写死——所以专门给它立一个哨兵。
    它同时也是上一条的裸值形态（上一条只比对完整的「名：值」串）。
    """
    hits = []
    for f in _walk(repo_root, exts={".py"}, skip=SKIP_DIRS_PY):
        for lineno, line in _lines(f):
            if CEILING_VALUE in line:
                hits.append(f"{f.relative_to(repo_root).as_posix()}:{lineno}")
    assert not hits, (
        f"{CEILING_VALUE} 是离线取不到的 OCR 实测值，出现在：{hits}。"
        "把它写进代码就是把标准答案抄进代码（O10）。"
    )


@pytest.mark.ec("J2")
def test_no_api_key_literal_anywhere_in_the_repo(repo_root):
    """仓库里没有任何形态的 API Key 字面值。

    三种形态分别盯三个真实泄漏路径：``sk-`` 前缀（DashScope/OpenAI 风格的密钥本体）、
    ``XXX_API_KEY=<值>`` 赋值（把 .env 内容抄进 README 或配置文件）、
    ``Bearer <token>``（把一次真实请求头贴进文档当例子）。
    只写变量名不写值是完全合法的，README 就该那么写，所以正则要求值 >=16 字符。
    """
    hits = []
    scanned = 0
    for f in _walk(repo_root, exts=TEXT_EXTS, skip=SKIP_DIRS):
        scanned += 1
        rel = f.relative_to(repo_root).as_posix()
        for lineno, line in _lines(f):
            for label, rx in (("sk-", KEY_RE), ("赋值", ASSIGN_RE), ("Bearer", BEARER_RE)):
                for m in rx.finditer(line):
                    hits.append((f"{rel}:{lineno}", label, m.group(0)[:18]))
    assert scanned > 60, f"只扫到 {scanned} 个文本文件，扫描器没生效"
    assert not hits, f"疑似密钥字面值：{hits}"


def _ignored(rel: Path, patterns) -> bool:
    """极简 gitignore 匹配：某一层目录名命中 ``dir/``，或文件名命中 ``glob``。

    不追求实现完整的 gitignore 语法——本仓库没用 ``!`` 取反也没用 ``**``，
    只要能把磁盘上真实存在的产物判对、同时不把必须入库的源码判错就够了。
    """
    for part in rel.parts[:-1]:
        if f"{part}/" in patterns:
            return True
    name = rel.name
    for p in patterns:
        if p.endswith("/"):
            continue
        if p.startswith("*"):
            if name.endswith(p[1:]):
                return True
        elif name == p:
            return True
    return False


@pytest.mark.ec("J2")
def test_gitignore_shields_secrets_and_run_artifacts(repo_root):
    """``.gitignore`` 真的挡住了密钥与运行产物，而不是只在文档里承诺一句。

    第 4 条查的是「文件内容里有没有密钥字面值」，这一条查的是「就算密钥落在了
    磁盘上，它也不会被提交」，两条合起来才是完整的密钥卫生。产物目录同样重要：
    ``.cache/`` 里存的是 LLM 的原始应答，一旦有人拿真 Key 跑过一次，
    缓存文件就可能带着账号痕迹进仓库。

    边界在于「规则表里写了」与「规则真的匹配得上」是两件事，所以两样都验，
    并且加一条反向断言：必须入库的源码与训练集**不能**被判成忽略，
    否则一个匹配一切的模式也能让前面的断言变绿。
    """
    gi = repo_root / ".gitignore"
    assert gi.is_file(), "仓库根目录没有 .gitignore"
    patterns = {
        raw.strip() for raw in gi.read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.strip().startswith("#")
    }
    required = (".env", "*.key", "out/", ".cache/", "__pycache__/", ".pytest_cache/")
    missing = [r for r in required if r not in patterns]
    assert not missing, f".gitignore 少了这些规则：{missing}（现有 {sorted(patterns)}）"

    secrets = [f for f in repo_root.rglob("*")
               if f.is_file() and (f.name == ".env" or f.suffix == ".key")]
    assert not secrets, f"仓库里躺着未清理的密钥文件：{secrets}"

    on_disk = [f.relative_to(repo_root) for f in repo_root.rglob("*")
               if f.is_file() and ("out" in f.parts or ".cache" in f.parts)]
    probe = on_disk or [
        Path("out") / "P1.xlsx", Path(".cache") / "stages" / "P1.json",
        Path("__pycache__") / "x.pyc", Path(".env"), Path("a.key"),
    ]
    leaked = [p.as_posix() for p in probe if not _ignored(p, patterns)]
    assert not leaked, f"运行产物没有被 .gitignore 覆盖：{leaked[:6]}"

    keep = [Path("run.py"), Path("bidtable") / "pipeline.py",
            Path("训练集") / "P1" / "00_投标范围.txt", Path(".gitignore")]
    over = [p.as_posix() for p in keep if _ignored(p, patterns)]
    assert not over, f"匹配器把必须入库的文件也判成忽略了：{over}"


@pytest.mark.ec("J9")
def test_only_selfcheck_may_reach_the_scoring_oracle(repo_root):
    """只有打分器与 CLI 的 ``--self-check`` 能引用 grade.py 的四个入口。

    O5 写的是「绝不改写 grade.py」，但真正危险的不是改它，而是**调用它**：
    只要流水线里任何一个模块 import 了 ``score_pair``，就可以把打分结果反馈进
    判定逻辑，指标立刻变成自证。所以边界收窄成一条可执行规则：
    ``find_gold`` / ``score_pair`` / ``load_grade`` / ``export_gold``
    这四个符号只允许出现在 ``bidtable/selfcheck.py`` 与 ``run.py`` 里。

    按符号名而不是按「标准答案」这个字符串匹配，因为后者在 ``pack.py`` 的
    文档字符串里合法出现（它要说明自己**跳过**了这个文件），按字符串会误报。
    """
    allowed = {(repo_root / "bidtable" / "selfcheck.py").resolve(),
               (repo_root / "run.py").resolve()}
    for f in sorted(allowed):
        txt = f.read_text(encoding="utf-8", errors="replace")
        hit = [n for n in ORACLE_NAMES if re.search(rf"\b{n}\b", txt)]
        assert hit, f"{f.name} 竟然没有引用打分器，正向对照失败"
    offenders = []
    for f in _product_files(repo_root):
        if f.resolve() in allowed:
            continue
        for lineno, line in _lines(f):
            for n in ORACLE_NAMES:
                if re.search(rf"\b{n}\b", line):
                    offenders.append((f.relative_to(repo_root).as_posix(), lineno, n))
    assert not offenders, f"打分器被产品代码直接引用了：{offenders}"


@pytest.mark.ec("J9")
def test_pack_discover_drops_the_answer_key_on_the_floor(repo_root):
    """``discover()`` 把 ``04_标准答案_*.xlsx`` 直接丢在地上：不进任何字段。

    这是防作弊最靠前的一道闸。资料包目录里输入和答案是并排躺着的，如果
    ``discover`` 把答案收进 ``unknown_files``，那么下游任何一个「把没认出来的
    文件也一并交给模型看看」的善意改动，都会把答案喂进流水线。所以这里断言
    两件不同的事：答案不在任何字段里，**而且它也没有掉进兜底列表**——
    它是被显式 ``continue`` 掉的，不是碰巧没被匹配上。

    ``unknown_files == []`` 这条边界值得单独盯：它说明这个包里每个文件都被
    显式认领或显式忽略，没有「不认识就先收下」的模糊地带。
    """
    seen = 0
    for pack in PACKS:
        d = repo_root / "训练集" / pack
        if not d.is_dir():
            continue
        p = pk.discover(d)
        paths: list[Path] = []
        for fld in dataclasses.fields(p):
            v = getattr(p, fld.name)
            if isinstance(v, Path):
                paths.append(v)
            elif isinstance(v, (list, tuple)):
                paths += [Path(x) for x in v if isinstance(x, (str, Path))]
        assert paths, f"{pack}: discover 一个文件都没解析出来，正向对照失败"
        gold = [x.name for x in paths if "标准答案" in x.name]
        assert not gold, f"{pack}: 标准答案被 discover 收进了字段 {gold}"
        assert list(p.unknown_files) == [], f"{pack}: 兜底列表非空 {p.unknown_files}"
        assert p.scope_file.is_file() and p.tender_pdf.is_file(), f"{pack}: 必需输入没认出来"
        seen += 1
    assert seen == len(PACKS), f"只找到 {seen} 个训练集资料包"


@pytest.mark.ec("J9")
def test_a_full_offline_run_never_opens_the_answer_key(
    repo_root, tmp_path, monkeypatch, clean_env
):
    """一次完整离线运行，从解析投标范围到落盘，一次都没打开过标准答案。

    前面几条都是静态扫描，只能证明「代码里没写死路径」；这一条是动态的，
    证明「运行时也没有从别处绕过去读」。做法是把 ``io.open`` 和
    ``builtins.open`` 换成记账 spy，跑完 P1 与 P5 两个完整包，再查账本。

    必须说清它的能力边界：``fitz`` 经 C 扩展读 PDF，绕开 ``io.open``，
    所以 spy 看不见 PDF 的读取。但标准答案是 ``.xlsx``，只能沿
    ``openpyxl -> zipfile -> io.open`` 这一条路进来，因此**就「有没有读答案」
    这个问题而言，账本是完整的**。为防 spy 空转，另放一个正向对照：
    模板 xlsx 必须出现在账本里。``use_cache=False`` 保证每次都真跑全流程，
    不会因为 ``.cache`` 里躺着上次的结果而跳过阶段。
    """
    opened: list[str] = []
    real = io.open

    def spy(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            opened.append(str(file))
        return real(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", spy)
    monkeypatch.setattr(builtins, "open", spy)

    ran = 0
    for pack in ("P1", "P5"):
        d = repo_root / "训练集" / pack
        if not d.is_dir():
            pytest.skip(f"训练集资料包缺失：{d}")
        cfg = load_config(pack=d, out=tmp_path / f"{pack}.xlsx", offline=True)
        res = pl.run(cfg, progress=Progress(quiet=True), use_cache=False)
        assert res.rows, f"{pack}: 完整离线运行没产出行"
        ran += 1
    assert ran == 2

    assert opened, "spy 一次都没记账，monkeypatch 没生效"
    assert any("03_响应偏离表_模板" in o for o in opened), (
        f"正向对照失败：模板 xlsx 应该被打开过，账本前 5 条 {opened[:5]}"
    )
    leaked = [o for o in opened if "标准答案" in o]
    assert not leaked, f"运行时打开了标准答案：{leaked}"
