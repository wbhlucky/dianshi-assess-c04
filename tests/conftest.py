"""pytest 共享 fixture 与离线守卫（agent.md 6.3）。

三条铁律：

1. **默认离线**。autouse 的网络守卫把 ``socket.socket`` 换成会炸的桩，
   只有带 ``eval`` / ``network`` 标记的测试才放行真实网络。
2. **不复制 grade.py 的口径**。所有打分都经 :func:`bidtable.selfcheck.load_grade`
   动态加载仓库根目录那份原文件——评分契约一变，测试立刻跟着变。
3. **fixture 只读**。``tests/fixtures/{packs,gold}`` 由
   ``python -m bidtable.selfcheck --freeze`` 生成；需要往里写的测试
   一律先复制到 ``tmp_path``。
"""

from __future__ import annotations

import json
import shutil
import socket
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # 让 `import bidtable` 在任何 cwd 下都成立
    sys.path.insert(0, str(REPO_ROOT))

from bidtable import pipeline as pl  # noqa: E402
from bidtable import selfcheck  # noqa: E402
from bidtable.config import load_config  # noqa: E402
from bidtable.llm.client import LLMClient  # noqa: E402
from bidtable.llm.transport import FakeTransport  # noqa: E402
from bidtable.logging_ import Progress  # noqa: E402
from bidtable.models import Evidence, Match, Requirement, Row, Scope  # noqa: E402

PACKS = ("P1", "P2", "P3", "P4", "P5")
TRAINING = REPO_ROOT / "训练集"
FIXTURES = REPO_ROOT / "tests" / "fixtures"


# --------------------------------------------------------------------------- #
# 网络守卫
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    """离线层一旦发出真实网络调用就立刻失败，而不是等超时。"""
    if any(request.node.iter_markers(m) for m in ("eval", "network")):
        return

    def _boom(*a, **k):
        raise AssertionError(
            "测试触发了真实网络调用。离线层必须靠 FakeTransport 或阶段缓存覆盖；"
            "确实需要联网的测试请加 @pytest.mark.network 或 @pytest.mark.eval。"
        )

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)


AMBIENT_ENV = (
    "LLM_API_KEY", "DASHSCOPE_API_KEY", "LLM_BASE_URL",
    "MODEL_TEXT", "MODEL_STRONG", "MODEL_VISION", "MODEL_OCR", "BIDTABLE_CACHE",
)


@pytest.fixture
def clean_env(monkeypatch):
    """抹掉环境里真实存在的 Key 与模型覆盖项。

    开发机上 ``DASHSCOPE_API_KEY`` 是有值的，不清掉的话 ``load_config`` 会判成
    「在线」，一批断言 ``offline is True`` 的测试就会随机器状态飘。
    """
    for key in AMBIENT_ENV:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


# --------------------------------------------------------------------------- #
# 路径与冻结 fixture
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def pack_path() -> Callable[[str], Path]:
    """``pack_path("P1")`` -> ``训练集/P1``。资料包不在就 skip 而不是 fail。"""

    def _get(name: str) -> Path:
        p = TRAINING / name
        if not p.is_dir():
            pytest.skip(f"训练集资料包缺失：{p}")
        return p

    return _get


def _need(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"缺少 {path}；先跑 python -m bidtable.selfcheck --packs 训练集 --freeze")


@pytest.fixture
def frozen() -> Callable[[str, str], dict[str, Any]]:
    """``frozen("P1", "tender")`` -> 该阶段冻结下来的 payload。"""

    def _get(pack: str, stage: str) -> dict[str, Any]:
        p = FIXTURES / "packs" / pack / f"{stage}.json"
        _need(p)
        return json.loads(p.read_text(encoding="utf-8"))

    return _get


@pytest.fixture
def gold_map() -> Callable[[str], dict[tuple[str, str, str], dict[str, str]]]:
    """``gold_map("P1")`` -> ``{(标的, 分节, 序号): 七列}``，已剔除「不判分」。"""

    def _get(pack: str) -> dict[tuple[str, str, str], dict[str, str]]:
        p = FIXTURES / "gold" / f"{pack}.json"
        _need(p)
        raw = json.loads(p.read_text(encoding="utf-8"))
        out = {}
        for k, v in raw.items():
            if v.get("dev") == "不判分":
                continue
            lot, sec, no = k.split("|")
            out[(lot, sec, no)] = v
        return out

    return _get


@pytest.fixture
def gold_rows(gold_map) -> Callable[[str], list[dict[str, str]]]:
    def _get(pack: str) -> list[dict[str, str]]:
        return list(gold_map(pack).values())

    return _get


@pytest.fixture(scope="session")
def grade() -> Callable[[Path, Path], dict[str, Any]]:
    """``grade(gold_xlsx, sub_xlsx)`` -> grade.py 打印的那个 dict。"""

    def _run(gold_xlsx: Path, sub_xlsx: Path) -> dict[str, Any]:
        return selfcheck.score_pair(Path(gold_xlsx), Path(sub_xlsx))

    return _run


# --------------------------------------------------------------------------- #
# 离线跑流水线：拿冻结 fixture 当阶段缓存，因此不碰 PDF/docx/jpg
# --------------------------------------------------------------------------- #


_OFFLINE_MEMO: dict[tuple, Any] = {}


@pytest.fixture
def offline_run(tmp_path):
    """``offline_run("P1")`` -> ``PackResult``。

    阶段 1~4 全部从 ``tests/fixtures/packs/<P>`` 读缓存，所以这条路径
    既离线又快（毫秒级）。缓存目录先复制到 tmp，保证 fixture 只读。

    同一组参数（包名 / 起始阶段 / Config 覆盖项）在整个会话里只真跑一次。
    ``process_pack`` 在这条路径上不改缓存目录、也不写 ``cfg.out``，返回值是只读的，
    所以复用是安全的；换来的是「五条用例各把五个包跑一遍」这种重复开销
    从几十秒降到几秒（离线全绿 <60s 的门禁靠它才守得住）。
    **任何测试都不许就地修改返回的 ``PackResult``。**
    """

    def _run(pack: str, *, from_stage: str = "match", **cfg_kw: Any):
        memo_key = (pack, from_stage, tuple(sorted((k, repr(v)) for k, v in cfg_kw.items())))
        if memo_key in _OFFLINE_MEMO:
            return _OFFLINE_MEMO[memo_key]
        root = TRAINING / pack
        if not root.is_dir():
            pytest.skip(f"训练集资料包缺失：{root}")
        src = FIXTURES / "packs" / pack
        _need(src / "scope.json")
        cache = tmp_path / f"cache_{pack}"
        if cache.exists():
            # 同一个测试里对同一个包取两次结果时，先清掉旧缓存再复制，
            # 否则 copytree 会因目录已存在而报 FileExistsError。
            shutil.rmtree(cache)
        shutil.copytree(src, cache)
        cfg = load_config(pack=root, out=tmp_path / f"{pack}.xlsx", offline=True, **cfg_kw)
        result = pl.process_pack(
            root, cfg,
            progress=Progress(quiet=True),
            from_stage=from_stage, use_cache=True, cache_root=cache,
        )
        _OFFLINE_MEMO[memo_key] = result
        return result

    return _run


@pytest.fixture
def offline_rows(offline_run) -> Callable[[str], list[Row]]:
    def _rows(pack: str) -> list[Row]:
        return offline_run(pack).rows

    return _rows


# --------------------------------------------------------------------------- #
# LLM 桩
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_llm(tmp_path):
    """``fake_llm([...])`` -> 带 ``FakeTransport`` 的离线 ``LLMClient``。

    返回的 client ``enabled is True``，因此被测代码会真的走进 LLM 分支；
    传输层是桩，所以既不发网络也不读磁盘缓存。
    """

    def _make(responses: Any = None, *, default: str = "", cache: bool = False) -> LLMClient:
        cfg = load_config(
            pack=TRAINING / "P1", out=tmp_path / "x.xlsx",
            offline=not cache, cache_dir=tmp_path / "llmcache",
        )
        transport = FakeTransport(responses, default=default)
        client = LLMClient(cfg, transport, sleep=lambda *_: None)
        client.transport = transport  # 便于测试直接读 .calls / .seen
        return client

    return _make


# --------------------------------------------------------------------------- #
# 数据构造器：让每条测试只写它真正关心的那几个字段
# --------------------------------------------------------------------------- #


@pytest.fixture
def mk_scope() -> Callable[..., Scope]:
    def _mk(lot_no: str = "1", device_name: str = "设备", quantity: str = "1台") -> Scope:
        return Scope.make(lot_no, device_name, quantity)

    return _mk


@pytest.fixture
def mk_req() -> Callable[..., Requirement]:
    def _mk(
        text: str,
        *,
        no: str = "1",
        lot_no: str = "1",
        section: str = "",
        mark: str = "",
        device_name: str = "设备",
        page: int = 8,
        is_heading: bool = False,
        layout: str = "A",
        head_text: str = "",
    ) -> Requirement:
        return Requirement(
            lot_no=lot_no, device_name=device_name, section=section, no=no,
            mark=mark, text=text, page=page, is_heading=is_heading,
            layout=layout, head_text=head_text or text,
        )

    return _mk


@pytest.fixture
def mk_ev() -> Callable[..., Evidence]:
    def _mk(
        name: str,
        value: str = "",
        *,
        lot_no: str = "1",
        model: str = "M-1",
        doc_type: str = "技术规格书",
        filename: str = "X_技术规格书.docx",
        row_no: int = 1,
        page: int = 1,
        priority: int | None = None,
        is_pointer: bool = False,
        confidence: float = 1.0,
        response_text: str | None = None,
    ) -> Evidence:
        from bidtable.config import DOC_PRIORITY

        if doc_type == "产品综合样本":
            locator = f"产品综合样本 第{page}页"
            filename = filename or "产品综合样本.pdf"
        elif doc_type == "检测报告":
            locator = f"检测报告 第{page}页 表行{row_no}"
        else:
            locator = f"{doc_type} 表行{row_no}"
        return Evidence(
            lot_no=lot_no, model=model, doc_type=doc_type, filename=filename,
            locator=locator, name=name, value=value,
            response_text=response_text if response_text is not None else f"{name}：{value}",
            priority=DOC_PRIORITY[doc_type] if priority is None else priority,
            is_pointer=is_pointer, confidence=confidence,
        )

    return _mk


@pytest.fixture
def mk_match(mk_req, mk_ev) -> Callable[..., Match]:
    """``mk_match(req_text, ev_name, ev_value, **kw)`` 一步造出配对。"""

    def _mk(
        req_text: str,
        ev_name: str | None = "指标",
        ev_value: str = "",
        *,
        score: float = 0.9,
        method: str = "exact",
        confidence: float = 0.9,
        flags: tuple[str, ...] = (),
        req_kw: dict[str, Any] | None = None,
        ev_kw: dict[str, Any] | None = None,
    ) -> Match:
        req = mk_req(req_text, **(req_kw or {}))
        ev = None if ev_name is None else mk_ev(ev_name, ev_value, **(ev_kw or {}))
        return Match(
            requirement=req, evidence=ev, score=score, method=method,
            confidence=confidence, reason="", flags=flags,
        )

    return _mk


@pytest.fixture
def mk_row() -> Callable[..., Row]:
    def _mk(
        *,
        lot_no: str = "1",
        section: str = "",
        no: str = "1",
        mark: str = "",
        req_text: str = "要求",
        resp_text: str = "",
        deviation: str = "无偏离",
        evidence_file: str = "",
        location: str = "",
        confidence: float = 1.0,
        flags: tuple[str, ...] = (),
        note: str = "",
    ) -> Row:
        return Row(
            lot_no=lot_no, device_name="设备", section=section, no=no, mark=mark,
            req_text=req_text, resp_text=resp_text, deviation=deviation,
            evidence_file=evidence_file, location=location,
            confidence=confidence, flags=flags, note=note,
        )

    return _mk


# --------------------------------------------------------------------------- #
# 合成资料包：负面测试（N1~N15）不能拿真资料包去破坏
# --------------------------------------------------------------------------- #


@pytest.fixture
def make_pack(tmp_path):
    """``make_pack(name, files={相对路径: 内容或 bytes})`` -> 合成资料包目录。"""

    def _mk(name: str = "SYN", files: dict[str, Any] | None = None, *, vendor: bool = True) -> Path:
        root = tmp_path / name
        root.mkdir(parents=True, exist_ok=True)
        for rel, content in (files or {}).items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                p.write_bytes(content)
            else:
                p.write_text(str(content), encoding="utf-8")
        if vendor and not any(r.startswith("02_") for r in (files or {})):
            (root / "02_厂商资料").mkdir(exist_ok=True)
        return root

    return _mk


SCOPE_TEXT = """项目名称：合成测试项目
采购人：某医院
投标人：某公司
标的 1：合成设备  数量 1台
"""
