"""全部数据契约（agent.md 4.2）。字段名即文档。

输出类 ``frozen=True``，保证跨阶段传递时不会被就地改写。
每个类都带 ``to_dict`` / ``from_dict``，用于阶段缓存与冻结 fixture。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .config import SERVICE_LOT_NAME


def _dc_to_dict(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return list(obj)
    return obj


class _Dictable:
    def to_dict(self) -> dict[str, Any]:
        return {f.name: _dc_to_dict(getattr(self, f.name)) for f in fields(self)}  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in names:
                continue
            f = next(x for x in fields(cls) if x.name == key)  # type: ignore[arg-type]
            if isinstance(value, list) and f.type in ("tuple[str, ...]", tuple):
                value = tuple(value)
            kwargs[key] = value
        return cls(**kwargs)


@dataclass(frozen=True)
class Scope(_Dictable):
    """一条投标标的，来自 ``00_投标范围.txt``。"""

    lot_no: str
    device_name: str
    quantity: str = ""
    is_service: bool = False

    @staticmethod
    def make(lot_no: str, device_name: str, quantity: str = "") -> "Scope":
        return Scope(
            lot_no=str(lot_no).strip(),
            device_name=device_name.strip(),
            quantity=quantity,
            is_service=device_name.strip() == SERVICE_LOT_NAME,
        )


@dataclass(frozen=True)
class Requirement(_Dictable):
    """招标侧一条要求。"""

    lot_no: str
    device_name: str
    section: str
    no: str
    mark: str
    text: str
    page: int = 0
    is_heading: bool = False
    layout: str = "A"
    # 标题行的**逐字原文**（含编号与标记）。A/B 族的 ``text`` 本身就是逐字原文，
    # 而 C 族的 ``text`` 按 gold 约定剥掉了编号，所以 ``分节`` 路径必须用这个字段拼，
    # 否则 ``1、治疗源`` 会退化成 ``治疗源``（S4 规则 6：标题文本逐字保留）。
    head_text: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.lot_no, self.section, self.no)


@dataclass(frozen=True)
class Evidence(_Dictable):
    """厂商侧一条可作证明材料的事实。"""

    lot_no: str
    model: str
    doc_type: str
    filename: str
    locator: str
    name: str
    value: str
    response_text: str
    priority: int = 1
    is_pointer: bool = False
    confidence: float = 1.0


@dataclass(frozen=True)
class Match(_Dictable):
    """一条要求与其证据的匹配结果。"""

    requirement: Requirement
    evidence: Evidence | None
    score: float
    method: str  # exact/contain/alias/fingerprint/llm/none
    confidence: float
    reason: str = ""
    # 匹配阶段产生的降信/消歧标记，原样透传给 Row.flags 进复核报告。
    # agent.md 4.2 的 Match 契约里没有这一项：Row.flags 需要一个来源，
    # 而匹配层是唯一知道「这条为什么可信度低」的地方。带默认值，向后兼容。
    flags: tuple[str, ...] = ()


@dataclass
class Row(_Dictable):
    """输出一行。前 10 个字段即 xlsx 的 10 列，顺序不可变。"""

    lot_no: str = ""
    device_name: str = ""
    section: str = ""
    no: str = ""
    mark: str = ""
    req_text: str = ""
    resp_text: str = ""
    deviation: str = ""
    evidence_file: str = ""
    location: str = ""
    # 以下三项不写进 xlsx，只进复核报告
    confidence: float = 1.0
    flags: tuple[str, ...] = ()
    note: str = ""

    def cells(self) -> list[str]:
        return [
            self.lot_no,
            self.device_name,
            self.section,
            self.no,
            self.mark,
            self.req_text,
            self.resp_text,
            self.deviation,
            self.evidence_file,
            self.location,
        ]


@dataclass
class PackResult:
    """一个资料包的完整产物。"""

    pack: str
    scopes: list[Scope] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_warning(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def add_skipped(self, msg: str) -> None:
        if msg not in self.skipped:
            self.skipped.append(msg)


def rows_to_records(rows: list[Row]) -> list[dict[str, Any]]:
    return [asdict(r) for r in rows]
