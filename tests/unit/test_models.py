"""数据契约（agent.md 4.2）。

这些 dataclass 是各阶段之间、以及阶段缓存 JSON 之间的唯一接口。
契约漂一点，冻结 fixture 就整体失效，所以每条测试盯一个不同的契约面。
"""

from __future__ import annotations

import json

import pytest

from bidtable.config import COLUMNS, SERVICE_LOT_NAME
from bidtable.models import (
    Evidence, Match, PackResult, Requirement, Row, Scope, rows_to_records,
)

pytestmark = [pytest.mark.unit]


def _req(**kw) -> Requirement:
    base = dict(lot_no="1", device_name="设备", section="", no="1", mark="", text="要求")
    base.update(kw)
    return Requirement(**base)


def _ev(**kw) -> Evidence:
    base = dict(lot_no="1", model="M-1", doc_type="技术规格书", filename="a.docx",
                locator="技术规格书 表行1", name="指标", value="5mW",
                response_text="指标：5mW")
    base.update(kw)
    return Evidence(**base)


@pytest.mark.ec("J4")
def test_flat_models_survive_a_json_round_trip():
    """阶段缓存就是 ``to_dict`` -> json -> ``from_dict``，tuple 必须还原成 tuple。"""
    for obj in (_req(), _ev(), Scope.make("1", "设备", "1台")):
        restored = type(obj).from_dict(json.loads(json.dumps(obj.to_dict(), ensure_ascii=False)))
        assert restored == obj


@pytest.mark.ec("J4")
def test_from_dict_ignores_unknown_keys():
    """fixture 里多出一列（例如 gold 的「判定说明」）不能让加载崩掉——向后兼容的关键。"""
    r = Requirement.from_dict({**_req().to_dict(), "判定说明": "无", "future_field": 1})
    assert r == _req()


@pytest.mark.ec("H1")
def test_tuple_fields_are_restored_as_tuples():
    """``flags`` 若退化成 list，后续 ``in`` 判断照旧但 ``==`` 比较会全线失败。"""
    row = Row(flags=("low_margin", "unit_converted"))
    back = Row.from_dict(json.loads(json.dumps(row.to_dict(), ensure_ascii=False)))
    assert back.flags == ("low_margin", "unit_converted")
    assert isinstance(back.flags, tuple)


@pytest.mark.ec("I1")
def test_requirement_key_is_the_grading_key():
    """``(标的, 分节, 序号)`` 就是 grade.py 的行键；设备名/页码都不参与，否则去重口径会错位。"""
    assert _req().key == ("1", "", "1")
    a = _req(device_name="甲", page=8, mark="★", text="文本A")
    b = _req(device_name="乙", page=9, mark="", text="文本B")
    assert a.key == b.key
    assert _req(section="一、摄像系统").key != a.key


@pytest.mark.ec("B2")
def test_scope_make_detects_the_service_lot_by_name():
    """服务标的靠设备名精确匹配识别，X15 的「服务标的不裁售后服务」全靠这个布尔。"""
    assert Scope.make(" 2 ", f" {SERVICE_LOT_NAME} ", "5年").is_service is True
    assert Scope.make("1", "钬激光治疗机", "1台").is_service is False
    assert Scope.make(" 2 ", "x").lot_no == "2"          # lot_no 归一成字符串并去空白
    assert Scope.make("12", "激光治疗仪", "1None").quantity == "1None"  # B4：瑕疵原样保留


@pytest.mark.ec("J3")
def test_row_cells_are_exactly_the_ten_written_columns():
    """``cells()`` 与 COLUMNS 一一对应；confidence/flags/note 只进复核报告，不进 xlsx。"""
    row = Row(lot_no="1", device_name="设备", section="二、", no="3", mark="★",
              req_text="要求", resp_text="响应", deviation="正偏离",
              evidence_file="a.docx", location="技术规格书 表行3",
              confidence=0.4, flags=("low_margin",), note="备注")
    assert len(row.cells()) == len(COLUMNS) == 10
    assert not any(x in row.cells() for x in ("0.4", "low_margin", "备注"))
    assert row.cells()[7] == "正偏离" and row.cells()[9] == "技术规格书 表行3"


@pytest.mark.ec("J3")
def test_upstream_models_are_frozen_but_row_is_not():
    """招标要求/证据/配对跨阶段只读；Row 是产物，写出前允许被去重与补注。"""
    for obj, attr in ((_req(), "text"), (_ev(), "value"), (Scope.make("1", "x"), "lot_no")):
        with pytest.raises(Exception):
            setattr(obj, attr, "改")
    m = Match(requirement=_req(), evidence=_ev(), score=0.9, method="exact", confidence=0.9)
    with pytest.raises(Exception):
        m.score = 1.0
    row = Row()
    row.deviation = "无偏离"
    assert row.deviation == "无偏离"


@pytest.mark.ec("J3")
def test_result_collectors_dedupe_warnings_and_skips():
    """同一条告警在多标的循环里会重复触发；不去重就会把复核报告第 5 节淹掉。"""
    res = PackResult(pack="SYN")
    res.add_warning("版面告警")
    res.add_warning("版面告警")
    res.add_skipped("第 3 页无文本")
    res.add_skipped("第 3 页无文本")
    assert res.warnings == ["版面告警"] and res.skipped == ["第 3 页无文本"]
    assert (res.rows, res.scopes, res.stats) == ([], [], {})


@pytest.mark.ec("J4")
def test_rows_to_records_is_json_serialisable_and_lossless():
    """自测把产物落成 JSON 对照，所以记录形态必须可序列化且带上全部 13 个字段。"""
    rows = [Row(no="1", flags=("ocr_failed",)), Row(no="2")]
    recs = rows_to_records(rows)
    assert len(recs) == 2 and recs[0]["flags"] == ("ocr_failed",)  # asdict 保留 tuple
    blob = json.loads(json.dumps(recs, ensure_ascii=False))        # 可序列化即通过
    assert blob[0]["flags"] == ["ocr_failed"]                      # 落盘后退化成 list
    assert Row.from_dict(blob[0]) == rows[0]                       # 再读回来必须还原
    assert set(recs[0]) == set(Row.__dataclass_fields__) == {
        "lot_no", "device_name", "section", "no", "mark", "req_text", "resp_text",
        "deviation", "evidence_file", "location", "confidence", "flags", "note",
    }


@pytest.mark.ec("I6")
def test_head_text_preserves_the_verbatim_heading_line():
    """C 族的 ``text`` 按 gold 约定剥掉了编号，分节路径必须改用 ``head_text``，
    否则 ``1、治疗源`` 会退化成 ``治疗源``（S4 规则 6）。"""
    c = _req(text="治疗源", head_text="1、治疗源", is_heading=True, layout="C")
    assert c.head_text == "1、治疗源" and c.text == "治疗源"
    assert _req(text="逐字原文").head_text == ""     # A/B 族留空，由 scope.body_of 回落
