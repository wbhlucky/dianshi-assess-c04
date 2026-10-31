"""判定层·配置、模型仲裁与告警（EC-J1 降级 / EC-J3 取值域 / EC-J6 角色 / EC-J7 错误契约）。

模型是"第三个口子"，不是主判据：只在两种低置信边界上花额度，答非所问就原样退回
确定性判定，改判了也要封顶置信度并留痕。告警按类别汇总，不是每行一条。
"""

from __future__ import annotations

import types
from decimal import Decimal as D

import pytest

from bidtable.config import DEV_NEGATIVE, DEV_NONE, DEV_POSITIVE
from bidtable.ingest.scope import body_of
from bidtable.judge import decide as J
from bidtable.judge.parse_req import parse_requirement
from bidtable.judge.parse_val import parse_value

pytestmark = [pytest.mark.unit]


class _Resp:
    def __init__(self, text: str) -> None:
        self.text, self.raw, self.usage = text, {}, {}


class _StubClient:
    """最小可用客户端：记录调用，按预设文本应答；``text=None`` 表示模型不可用。"""

    def __init__(self, text: str | None = None, *, enabled: bool = True) -> None:
        self._text, self.enabled, self.calls = text, enabled, []

    def complete(self, role, messages, **kw):
        self.calls.append((role, messages))
        return None if self._text is None else _Resp(self._text)


def _ctx(m):
    """直接调 ``arbitrate`` 需要的三件套：判定结果、要求规格、响应规格。"""
    return (J.decide(m), parse_requirement(body_of(m.requirement)), parse_value(m.evidence))


@pytest.mark.ec("J3")
@pytest.mark.ec("J1")
def test_settings_read_the_knobs_and_survive_junk():
    """τ 与低置信带可配，但**取值非法时必须退回默认**而不是崩或静默放行。

    ``positive_margin="bogus"`` 退回 1.10；``low_margin_band=None`` 退回 (1.05,1.15)。
    已知缺口：只给一个元素的 band 会拼出 (1.2,1.05) 这种倒挂区间，
    于是 low_margin 永不触发——该键没有 CLI 入口，只能编程设置，故只做特征化记录。
    """
    for cfg in (None, types.SimpleNamespace()):
        st = J.settings_of(cfg)
        assert (st.tau, st.band, st.ocr_fallback) == (D("1.1"), (D("1.05"), D("1.15")), "assume")

    tuned = J.settings_of(types.SimpleNamespace(
        positive_margin=1.5, low_margin_band=(1.2, 1.8), ocr_fallback="no-response"))
    assert (tuned.tau, tuned.band, tuned.ocr_fallback) == (D("1.5"), (D("1.2"), D("1.8")), "no-response")

    assert J.settings_of(types.SimpleNamespace(positive_margin="bogus")).tau == D("1.1")
    assert J.settings_of(types.SimpleNamespace(low_margin_band=None)).band == (D("1.05"), D("1.15"))
    assert J.settings_of(types.SimpleNamespace(low_margin_band=(1.2,))).band == (D("1.2"), D("1.05"))


@pytest.mark.ec("J7")
def test_model_output_is_parsed_as_json_first_then_as_a_substring():
    """模型不一定听话：优先 JSON，退化到子串扫描，两者都拿不到就返回 None。

    ``{"deviation":"bogus"}`` 与 ``{"reason":"x"}`` 都是 None（触发原样退回），
    而 ``{"deviation":"未响应"}`` 不在三分类里，却由子串扫描捞回——这是有意的宽容，
    因为仲裁只在低置信边界上跑，捞到就用、捞不到不猜。
    """
    assert J._parse_deviation('{"deviation":"负偏离"}') == DEV_NEGATIVE
    assert J._parse_deviation('  {"deviation": "无偏离"}  ') == DEV_NONE
    assert J._parse_deviation('{"deviation":"正偏离","reason":"r"}') == DEV_POSITIVE
    assert J._parse_deviation('{"deviation":"未响应"}') == "未响应"
    assert J._parse_deviation("我认为是正偏离的") == DEV_POSITIVE
    for junk in ('{"deviation":"bogus"}', '{"reason":"x"}', "[1,2]", "garbage", "", None):
        assert J._parse_deviation(junk) is None, junk


@pytest.mark.ec("J1")
@pytest.mark.ec("E5")
def test_arbitration_only_fires_on_the_two_low_confidence_boundaries(mk_match):
    """花模型额度的闸只有两道：比值卡在 τ 附近，或要求可比却找不到可比结构。

    文字型要求（``rs.comparable is False``）即使打了 low_confidence 也不仲裁——
    模型对"支持/不支持"没有比确定性规则更好的判断，白花额度还引入不确定性。
    """
    numeric = parse_requirement("工作长度：≥670 mm")
    textual = parse_requirement("具备软硬镜模式切换功能")
    assert numeric.comparable and not textual.comparable
    assert J._needs_arbitration(("low_margin",), textual) is True
    assert J._needs_arbitration(("low_confidence",), numeric) is True
    assert J._needs_arbitration(("low_confidence",), textual) is False
    assert J._needs_arbitration((), numeric) is False
    assert J._needs_arbitration(("unit_converted", "unit_assumed"), numeric) is False

    clean = mk_match("流量：≥40L/min", "流量", "40L/min")
    client = _StubClient('{"deviation":"负偏离"}')
    dec, rs, vs = _ctx(clean)
    assert J.arbitrate(dec, clean, rs, vs, client) is dec and client.calls == []


@pytest.mark.ec("J1")
@pytest.mark.ec("J7")
def test_arbitrate_returns_the_deterministic_verdict_whenever_the_model_cannot_help(mk_match):
    """六种"模型帮不上忙"的情形都必须**原样返回同一个对象**，且绝不抛异常。

    这是离线/欠费/限流场景下的核心不变量：判定层的可用性不依赖模型可用性。
    """
    m = mk_match("预热时间：≥30秒", "预热时间", "34秒")  # ratio 1.133 -> low_margin
    dec, rs, vs = _ctx(m)
    assert (dec.deviation, dec.flags) == (DEV_POSITIVE, ("low_margin",))

    clients = [None, _StubClient('{"deviation":"无偏离"}', enabled=False), _StubClient(None),
               _StubClient("garbage"), _StubClient('{"deviation":"bogus"}'),
               _StubClient('{"deviation":"正偏离"}')]
    for client in clients:
        assert J.arbitrate(dec, m, rs, vs, client) is dec
    assert dec.method == "compare" and dec.confidence == 0.6


@pytest.mark.ec("J6")
@pytest.mark.ec("E5")
def test_a_contradicting_verdict_is_adopted_capped_and_traceable(mk_match):
    """改判要留三处痕迹：``llm_arbitrated`` 旗标、``method="llm"``、理由后缀。

    置信度取 ``min(原判, 0.6)``——模型改判不会比确定性判定更可信，
    原本只有 0.42 的行改判后仍是 0.42，封顶永远不上调。
    角色走 ``judge``（映射到 MODEL_STRONG），提示词里带上仲裁规则原文。
    """
    m = mk_match("预热时间：≥30秒", "预热时间", "34秒")
    dec, rs, vs = _ctx(m)
    client = _StubClient('{"deviation":"无偏离"}')
    got = J.arbitrate(dec, m, rs, vs, client)

    assert (got.deviation, got.method, got.confidence) == (DEV_NONE, "llm", 0.6)
    assert got.flags == ("low_margin", "llm_arbitrated")
    assert got.reason == f"{dec.reason}｜模型仲裁改判为 {DEV_NONE}"
    assert (got.resp_text, got.evidence_file, got.location) == (dec.resp_text, dec.evidence_file, dec.location)

    (role, messages), = client.calls
    assert role == "judge" and [x["role"] for x in messages] == ["system", "user"]
    assert J.ARBITRATION_RULES in messages[-1]["content"]

    weak = mk_match("预热时间：≥30秒", "预热时间", "34秒", confidence=0.42,
                    flags=("low_confidence",))
    wdec, wrs, wvs = _ctx(weak)
    assert wdec.confidence == 0.42
    assert J.arbitrate(wdec, weak, wrs, wvs, _StubClient('{"deviation":"无偏离"}')).confidence == 0.42


@pytest.mark.ec("J8")
def test_judge_all_warns_per_category_not_per_row(mk_match):
    """6 行里有 5 行带旗标，但告警只有 4 条——按类别计数，供复核报告和人看。

    顺序固定为 OCR → 低置信带 → 无可比量 → 单位换算 → 模型改判；
    全干净的批次返回空列表，不制造噪音。OCR 那条的措辞随 --ocr-fallback 变。
    """
    ptr = {"is_pointer": True, "doc_type": "检测报告"}
    matches = [
        mk_match("脉冲宽度：200μs～800μs", "脉冲宽度", "详见检测报告", ev_kw=dict(ptr, row_no=1)),
        mk_match("脉冲宽度：201μs～801μs", "脉冲宽度", "详见检测报告", ev_kw=dict(ptr, row_no=2)),
        mk_match("预热时间：≥30秒", "预热时间", "34秒"),
        mk_match("工作长度：≥670 mm", "工作长度", "详见样机"),
        mk_match("工作电流：≤10A", "工作电流", "9360mA"),
        mk_match("流量：≥40L/min", "流量", "40L/min"),
    ]
    decisions, warnings = J.judge_all(matches)
    assert len(decisions) == 6 and len(warnings) == 4
    assert warnings == [
        "2 条要求指向检测报告，OCR 不可用，已按「无偏离（推定）」输出，需人工复核实测值。",
        "2 条判定的比值落在 1.05~1.15 的低置信带内，建议优先复核。",
        "1 条要求含数值门槛，但响应侧找不到同量纲的量，已按无偏离输出。",
        "1 条判定做过单位换算（如 mA→A、g→kg），换算依据见复核报告。",
    ]

    strict, warns = J.judge_all(matches[:1], cfg=types.SimpleNamespace(ocr_fallback="no-response"))
    assert strict[0].deviation == "未响应" and "按「未响应」输出" in warns[0]

    client = _StubClient('{"deviation":"无偏离"}')
    _, arb = J.judge_all([matches[2]], client=client)
    assert arb[-1] == "1 条判定被模型仲裁改判，已在复核报告中标出。"

    assert J.judge_all([matches[5]])[1] == []
