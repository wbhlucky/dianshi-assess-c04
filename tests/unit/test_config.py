"""配置层：Key 解析、离线降级、策略取值域、路径与编码形态（EC-J1~J6）。

这一层是「三个模型口子」的总开关，也是最容易被环境变量悄悄改写的地方，
所以每条测试都先把环境清干净，再只改自己关心的那一个变量。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bidtable import config as C

pytestmark = [pytest.mark.unit]

ENV_KEYS = (
    "LLM_API_KEY", "DASHSCOPE_API_KEY", "LLM_BASE_URL", "BIDTABLE_CACHE",
    "MODEL_TEXT", "MODEL_STRONG", "MODEL_VISION", "MODEL_OCR",
)

PACK = Path("训练集/P1")
OUT = Path("out/x.xlsx")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """环境里本来就带着一把（已欠费的）DASHSCOPE Key，不清就测不到「无 Key」分支。"""
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def load(**kw):
    return C.load_config(pack=kw.pop("pack", PACK), out=kw.pop("out", OUT), **kw)


# --------------------------------------------------------------------------- #
# J2 Key 来源与 base_url 推导
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J2")
def test_no_key_yields_offline_config(monkeypatch):
    """无 Key：api_key/base_url 全空，offline 自动置真，llm_available 为假。"""
    cfg = load()
    assert (cfg.api_key, cfg.key_source, cfg.base_url) == (None, None, None)
    assert cfg.offline is True and cfg.llm_available is False
    # 显式要求联网也会被压回离线：没有 Key 就没有可发的请求，这一条是
    # test_client.py 里必须 dataclasses.replace 掰回来的根因。
    forced = load(offline=False)
    assert forced.offline is True and forced.llm_available is False


@pytest.mark.ec("J2")
def test_llm_api_key_wins_over_dashscope(monkeypatch):
    """两把 Key 同时存在时按固定顺序取第一把，并把来源记进 key_source。"""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dk-123")
    assert load().api_key == "dk-123"
    assert load().key_source == "DASHSCOPE_API_KEY"
    monkeypatch.setenv("LLM_API_KEY", "lk-456")
    cfg = load()
    assert cfg.api_key == "lk-456" and cfg.key_source == "LLM_API_KEY"


@pytest.mark.ec("J2")
def test_base_url_follows_the_key_source(monkeypatch):
    """不同发行方的 Key 对应不同兼容端点，写死一个就会全线 404。"""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dk")
    assert load().base_url == C._BASE_URL_FOR_KEY["DASHSCOPE_API_KEY"]
    monkeypatch.setenv("LLM_API_KEY", "lk")
    assert load().base_url == C._BASE_URL_FOR_KEY["LLM_API_KEY"]
    assert len(set(C._BASE_URL_FOR_KEY.values())) == 2


@pytest.mark.ec("J5")
def test_base_url_precedence_cli_over_env_over_table(monkeypatch):
    """CLI > 环境变量 > Key 推导；末尾斜杠会被吃掉，避免拼出 ``//chat``。"""
    monkeypatch.setenv("LLM_API_KEY", "lk")
    monkeypatch.setenv("LLM_BASE_URL", "http://env.base/v1/")
    assert load().base_url == "http://env.base/v1"
    assert load(base_url="http://cli.base/v2/").base_url == "http://cli.base/v2"


# --------------------------------------------------------------------------- #
# J1 离线开关与降级
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J1")
def test_offline_flag_wins_even_with_a_valid_key(monkeypatch):
    """``--offline`` 是硬开关：自测口径必须可复现，不能因为环境里有 Key 就偷偷联网。"""
    monkeypatch.setenv("LLM_API_KEY", "lk")
    cfg = load(offline=True)
    assert cfg.api_key == "lk" and cfg.offline is True and cfg.llm_available is False


@pytest.mark.ec("J1")
def test_key_present_switches_offline_off(monkeypatch):
    """反向：有 Key 且未指定 --offline 时默认允许联网（这也是 selfcheck 需要 --offline 的原因）。"""
    monkeypatch.setenv("LLM_API_KEY", "lk")
    cfg = load()
    assert cfg.offline is False and cfg.llm_available is True


# --------------------------------------------------------------------------- #
# J6 模型角色映射
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J6")
def test_model_env_beats_cli_kwarg(monkeypatch):
    """刻意保留的优先级：环境变量是账户级事实，CLI 只是本次调用的建议值。"""
    monkeypatch.setenv("LLM_API_KEY", "lk")
    monkeypatch.setenv("MODEL_TEXT", "env-model")
    assert load(model_text="cli-model").model_text == "env-model"
    monkeypatch.delenv("MODEL_TEXT")
    assert load(model_text="cli-model").model_text == "cli-model"


@pytest.mark.ec("J6")
def test_four_roles_have_distinct_defaults(monkeypatch):
    """四个角色各有一套默认模型；judge 用强模型、ocr 用视觉模型是刻意的分工。"""
    monkeypatch.setenv("LLM_API_KEY", "lk")
    cfg = load()
    got = (cfg.model_text, cfg.model_strong, cfg.model_vision, cfg.model_ocr)
    assert got == tuple(C._MODEL_DEFAULTS[k] for k in
                        ("MODEL_TEXT", "MODEL_STRONG", "MODEL_VISION", "MODEL_OCR"))
    assert cfg.model_strong != cfg.model_text and cfg.model_ocr != cfg.model_vision


# --------------------------------------------------------------------------- #
# J4 缓存路径
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J4")
def test_cache_dir_kwarg_over_env_over_default(monkeypatch):
    """``BIDTABLE_CACHE`` 让 CI 能把缓存挂到工作区外，CLI 参数仍可覆盖它。"""
    assert load().cache_dir == Path(".cache")
    monkeypatch.setenv("BIDTABLE_CACHE", "envcache")
    assert load().cache_dir == Path("envcache")
    assert load(cache_dir="kwcache").cache_dir == Path("kwcache")


@pytest.mark.ec("J4")
def test_cache_subdirs_are_namespaced(monkeypatch):
    """阶段缓存与 LLM 缓存分目录：清一个不会误伤另一个（--freeze 与 --no-cache 各管一半）。"""
    cfg = load(cache_dir="c")
    assert cfg.llm_cache_dir == Path("c") / "llm"
    assert cfg.stage_cache_dir == Path("c") / "stages"


# --------------------------------------------------------------------------- #
# J5 / J3 取值域与常量契约
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J5")
def test_env_helper_strips_and_treats_blank_as_absent(monkeypatch):
    """空串和纯空白等于「没设」，否则 Key 会被读成 ``""`` 而绕开无 Key 分支。"""
    monkeypatch.setenv("LLM_API_KEY", "  lk  ")
    assert C._env("LLM_API_KEY") == "lk"
    monkeypatch.setenv("LLM_API_KEY", "   ")
    assert C._env("LLM_API_KEY", "D") == "D"
    assert C._env("BIDTABLE_NOPE_ZZZ", "D") == "D"


@pytest.mark.ec("J3")
@pytest.mark.case("X28")
def test_columns_are_exactly_the_ten_template_columns():
    """列集合与模板逐字对齐；gold 里的第 11 列「判定说明」绝不能输出。"""
    assert C.COLUMNS == (
        "标的", "设备名称", "分节", "序号", "标记",
        "招标要求（原文）", "投标响应", "偏离类型", "证明材料文件", "位置",
    )
    assert "判定说明" not in C.COLUMNS and len(C.COLUMNS) == 10
    assert C.SHEET_NAME == "响应偏离表"


@pytest.mark.ec("J3")
def test_policy_value_domains_are_closed_tuples():
    """三个策略开关的取值域是闭集，run.py 的校验直接拿它当白名单。"""
    assert C.SECTION_POLICIES == ("s4", "s2", "s1", "off")
    assert C.PAREN_POLICIES == ("auto", "fold", "split")
    assert C.OCR_FALLBACKS == ("assume", "no-response")
    assert C.DEVIATIONS == ("无偏离", "正偏离", "负偏离", "未响应")
    assert (C.LAYOUT_A, C.LAYOUT_B, C.LAYOUT_C, C.LAYOUT_UNKNOWN) == ("A", "B", "C", "?")


@pytest.mark.ec("J3")
def test_negative_vocabulary_is_the_four_closed_phrases():
    """负偏离词表是闭集（X21 的整短语匹配以它为准），改一个词就会动摇判定口径。"""
    assert C.NEGATIVE_PHRASES == (
        "暂不支持", "选配（需另行采购）",
        "标准配置不含，可按需定制", "部分支持（完整功能需升级版本）",
    )
    assert C.MARK_STAR == "★" and C.MARK_TRIANGLE == "▲"
    assert C.SERVICE_LOT_NAME == "设备运维技术服务"


@pytest.mark.ec("D6")
def test_doc_priority_ranks_report_first():
    """证据优先级 检测报告 > 规格书/服务方案 > 样本，是 X3/X4 冲突裁决的唯一依据。"""
    assert C.DOC_PRIORITY[C.DOC_REPORT] == 0
    assert C.DOC_PRIORITY[C.DOC_SPEC] == C.DOC_PRIORITY[C.DOC_SERVICE] == 1
    assert C.DOC_PRIORITY[C.DOC_CATALOG] == 2
    assert set(C.DOC_PRIORITY) == {C.DOC_REPORT, C.DOC_SPEC, C.DOC_SERVICE, C.DOC_CATALOG}


@pytest.mark.ec("J3")
def test_config_is_frozen():
    """Config 全程只读：判定阶段偷偷改阈值会让自测口径失去可复现性。"""
    assert C.Config.__dataclass_params__.frozen is True
    cfg = load()
    with pytest.raises(Exception):
        cfg.positive_margin = 2.0
    assert (cfg.positive_margin, cfg.low_margin_band) == (1.10, (1.05, 1.15))
    assert (cfg.section_policy, cfg.paren_policy, cfg.ocr_fallback) == ("s4", "auto", "assume")
