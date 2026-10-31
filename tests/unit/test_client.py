"""LLMClient 门面：缓存优先、退避重试、错误降级（EC-J1/J2/J4/J6，用例 N10=B18）。

这一层的硬契约只有一句：**``complete()`` 绝不向上抛异常**。三个窄口子
（OCR / 语义匹配 / 难判定裁决）都必须在模型不可用时安静退化成 ``None``，
由调用方走确定性分支——所以「抛不抛」和「重试几次」都是被测行为。
"""

from __future__ import annotations

import dataclasses

import pytest

from bidtable import errors
from bidtable.config import Config, load_config
from bidtable.llm.client import LLMClient, build_transport
from bidtable.llm.transport import (
    FakeTransport,
    LiveTransport,
    RecordingTransport,
    ReplayTransport,
)

pytestmark = [pytest.mark.unit]

MSGS = [{"role": "user", "content": "输出功率：≥750w"}]


def _cfg(tmp_path, *, offline: bool = True, **kw) -> Config:
    """load_config 在环境里没有 Key 时会把 offline 强制成真（config.py 的
    ``offline=offline or api_key is None``）。要测缓存与重试就得把它掰回来，
    否则 DiskCache 被关掉、每个请求都直达传输层，命中路径永远测不到。"""
    cfg = load_config(pack="训练集/P1", out=tmp_path / "o.xlsx",
                      cache_dir=tmp_path / "cache", offline=offline, **kw)
    return dataclasses.replace(cfg, offline=offline)


@pytest.mark.ec("J1")
def test_build_transport_mode_table(clean_env, tmp_path):
    """``--transport`` 的每种取值都必须落到确定的传输类，且离线/无 Key 一律 None。"""
    keyed = dataclasses.replace(
        _cfg(tmp_path, offline=False), api_key="sk-test", base_url="https://x.invalid/v1"
    )
    assert isinstance(build_transport(dataclasses.replace(keyed, transport="fake")), FakeTransport)
    assert isinstance(build_transport(dataclasses.replace(keyed, transport="live")), LiveTransport)
    rec = build_transport(dataclasses.replace(keyed, transport="record"), replay_path="a/b.jsonl")
    assert isinstance(rec, RecordingTransport) and rec.path.as_posix() == "a/b.jsonl"
    rep = build_transport(dataclasses.replace(keyed, transport="replay"))
    assert isinstance(rep, ReplayTransport)
    assert rep.path.as_posix().endswith("tests/fixtures/llm/recorded.jsonl")   # 默认语料路径

    nokey = _cfg(tmp_path, offline=False)          # 没 Key：live/record 建不出真传输
    assert nokey.api_key is None and nokey.offline is False
    for mode in ("live", "record"):
        assert build_transport(dataclasses.replace(nokey, transport=mode)) is None
    assert build_transport(dataclasses.replace(nokey, transport="auto")) is None
    offline = dataclasses.replace(keyed, transport="auto", offline=True)
    assert build_transport(offline) is None                       # --offline 压过已配置的 Key
    no_base = dataclasses.replace(keyed, transport="auto", base_url=None)
    assert build_transport(no_base) is None                       # 有 Key 没端点也不能发


@pytest.mark.ec("J6")
def test_model_for_maps_each_role_and_falls_back_to_the_text_model(clean_env, tmp_path):
    cfg = _cfg(tmp_path, model_text="T", model_strong="S", model_vision="V", model_ocr="O")
    c = LLMClient(cfg, FakeTransport())
    assert c.model_for("ocr") == "O" and c.model_for("vision") == "V"
    assert c.model_for("match") == "T" and c.model_for("judge") == "S"
    assert c.model_for("text") == "T" and c.model_for("没听说过的角色") == "T"


@pytest.mark.ec("J1")
def test_offline_client_degrades_with_a_readable_reason(clean_env, tmp_path):
    """离线时 enabled=False、complete 返回 None，并留下一句人话进复核报告。"""
    c = LLMClient(_cfg(tmp_path, offline=True), None)
    assert c.enabled is False and c.calls == 0
    assert c.complete("ocr", MSGS) is None
    assert c.degraded == ["[ocr] 离线模式：未配置可用的 API Key 或指定了 --offline"]


@pytest.mark.ec("J4")
@pytest.mark.case("N10")
def test_second_run_makes_zero_api_calls(tmp_path, clean_env):
    """B18/N10 的验收口径：同一个缓存目录下跑第二遍，传输层一次都不该被调用。"""
    cfg = _cfg(tmp_path, offline=False)
    first = LLMClient(cfg, FakeTransport(["第一次的回答"]), sleep=lambda *_: None)
    assert first.complete("judge", MSGS).text == "第一次的回答"
    assert (first.calls, first.cache_hits) == (1, 0)

    second = LLMClient(cfg, FakeTransport(["不该被用到"]), sleep=lambda *_: None)
    resp = second.complete("judge", MSGS)
    assert resp.text == "第一次的回答"                            # 命中的是旧答案
    assert (second.calls, second.cache_hits) == (0, 1)
    assert second.transport.calls == 0


@pytest.mark.ec("J4")
def test_explicit_model_and_params_take_part_in_the_cache_key(tmp_path, clean_env):
    """换了模型或 temperature 就是另一个请求，不能复用上一个模型的缓存。"""
    cfg = _cfg(tmp_path, offline=False)
    c = LLMClient(cfg, FakeTransport(["a", "b", "c", "d"]), sleep=lambda *_: None)
    assert c.complete("judge", MSGS).text == "a"
    assert c.complete("judge", MSGS, model="别的模型").text == "b"
    assert c.complete("judge", MSGS, params={"temperature": 0.9}).text == "c"
    assert c.complete("judge", MSGS).text == "a" and c.cache_hits == 1
    assert c.complete("judge", MSGS, model="别的模型").text == "b" and c.cache_hits == 2
    assert c.calls == 3                                            # 三次真发，两次命中
    assert c.transport.seen[1].model == "别的模型"


@pytest.mark.ec("J1")
def test_offline_config_disables_the_disk_cache_entirely(tmp_path, clean_env):
    """``--offline`` 不只是不发请求：连读缓存也要关掉，避免用陈旧答案冒充本次结果。"""
    cfg = _cfg(tmp_path, offline=True)
    warm = LLMClient(dataclasses.replace(cfg, offline=False), FakeTransport(["热的"]),
                     sleep=lambda *_: None)
    warm.complete("judge", MSGS)
    cold = LLMClient(cfg, FakeTransport(["冷的"]), sleep=lambda *_: None)
    assert cold.cache.enabled is False
    assert cold.complete("judge", MSGS).text == "冷的"
    assert (cold.calls, cold.cache_hits) == (1, 0)


@pytest.mark.ec("J7")
@pytest.mark.case("B19")
def test_retryable_errors_back_off_exponentially_then_degrade(tmp_path, clean_env):
    """限流/超时要退避重试，重试用尽后返回 None 而不是抛出去。"""
    sleeps: list[float] = []
    cfg = dataclasses.replace(_cfg(tmp_path, offline=False), max_retries=3)
    t = FakeTransport([errors.LLMRateLimitError("限流")] * 4)
    c = LLMClient(cfg, t, sleep=sleeps.append)
    assert c.complete("ocr", MSGS) is None
    assert c.calls == 4 == cfg.max_retries + 1                    # 首发 + 3 次重试
    assert sleeps == [1, 2, 4]                                    # 2**(n-1)，上限 8
    assert c.degraded == ["[ocr] 重试 3 次后仍失败：限流"]


@pytest.mark.ec("J7")
def test_backoff_is_capped_at_eight_seconds(tmp_path, clean_env):
    """退避不能无限翻倍：重试次数调大时 sleep 必须停在 8 秒。"""
    sleeps: list[float] = []
    cfg = dataclasses.replace(_cfg(tmp_path, offline=False), max_retries=6)
    c = LLMClient(cfg, FakeTransport([errors.LLMTimeoutError("超时")] * 7), sleep=sleeps.append)
    assert c.complete("match", MSGS) is None
    assert sleeps == [1, 2, 4, 8, 8, 8]


@pytest.mark.ec("J7")
def test_non_retryable_errors_fail_fast_without_burning_the_budget(tmp_path, clean_env):
    """鉴权/配额错误重试没有意义：立刻降级，一次调用都不多打。"""
    sleeps: list[float] = []
    t = FakeTransport([errors.LLMAuthError("令牌无效"), "永远用不到"])
    c = LLMClient(_cfg(tmp_path, offline=False), t, sleep=sleeps.append)
    assert c.complete("judge", MSGS) is None
    assert (c.calls, sleeps) == (1, [])
    assert c.degraded == ["[judge] 令牌无效"]


@pytest.mark.ec("J4")
def test_a_late_success_after_retries_is_returned_and_cached(tmp_path, clean_env):
    """重试成功要走正常返回路径，并且同样落缓存（否则第三次运行又会重打）。"""
    sleeps: list[float] = []
    cfg = _cfg(tmp_path, offline=False)
    c = LLMClient(cfg, FakeTransport([errors.LLMRateLimitError("限流"), "终于成功了"]),
                  sleep=sleeps.append)
    assert c.complete("judge", MSGS).text == "终于成功了"
    assert (c.calls, sleeps) == (2, [1])
    again = LLMClient(cfg, FakeTransport(["不该被用到"]), sleep=sleeps.append)
    assert again.complete("judge", MSGS).text == "终于成功了" and again.calls == 0


@pytest.mark.ec("J7")
def test_unexpected_transport_blowup_is_swallowed_with_its_type_name(tmp_path, clean_env):
    """传输层抛出计划外的异常（SDK 版本漂移）也只能降级，消息要带上类型便于定位。"""
    c = LLMClient(_cfg(tmp_path, offline=False), FakeTransport([ValueError("boom")]),
                  sleep=lambda *_: None)
    assert c.complete("ocr", MSGS) is None
    assert c.degraded == ["[ocr] 传输层意外错误：ValueError: boom"]


@pytest.mark.ec("J8")
def test_degradation_notes_are_deduplicated_per_role(tmp_path, clean_env):
    """同一原因失败 5 次只在复核报告里出现一行，否则告警会把真正的问题淹掉。"""
    c = LLMClient(_cfg(tmp_path, offline=True), None)
    for _ in range(5):
        assert c.complete("ocr", MSGS) is None
    assert len(c.degraded) == 1
    c.complete("judge", MSGS)                                     # 换个 role 才会有新的一行
    assert len(c.degraded) == 2 and c.degraded[1].startswith("[judge]")
