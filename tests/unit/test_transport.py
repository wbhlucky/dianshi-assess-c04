"""四种传输与 HTTP 错误分类（EC-J1 降级 / EC-J7 错误契约，用例 N9=B19）。

分类表不是装饰：欠费、令牌失效、模型名不存在三种情况的**处置动作完全不同**
（换 Key / 换模型 / 退避重试），所以消息文本本身也是被测契约的一部分。
"""

from __future__ import annotations

import json

import pytest

from bidtable import errors
from bidtable.llm import client as cli
from bidtable.llm.transport import (
    FakeTransport,
    LiveTransport,
    RecordingTransport,
    ReplayError,
    ReplayTransport,
    Request,
    Response,
    Transport,
    _from_sdk_exception,
    classify_http,
)

pytestmark = [pytest.mark.unit]

REQ = Request(model="qwen-vl-ocr", messages=[{"role": "user", "content": "识别"}])


@pytest.mark.ec("J7")
@pytest.mark.case("N9")
@pytest.mark.parametrize("status,body,want", [
    (401, "", errors.LLMAuthError),
    (200, '{"error":{"type":"one_api_error","message":"无效的令牌"}}', errors.LLMAuthError),
    (400, "Invalid API key provided", errors.LLMAuthError),
    (403, "", errors.LLMQuotaError),
    (400, "Free quota exhausted", errors.LLMQuotaError),
    (404, "model not found", errors.LLMQuotaError),
    (429, "", errors.LLMRateLimitError),
    (400, '{"error":{"type":"Arrearage","code":"overdue-payment"}}', errors.LLMQuotaError),
    (400, "账户欠费，请充值", errors.LLMQuotaError),
    (400, "bad request", errors.LLMResponseError),
    (500, "internal error", errors.LLMRateLimitError),
    (503, "", errors.LLMRateLimitError),
    (302, "redirect", errors.LLMResponseError),
])
def test_classify_http_maps_status_and_body_to_the_right_error(status, body, want):
    exc = classify_http(status, body)
    assert isinstance(exc, want) and isinstance(exc, errors.LLMError)
    assert exc.exit_code == 2                       # LLM 层错误一律退出码 2


@pytest.mark.ec("J7")
def test_error_messages_tell_the_operator_what_to_do_next():
    """N9 的验收口径是「可读消息」：三类错误各自指向不同的修复动作。"""
    assert "LLM_API_KEY" in str(classify_http(401, "无效的令牌"))
    assert "--model-ocr" in str(classify_http(404, "not found"))
    arrear = str(classify_http(400, "overdue-payment"))
    assert "离线模式" in arrear and "未响应" in arrear and "不会重复计费" in arrear
    assert "退避重试" in str(classify_http(429, ""))
    long_body = "x" * 500
    assert len(str(classify_http(400, long_body))) < 400          # 原始响应截断到 200 字


@pytest.mark.ec("J3")
def test_retryable_flag_agrees_with_the_client_retry_set():
    """可重试集合只有一处定义，两边的口径必须一致，否则退避逻辑会漏掉超时。"""
    assert errors.LLMRateLimitError("x").retryable is True
    assert errors.LLMTimeoutError("x").retryable is True
    assert cli.RETRYABLE == (errors.LLMRateLimitError, errors.LLMTimeoutError)
    for exc in (errors.LLMAuthError("x"), errors.LLMQuotaError("x"), errors.LLMResponseError("x")):
        assert not isinstance(exc, cli.RETRYABLE)


@pytest.mark.ec("J4")
def test_request_digest_is_content_addressed():
    """录制/回放靠 digest 命中，所以它必须只随内容变化，不随对象身份变化。"""
    same = Request(model="qwen-vl-ocr", messages=[{"role": "user", "content": "识别"}])
    assert same is not REQ and REQ.digest() == same.digest() and len(REQ.digest()) == 64
    assert REQ.digest() != Request(model="qwen-vl-plus", messages=REQ.messages).digest()
    assert REQ.digest() != Request(
        model="qwen-vl-ocr", messages=REQ.messages, params={"temperature": 0}
    ).digest()


@pytest.mark.ec("J1")
def test_fake_transport_serves_by_call_index_then_falls_back_to_default():
    """离线测试的主力桩：按调用序发牌，牌发完给 default，且如实计数。"""
    t = FakeTransport(["一", "二"], default="兜底")
    assert [t.send(REQ).text for _ in range(4)] == ["一", "二", "兜底", "兜底"]
    assert t.calls == 4 and len(t.seen) == 4 and t.seen[0] is REQ


@pytest.mark.ec("J7")
def test_fake_transport_can_raise_and_can_return_a_ready_response():
    """桩既能模拟 HTTP 错误（N9），也能透传带 usage 的 Response（缓存断言要用）。"""
    boom = FakeTransport([errors.LLMRateLimitError("限流")])
    with pytest.raises(errors.LLMRateLimitError):
        boom.send(REQ)
    assert boom.calls == 1                                  # 抛错也算一次调用
    rich = FakeTransport([Response(text="ok", usage={"prompt_tokens": 7})])
    assert rich.send(REQ).usage == {"prompt_tokens": 7}
    fn = FakeTransport(lambda r: f"回声:{r.model}")
    assert fn.send(REQ).text == "回声:qwen-vl-ocr"


@pytest.mark.ec("J1")
def test_fake_transport_dict_lookup_prefers_digest_over_model():
    """dict 形态支持两种粒度：整段对话（digest）优先，模型级默认兜底。"""
    t = FakeTransport({REQ.digest(): "精确", "qwen-vl-ocr": "按模型"}, default="都没中")
    assert t.send(REQ).text == "精确"
    assert t.send(Request(model="qwen-vl-ocr", messages=[{"role": "user", "content": "别的"}])).text == "按模型"
    assert t.send(Request(model="gpt", messages=[])).text == "都没中"


@pytest.mark.ec("J1")
def test_replay_transport_reports_the_missing_key_loudly(tmp_path):
    """回放未命中必须报出是哪一次调用没录到，否则排查成本极高（agent.md 6.5）。"""
    path = tmp_path / "recorded.jsonl"
    path.write_text(
        json.dumps({"key": REQ.digest(), "text": "录制的回答", "usage": {"prompt_tokens": 3}},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    t = ReplayTransport(path)
    assert t.send(REQ).text == "录制的回答" and t.calls == 1
    other = Request(model="m", messages=[{"role": "user", "content": "没录过"}])
    with pytest.raises(ReplayError) as ei:
        t.send(other)
    msg = str(ei.value)
    assert other.digest()[:16] in msg and str(path) in msg and "--offline" in msg
    assert t.misses == [other.digest()] and t.calls == 1        # 未命中不计入成功调用


@pytest.mark.ec("J5")
def test_replay_transport_skips_blank_and_malformed_lines(tmp_path):
    """录制文件被手工编辑坏一行，不该让整个回放表加载失败。"""
    path = tmp_path / "recorded.jsonl"
    good = json.dumps({"key": REQ.digest(), "text": "有效行"}, ensure_ascii=False)
    path.write_text(f"\n{good}\n{{ 坏行\n\n", encoding="utf-8")
    t = ReplayTransport(path)
    assert list(t.table) == [REQ.digest()] and t.send(REQ).text == "有效行"
    assert ReplayTransport(tmp_path / "不存在.jsonl").table == {}    # 文件缺失也不抛


@pytest.mark.ec("J1")
def test_recording_transport_produces_a_replayable_corpus(tmp_path):
    """录制 -> 回放闭环：这是 ``recorded.jsonl`` 唯一的生成路径。"""
    corpus = tmp_path / "sub" / "recorded.jsonl"
    rec = RecordingTransport(FakeTransport(["第一次", "第二次"]), corpus)
    r1 = Request(model="m", messages=[{"role": "user", "content": "A"}])
    r2 = Request(model="m", messages=[{"role": "user", "content": "B"}])
    assert rec.send(r1).text == "第一次" and rec.send(r2).text == "第二次"
    assert rec.calls == 2 and rec.close() is None

    lines = [json.loads(x) for x in corpus.read_text(encoding="utf-8").splitlines()]
    assert [ln["key"] for ln in lines] == [r1.digest(), r2.digest()]
    assert lines[0]["request"]["model"] == "m" and lines[0]["text"] == "第一次"

    replay = ReplayTransport(corpus)
    assert replay.send(r2).text == "第二次" and replay.send(r1).text == "第一次"


@pytest.mark.ec("J2")
def test_live_transport_refuses_to_be_built_without_a_key():
    """没有 Key 就不该走到 HTTP 层：早失败、消息直指配置项。"""
    with pytest.raises(errors.LLMAuthError) as ei:
        LiveTransport("", "https://example.invalid/v1")
    assert "LLM_API_KEY" in str(ei.value)


@pytest.mark.ec("J7")
def test_sdk_exceptions_are_translated_into_the_error_hierarchy():
    """openai SDK 的异常类型跨版本不稳定，只能靠名字/status_code 归一。"""
    class Timeout(Exception):
        pass

    class RateLimitError(Exception):
        pass

    class BadRequest(Exception):
        status_code = 400
        response = "overdue-payment"

    assert isinstance(_from_sdk_exception(Timeout("timed out")), errors.LLMTimeoutError)
    assert isinstance(_from_sdk_exception(RateLimitError("slow down")), errors.LLMRateLimitError)
    assert isinstance(_from_sdk_exception(BadRequest()), errors.LLMQuotaError)
    assert isinstance(_from_sdk_exception(ValueError("莫名其妙")), errors.LLMResponseError)


@pytest.mark.ec("J7")
def test_base_transport_is_abstract_and_close_is_idempotent():
    with pytest.raises(NotImplementedError):
        Transport().send(REQ)
    t = FakeTransport()
    assert t.close() is None and t.close() is None and t.name == "fake"
