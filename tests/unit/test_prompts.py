"""提示词与常量的一致性（EC-J3 策略取值域与常量契约 / EC-J6 模型角色映射）。

提示词是「第三份判定规则」：确定性判定器有一份、``config`` 常量有一份、
提示词里还有一份。三份一旦漂移，离线跑和在线跑就会给出不同答案，而且
这种偏差在训练集上很难被发现（模型不可用时提示词根本不被执行）。
所以这里测的不是文案好不好，而是**它和代码里的常量还对不对得上**。
"""

from __future__ import annotations

import pytest

from bidtable import config
from bidtable.judge import decide
from bidtable.llm import prompts

pytestmark = [pytest.mark.unit]

B64 = "ZmFrZQ=="


@pytest.mark.ec("J6")
def test_ocr_prompt_defaults_to_plain_text_and_only_switches_on_demand():
    """实测纯文本 2.8s / JSON 12.8s，所以默认必须是纯文本；
    structured=True 才换成 JSON 契约（``rows`` 四列与 :mod:`vendor_report` 的解析器对齐）。"""
    plain = prompts.ocr_messages(B64)
    assert plain[0]["content"] == prompts.OCR_SYSTEM
    assert plain[1]["content"][0]["image_url"]["url"] == f"data:image/jpeg;base64,{B64}"
    assert plain[1]["content"][1]["text"] == prompts.OCR_USER
    assert "JSON" not in prompts.OCR_USER

    structured = prompts.ocr_messages(B64, structured=True)
    assert structured[0]["content"] == prompts.OCR_JSON_SYSTEM
    body = structured[1]["content"][1]["text"]
    for field in ("report_no", "sample_name", "sample_model", "test_date", "rows"):
        assert field in body
    assert "以下空白" in body                     # 报告表格的尾部噪声要显式排除


@pytest.mark.ec("J7")
def test_ocr_prompt_forbids_inventing_content():
    """取证层的红线：模型不能补出图里没有的数字。这句话被删掉，
    O10（不得内嵌标准答案）在在线模式下就会失守。"""
    assert "不要补充" in prompts.OCR_SYSTEM and "不要添加任何图片中没有的内容" in prompts.OCR_SYSTEM
    assert "不要翻译" in prompts.OCR_SYSTEM


@pytest.mark.ec("J7")
def test_match_prompt_indexes_from_zero_and_pins_the_output_schema():
    """``semantic._parse_index`` 只认 ``{"index": …}``；提示词里的下标基准
    必须是 0，且和合格池的顺序一致，否则模型选的行会整体错位。"""
    msgs = prompts.match_messages("输出功率：≥750w", [("输出功率", "600W"), ("治疗深度", "7cm")])
    assert msgs[0]["content"] == prompts.MATCH_SYSTEM
    user = msgs[1]["content"]
    assert "[0] 输出功率 = 600W" in user and "[1] 治疗深度 = 7cm" in user
    assert "输出功率：≥750w" in user
    assert '{"index": <整数或 null>}' in user and "null" in user


@pytest.mark.ec("J3")
def test_judge_prompt_vocabulary_matches_the_code_constants():
    """提示词里的负偏离话术表必须逐字等于 ``config.NEGATIVE_PHRASES``，
    正偏离的 10% 门槛必须等于 ``decide.DEFAULT_MARGIN``。"""
    for phrase in config.NEGATIVE_PHRASES:
        assert phrase in prompts.JUDGE_SYSTEM, phrase
    pct = round((decide.DEFAULT_MARGIN - 1) * 100)
    assert f"超出 {pct}% 以上" in prompts.JUDGE_SYSTEM
    for dev in ("无偏离", "正偏离", "负偏离"):
        assert dev in prompts.JUDGE_SYSTEM
    # 「未响应」是结构化判定（证据缺失）的结论，不该让模型来猜
    assert "未响应" not in prompts.JUDGE_SYSTEM


@pytest.mark.ec("J3")
def test_judge_prompt_keeps_text_requirements_out_of_positive_deviation():
    """X22 的提示词侧对应物：文字型要求即使客观更优也只判无偏离。"""
    assert "文字型要求即使客观更优也只判「无偏离」" in prompts.JUDGE_SYSTEM
    assert "只输出 JSON" in prompts.JUDGE_SYSTEM


@pytest.mark.ec("J3")
def test_judge_prompt_includes_supplementary_rules_only_when_given():
    msgs = prompts.judge_messages("≥750w", "600W")
    assert "补充规则" not in msgs[1]["content"]
    assert "招标要求：≥750w" in msgs[1]["content"] and "投标响应：600W" in msgs[1]["content"]
    with_rules = prompts.judge_messages("≥750w", "600W", rules="取等视为满足")
    assert "补充规则：取等视为满足" in with_rules[1]["content"]
    assert with_rules[0]["content"] == prompts.JUDGE_SYSTEM
