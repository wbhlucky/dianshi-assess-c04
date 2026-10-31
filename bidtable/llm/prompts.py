"""提示词集中管理，便于 README 引用（agent.md 11.3）。

实测取舍：``qwen-vl-ocr`` 出纯文本约 2.8s / 181 token，
而 ``qwen3.7-plus`` 出 JSON 约 12.8s / 773 token。
所以 OCR 默认走纯文本 + 确定性正则解析，只在解析失败时才退回结构化 JSON。
"""

from __future__ import annotations

from typing import Any

# --- L1 检测报告 OCR --------------------------------------------------------

OCR_SYSTEM = (
    "你是检测报告的文字识别引擎。只输出图片中出现的文字，"
    "逐行还原，保持原有的行顺序与标点。"
    "不要解释、不要补充、不要翻译、不要添加任何图片中没有的内容。"
)

OCR_USER = (
    "请把这张检测报告图片中的所有文字按行输出，"
    "表格部分按「序号 检测项目 实测值 备注」的列顺序逐行输出，列之间用一个空格分隔。"
)

OCR_JSON_SYSTEM = (
    "你是检测报告结构化抽取引擎。只输出 JSON，不要任何解释文字。"
)

OCR_JSON_USER = (
    "请把这张检测报告解析为 JSON，字段如下：\n"
    '{"report_no": string, "sample_name": string, "sample_model": string,\n'
    ' "test_date": string,\n'
    ' "rows": [{"no": string, "item": string, "value": string, "note": string}]}\n'
    "rows 只包含检测结果表格中的数据行，不含表头与「以下空白」。"
)


def ocr_messages(image_b64: str, *, structured: bool = False) -> list[dict[str, Any]]:
    system = OCR_JSON_SYSTEM if structured else OCR_SYSTEM
    user_text = OCR_JSON_USER if structured else OCR_USER
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": user_text},
            ],
        },
    ]


# --- L2 语义匹配兜底 --------------------------------------------------------

MATCH_SYSTEM = (
    "你是招投标技术参数比对助手。判断哪一条厂商指标最能证明招标要求被满足。"
    "只输出 JSON，不要解释。"
)


def match_messages(requirement: str, candidates: list[tuple[str, str]]) -> list[dict[str, Any]]:
    listing = "\n".join(
        f"[{i}] {name} = {value}" for i, (name, value) in enumerate(candidates)
    )
    user = (
        "招标要求：\n"
        f"{requirement}\n\n"
        "候选厂商指标：\n"
        f"{listing}\n\n"
        '请选出唯一最能对应该要求的候选下标，输出 JSON：{"index": <整数或 null>}。'
        "若没有任何候选与该要求相关，index 填 null。"
    )
    return [
        {"role": "system", "content": MATCH_SYSTEM},
        {"role": "user", "content": user},
    ]


# --- L3 难判定裁决 ----------------------------------------------------------

JUDGE_SYSTEM = (
    "你是招投标偏离判定专家。判据：投标响应达到或优于招标要求为「无偏离」；"
    "数值型要求且明显优于（超出 10% 以上）为「正偏离」；"
    "未达到要求、或明确写「暂不支持」「选配（需另行采购）」"
    "「标准配置不含，可按需定制」「部分支持（完整功能需升级版本）」为「负偏离」。"
    "文字型要求即使客观更优也只判「无偏离」。只输出 JSON，不要解释。"
)


def judge_messages(requirement: str, response: str, rules: str = "") -> list[dict[str, Any]]:
    user = (
        f"招标要求：{requirement}\n"
        f"投标响应：{response}\n"
        + (f"补充规则：{rules}\n" if rules else "")
        + '\n输出 JSON：{"deviation": "无偏离|正偏离|负偏离", "reason": "<一句话>"}'
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
