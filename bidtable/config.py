"""配置：环境变量、模型角色、阈值、路径。本模块无副作用（不建目录、不发请求）。

Key 读取顺序见 agent.md 4.4。本模块**不修改**任何 Key 配置，
只把「从哪儿读」做成 Provider 无关的两级兜底。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- 输出 schema 常量（与 03_响应偏离表_模板.xlsx 完全一致） -----------------
COLUMNS: tuple[str, ...] = (
    "标的",
    "设备名称",
    "分节",
    "序号",
    "标记",
    "招标要求（原文）",
    "投标响应",
    "偏离类型",
    "证明材料文件",
    "位置",
)
SHEET_NAME = "响应偏离表"

DEV_NONE = "无偏离"
DEV_POSITIVE = "正偏离"
DEV_NEGATIVE = "负偏离"
DEV_NO_RESPONSE = "未响应"
DEVIATIONS: tuple[str, ...] = (DEV_NONE, DEV_POSITIVE, DEV_NEGATIVE, DEV_NO_RESPONSE)

MARK_STAR = "★"
MARK_TRIANGLE = "▲"

# 负偏离触发词表：**封闭集合，整短语匹配**（agent.md 3.9.3 / X21）。
# 这四个短语在训练集里只出现在负偏离行，所以是确定性判据；
# 但 `支持` 这个子串同时出现在 31 条无偏离与 7 条负偏离里，
# 因此**绝不能**退化成子串匹配（`不支持` / `支持` 都会误伤）。
# 匹配侧也用它识别「厂商明确拒答」的证据（见 match.candidates 的 residual 通道）。
NEGATIVE_PHRASES: tuple[str, ...] = (
    "暂不支持",
    "选配（需另行采购）",
    "标准配置不含，可按需定制",
    "部分支持（完整功能需升级版本）",
)

# --- 文档类型（``位置`` 列前缀，不是文件名） --------------------------------
DOC_SPEC = "技术规格书"
DOC_CATALOG = "产品综合样本"
DOC_REPORT = "检测报告"
DOC_SERVICE = "服务方案"
DOC_PRIORITY = {DOC_REPORT: 0, DOC_SPEC: 1, DOC_SERVICE: 1, DOC_CATALOG: 2}

# 服务标的的固定名称（00_投标范围.txt 中以此判定 is_service）
SERVICE_LOT_NAME = "设备运维技术服务"

# 版面族
LAYOUT_A = "A"  # 采购清单 6 列，参数全在一个跨页大单元格（P1/P2/P3）
LAYOUT_B = "B"  # 独立 序号|技术要求 双列表（P4）
LAYOUT_C = "C"  # N-科室-设备名技术参数 多块（P5）
LAYOUT_UNKNOWN = "?"

# --- 可插拔策略的合法取值（CLI choices 与校验共用同一份常量） ----------------
SECTION_POLICIES = ("s4", "s2", "s1", "off")
PAREN_POLICIES = ("auto", "fold", "split")
OCR_FALLBACKS = ("assume", "no-response")

_BASE_URL_FOR_KEY = {
    "LLM_API_KEY": "https://parvis.zjugpt.com/one-api/v1",
    "DASHSCOPE_API_KEY": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

_MODEL_DEFAULTS = {
    "MODEL_TEXT": "deepseek-v4-flash",
    "MODEL_STRONG": "deepseek-v4-pro",
    "MODEL_VISION": "qwen3.7-plus",
    "MODEL_OCR": "qwen-vl-ocr",
}


@dataclass(frozen=True)
class Config:
    """一次运行所需的全部可调项。由 :func:`load_config` 构造。"""

    pack: Path
    out: Path
    report: Path | None = None

    # LLM
    api_key: str | None = None
    base_url: str | None = None
    key_source: str | None = None
    model_text: str = _MODEL_DEFAULTS["MODEL_TEXT"]
    model_strong: str = _MODEL_DEFAULTS["MODEL_STRONG"]
    model_vision: str = _MODEL_DEFAULTS["MODEL_VISION"]
    model_ocr: str = _MODEL_DEFAULTS["MODEL_OCR"]
    offline: bool = True
    transport: str = "auto"  # auto/live/record/replay/fake
    timeout: float = 60.0
    max_retries: int = 3

    # 判定阈值
    positive_margin: float = 1.10  # τ：ratio > tau 才判正偏离
    low_margin_band: tuple[float, float] = (1.05, 1.15)

    # 策略
    section_policy: str = "s4"  # s4/s2/s1/off
    # ``（N）`` 编号：auto = 服务标的展开成 ``父.N``、设备标的并入父项；
    # fold/split = 强制统一。标准答案在 P1 与 P2 上对此**不自洽**，
    # 所以做成旋钮而不是硬编码（README「已知问题」会写明）。
    paren_policy: str = "auto"
    # 检测报告 OCR 不可用时，指向报告的 ``● 指标名：详见检测报告`` 行怎么处理：
    # assume = 仍给出材料/位置并按「无偏离（推定）」输出 + ``ocr_failed`` 旗标；
    # no-response = 一律判「未响应」，三列留空。
    ocr_fallback: str = "assume"
    verbose: bool = False

    # 缓存目录
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))

    @property
    def llm_available(self) -> bool:
        """是否可能真的调用模型。离线模式或无 Key 时为 False。"""
        return bool(self.api_key) and not self.offline

    @property
    def llm_cache_dir(self) -> Path:
        return self.cache_dir / "llm"

    @property
    def stage_cache_dir(self) -> Path:
        return self.cache_dir / "stages"


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip()
    return v or default


def load_config(
    *,
    pack: str | Path,
    out: str | Path,
    report: str | Path | None = None,
    offline: bool = False,
    base_url: str | None = None,
    model_text: str | None = None,
    model_strong: str | None = None,
    model_vision: str | None = None,
    model_ocr: str | None = None,
    positive_margin: float = 1.10,
    section_policy: str = "s4",
    paren_policy: str = "auto",
    ocr_fallback: str = "assume",
    verbose: bool = False,
    cache_dir: str | Path | None = None,
    transport: str = "auto",
    timeout: float = 60.0,
    max_retries: int = 3,
) -> Config:
    """从环境变量 + CLI 覆盖项组装 :class:`Config`。

    Key 优先级：``LLM_API_KEY`` > ``DASHSCOPE_API_KEY``。
    Base URL 优先级：CLI ``--base-url`` > ``LLM_BASE_URL`` > 按 Key 来源推断。
    """
    api_key: str | None = None
    key_source: str | None = None
    for var in ("LLM_API_KEY", "DASHSCOPE_API_KEY"):
        val = _env(var)
        if val:
            api_key, key_source = val, var
            break

    if base_url:
        resolved_base = base_url.rstrip("/")
    else:
        resolved_base = _env("LLM_BASE_URL") or (
            _BASE_URL_FOR_KEY.get(key_source or "", "") or None
        )
        if resolved_base:
            resolved_base = resolved_base.rstrip("/")

    cdir = Path(cache_dir) if cache_dir else Path(os.environ.get("BIDTABLE_CACHE", ".cache"))

    return Config(
        pack=Path(pack),
        out=Path(out),
        report=Path(report) if report else None,
        api_key=api_key,
        base_url=resolved_base,
        key_source=key_source,
        model_text=_env("MODEL_TEXT", model_text or _MODEL_DEFAULTS["MODEL_TEXT"]) or "",
        model_strong=_env("MODEL_STRONG", model_strong or _MODEL_DEFAULTS["MODEL_STRONG"]) or "",
        model_vision=_env("MODEL_VISION", model_vision or _MODEL_DEFAULTS["MODEL_VISION"]) or "",
        model_ocr=_env("MODEL_OCR", model_ocr or _MODEL_DEFAULTS["MODEL_OCR"]) or "",
        offline=offline or api_key is None,
        transport=transport,
        timeout=timeout,
        max_retries=max_retries,
        positive_margin=positive_margin,
        section_policy=section_policy,
        paren_policy=paren_policy,
        ocr_fallback=ocr_fallback,
        verbose=verbose,
        cache_dir=cdir,
    )
