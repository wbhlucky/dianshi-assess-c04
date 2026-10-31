"""编号识别、层级、标题 vs 参数项判别（agent.md 3.5 / EC-H）。

训练集里**实际出现过**的编号形态（每一种都必须被认出来，且只被认一次）：

=========== ========================================== =======
形态         实例                                        出处
=========== ========================================== =======
``N、``      ``1、技术培训：…``                          P1 服务
``N.``       ``19.提供医疗器械注册证``                    P5 标的5
``N.k``      ``1.1、分辨率…`` / ``1.1 分辨率…``          P3 / P5
``N.k、``    ``13.1、维修响应速度：…``                    P5 标的8
``N.k.m``    ``1.9.3``（无分隔符，独占序号列）             P4
标记前缀      ``▲2.最大单脉冲能量5J。`` / ``*1.2 输出功率``  P1 / P5
标记尾缀      ``1.7★``（追加在序号单元格里）               P4
全角点       ``1．2``                                    P5
括号编号      ``（1）分辨率≥16 万像素…``                   P1 续页
编号独占行    ``1.`` 换行 ``基础率时段：≥48 段。``          P5 标的8
=========== ========================================== =======

**必须拒绝**的伪编号（混淆 case，X 系列）：
``3840*2160``（乘号）、``10ml/min-450ml/min``（无分隔符）、
``0.l-5ml／每次最大给药量``（续行，``0`` 不是合法顶层号）。
拒绝靠两道闸：正则要求编号后紧跟**分隔符**，:class:`Sequencer` 再校验序号是否**合理递增**。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from . import marks

# 顶层号之间允许的最大跳号。P5 标的1 是 15->17->19->20，最大跳 2；
# 放宽到 5 以容纳未见包，同时仍能挡住 ``0.l-5ml`` 这类伪编号。
MAX_GAP = 5

_NUM = r"\d+(?:[.．]\d+)*"
_SEPS = r"(?:、|[.．]|\s+|$)"

ITEM_RE = re.compile(rf"^\s*(?P<num>{_NUM})\s*(?P<sep>{_SEPS})\s*(?P<rest>.*)$", re.S)
CHINESE_RE = re.compile(r"^\s*(?P<num>[一二三四五六七八九十]+)\s*、\s*(?P<rest>.*)$", re.S)
PAREN_RE = re.compile(r"^\s*[（(]\s*(?P<num>\d{1,2})\s*[）)]\s*(?P<rest>.*)$", re.S)
LEAD_RE = re.compile(r"[：:]\s*$")

_CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

KIND_ITEM = "item"
KIND_CHINESE = "chinese"
KIND_PAREN = "paren"
KIND_LEAD = "lead"

# ``（N）`` 编号的三种处理策略，见 :func:`split_items`。
PAREN_FOLD = "fold"  # 并入上一条目（设备标的）
PAREN_SPLIT = "split"  # 独立成条，由调用方展开成 ``父.N``（服务标的）
PAREN_DROP = "drop"  # 整行丢弃（``（N）`` 只是排版噪声）

# 裸整数编号（``4 次的…``）的第三道闸：前一行必须以终止标点收尾。见 :func:`split_items`。
TERMINAL_END = "。；;：:！!？?）)】」』》”’"

def normalize_no(raw: str) -> str:
    """序号归一：全角点 -> 半角点，吃掉空白与残留分隔符。

    与 ``grade.py`` 的 ``norm()`` 对齐（``re.sub(r"\\s+","",s).replace("．",".")``），
    这样对齐键不会因为标点写法不同而丢行。
    """
    s = re.sub(r"\s+", "", raw or "")
    s = s.replace("．", ".")
    return s.strip("、.")


def chinese_to_int(label: str) -> int:
    """``一``->1、``十``->10、``十一``->11、``二十``->20、``二十一``->21。"""
    s = (label or "").strip()
    if not s or any(ch not in _CN_DIGITS and ch != "十" for ch in s):
        return 0
    if "十" not in s:
        value = 0
        for ch in s:
            value = value * 10 + _CN_DIGITS[ch]
        return value
    head, _, tail = s.partition("十")
    tens = _CN_DIGITS.get(head, 1) if head else 1
    ones = _CN_DIGITS.get(tail, 0) if tail else 0
    return tens * 10 + ones


def int_parts(no: str) -> list[int]:
    """``"1.9.1"`` -> ``[1, 9, 1]``；含非数字段（中文序数、``""``）-> ``[]``。"""
    out: list[int] = []
    for chunk in (no or "").split("."):
        if not chunk.isdigit():
            return []
        out.append(int(chunk))
    return out


@dataclass(frozen=True)
class Numbered:
    """一行文本的编号解析结果。"""

    kind: str
    no: str  # 归一后的裸编号；中文序数保留原文（``一``/``十一``）
    mark: str
    text: str  # 去掉编号与分隔符后的正文；引导句为整行
    top: int = 0  # 顶层号（``2.1.3`` -> 2）；中文序数为其整数值
    level: int = 1  # 层级深度；引导句为 0
    raw: str = ""  # 原始整行，诊断与原文列用
    sep: str = ""  # 编号与正文之间**实际匹配到的分隔符**（``、``/``.``/``．``/空白/``""``）

    @property
    def is_heading_like(self) -> bool:
        """**候选**标题（顶层的 ``N、Title`` / ``一、Title``）。

        是否真为标题由 :func:`has_children` 用「有无 ``N.k`` 子项」确定，
        这里只做形态上的初筛（引导句、括号编号、子项都不算候选）。
        """
        return self.kind in (KIND_ITEM, KIND_CHINESE) and self.level == 1


def _mk_item(m: re.Match[str], raw: str, mark: str) -> Numbered:
    no = normalize_no(m.group("num"))
    parts = no.split(".")
    return Numbered(
        kind=KIND_ITEM,
        no=no,
        mark=mark,
        text=m.group("rest").strip(),
        top=int(parts[0]) if parts[0].isdigit() else 0,
        level=len(parts),
        raw=raw,
        sep=m.group("sep") or "",
    )


def parse_item(line: str) -> Numbered | None:
    """识别阿拉伯数字编号项。识别不出返回 ``None``（不抛异常）。

    标记可能在编号**前**（``▲2.`` / ``▲1.4 、`` / ``*1.2``）也可能在**后**
    （``1.7★``——B 族把 ``★`` 追加在序号单元格里）。两种都要先剥掉，
    否则正则整条失配，该条目会被静默并进上一条，覆盖率直接塌。
    """
    if not line or not line.strip():
        return None
    mark, body = marks.edge_mark(line)
    tail_mark, stripped = marks.strip_trailing_mark(body)
    m = ITEM_RE.match(stripped)
    if m:
        return _mk_item(m, line, mark or tail_mark)
    if not tail_mark:
        return None
    # 尾部标记不是紧贴编号（是正文里的 ``★``），退回未剥离的版本再试一次。
    m = ITEM_RE.match(body)
    return _mk_item(m, line, mark) if m else None


def parse_chinese(line: str) -> Numbered | None:
    """识别中文序数标题（``一、4K3D 摄像系统``、``十一、妇科电动检查床：``）。"""
    if not line:
        return None
    m = CHINESE_RE.match(line)
    if not m:
        return None
    label = m.group("num")
    return Numbered(
        kind=KIND_CHINESE,
        no=label,
        mark=marks.find_marks(line),
        text=m.group("rest").strip(),
        top=chinese_to_int(label),
        level=1,
        raw=line,
    )


def parse_paren(line: str) -> Numbered | None:
    """识别 ``（N）`` 编号（P1 续页 ``（1）分辨率≥16 万像素…``）。

    这类编号**没有自己的层级**：在服务标的里展开成 ``父.N``，在设备标的里并入父项
    （见 :mod:`bidtable.ingest.section`，agent.md 的 ``（N）`` 策略）。
    """
    if not line:
        return None
    m = PAREN_RE.match(line)
    if not m:
        return None
    return Numbered(
        kind=KIND_PAREN,
        no=m.group("num"),
        mark=marks.find_marks(line),
        text=m.group("rest").strip(),
        top=int(m.group("num")),
        level=1,
        raw=line,
    )


def parse_lead(line: str) -> Numbered | None:
    """识别**引导句**：以 ``：`` 结尾、且自身不带编号。

    ``设备运维技术服务要求如下：`` -> 是；``具体技术参数：`` -> 是；
    ``十一、妇科电动检查床：`` -> 否（它带中文序数，走 :func:`parse_chinese`）。
    """
    s = (line or "").strip()
    if not s or not LEAD_RE.search(s):
        return None
    if parse_item(s) or parse_chinese(s) or parse_paren(s):
        return None
    return Numbered(kind=KIND_LEAD, no="", mark="", text=s, top=0, level=0, raw=line)


def parse_line(line: str) -> Numbered | None:
    """按优先级尝试所有形态：中文序数 > 阿拉伯编号 > 括号编号 > 引导句。"""
    return parse_chinese(line) or parse_item(line) or parse_paren(line) or parse_lead(line)


def starts_with_paren(text: str) -> bool:
    """这一行是不是以 ``（N）`` **项目符号**开头。

    ``（N）`` 是排版符号，不是文档自己的层级编号，所以它撑起来的「父子关系」
    不能用来判定标题：P1/P2 标的2 的 ``2、设备运行维护：`` 在原文里并没有
    ``2.1`` 子项，是我们把 ``（1）（2）（3）`` 展开才产生的（见
    :mod:`bidtable.ingest.fold`）。而 P3 标的2 的 ``2.1``/``2.2``/``2.3`` 是原文
    里真写着的编号 —— 两者在 ``分节`` 策略上必须区别对待。
    """
    return bool(PAREN_RE.match((text or "").strip()))


class Sequencer:
    """序号合理性闸门。

    光靠正则会把 ``0.l-5ml／每次最大给药量`` 当成编号 0 的新条目（X 系列混淆 case）。
    这里再加一条**上下文**约束：

    * 顶层号必须落在 ``(last_top, last_top + MAX_GAP]``；``0`` 永远不合法。
    * 子层号必须与**上一条被接受的编号**同父前缀，且在该父下严格递增；深度不限。
      P4 是三层：``1.9`` -> ``1.9.1`` -> … -> ``1.9.6`` -> ``1.10``。
      只记「二级号最大值」会把 ``1.9.1`` 判成 ``9 > 9`` 的倒退而整块丢掉。
    * 中文序数标题 / 引导句总是**重置**游标（它们之下的编号重新开始）。
    """

    def __init__(self, max_gap: int = MAX_GAP) -> None:
        self.max_gap = max_gap
        self.reset()

    def reset(self) -> None:
        self.last_top = 0
        self._last_parts: list[int] = []

    @property
    def last_no(self) -> str:
        """上一条被接受的编号（``"1.9.3"``）；还没有则为 ``""``。诊断/告警用。"""
        return ".".join(str(x) for x in self._last_parts)

    def plausible(self, n: Numbered) -> bool:
        if n.kind == KIND_CHINESE:
            return True
        if n.kind == KIND_LEAD:
            return True
        if n.kind == KIND_PAREN:
            return n.top >= 1 and n.top <= 30
        if n.level == 1:
            if n.top < 1:
                return False
            if self.last_top == 0:
                return n.top == 1 or n.top <= self.max_gap
            return self.last_top < n.top <= self.last_top + self.max_gap
        # 子层：父号必须是当前顶层号，且与上一条同父前缀、本级严格递增
        parts = int_parts(n.no)
        if not parts or parts[0] != self.last_top or parts[0] < 1:
            return False
        depth = len(parts) - 1  # 父前缀长度
        if parts[:depth] != self._last_parts[:depth]:
            return False
        if len(self._last_parts) <= depth:
            return True  # 新父项下的第一个子项（``1.9`` -> ``1.9.1``）
        return parts[depth] > self._last_parts[depth]

    def advance(self, n: Numbered) -> None:
        if n.kind in (KIND_CHINESE, KIND_LEAD):
            # 中文序数标题 / 引导句之下的编号是**重新开始**的：P3 ``一、`` 下有 1..6，
            # ``二、`` 下又从 ``1、4K 采集摄像头`` 起。不重置游标的话，``二、`` 之后的
            # ``1`` 会被判成「1 <= 6」的倒退而整条丢弃（P3 从 71 行掉到 26 行的根因）。
            self.reset()
            return
        if n.level == 1:
            self.last_top = n.top
        self._last_parts = int_parts(n.no)

    def offer(self, n: Numbered) -> bool:
        """:meth:`plausible` + :meth:`advance` 的合并入口。"""
        if not self.plausible(n):
            return False
        self.advance(n)
        return True


def has_children(items: list[Numbered], parent: Numbered) -> bool:
    """标题 vs 参数项的**确定性判别式**（agent.md 3.5）：

    ``N、Title`` 是标题 **当且仅当** 存在编号为 ``N.k`` 的子项。

    已验证：P3 ``1、4K 采集摄像头`` -> 标题（有 1.1~1.6）；
    P3 ``三、`` 下的 ``1、数量：2 条`` -> 参数项（下一条是 ``2、``）；
    P5 标的5 ``1、治疗源`` -> 标题；P1 ``1、技术培训：…`` -> 参数项。
    """
    if parent.kind == KIND_CHINESE:
        return True  # 中文序数标题总是进路径（S4 规则 2）
    if parent.level != 1:
        return False
    prefix = normalize_no(parent.no) + "."
    return any(x.kind == KIND_ITEM and normalize_no(x.no).startswith(prefix) for x in items)


def mark_headings(items: list["Item"]) -> list[bool]:
    """:func:`has_children` 的**窗口化**版本，实际抽取走这一条。

    为什么不能全块扫描：P3 标的1 有 ``一、``~``十一、`` 十一个中文序数节，
    每节下面都从 ``1、`` 重新编号。全块扫描时，``一、`` 下的 ``1、3D 输出方式``
    会因为 ``二、`` 下存在 ``1.1`` 而被误判成标题——P3 因此从 71 行掉到 57 行，
    并产生 35 个重复键。

    窗口规则：从当前项向后扫，遇到**同级兄弟**（``level == 1`` 的整数项）
    或**中文序数标题 / 引导句**即停；窗口内出现 ``N.k`` 才算有子项。
    """
    out: list[bool] = []
    for i, it in enumerate(items):
        n = it.numbered
        if n is None:
            # 块首孤儿续行（跨页单元格的尾巴）：没有编号，谈不上标题。
            # 这一支必须在访问 n.kind 之前，否则 AttributeError。
            out.append(False)
            continue
        if n.kind in (KIND_CHINESE, KIND_LEAD):
            out.append(True)  # S4 规则 2 / 规则 4：中文序数与引导句总是进路径
            continue
        if n.kind == KIND_PAREN or n.level != 1:
            out.append(False)
            continue
        prefix = normalize_no(n.no) + "."
        found = False
        for other in items[i + 1 :]:
            m = other.numbered
            if m is None:
                continue
            if m.kind in (KIND_CHINESE, KIND_LEAD):
                break
            if m.kind == KIND_ITEM and m.level == 1:
                break
            if m.kind == KIND_ITEM and normalize_no(m.no).startswith(prefix):
                found = True
                break
        out.append(found)
    return out


@dataclass
class Item:
    """参数单元格里的**一个条目**（编号 + 其全部续行）。

    ``lines`` 保留 PDF 的原始行切分；:attr:`text` 按 agent.md 3.6 的规则
    **直接拼接**（不补空格），得到与 gold 逐字一致的原文。
    """

    numbered: Numbered | None
    lines: list[str] = field(default_factory=list)
    page: int = 0

    @property
    def text(self) -> str:
        return "".join(self.lines)

    @property
    def no(self) -> str:
        return self.numbered.no if self.numbered else ""

    @property
    def mark(self) -> str:
        return self.numbered.mark if self.numbered else ""

    @property
    def kind(self) -> str:
        return self.numbered.kind if self.numbered else ""

    @property
    def level(self) -> int:
        return self.numbered.level if self.numbered else 0


def split_items(
    lines: Iterable[tuple[str, int]],
    *,
    seq: Sequencer | None = None,
    allow_chinese: bool = True,
    allow_lead: bool = True,
    paren: str = PAREN_FOLD,
    guard_bare: bool = True,
) -> list[Item]:
    """把「(行文本, 页码)」序列切成条目列表。

    判据是**两道闸**：

    1. :func:`parse_line` 认得出编号形态；
    2. :class:`Sequencer` 认为该编号在当前上下文里**合理递增**。

    两闸都过 -> 新条目；否则 -> 上一条目的续行（拼接）。
    第 2 闸是必须的：``0.l-5ml／每次最大给药量`` 能过第 1 闸（编号 ``0``），
    但顶层号 ``0`` 不合理，于是被正确判成续行。

    ``paren`` 控制 ``（N）`` 的处理（agent.md 的 ``（N）`` 策略，**跨包不一致**）：

    * :data:`PAREN_FOLD` —— 并入上一条目。设备标的用这个：P1 序号14 的
      ``（1）分辨率≥16 万像素…（2）…（3）…`` 在 gold 里就是一整条。
    * :data:`PAREN_SPLIT` —— 独立成条（``kind=paren``），由调用方展开成 ``父.N``。
      服务标的用这个：P2 标的2 的 gold 序号是 ``2.1``/``2.2``/``2.3``。
    * :data:`PAREN_DROP` —— 整行丢弃（``（N）`` 只是排版噪声时用）。

    ``guard_bare`` 是**第三道闸**（X 系列混淆 case）。训练集里有两类续行会被
    正则当成新条目，而且都能骗过 :class:`Sequencer` 的跳号检查::

        P3 标的2   …每年提供≥        <- 上一行不以终止标点收尾
                   4 次的设备维护保养服务…    -> 误判成序号 4（2<4<=7 合理）
        P3 标的1   …具备
                   6 孔及4 孔的脚控接口。      -> 误判成序号 6

    判别式：**编号与正文之间是空白**（不是 ``、``/``.``/``．``）、**顶层条目**、
    **正文非空** 三者同时成立时，额外要求**上一被接受的行以终止标点收尾**；
    否则整行降级为上一条的续行。对真条目无副作用：``2.1``/``1.9.3`` 是子层或
    正文为空，``1、``/``19.``/``13.1、`` 用的是显式标点分隔符。
    """
    seq = seq if seq is not None else Sequencer()
    out: list[Item] = []
    prev_terminal = True  # 块首无前文，按「可以起新条」处理
    for raw, page in lines:
        s = raw.strip()
        if not s:
            continue
        if paren == PAREN_DROP and PAREN_RE.match(s):
            continue
        n = _try_parse(s, allow_chinese, allow_lead, paren)
        if n is not None and guard_bare and _is_bare_continuation(n, prev_terminal):
            n = None
        if n is not None and seq.plausible(n):
            seq.advance(n)
            out.append(Item(numbered=n, lines=[s], page=page))
            prev_terminal = bool(s) and s[-1] in TERMINAL_END
        elif out:
            out[-1].lines.append(s)
            prev_terminal = bool(s) and s[-1] in TERMINAL_END
        else:
            # 块首就是续行（例如跨页单元格的尾巴）：单独成一个无编号条目，
            # 交给调用方决定并入还是丢弃。
            out.append(Item(numbered=None, lines=[s], page=page))
            prev_terminal = bool(s) and s[-1] in TERMINAL_END
    return out


def _is_bare_continuation(n: Numbered, prev_terminal: bool) -> bool:
    """第三道闸：这行是不是「被误判成条目」的续行。

    两类形态，判据都是**上一被接受的行没有以终止标点收尾**：

    1. 空白分隔符的假条目（``4 次的设备维护保养服务…``、``6 孔及4 孔的脚控接口。``）。
       额外要求顶层 + 正文非空 + 分隔符是空白，免得误伤 ``2.1``/``1.9.3`` 这类真子项。
    2. **被折行的标题尾巴**。P3 标的1 的 ``九、配置适合腹腔镜及开腹手术使用电刀：``
       在 PDF 单元格里折成两行，第二行 ``使用电刀：`` 以 ``：`` 结尾，
       会被 :func:`parse_lead` 认成引导句而另起一条 —— 于是 ``分节`` 变成截断的
       ``九、配置适合腹腔镜及开腹手术``，该节 4 行的对齐键全错（覆盖率 71→67）。
       真正的引导句（``设备运维技术服务要求如下：``）都出现在块首或上一句已收尾之后，
       所以这条闸不会误伤它们。
    """
    if n.kind == KIND_LEAD:
        return not prev_terminal
    if n.kind != KIND_ITEM or n.level != 1 or not n.text:
        return False
    if n.sep.strip() != "":  # ``、`` / ``.`` / ``．`` 是真分隔符，放行
        return False
    return not prev_terminal


def _try_parse(s: str, allow_chinese: bool, allow_lead: bool, paren: str) -> Numbered | None:
    if allow_chinese:
        n = parse_chinese(s)
        if n:
            return n
    n = parse_item(s)
    if n:
        return n
    if paren == PAREN_SPLIT:
        n = parse_paren(s)
        if n:
            return n
    if allow_lead:
        return parse_lead(s)
    return None


def iter_numbered(items: Iterable[Item]) -> Iterator[Item]:
    """只保留带编号的条目（引导句 ``kind=lead`` 保留，无编号的孤儿续行剔除）。"""
    for it in items:
        if it.numbered is not None:
            yield it
