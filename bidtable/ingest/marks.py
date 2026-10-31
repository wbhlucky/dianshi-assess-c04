"""``★`` / 半角 ``*`` / ``▲`` 的抽取与剥离。

三条硬事实（agent.md 3.9.4）：

1. 半角 ``*`` 一律当 ``★``（每个 ``00_投标范围.txt`` 都这么写）。
2. ``序号`` 必须是**裸编号**：``▲2.最大单脉冲能量5J。`` -> 序号 ``2``、标记 ``▲``。
3. ``招标要求（原文）`` **逐字保留**标记字符（P1 就把 ``▲`` 留在原文里），
   因为原文不参与打分，而剥掉反而会让复核的人对不上 PDF。

陷阱：``3840*2160``、``0.1~20m1／h`` 这类内容里的 ``*`` 是**乘号**，不是标记。
所以 ``*`` 只在「整行首字符」或「整行末字符」两个位置才被认作标记。
"""

from __future__ import annotations

MARK_STAR = "★"
MARK_TRIANGLE = "▲"
MARK_CHARS = (MARK_STAR, MARK_TRIANGLE)
ALL_MARK_TOKENS = (MARK_STAR, MARK_TRIANGLE, "*")

# 两个标记同时出现时 ``★`` 优先：它是实质性要求，风险等级更高。
_PRIORITY = {MARK_STAR: 0, MARK_TRIANGLE: 1}


def normalize_token(ch: str) -> str:
    """把半角 ``*`` 归一成 ``★``；其余原样返回。"""
    if ch == "*":
        return MARK_STAR
    return ch


def find_marks(text: str) -> str:
    """返回文本中出现的最高优先级标记，没有则 ``""``。

    只统计 ``★``/``▲``（这两个字符不会当标点用）；半角 ``*`` 需要位置判定，
    由 :func:`edge_mark` 处理。
    """
    best = ""
    for ch in text or "":
        if ch in _PRIORITY and (not best or _PRIORITY[ch] < _PRIORITY[best]):
            best = ch
    return best


def edge_mark(text: str) -> tuple[str, str]:
    """识别「贴边」的标记字符，返回 ``(标记, 去掉该标记后的文本)``。

    ``*1.2 输出功率：≥750w``    -> ``("★", "1.2 输出功率：≥750w")``
    ``▲2.最大单脉冲能量5J。``    -> ``("▲", "2.最大单脉冲能量5J。")``
    ``3840*2160``              -> ``("",  "3840*2160")``（乘号，必须原样保留）

    ``▲`` 前缀是 A 族最常见的一类（P1 ``▲2.``、P2 ``▲1.4 、``、P3 ``▲1、``），
    不在这里剥掉，编号正则就会整条失配，导致该条目被并进上一条。
    """
    s = text or ""
    stripped_left = s.lstrip()
    if stripped_left[:1] in ("*", MARK_STAR, MARK_TRIANGLE):
        indent = s[: len(s) - len(stripped_left)]
        return normalize_token(stripped_left[0]), indent + stripped_left[1:]
    return "", s


def strip_all(text: str) -> str:
    """删除所有标记字符（含贴边 ``*``）。**仅用于编号解析**，不用于输出原文。"""
    _, s = edge_mark(text)
    out = []
    for i, ch in enumerate(s):
        if ch in MARK_CHARS:
            continue
        if ch == "*" and (i == 0 or i == len(s) - 1):
            continue
        out.append(ch)
    return "".join(out)


def strip_trailing_mark(text: str) -> tuple[str, str]:
    """剥离**紧跟在编号后面**的尾缀标记，例如 ``1.7★`` -> ``("★", "1.7")``。

    只有当尾缀标记之前全是编号字符（数字、点、顿号）时才剥，
    否则 ``2.最大单脉冲能量5J。（大于5J 为优于）★`` 这种会把正文吃掉。
    """
    s = (text or "").rstrip()
    if not s:
        return "", text
    last = s[-1]
    token = normalize_token(last) if last in ("*", *MARK_CHARS) else ""
    if not token:
        return "", text
    head = s[:-1].rstrip()
    if head and all(c in "0123456789.．、" for c in head):
        return token, head
    return "", text


# 兼容旧名（``trailing_mark`` 曾是本函数的名字）。
trailing_mark = strip_trailing_mark
