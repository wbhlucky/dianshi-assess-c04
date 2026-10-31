"""表格几何：横线/竖线探测、列切分、行带切分、单元格取词、跨页拼接。

**头号陷阱**（agent.md 3.6）：PyMuPDF 的 ``find_tables()`` 会截断跨页单元格，
还会重新插入空格。实测 P1 p8 用 ``find_tables()`` 只能拿到 15 条参数里的 3 条。
⇒ 本模块**完全不用** ``find_tables()``，改为自己从矢量绘图指令里恢复表格几何，
再用 ``get_text(clip=...)`` 取词。这样跨页单元格的续页也能按 x 区间正确切出来。

所有阈值都是**相对页面尺寸**或来自训练集实测的宽松常量，不写死任何一页的坐标。
"""

from __future__ import annotations

from dataclasses import dataclass

import fitz

# 一条线要被当成「表格线」，跨度至少要这么长（pt）。用于滤掉页眉/页脚的短装饰线。
MIN_H_SPAN = 200.0
MIN_V_SPAN = 60.0
# 离页面边缘多近就不算表格线（页眉横线常常贴边且 x 范围与表体不同）。
EDGE_MARGIN = 5.0
# 判断「水平/垂直」的容差（pt）。扫描件里线条会有轻微倾斜。
FLAT_TOL = 0.6


@dataclass(frozen=True)
class HRule:
    """一条水平表格线：``y`` 坐标 + 左右端点。"""

    y: float
    x0: float
    x1: float

    @property
    def span(self) -> float:
        return self.x1 - self.x0


@dataclass(frozen=True)
class VRule:
    """一条垂直表格线：``x`` 坐标 + 上下端点。"""

    x: float
    y0: float
    y1: float

    @property
    def span(self) -> float:
        return self.y1 - self.y0


def page_rules(page: fitz.Page) -> tuple[list[HRule], list[VRule]]:
    """从矢量绘图指令恢复横线与竖线。

    同时处理 ``l``（直线段）与 ``re``（矩形）两种指令：细长的矩形就是表格线，
    有的 PDF 用矩形描边画线。
    """
    hs: list[HRule] = []
    vs: list[VRule] = []
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                if abs(p1.y - p2.y) < FLAT_TOL:
                    hs.append(HRule(round(p1.y, 2), min(p1.x, p2.x), max(p1.x, p2.x)))
                elif abs(p1.x - p2.x) < FLAT_TOL:
                    vs.append(VRule(round(p1.x, 2), min(p1.y, p2.y), max(p1.y, p2.y)))
            elif it[0] == "re":
                r = it[1]
                if r.height < 1.5 and r.width > MIN_H_SPAN * 0.5:
                    hs.append(HRule(round(r.y0, 2), r.x0, r.x1))
                elif r.width < 1.5 and r.height > MIN_V_SPAN * 0.5:
                    vs.append(VRule(round(r.x0, 2), r.y0, r.y1))
    return hs, vs


def column_edges(page: fitz.Page, vs: list[VRule], min_span: float = MIN_V_SPAN) -> list[int]:
    """表格的列边界（去重、排序后的整数 x）。

    只保留**贯穿表体**的长线（``span > min_span``）并排除贴边的，
    这样页眉/页脚里的短竖线不会污染列切分。
    """
    w = page.rect.width
    xs = {
        round(v.x)
        for v in vs
        if v.span > min_span and EDGE_MARGIN < v.x < w - EDGE_MARGIN
    }
    return sorted(xs)


def row_bands(page: fitz.Page, hs: list[HRule], min_span: float = MIN_H_SPAN) -> list[float]:
    """表格的行分界 y 坐标。相邻两个构一个「行带」。"""
    h = page.rect.height
    ys = {
        round(x.y, 1)
        for x in hs
        if x.span > min_span and EDGE_MARGIN < x.y < h - EDGE_MARGIN
    }
    return sorted(ys)


def bands_from(band_ys: list[float]) -> list[tuple[float, float]]:
    """把 y 坐标列表转成 ``(y0, y1)`` 区间列表。"""
    return [(band_ys[i], band_ys[i + 1]) for i in range(len(band_ys) - 1)]


def join_lines(text: str, sep: str = "") -> str:
    """单元格文本重建（agent.md 3.6）：**换行直接删除，不补空格**，保留原有内部空格。

    对照实例：P1 ``序号2`` 的 gold 原文 = ``▲2.最大单脉冲能量5J。（大于5J 为优于）``，
    而 PDF 原始是 ``▲2.最大单脉冲能量5J。（大于5J 为\\n优于）``。
    PyMuPDF 把行尾空格留在行内，所以直接拼接即可得到与 gold 逐字一致的结果。
    """
    return sep.join((text or "").split("\n"))


def cell_text(page: fitz.Page, x0: float, x1: float, y0: float, y1: float, *, sep: str = "") -> str:
    """按矩形裁切取词并重建文本。四边各内缩一点，避免吃到相邻单元格的字。"""
    clip = fitz.Rect(x0 + FLAT_TOL, y0 + FLAT_TOL, x1 - FLAT_TOL, y1 - FLAT_TOL)
    raw = page.get_text("text", clip=clip)
    return join_lines(raw, sep)


@dataclass
class BandCells:
    """一个行带在各列上的文本。"""

    page_no: int  # 1 起，与 ``位置`` 列的「第N页」口径一致
    y0: float
    y1: float
    cells: list[str]

    def get(self, idx: int) -> str:
        return self.cells[idx] if 0 <= idx < len(self.cells) else ""

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def band_cells(
    page: fitz.Page,
    cols: list[int],
    band_ys: list[float],
    page_no: int,
    *,
    sep: str = "",
) -> list[BandCells]:
    """在一页上按 ``cols × bands`` 网格取词。"""
    out: list[BandCells] = []
    if len(cols) < 2:
        return out
    for y0, y1 in bands_from(band_ys):
        cells = [cell_text(page, cols[c], cols[c + 1], y0, y1, sep=sep) for c in range(len(cols) - 1)]
        out.append(BandCells(page_no=page_no, y0=y0, y1=y1, cells=cells))
    return out


def is_empty_band(band: BandCells) -> bool:
    return not any(c.strip() for c in band.cells)


def table_xspan(page: fitz.Page, hs: list[HRule], min_span: float = MIN_H_SPAN) -> tuple[float, float] | None:
    """表体横线的公共 x 区间。B 族靠它把 ``x0`` 当列分割线（B 族没有竖线）。"""
    long_hs = [h for h in hs if h.span > min_span]
    if not long_hs:
        return None
    x0 = max(h.x0 for h in long_hs)
    x1 = min(h.x1 for h in long_hs)
    return (x0, x1) if x1 > x0 else None


def dominant_hspan(page: fitz.Page, hs: list[HRule], min_span: float = MIN_H_SPAN) -> tuple[float, float] | None:
    """出现次数最多的横线 x 区间（表体主区间），用于剔除页眉单线。"""
    counts: dict[tuple[int, int], int] = {}
    for h in hs:
        if h.span <= min_span:
            continue
        key = (round(h.x0), round(h.x1))
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    best = max(counts.items(), key=lambda kv: kv[1])[0]
    return (float(best[0]), float(best[1]))


def filter_bands_to_table(band_ys: list[float], lo: float, hi: float) -> list[float]:
    """只保留落在 ``[lo, hi]`` 内的分界线（用于 B 族剔掉页眉线）。"""
    return [y for y in band_ys if lo - 1.0 <= y <= hi + 1.0]
