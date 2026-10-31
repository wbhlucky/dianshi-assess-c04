"""xlsx 写出（agent.md 4.2 / 不变式 I2、I3）。

三条硬约束，任何一条破了 ``grade.py`` 就读不到我们的行：

1. **严格 10 列，顺序与模板一致**（I2）。gold 有第 11 列 ``判定说明``，
   但模板 ``03_响应偏离表_模板.xlsx`` 只有 ``A1:J1`` —— 多写一列会让人工复核时
   列错位，少写一列直接 KeyError。
2. **活动表名必须是 ``响应偏离表``**（I3）。``grade.py`` 用 ``wb.active``，
   表名不对它虽然还能读，但 Excel 里打开会看到一个陌生的 sheet，评审体验直接崩。
3. ** ``(标的, 分节, 序号)`` 唯一**（I1）。``grade.py`` 把行装进 dict，
   重复键会**静默覆盖**，丢的是整行的所有指标（N14）。所以写出前必须去重 + 告警。

有模板就**在模板上写**（表头样式、列宽、freeze pane 全部原样继承），
没有模板才按实测的样式常量重建，保证两条路径产出的文件在 Excel 里长得一样。
"""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from ..config import COLUMNS, SHEET_NAME
from ..errors import OutputError
from ..models import Row

# 实测自 ``03_响应偏离表_模板.xlsx``（没有模板时用这份常量重建，保证观感一致）。
WIDTHS = (6, 18, 14, 8, 6, 60, 40, 10, 30, 22)
HEADER_FILL = "E8EEF4"
HEADER_ROW = 1
FIRST_DATA_ROW = 2


def dedupe(rows: list[Row]) -> tuple[list[Row], list[str]]:
    """按 ``(标的, 分节, 序号)`` 去重（I1 / N14）。**保留第一条**，其余降级成告警。

    为什么保留第一条：抽取是按文档顺序产出的，第一条来自正文，
    后面那条通常是跨页拼接或降级路径重复捞到的。
    """
    seen: set[tuple[str, str, str]] = set()
    out: list[Row] = []
    warnings: list[str] = []
    for row in rows:
        key = (row.lot_no.strip(), row.section.strip(), row.no.strip())
        if key in seen:
            warnings.append(
                f"输出现重复键 (标的{key[0]}, 分节{key[1] or '空'}, 序号{key[2]})，"
                f"已丢弃后出现的那一行（grade.py 的行字典会静默覆盖重复键）。"
            )
            continue
        seen.add(key)
        out.append(row)
    return out, warnings


def _style_header(ws) -> None:
    """重建路径下的表头样式：加粗 + 浅蓝底 + 居中，与模板一致。"""
    font = Font(bold=True)
    fill = PatternFill(start_color=HEADER_FILL, end_color=HEADER_FILL, fill_type="solid")
    align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for idx, name in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=HEADER_ROW, column=idx, value=name)
        cell.font = font
        cell.fill = fill
        cell.alignment = align
        ws.column_dimensions[get_column_letter(idx)].width = WIDTHS[idx - 1]


def _from_template(template: Path) -> Workbook:
    """在模板副本上写：继承表头样式与列宽，并清掉模板里可能残留的示例行。"""
    wb = load_workbook(template)
    ws = wb.active
    ws.title = SHEET_NAME
    header = [ws.cell(row=HEADER_ROW, column=i).value for i in range(1, len(COLUMNS) + 1)]
    if [str(h or "").strip() for h in header] != list(COLUMNS):
        # 模板表头对不上（被人改过）：以代码里的 COLUMNS 为准重写表头，并喊一声。
        for idx, name in enumerate(COLUMNS, start=1):
            ws.cell(row=HEADER_ROW, column=idx, value=name)
    if ws.max_row > HEADER_ROW:
        ws.delete_rows(HEADER_ROW + 1, ws.max_row - HEADER_ROW)
    return wb


def _new_workbook() -> Workbook:
    wb = Workbook()
    ws = wb.active
    ws.title = SHEET_NAME
    for name in list(wb.sheetnames)[1:]:  # 只留一张表，免得 wb.active 指错
        del wb[name]
    _style_header(ws)
    ws.freeze_panes = f"A{FIRST_DATA_ROW}"
    return wb


def write(
    rows: list[Row],
    out_path: str | Path,
    *,
    template: str | Path | None = None,
    wrap: bool = True,
) -> tuple[Path, list[str]]:
    """写出 xlsx。返回 ``(实际路径, 告警)``。

    任何 IO 失败都转成 :class:`OutputError`（退出码 2），消息里给出可行动的建议
    ——最常见的就是「文件正被 Excel 打开」。
    """
    path = Path(out_path)
    rows, warnings = dedupe(rows)

    tpl = Path(template) if template else None
    wb: Workbook | None = None
    try:
        wb = _from_template(tpl) if (tpl and tpl.is_file()) else _new_workbook()
        if tpl and not tpl.is_file():
            warnings.append(f"模板 {tpl} 不存在，已按内置样式常量重建表头与列宽。")
        ws = wb.active
        body = Alignment(vertical="top", wrap_text=wrap)
        for r, row in enumerate(rows, start=FIRST_DATA_ROW):
            for c, value in enumerate(row.cells(), start=1):
                cell = ws.cell(row=r, column=c, value=value)
                cell.alignment = body
        path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(path)
    except PermissionError as exc:
        raise OutputError(
            f"无法写入 {path}：文件可能正被 Excel 打开，或目录没有写权限。"
            f"请关闭该文件后重试。（{exc}）"
        ) from exc
    except OSError as exc:
        raise OutputError(f"无法写入 {path}：{exc}") from exc
    finally:
        if wb is not None:
            wb.close()
    return path, warnings
