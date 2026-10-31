# -*- coding: utf-8 -*-
"""golden 层的共享夹具：把「离线出表 + ``grade.py`` 打分」做成每包只跑一次。

golden 的六个文件都要拿同一个东西——某个包的成品 xlsx 与它对标准答案的得分。
如果每个文件各跑一遍，五个包会被重复构建五次；所以这里按包记忆化。
记忆化能成立的前提是打分**只读**：``score_pair`` 不改 ``PackResult``，
``writer.write`` 写到调用方给的临时路径，因此第二次拿到的结果与第一次全等。

与 ``tests/conftest.py`` 的分工：那边记忆化的是 ``process_pack`` 的产物
（阶段缓存命中，不碰 PDF/docx/jpg），这边记忆化的是它的**下游**——
落盘成 xlsx 再交给 ``grade.py``。两层缓存互相独立，谁先跑都不影响结果。
"""

from __future__ import annotations

import pathlib
from typing import Any, Callable

import pytest

from bidtable import selfcheck as selfcheck_mod
from bidtable.models import PackResult
from bidtable.output import writer

# 包名 -> (PackResult, grade.py 的得分 dict)
_SCORE_MEMO: dict[str, tuple[PackResult, dict[str, Any]]] = {}


@pytest.fixture
def score_pack(
    tmp_path: pathlib.Path,
    offline_run: Callable[..., PackResult],
    grade: Callable[[pathlib.Path, pathlib.Path], dict[str, Any]],
    repo_root: pathlib.Path,
) -> Callable[[str], tuple[PackResult, dict[str, Any]]]:
    """``score_pack("P1")`` -> ``(PackResult, 得分 dict)``，同一包只真跑一次。"""

    def _get(pack: str) -> tuple[PackResult, dict[str, Any]]:
        if pack in _SCORE_MEMO:
            return _SCORE_MEMO[pack]
        result = offline_run(pack)
        gold = selfcheck_mod.find_gold(repo_root / "训练集" / pack)
        if gold is None:
            pytest.skip(f"{pack} 里没有 04_标准答案_*.xlsx，无法打分")
        template = repo_root / result.stats["template_xlsx"]
        out = tmp_path / f"{pack}.xlsx"
        written, warns = writer.write(
            result.rows, out, template=template if template.is_file() else None
        )
        # 模板路径完好时写出必须是零告警的；有告警说明成品本身就有问题。
        assert warns == [], warns
        _SCORE_MEMO[pack] = (result, grade(gold, written))
        return _SCORE_MEMO[pack]

    return _get
