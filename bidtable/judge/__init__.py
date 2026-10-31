"""偏离判定层（agent.md 4.1 / 4.3 L3）。

**本文件刻意只有 docstring**，理由同 :mod:`bidtable.match`：
``judge.units`` 是 ``match.normalize`` 的下游，包初始化不能引入反向依赖。

对外入口：:func:`bidtable.judge.decide.judge_all`。
"""
