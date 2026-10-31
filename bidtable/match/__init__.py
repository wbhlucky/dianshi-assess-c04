"""要求 -> 证据匹配层（agent.md 4.1）。

**本文件刻意只有 docstring。** 依赖方向是 ``judge.units -> match.normalize``：
单位量纲表复用匹配层的同一份分词表，避免两处维护。如果这里 eager import
``candidates``（它又要 ``judge.units``），先导入 judge 的一侧就会拿到一个
半初始化的 units 模块。保持包初始化为空，两个子包就都能从任意方向先加载。

对外入口：:func:`bidtable.match.candidates.match_all`。
"""
