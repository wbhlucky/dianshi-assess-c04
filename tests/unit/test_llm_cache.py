"""内容哈希磁盘缓存（EC-J4 缓存开关与命中/失效）。

B18 / N10 的「第二次运行 API 调用数为 0」全靠这一层，所以它的失效路径
（关掉、损坏、写不进去）必须和命中路径一样被钉住：缓存出问题时只能退化成
「多打一次 API」，绝不能让主流程崩掉。
"""

from __future__ import annotations

import json

import pytest

from bidtable.llm.cache import DiskCache, cache_key

pytestmark = [pytest.mark.unit]

MSGS = [{"role": "user", "content": "输出功率：≥750w"}]


@pytest.mark.ec("J4")
def test_cache_key_ignores_layout_whitespace_but_not_semantics():
    """排版抖动（多空格、制表符、首尾空白、字典键序）不该造成 miss；
    而 role / model / temperature / 正文用词必须造成不同的键。"""
    # _normalize_text 把连续空白折叠成「一个」空格再去首尾，所以基线自己得带那个空格，
    # 排版抖动才折叠得回去；它并不删除词内空白。
    plain = [{"role": "user", "content": "输出功率： ≥750w"}]
    base = cache_key("ocr", "m", plain)
    padded = [{"role": "user", "content": "  输出功率：\t \t ≥750w   "}]
    assert cache_key("ocr", "m", padded) == base
    assert cache_key("ocr", "m", [{"content": "输出功率： ≥750w", "role": "user"}]) == base
    assert cache_key("ocr", "m", plain, temperature=None) == base      # None 参数被丢弃
    assert cache_key("ocr", "m", [{"role": "user", "content": "输出功率：≥750w"}]) != base
    assert cache_key("ocr", "m", [{"role": "user", "content": "输出功率： ≥751w"}]) != base
    assert cache_key("judge", "m", plain) != base
    assert cache_key("ocr", "other", plain) != base
    assert cache_key("ocr", "m", plain, temperature=0) != cache_key("ocr", "m", plain, temperature=1)
    assert len(base) == 64 and all(c in "0123456789abcdef" for c in base)


@pytest.mark.ec("J4")
def test_round_trip_shards_by_key_prefix_and_leaves_no_temp_file(tmp_path):
    """写入走 ``.tmp`` + ``replace``（原子），读回内容一致，中文不被转义。"""
    c = DiskCache(tmp_path / "llm")
    key = cache_key("ocr", "m", MSGS)
    assert c.get(key) is None and c.misses == 1

    c.put(key, {"text": "输出功率：600W", "usage": {"prompt_tokens": 181}})
    path = c._path(key)
    assert path.parent.name == key[:2] and path.name == f"{key}.json"
    assert not path.with_suffix(".tmp").exists()
    assert json.loads(path.read_text(encoding="utf-8"))["text"] == "输出功率：600W"

    got = c.get(key)
    assert got == {"text": "输出功率：600W", "usage": {"prompt_tokens": 181}}
    assert (c.hits, c.misses) == (1, 1)


@pytest.mark.ec("J1")
def test_disabled_cache_neither_reads_nor_writes(tmp_path):
    """``--offline`` 时 enabled=False：既不能命中已有缓存，也不能往磁盘落东西。"""
    DiskCache(tmp_path / "llm").put("k" * 64, {"text": "预热"})
    off = DiskCache(tmp_path / "llm", enabled=False)
    assert off.get("k" * 64) is None and off.misses == 1 and off.hits == 0
    off.put("j" * 64, {"text": "不该落盘"})
    assert not (tmp_path / "llm" / "jj" / f"{'j' * 64}.json").exists()


@pytest.mark.ec("J4")
def test_corrupt_or_non_dict_cache_entry_counts_as_a_miss(tmp_path):
    """半截 JSON 只能算未命中——缓存损坏不该把整条流水线带崩。"""
    c = DiskCache(tmp_path / "llm")
    for i, blob in enumerate(["{ 不是 json", '["列表不是字典"]', ""]):
        key = f"{i:064d}"
        c._path(key).parent.mkdir(parents=True, exist_ok=True)
        c._path(key).write_text(blob, encoding="utf-8")
        assert c.get(key) is None, blob
    assert (c.hits, c.misses) == (0, 3)


@pytest.mark.ec("J7")
def test_unwritable_cache_directory_is_swallowed(tmp_path):
    """缓存目录不可写（例如被一个同名普通文件占住）时 put 静默失败，主流程继续。"""
    blocker = tmp_path / "llm"
    blocker.write_text("我是一个文件，不是目录", encoding="utf-8")
    c = DiskCache(blocker)
    c.put(cache_key("ocr", "m", MSGS), {"text": "x"})          # 不抛
    assert c.get(cache_key("ocr", "m", MSGS)) is None


@pytest.mark.ec("J4")
def test_clear_drops_every_shard_but_keeps_the_object_usable(tmp_path):
    c = DiskCache(tmp_path / "llm")
    for i in range(3):
        c.put(f"{i:064x}", {"text": str(i)})
    assert len(list((tmp_path / "llm").rglob("*.json"))) == 3
    c.clear()
    assert not (tmp_path / "llm").exists()
    c.put(f"{9:064x}", {"text": "9"})                            # 清空后仍可继续用
    assert c.get(f"{9:064x}") == {"text": "9"}
    DiskCache(tmp_path / "从不存在的目录").clear()                 # 幂等，不抛
