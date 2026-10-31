"""``bidtable/pipeline.py``：七阶段编排层。

这一层自己不做业务判断，它的职责是"把六个子系统按固定顺序串起来，并且在
任何一环出问题时给出可读的降级信息"。所以这里的每条用例盯的都是编排契约，
互不重叠：

1. ``one_line`` 只在文本真含换行时才改写（I13 的实现点）
2. ``pack_hash`` 是元数据指纹，**不读文件内容**
3. ``stage_index`` 的合法域：阶段名 / 1..7 / 空值
4. ``stage_index`` 的非法域：bool 先于 int 被拒、越界、未知名，且错误信息自带选项
5. ``StageCache`` 在关闭 / 损坏 / 顶层非 dict / 目录不可写时一律静默降级
6. ``_take`` 只对"重启点之前"的阶段读缓存，且只回写 ``CACHED_STAGES``
7. ``STAGES`` / ``CACHED_STAGES`` 与进度层的 7 阶段名字表对齐
8. ``build_rows`` 把三条并行列表按 zip 对齐，判定字典恰好 5 个键
9. ``_dedupe_aligned`` 去重时让行与判定字典同步收缩（N14）
10. ``make_client`` 在离线配置下产出 disabled 客户端，且角色->模型映射稳定
11. ``process_pack`` 对不存在 / 空壳资料包抛带绝对路径与缺件清单的 InputError 子类
12. 冷跑落盘四个阶段缓存，热跑零模型调用且逐格相同（B18）
13. ``process_pack`` 的 ``stats`` 覆盖全部阶段的口径，且偏离分布只列非零桶
"""

from __future__ import annotations

import dataclasses
import json
import os

import pytest

from bidtable import errors
from bidtable import pipeline as pl
from bidtable.config import load_config
from bidtable.judge.decide import Decision
from bidtable.logging_ import STAGE_NAMES, Progress
from bidtable.models import PackResult

pytestmark = [pytest.mark.unit]


# --------------------------------------------------------------------------- #
# 1. one_line
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J5")
@pytest.mark.ec("J8")
def test_one_line_only_rewrites_text_that_actually_has_newlines():
    """单行化是"最小改动"：没有换行就原样返回，连首尾空格都不碰。

    这是个真边界——招标 PDF 的单元格里经常出现 ``"≥40L/min\\n（连续）"``，
    换行必须压成空格；但 ``one_line`` 同时被用在已经干净的字段上，
    如果顺手 strip 就会把 ``位置`` 之类的前导空格吃掉，破坏逐格对齐。
    """
    assert pl.one_line("  x  ") == "  x  "          # 无换行 -> 逐字返回
    assert pl.one_line("a\nb") == "a b"
    assert pl.one_line("a\rb  c") == "a b c"        # \r 也算换行，连续空白压成一个
    assert pl.one_line("  a\n b  ") == "a b"        # 有换行时才 strip
    assert pl.one_line("\n") == ""
    assert pl.one_line(None) == ""                  # 招标要求缺列时不炸
    assert pl.one_line(5) == "5"                    # 非字符串走 str() 强转


# --------------------------------------------------------------------------- #
# 2. pack_hash
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J4")
@pytest.mark.ec("J5")
def test_pack_hash_fingerprints_metadata_not_content(tmp_path):
    """缓存键必须能察觉"资料包换了"，但不能贵到去读一遍 PDF。

    实现取 ``相对路径|字节数|mtime_ns``，所以同尺寸同 mtime 的改写是不可见的
    ——这是有意的取舍（训练集是只读的），必须被钉住而不是被误当成 bug。
    """
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    first = pl.pack_hash(tmp_path)
    assert len(first) == 16                          # sha256 前 16 位
    assert pl.pack_hash(tmp_path) == first           # 稳定，可跨进程复用

    st = f.stat()
    f.write_text("world", encoding="utf-8")          # 同样 5 字节，内容不同
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert pl.pack_hash(tmp_path) == first           # 证明它不读内容

    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    bumped = pl.pack_hash(tmp_path)
    assert bumped != first                           # mtime 变了 -> 缓存失效

    (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    assert pl.pack_hash(tmp_path) != bumped          # 新增文件 -> 缓存失效


# --------------------------------------------------------------------------- #
# 3~4. stage_index
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J3")
@pytest.mark.ec("J7")
def test_stage_index_accepts_names_one_based_numbers_and_blank():
    """``--from-stage`` 的合法域同时接受阶段名和 1..7，空值表示"从头跑"。

    1-based 是给 CLI 用的（进度打印是 ``[3/7]``），0-based 是给内部用的，
    两套编号在同一个函数里换算，是最容易写错的地方。
    """
    assert pl.stage_index(None) == 0
    assert pl.stage_index("") == 0
    assert pl.stage_index("   ") == 0                # 纯空白等同于没传
    assert pl.stage_index("scope") == 0
    assert pl.stage_index("  write ") == 6           # 阶段名两侧空白被容忍
    assert pl.stage_index(1) == 0
    assert pl.stage_index("1") == 0
    assert pl.stage_index(7) == 6
    assert pl.stage_index("7") == 6
    assert pl.stage_index("cut") == 3
    assert pl.stage_index("judge") == 5


@pytest.mark.ec("J3")
@pytest.mark.ec("J7")
def test_stage_index_rejects_the_out_of_domain_values_with_a_useful_message():
    """非法值必须是 ``InputError``（退出码 1，用户输错），不是崩溃。

    ``True`` 在 Python 里 ``== 1``，如果 bool 分支不先拦，``--from-stage True``
    会被当成"从第 1 阶段跑"而静默成功；错误信息里把 7 个阶段名全列出来，
    这样用户不用去翻 README。
    """
    with pytest.raises(errors.InputError) as exc:
        pl.stage_index(True)
    assert "不接受" in str(exc.value)
    assert exc.value.exit_code == 1
    with pytest.raises(errors.InputError):
        pl.stage_index(False)

    with pytest.raises(errors.InputError) as exc:
        pl.stage_index("bogus")
    msg = str(exc.value)
    for name in pl.STAGES:
        assert name in msg                           # 选项自带在错误信息里

    for bad in ("0", "8", 0, 8, -1, "-1"):
        with pytest.raises(errors.InputError):
            pl.stage_index(bad)


# --------------------------------------------------------------------------- #
# 5. StageCache
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J4")
@pytest.mark.ec("J1")
def test_stage_cache_never_raises_and_never_returns_a_half_parsed_payload(tmp_path):
    """阶段缓存是"加速器"，不是"依赖"：任何异常都退化成未命中。

    四种真实故障都在这里：``--no-cache``（root=None）、上一次被 Ctrl-C 打断
    留下的半截 JSON、被人手改成数组的 JSON、缓存目录位置被一个同名文件占住。
    """
    off = pl.StageCache(None)
    assert off.path("scope") is None
    assert off.get("scope") is None                  # 关缓存时 get 恒 None
    assert (off.hits, off.misses) == (0, 1)          # 但仍然计入未命中，便于统计
    off.put("scope", {"a": 1})                       # put 是 no-op，不抛
    assert off.root is None

    root = tmp_path / "stages"
    cache = pl.StageCache(root)
    assert cache.get("scope") is None                # 冷：文件还不存在
    assert (cache.hits, cache.misses) == (0, 1)
    cache.put("scope", {"a": 1, "b": [1, 2]})
    assert (root / "scope.json").is_file()
    assert cache.get("scope") == {"a": 1, "b": [1, 2]}
    assert (cache.hits, cache.misses) == (1, 1)

    (root / "scope.json").write_text("{ 这不是 JSON", encoding="utf-8")
    broken = pl.StageCache(root)
    assert broken.get("scope") is None               # 损坏 -> 未命中，不抛
    assert (broken.hits, broken.misses) == (0, 1)

    (root / "scope.json").write_text(json.dumps([1, 2]), encoding="utf-8")
    assert pl.StageCache(root).get("scope") is None  # 顶层不是 dict 也当未命中

    cache.put("cut", {"a": object()})                # 不可序列化 -> 静默丢弃
    assert not (root / "cut.json").exists()

    occupied = tmp_path / "occupied"
    occupied.write_text("x", encoding="utf-8")       # 目录位置被文件占住
    pl.StageCache(occupied).put("scope", {"a": 1})   # mkdir 失败也不抛


# --------------------------------------------------------------------------- #
# 6. _take
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J4")
@pytest.mark.ec("J8")
def test_take_reads_cache_only_for_stages_before_the_restart_point(tmp_path):
    """``--from-stage match`` 的语义是"match 及其之后重算，之前全部读缓存"。

    三个边界：命中要说出来（用户在等进度）、该读却没读到要改口重算而不是崩、
    以及 ``start <= 本阶段下标`` 时**连缓存文件都不该打开**——否则 ``scope``
    阶段会在每次冷跑时白白统计一次未命中。
    """
    warm_root = tmp_path / "warm"
    warm = pl.StageCache(warm_root)
    warm.put("vendor", {"evidence": ["cached"]})

    def explode():
        raise AssertionError("命中缓存时不该重算")

    prog = Progress(quiet=True)
    payload, hit = pl._take(prog, warm, "vendor", 3, explode)
    assert (payload, hit) == ({"evidence": ["cached"]}, True)
    assert prog.log == ["      （vendor 阶段命中缓存，跳过重算）"]

    fresh_root = tmp_path / "fresh"
    warm2 = pl.StageCache(fresh_root)
    warm2.put("vendor", {"evidence": ["cached"]})
    prog = Progress(quiet=True)
    payload, hit = pl._take(prog, warm2, "vendor", 0, lambda: {"evidence": ["fresh"]})
    assert (payload, hit) == ({"evidence": ["fresh"]}, False)
    assert prog.log == []                            # 重算路径不打缓存日志
    assert (warm2.hits, warm2.misses) == (0, 0)      # 根本没读缓存

    empty = pl.StageCache(tmp_path / "empty")
    prog = Progress(quiet=True)
    payload, hit = pl._take(prog, empty, "vendor", 3, lambda: {"evidence": []})
    assert hit is False
    assert prog.log == ["      （vendor 阶段缓存缺失或已损坏，改为重新计算）"]
    assert empty.path("vendor").is_file()            # 重算结果落盘，下次就能命中

    uncached = pl.StageCache(tmp_path / "uncached")
    pl._take(Progress(quiet=True), uncached, "match", 3, lambda: {"m": 1})
    assert not (tmp_path / "uncached").exists()      # match 不在 CACHED_STAGES


# --------------------------------------------------------------------------- #
# 7. 常量契约
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J3")
@pytest.mark.ec("J8")
def test_stage_constants_agree_with_the_progress_layer():
    """``STAGES`` 与 ``logging_.STAGE_NAMES`` 必须同长同序。

    进度打印是 ``[i/7]``，缓存文件名是 ``STAGES[i]``，两者靠下标对齐；
    谁往中间插一个阶段而忘了另一边，用户看到的就会是"[5/7] 偏离判定"
    却读到了 ``cut.json``。这条用例就是那道锁。
    """
    assert len(pl.STAGES) == len(STAGE_NAMES) == 7
    assert pl.STAGES[0] == "scope"
    assert pl.STAGES[-1] == "write"
    assert pl.CACHED_STAGES == pl.STAGES[:4]         # 前四阶段可缓存，后三阶段总是重算
    for i, name in enumerate(pl.STAGES):
        assert pl.stage_index(name) == i
        assert pl.stage_index(str(i + 1)) == i


# --------------------------------------------------------------------------- #
# 8. build_rows
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J8")
@pytest.mark.ec("D7")
def test_build_rows_zips_requirement_match_and_decision_into_output_rows(mk_req, mk_match):
    """匹配层、判定层、输出层之间靠"下标对齐"传话，这里钉住对齐本身。

    三个能力点：招标要求原文经 ``one_line`` 后进入第 6 列（I13）；判定字典
    **恰好** 5 个键（复核报告只认这 5 个，多一个键就说明有人往 xlsx 里塞了
    第 11 列的料，X28）；以及 ``zip`` 在长度不齐时静默截断而不抛。
    """
    reqs = [mk_req("工作电流：≤10A\n（连续）", no="1"), mk_req("流量：≥40L/min", no="2")]
    matches = [
        dataclasses.replace(mk_match("工作电流：≤10A", "工作电流", "9360mA",
                                     score=0.1234567), reason="单位指纹一致"),
        mk_match("流量：≥40L/min", ev_name=None, method="none", confidence=0.42),
    ]
    decisions = [
        Decision(deviation="无偏离", resp_text="工作电流：9360mA",
                 evidence_file="X_技术规格书.docx", location="技术规格书 表行1",
                 confidence=0.6, flags=("unit_converted",), note="n",
                 reason="比值 1.07", method="compare"),
        Decision(deviation="未响应", resp_text="", evidence_file="", location="",
                 confidence=0.42, flags=("alias_rescue",), reason="找不到配对",
                 method="none"),
    ]

    rows, decs = pl.build_rows(reqs, matches, decisions)
    assert [r.no for r in rows] == ["1", "2"]
    assert rows[0].req_text == "工作电流：≤10A （连续）"    # 换行压成空格
    assert rows[1].req_text == "流量：≥40L/min"
    assert rows[0].flags == ("unit_converted",)            # flags 必须是 tuple
    assert isinstance(rows[0].flags, tuple)
    assert (rows[1].deviation, rows[1].resp_text) == ("未响应", "")
    assert rows[0].device_name == "设备"                   # 来自 Requirement，不是 Evidence

    assert all(set(d) == {"reason", "method", "match_method", "match_score", "match_reason"}
               for d in decs)
    assert decs[0]["match_score"] == 0.123                 # round(...,3) 的数值，不是字符串
    assert decs[0]["match_reason"] == "单位指纹一致"
    assert decs[1]["match_method"] == "none"

    short_rows, short_decs = pl.build_rows(reqs, matches[:1], decisions)
    assert len(short_rows) == len(short_decs) == 1         # 长度不齐 -> 静默截断


# --------------------------------------------------------------------------- #
# 9. _dedupe_aligned
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J1")
@pytest.mark.ec("J8")
@pytest.mark.case("N14")
def test_dedupe_aligned_shrinks_rows_and_decisions_together(mk_row):
    """写出去重后，判定字典必须跟着缩，否则复核报告会张冠李戴。

    真实触发场景：P3 的 71 条正文行只有 24 个互异的 ``(标的, 序号)``，
    分节策略一旦选错就会撞出重复键，而 ``grade.py`` 的行字典会静默覆盖
    重复键——所以必须在写盘前自检、去重、并**告警**。
    """
    result = PackResult(pack="P1")
    rows = [mk_row(no="1", req_text="first"), mk_row(no="1", req_text="dup"),
            mk_row(no="2", req_text="third")]
    decs = [{"i": 0}, {"i": 1}, {"i": 2}]

    kept, kept_decs = pl._dedupe_aligned(rows, decs, result)
    assert [r.req_text for r in kept] == ["first", "third"]   # 先出现的那条活下来
    assert kept_decs == [{"i": 0}, {"i": 2}]                  # 与行同步收缩
    assert len(result.warnings) == 1
    assert "重复键" in result.warnings[0]
    assert "序号1" in result.warnings[0]

    clean = PackResult(pack="P1")
    uniq = [mk_row(no="1"), mk_row(no="2")]
    uniq_decs = [{"i": 0}, {"i": 1}]
    kept2, decs2 = pl._dedupe_aligned(uniq, uniq_decs, clean)
    assert kept2 is uniq and decs2 is uniq_decs   # 无重复时原对象直接返回，不复制
    assert clean.warnings == []


# --------------------------------------------------------------------------- #
# 10. make_client
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J1")
@pytest.mark.ec("J6")
def test_make_client_is_disabled_offline_and_maps_roles_to_models(tmp_path):
    """离线是**默认**状态，disabled 客户端必须能被业务代码安全调用。

    ``complete()`` 返回 ``None`` 而不是抛异常，这是 §4.5 第 2 条（能降级就不崩）
    的落点；同时角色->模型映射要稳，否则 ``--model-ocr`` 会打到文本模型上。
    """
    cfg = load_config(pack=tmp_path, out=tmp_path / "o.xlsx", offline=True)
    client = pl.make_client(cfg)
    assert client.enabled is False
    assert client.complete("ocr", [{"role": "user", "content": "hi"}]) is None
    assert client.degraded and "离线模式" in client.degraded[0]
    assert client.calls == 0                                  # 一次都没发出去

    online = dataclasses.replace(cfg, offline=False, api_key="sk-unit-test",
                                 transport="fake")
    assert online.llm_available is True
    stub = pl.make_client(online)
    assert stub.enabled is True
    assert stub.model_for("ocr") == online.model_ocr
    assert stub.model_for("vision") == online.model_vision
    assert stub.model_for("judge") == online.model_strong
    assert stub.model_for("match") == online.model_text
    assert stub.model_for("没这个角色") == online.model_text   # 未知角色回落到文本模型


# --------------------------------------------------------------------------- #
# 11. process_pack 的资料包级失败
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J1")
@pytest.mark.ec("J5")
@pytest.mark.ec("J7")
def test_process_pack_rejects_a_missing_or_hollow_pack(tmp_path):
    """两种最常见的用户失误，都必须是退出码 1 + 说清楚"我看的是哪儿/缺了啥"。

    消息里带 **resolve 之后的绝对路径**很关键：用户可能传的是相对路径，
    而程序的工作目录未必是他以为的那个。
    """
    cfg = load_config(pack=tmp_path, out=tmp_path / "o.xlsx", offline=True)

    gone = tmp_path / "不存在的包"
    with pytest.raises(errors.PackNotFoundError) as exc:
        pl.process_pack(gone, cfg)
    assert str(gone.resolve()) in str(exc.value)
    assert exc.value.exit_code == 1

    hollow = tmp_path / "空壳"
    hollow.mkdir()
    with pytest.raises(errors.PackIncompleteError) as exc:
        pl.process_pack(hollow, cfg)
    assert exc.value.exit_code == 1
    assert exc.value.missing                              # 缺件清单非空
    assert any("01_" in item for item in exc.value.missing)
    assert "空壳" in str(exc.value)                        # 消息里点名是哪个包


# --------------------------------------------------------------------------- #
# 12. 缓存回放（B18）
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J4")
@pytest.mark.ec("J1")
@pytest.mark.case("B18")
def test_warm_restart_replays_a_pack_without_touching_the_model(pack_path, fake_llm, tmp_path):
    """B18：第二次运行的模型调用数必须是 0，且逐格结果与冷跑一致。

    这条用例是"钱"的边界——OCR 是唯一按张计费的环节，P1 有 1 张检测报告，
    冷跑会真发两次请求（1 次 OCR + 1 次语义兜底）。热跑从 ``match`` 重启时
    前四阶段全部读缓存，所以连图片都不该被打开。

    但**只有阶段缓存是不够的**：``match`` 阶段按定义要重算，语义兜底会再发
    一次请求。真正让"第二次运行零调用"成立的是阶段缓存 + LLM 磁盘缓存两层
    同时在位，所以最后一段专门把磁盘缓存关掉，证明这一层不是摆设。
    """
    root = pack_path("P1")
    cache_root = tmp_path / "stages"

    cold = fake_llm([""], default="", cache=True)   # cache=True -> 磁盘缓存开启
    assert cold.enabled is True and cold.cache.enabled is True
    result = pl.process_pack(root, cold.cfg, progress=Progress(quiet=True), client=cold,
                             from_stage=None, use_cache=True, cache_root=cache_root)
    assert cold.calls == 2                                     # 1 次 OCR + 1 次语义兜底
    assert cold.cache_hits == 0
    assert result.stats["cache"] == {"hits": 0, "misses": 0}    # 冷跑一次都不读
    assert sorted(p.name for p in cache_root.iterdir()) == [
        "cut.json", "scope.json", "tender.json", "vendor.json"]

    warm = fake_llm([""], default="", cache=True)
    replay = pl.process_pack(root, warm.cfg, progress=Progress(quiet=True), client=warm,
                             from_stage="match", use_cache=True, cache_root=cache_root)
    assert warm.calls == 0                                     # 一分钱没花
    assert warm.cache_hits == 1                                # 但 match 阶段确实重算过
    assert replay.stats["cache"] == {"hits": 4, "misses": 0}
    assert replay.stats["llm"] == {"enabled": True, "calls": 0, "cache_hits": 1}
    assert [r.cells() for r in replay.rows] == [r.cells() for r in result.rows]

    partial = fake_llm([""], default="")            # offline=True -> 磁盘缓存关闭
    assert partial.cache.enabled is False
    pl.process_pack(root, partial.cfg, progress=Progress(quiet=True), client=partial,
                    from_stage="match", use_cache=True, cache_root=cache_root)
    assert partial.calls == 1                                  # 少一层缓存就会漏一次请求


# --------------------------------------------------------------------------- #
# 13. stats 口径
# --------------------------------------------------------------------------- #


@pytest.mark.ec("J8")
@pytest.mark.ec("J3")
def test_process_pack_stats_report_every_stage_on_a_real_pack(offline_run):
    """``stats`` 是复核报告和自检的共同数据源，键名与口径都是对外契约。

    钉住 P1 的真实数字：21 条抽取 -> 折叠 18 -> 裁剪 18 -> 分节 18，
    其中 1 条标题不输出、17 条成行；偏离分布**只列非零桶**（否则
    ``stage_counts`` 会被四个 0 撑成噪音）。
    """
    result = offline_run("P1", from_stage="match")
    st = result.stats
    assert {"pack_dir", "template_xlsx", "vendor_counts", "layout", "cut_counts",
            "decisions", "stage_counts", "llm", "cache"} <= set(st)

    assert st["layout"] == "A"
    assert st["cut_counts"] == {"抽取条目": 21, "折叠后": 18, "裁剪后": 18,
                                "分节后": 18, "标题/引导句": 1, "参与判定": 17}
    assert st["vendor_counts"]["厂商证据"] == 20
    assert st["vendor_counts"]["检测报告指针"] == 3
    assert st["cache"] == {"hits": 4, "misses": 0}
    assert st["llm"] == {"enabled": False, "calls": 0, "cache_hits": 0}
    assert len(st["decisions"]) == len(result.rows) == 17

    counts = st["stage_counts"]
    assert counts["标的"] == 2
    assert counts["输出行"] == 17
    assert counts["标题/引导句（不输出）"] == 1
    assert counts["关键项风险（★/▲ 且负偏离或未响应）"] == 1
    assert counts["待复核"] == 15
    dist = {k[2:]: v for k, v in counts.items() if k.startswith("· ")}
    assert dist == {"无偏离": 10, "正偏离": 1, "负偏离": 3, "未响应": 3}
    assert sum(dist.values()) == 17

    # 离线跑 P1 必然带出 OCR 降级告警，这是"已知缺口要喊出来"的验收点
    assert len(result.warnings) == 2
    assert any("OCR 不可用" in w for w in result.warnings)
    assert len(result.skipped) == 1 and "折叠" in result.skipped[0]
