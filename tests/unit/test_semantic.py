"""L2 语义匹配兜底（EC-J1 离线降级 / EC-J7 错误契约）。

这一层最容易出的事故是「模型随口一说，我们照单全收」。所以测试的重点不在
能不能选中，而在**选不中时必须原样退回确定性结果**：下标越界、null、
JSON 损坏、候选池太薄，一律视作「LLM 没有意见」。
"""

from __future__ import annotations

import pytest

from bidtable.llm.transport import Response
from bidtable.match import semantic
from bidtable.match.candidates import Candidate

pytestmark = [pytest.mark.unit]


@pytest.mark.ec("J7")
@pytest.mark.parametrize("text,want", [
    (None, None), ("", None),
    ("没有任何花括号", None),
    ("{坏掉的 json", None),
    ('["不是字典"]', None),
    ('{"别的键": 1}', None),
    ('{"index": null}', None),
    ('{"index": true}', None),          # bool 是 int 的子类，必须显式挡掉
    ('{"index": "不是数字"}', None),
    ('{"index": 3}', None),             # 越界
    ('{"index": -1}', None),
    ('{"index": 1}', 1),
    ('{"index": "2"}', 2),              # 字符串数字要能吃
    ('好的，我的判断是 {"index": 0, "reason": "…"} 希望有帮助', 0),   # 夹在散文里
])
def test_parse_index_rejects_every_abnormal_shape(text, want):
    assert semantic._parse_index(text, 3) == want


def _cands(mk_req, mk_ev, scores):
    req = mk_req("用于泌尿系结石的碎石")
    return req, [
        Candidate(requirement=req, evidence=mk_ev(f"指标{i}", f"{i}"), score=s,
                  method="contain", name_sim=s, value_sim=s)
        for i, s in enumerate(scores)
    ]


@pytest.mark.ec("J1")
def test_suggest_never_asks_when_there_is_nothing_worth_asking(mk_req, mk_ev, fake_llm):
    """client 为空、候选为空、候选全都低于 ``LLM_FLOOR``：三种情况都不发请求。"""
    req, thin = _cands(mk_req, mk_ev, [0.10, 0.34])
    client = fake_llm(['{"index": 0}'])
    assert semantic.suggest(req, thin, None) is None
    assert semantic.suggest(req, [], client) is None
    assert semantic.suggest(req, thin, client) is None
    assert client.calls == 0 and client.transport.calls == 0
    assert all(c.method == "contain" and not c.flags for c in thin)   # 没被就地改过


@pytest.mark.ec("J1")
def test_suggest_mutates_the_chosen_candidate_in_place(mk_req, mk_ev, fake_llm):
    """返回的是入参里的**同一个对象**，调用方不需要重新打分。"""
    req, pool = _cands(mk_req, mk_ev, [0.40, 0.20, 0.50])
    client = fake_llm(['{"index": 1}'])
    chosen = semantic.suggest(req, pool, client)
    eligible = [c for c in pool if c.score >= semantic.LLM_FLOOR]
    assert chosen is eligible[1] is pool[2]                     # 下标是相对「合格池」的
    assert chosen.method == "llm" and chosen.flags == ("llm_fallback",)
    assert chosen.score == pytest.approx(0.50)                  # 已经高于 FLOOR 就不动
    assert eligible[0].score == pytest.approx(0.40) and eligible[0].method == "contain"
    assert "确定性打分" in chosen.reason and "语义模型选定" in chosen.reason
    assert client.cache_hits == 0 and client.transport.calls == 1


@pytest.mark.ec("J1")
def test_suggest_hides_ineligible_candidates_and_renders_pointers(mk_req, mk_ev, fake_llm):
    """提示词里的编号必须和合格池对齐，否则模型说的 ``index`` 会指错行；
    报告指针没有实测值，用「详见检测报告」占位而不是留空。"""
    req = mk_req("光纤终端最大平均输出功率")
    thin = Candidate(requirement=req, evidence=mk_ev("无关指标", "1"), score=0.05)
    keep = Candidate(requirement=req, evidence=mk_ev("脉冲宽度可调范围", "", is_pointer=True),
                     score=0.44, method="pointer")
    client = fake_llm(['{"index": 0}'])
    assert semantic.suggest(req, [thin, keep], client) is keep
    prompt = client.transport.seen[0].messages[1]["content"]
    assert "[0] 脉冲宽度可调范围 = 详见检测报告" in prompt
    assert "无关指标" not in prompt and "[1]" not in prompt


@pytest.mark.ec("J7")
def test_suggest_gives_up_cleanly_when_the_model_has_no_opinion(mk_req, mk_ev, fake_llm):
    """模型返回 null、返回空串、或整个不可用（complete -> None）都要安静退出。"""
    req, pool = _cands(mk_req, mk_ev, [0.40, 0.50])
    for reply in ['{"index": null}', "", Response(text="我不知道")]:
        client = fake_llm([reply])
        assert semantic.suggest(req, pool, client) is None
        assert all(c.method == "contain" and not c.flags for c in pool)

    offline = fake_llm(['{"index": 0}'])
    offline.transport = None                                    # 模拟 Key 失效
    assert offline.enabled is False
    assert semantic.suggest(req, pool, offline) is None
    assert offline.degraded and not pool[0].flags
