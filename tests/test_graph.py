"""The workflow end to end with fake dependencies and an in-memory checkpointer."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent.graph import build_graph
from app.services.judgments import JudgeOutcome


def _cfg(tid="t1"):
    return {"configurable": {"thread_id": tid}}


async def _pending(graph, cfg):
    snap = await graph.aget_state(cfg)
    return [i.value for t in snap.tasks for i in t.interrupts], snap


async def test_vague_request_is_clarified_then_vendors_are_listed(fake_deps):
    graph = build_graph(fake_deps).compile(checkpointer=InMemorySaver())
    cfg = _cfg()

    # 1) vague request -> pauses for clarification
    await graph.ainvoke({"request_id": "11111111-1111-1111-1111-111111111111",
                         "raw_text": "Need waterproof plywood, 18 mm", "log": []}, cfg)
    pending, _ = await _pending(graph, cfg)
    assert pending[0]["type"] == "clarification"
    assert "quantity" in pending[0]["missing"]

    # 2) answer -> retrieve / verify / rank -> report, and the run is finished
    out = await graph.ainvoke(Command(resume={"answers": {"quantity": 200, "unit": "sheets",
                                                          "delivery_location": "Udaipur", "deadline_days": 20}}), cfg)
    pending, snap = await _pending(graph, cfg)
    assert pending == [] and snap.next == ()
    report = out["report"]
    assert [v["vendor_id"] for v in report["vendors"]] == ["V-1", "V-4"]  # V-2 MOQ, V-3 lead time: ruled out
    first = report["vendors"][0]
    assert first["rank"] == 1 and first["contact_email"] and first["phone"] and first["confidence"] == "shortlist"
    reasons = {m["vendor_id"]: m["reason"] for m in report["near_misses"]}
    assert "minimum order" in reasons["V-2"] and "lead time" in reasons["V-3"]
    assert report["message"] == "2 vendor(s) can supply this."
    assert snap.values["requirement"]["delivery_state"] == "Rajasthan"
    actions = [e["action"] for e in fake_deps.audit.events]
    assert actions[-1] == "reported" and "clarified" in actions


async def test_complete_request_needs_no_pause(fake_deps):
    graph = build_graph(fake_deps).compile(checkpointer=InMemorySaver())
    out = await graph.ainvoke({"request_id": "22222222-2222-2222-2222-222222222222", "log": [],
                               "raw_text": "200 sheets of 18 mm waterproof plywood to Udaipur within 20 days"},
                              _cfg("t2"))
    assert len(out["report"]["vendors"]) == 2


async def test_no_candidates_says_so(fake_deps):
    async def nothing(req):
        return []

    fake_deps.retrieve = nothing
    graph = build_graph(fake_deps).compile(checkpointer=InMemorySaver())
    out = await graph.ainvoke({"request_id": "33333333-3333-3333-3333-333333333333", "log": [],
                               "raw_text": "50 kg of unobtainium to Jodhpur within 5 days"}, _cfg("t3"))
    assert out["report"]["vendors"] == [] and "No vendor in the database" in out["report"]["message"]


async def test_everything_ruled_out_points_to_near_misses(fake_deps):
    graph = build_graph(fake_deps).compile(checkpointer=InMemorySaver())
    out = await graph.ainvoke({"request_id": "44444444-4444-4444-4444-444444444444", "log": [],
                               "raw_text": "20 sheets of 18 mm waterproof plywood to Udaipur within 2 days"}, _cfg("t4"))
    assert out["report"]["vendors"] == [] and "near misses" in out["report"]["message"]
    assert len(out["report"]["near_misses"]) == 4


async def test_jev_probabilities_route_candidates(fake_deps):
    async def judge(req, cands):
        assert {c["id"] for c in cands} == {"V-1", "V-4"}  # code-check failures never reach Jev
        return JudgeOutcome(by_vendor={"V-1": {"p_spec": 0.9, "fit": 3.0}, "V-4": {"p_spec": 0.55, "fit": 2.0}},
                            usage=[{"step": "verify", "source": "jev:test", "input_tokens": 10,
                                    "output_tokens": 4, "latency_ms": 5, "error": None}])

    fake_deps.judge = judge
    graph = build_graph(fake_deps).compile(checkpointer=InMemorySaver())
    out = await graph.ainvoke({"request_id": "55555555-5555-5555-5555-555555555555", "log": [],
                               "raw_text": "200 sheets of 18 mm waterproof plywood to Udaipur within 20 days"},
                              _cfg("t5"))
    report = out["report"]
    assert {v["vendor_id"]: v["confidence"] for v in report["vendors"]} == {"V-1": "shortlist",
                                                                           "V-4": "needs_review"}
    assert report["vendors"][0]["match"]["p_spec"] == 0.9
    assert report["summary"]["needs_review"] == 1 and report["summary"]["llm"]["calls"] == 2


async def test_empty_search_index_is_reported_as_not_ready(fake_deps):
    async def nothing(req):
        return []

    async def status():
        return {"vendors": 500, "searchable": 120, "embedder": "gemini:test"}

    fake_deps.retrieve, fake_deps.index_status = nothing, status
    graph = build_graph(fake_deps).compile(checkpointer=InMemorySaver())
    out = await graph.ainvoke({"request_id": "66666666-6666-6666-6666-666666666666", "log": [],
                               "raw_text": "200 sheets of 18 mm waterproof plywood to Udaipur within 20 days"}, _cfg("t6"))
    assert "isn't ready yet: 120 of 500" in out["report"]["message"]
