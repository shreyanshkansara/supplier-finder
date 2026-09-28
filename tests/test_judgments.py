"""Jev judgments and the ranking policy. The TypeSafe SDK is real; only its HTTP transport is mocked."""

import json

import httpx2  # the HTTP library the TypeSafe SDK uses
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from app.agent.graph import build_graph
from app.agent.nodes import RankPolicy, rank_candidates
from app.agent.runtime import policy_from
from app.config import Settings
from app.services.judgments import JudgeOutcome, check_quotable, judge_candidates

OFF = Settings(typesafe_api_key=None, jev_batch_size=2)


def jev_client(answer=None, *, status=200, calls=None):
    """Real AsyncTypeSafeClient over a mock transport. `answer(qid, question) -> answer dict`."""

    def handler(request):
        body = json.loads(request.content)
        if calls is not None:
            calls.append(body)
        if status != 200:
            return httpx2.Response(status, json={"error": {"message": "boom"}})
        answers = {qid: answer(qid, q) for qid, q in body["questions"].items()}
        return httpx2.Response(200, json={"model": "jev-test", "usage": {"input_tokens": 300, "output_tokens": 12},
                                          "answers": answers})

    return AsyncTypeSafeClient(api_key="ts-test", transport=httpx2.MockTransport(handler),
                               retry=RetryPolicy(max_retries=0))


def noul(p):
    return {"type": "noul", "noul": p}


def score(x, conf=0.8):
    return {"type": "score", "score": x, "confidence": conf,
            "legend": {"0": "a", "1": "b", "2": "c", "3": "d"}, "probabilities": {"0": 0.1, "1": 0.1, "2": 0.2, "3": 0.6}}


CANDS = [{"id": f"V-{i}", "categories": ["Panel boards"], "specs": f"listing {i}", "moq": 10} for i in range(3)]
REQ = {"item": "waterproof plywood", "specs": "18 mm", "category": "Panel boards", "quantity": 200}


# ---------------------------------------------------------------- judge_candidates
async def test_judge_candidates_batches_and_parses():
    calls = []
    client = jev_client(lambda qid, q: noul(0.9 if qid.endswith("_0") else 0.3) if q["type"] == "noul" else score(2.5),
                        calls=calls)
    out = await judge_candidates(REQ, CANDS, settings=OFF, client=client)

    assert len(calls) == 2  # batch size 2 -> 2 requests for 3 vendors, over one client
    first = calls[0]
    assert set(first["state"]) == {"requirement", "vendors"}
    assert first["state"]["requirement"] == {"item": "waterproof plywood", "specs": "18 mm", "category": "Panel boards"}
    assert "moq" not in first["state"]["vendors"][0]  # numbers are code's job, not Jev's
    assert set(first["questions"]) == {"spec_0", "fit_0", "spec_1", "fit_1"}
    assert first["questions"]["spec_0"]["criteria"]["false"].startswith("A different product")
    # criteria examples must not leak the evaluation set's planted cases
    leaked = [w for w in ("sheesham", "plywood", "tarpaulin", "foam", "kiln") if w in json.dumps(first["questions"]).lower()]
    assert leaked == []
    assert len(first["questions"]["fit_0"]["criteria"]) == 4

    assert out.ok and out.by_vendor["V-0"] == {"p_spec": 0.9, "fit": 2.5, "fit_confidence": 0.8}
    assert out.by_vendor["V-2"]["p_spec"] == 0.9  # first item of the second batch
    assert [u["input_tokens"] for u in out.usage] == [300, 300]


async def test_judge_candidates_off_without_key():
    assert await judge_candidates(REQ, CANDS, settings=OFF) is None
    assert await judge_candidates(REQ, [], settings=OFF, client=jev_client(lambda *_: noul(1))) is None


@pytest.mark.parametrize("client", [
    jev_client(status=500),
    jev_client(lambda qid, q: noul(0.9)),  # every answer is a noul: the fit_* scores are missing
])
async def test_judge_candidates_failure_is_all_or_nothing(client):
    out = await judge_candidates(REQ, CANDS, settings=OFF, client=client)
    assert out is not None and not out.ok
    assert out.by_vendor == {}  # never mix judged and unjudged vendors
    assert any(u["error"] for u in out.usage)


# ---------------------------------------------------------------- check_quotable
async def test_check_quotable():
    calls = []
    p, usage = await check_quotable({"item": "foam", "specs": "foam"}, settings=OFF,
                                    client=jev_client(lambda *_: noul(0.12), calls=calls))
    assert p == 0.12 and usage["step"] == "completeness" and usage["error"] is None
    assert calls[0]["questions"]["quotable"]["type"] == "noul"
    assert await check_quotable({"item": "x"}, settings=OFF) == (None, None)  # Jev off
    p, usage = await check_quotable({"item": "x"}, settings=OFF, client=jev_client(status=503))
    assert p is None and usage["error"]


# ---------------------------------------------------------------- policy
def _c(vid, sim, p_spec=None, fit=None, moq_ok=True, lead_ok=True):
    return {"id": vid, "similarity": sim, "p_spec": p_spec, "fit": fit,
            "checks": {"unit_match": True, "moq_ok": moq_ok, "lead_ok": lead_ok}}


def test_rank_candidates_routes_by_thresholds():
    policy = RankPolicy(shortlist_at=0.7, review_at=0.4)
    shortlist, dropped = rank_candidates([
        _c("A", 0.5, p_spec=0.95, fit=3), _c("B", 0.9, p_spec=0.55, fit=2), _c("C", 0.99, p_spec=0.2, fit=0),
        _c("D", 0.8, p_spec=0.9, fit=3, moq_ok=False), _c("E", 0.4, p_spec=0.8, fit=2)], policy, top_n=2)
    assert [c["id"] for c in shortlist] == ["A", "B"]
    assert {c["id"]: c["review_flag"] for c in shortlist} == {"A": "shortlist", "B": "needs_review"}
    reasons = {c["id"]: c["drop_reason"] for c in dropped}
    assert reasons == {"C": "low spec match (0.20)", "D": "moq_ok", "E": "below top_n"}


def test_rank_candidates_without_jev_uses_similarity():
    shortlist, _ = rank_candidates([_c("A", 0.2), _c("B", 0.9)], RankPolicy(), top_n=5)
    assert [c["id"] for c in shortlist] == ["B", "A"] and {c["review_flag"] for c in shortlist} == {"shortlist"}


def test_policy_comes_from_settings():
    p = policy_from(Settings(jev_shortlist_at=0.8, jev_review_at=0.3, jev_quotable_at=0.6, rank_w_spec=0.6))
    assert (p.shortlist_at, p.review_at, p.quotable_at, p.w_spec) == (0.8, 0.3, 0.6, 0.6)


# ---------------------------------------------------------------- in the graph
async def test_unquotable_request_is_sent_back_for_specs(fake_deps):
    async def quotable(req):
        return 0.1, {"step": "completeness", "source": "jev:test", "input_tokens": 5, "output_tokens": 1,
                     "latency_ms": 1, "error": None}

    fake_deps.quotable = quotable
    graph = build_graph(fake_deps).compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "q1"}}
    # every field is present, but "some plywood" can't be quoted
    await graph.ainvoke({"request_id": "66666666-6666-6666-6666-666666666666", "log": [],
                         "raw_text": "200 sheets of some plywood to Jodhpur within 10 days"}, cfg)
    snap = await graph.aget_state(cfg)
    pending = snap.tasks[0].interrupts[0].value
    assert pending["type"] == "clarification" and pending["missing"] == ["specs"]
    assert snap.values["p_quotable"] == 0.1


async def test_failed_jev_falls_back_to_code_checks(fake_deps):
    async def broken_judge(req, cands):
        return JudgeOutcome(usage=[{"step": "verify", "source": "jev:test", "input_tokens": 0, "output_tokens": 0,
                                    "latency_ms": 1, "error": "TypeSafeRateLimitError: 429"}])

    fake_deps.judge = broken_judge
    graph = build_graph(fake_deps).compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "q2"}}
    await graph.ainvoke({"request_id": "77777777-7777-7777-7777-777777777777", "log": [],
                         "raw_text": "200 sheets of 18 mm waterproof plywood to Udaipur within 20 days"}, cfg)
    snap = await graph.aget_state(cfg)
    vendors = snap.values["report"]["vendors"]
    assert [v["vendor_id"] for v in vendors] == ["V-1", "V-4"]  # the run carried on
    assert all(v["match"]["p_spec"] is None for v in vendors)
    assert "verify: jev=failed -> code checks only" in snap.values["log"]
    verify_event = next(e for e in fake_deps.audit.events if e["action"] == "verified")
    assert verify_event["detail"]["jev"] == "failed -> code checks only"


async def test_find_missing_lets_jev_decide_specificity():
    from app.agent.nodes import find_missing

    base = {"quantity": 50, "unit": "kg", "delivery_location": "Jodhpur", "deadline_days": 5}

    async def says(p):
        async def q(req):
            return p, None
        return q

    no_jev = await says(None)
    # "memory foam" with no separate specs: specific enough per Jev, missing per the no-Jev rule
    assert (await find_missing(base | {"item": "memory foam"}, await says(0.9), RankPolicy()))[0] == []
    assert (await find_missing(base | {"item": "memory foam"}, no_jev, RankPolicy()))[0] == ["specs"]
    # "fabric" + a purpose: filled in, but not quotable
    assert (await find_missing(base | {"item": "fabric", "specs": "for the hotel sofas"}, await says(0.1),
                               RankPolicy()))[0] == ["specs"]
    assert (await find_missing({"item": None}, no_jev, RankPolicy()))[0][:2] == ["item", "specs"]
