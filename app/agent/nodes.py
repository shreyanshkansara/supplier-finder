"""Graph nodes.

The workflow ends with a list of suppliers; contacting them is left to the buyer.
Each node is small and does one job:

  parse          DONE: LLM structured output -> SourcingRequirement (rule-based without a key / on failure)
  completeness   DONE: code finds missing fields; Jev asks "specific enough to quote?" (when configured)
  clarify        DONE: interrupt() #1, resumes with the user's answers
  retrieve       DONE: SQL filters + pgvector
  verify         DONE: code checks + Jev spec/fit judgments for all candidates in one call per batch
  rank           DONE: RankPolicy (weights + thresholds from settings, tuned by eval/tune_thresholds.py)
  report         DONE: the answer: vendors that can supply the requirement, with contact details,
                 terms and why each one matched, plus the near misses and why they were ruled out

Note on interrupt(): when a run resumes, the paused node runs again from the top, so anything
before interrupt() must be safe to repeat. Side effects (like audit writes) go after it.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from langgraph.types import interrupt

from app.catalog import norm_unit
from app.schemas import SourcingRequirement
from app.services.audit import AuditSink
from app.services.geo import state_for
from app.services.llm import LLM, RuleBasedLLM, rule_based_parse

__all__ = ["AgentDeps", "Nodes", "code_checks", "norm_unit", "rule_based_parse"]

log = logging.getLogger(__name__)



@dataclass(frozen=True)
class RankPolicy:
    """How judgments become decisions. Pure policy: changing it never needs new model calls.

    p_spec >= shortlist_at            -> "shortlist"     (listed as a confident match)
    review_at <= p_spec < shortlist_at -> "needs_review" (listed, flagged: check the listing first)
    p_spec < review_at                -> dropped         (with the reason kept for the report)
    Without Jev (p_spec is None), code-check survivors are ranked by similarity and shortlisted.
    """

    w_similarity: float = 0.3
    w_spec: float = 0.5
    w_fit: float = 0.2
    shortlist_at: float = 0.7
    review_at: float = 0.4
    quotable_at: float = 0.5  # completeness: below this, the specs count as missing


def rank_candidates(verified: list[dict], policy: RankPolicy, top_n: int) -> tuple[list[dict], list[dict]]:
    """Shared by the rank node and the evaluation, so the eval measures exactly what the app does."""
    keep, dropped = [], []
    for c in verified:
        failed = [k for k in ("moq_ok", "lead_ok") if c["checks"][k] is False]
        p_spec = c.get("p_spec")
        if failed:
            dropped.append(c | {"drop_reason": ", ".join(failed)})
            continue
        if p_spec is not None and p_spec < policy.review_at:
            dropped.append(c | {"drop_reason": f"low spec match ({p_spec:.2f})"})
            continue
        spec = p_spec if p_spec is not None else c["similarity"]
        fit = (c["fit"] if c.get("fit") is not None else 1.5) / 3
        score = policy.w_similarity * c["similarity"] + policy.w_spec * spec + policy.w_fit * fit
        flag = "shortlist" if p_spec is None or p_spec >= policy.shortlist_at else "needs_review"
        keep.append(c | {"score": round(score, 4), "review_flag": flag})
    keep.sort(key=lambda c: c["score"], reverse=True)
    return keep[:top_n], dropped + [c | {"drop_reason": "below top_n"} for c in keep[top_n:]]


async def find_missing(req: dict, quotable, policy: RankPolicy) -> tuple[list[str], float | None, list[dict]]:
    """Which details to ask the buyer for. Shared by the completeness node and the parse eval.

    Empty fields are always missing. Whether the product is specific enough is a judgment, not a
    field check: "memory foam" is quotable with no separate specs, "fabric" + "for the hotel sofas"
    is not. So when Jev is available it decides (on item + specs together); without Jev the rule
    is simply "specs must be filled in".
    """
    r = SourcingRequirement(**req)
    missing = r.missing_fields()
    usage: list[dict] = []
    p_quote = None
    if r.item:
        p_quote, u = await quotable(req)
        if u:
            usage.append(u)
    if p_quote is None:
        if not r.specs:
            missing.append("specs")
    elif p_quote < policy.quotable_at:
        missing.append("specs")
    order = ["item", "specs", "quantity", "unit", "delivery_location", "deadline_days"]
    return sorted(set(missing), key=order.index), p_quote, usage


async def _no_judge(req: dict, cands: list[dict]):
    return None


async def _no_index_status():
    return None


async def _no_quotable(req: dict):
    return None, None

QUESTIONS = {
    "item": "What exactly do you need to buy?",
    "specs": "What specs matter (size, thickness, density, finish, material)?",
    "quantity": "How many / how much do you need?",
    "unit": "In what unit (sheets, cft, pcs, kg, m, L, sq ft)?",
    "delivery_location": "Which city should it be delivered to?",
    "deadline_days": "Within how many days do you need it delivered?",
}


@dataclass
class AgentDeps:
    retrieve: Callable[[dict], Awaitable[list[dict]]]
    audit: AuditSink
    llm: LLM = field(default_factory=RuleBasedLLM)
    judge: Callable[[dict, list[dict]], Awaitable] = _no_judge  # -> JudgeOutcome | None
    quotable: Callable[[dict], Awaitable[tuple]] = _no_quotable  # -> (p | None, usage | None)
    index_status: Callable[[], Awaitable] = _no_index_status  # -> {"vendors", "searchable", "embedder"} | None
    policy: RankPolicy = field(default_factory=RankPolicy)
    top_n: int = 10  # most vendors to list
    max_clarify_rounds: int = 3


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def code_checks(req: dict, v: dict) -> dict:
    """Hard rules. Plain Python, never a model: MOQ and lead time are arithmetic."""
    qty, dl = req.get("quantity"), req.get("deadline_days")
    unit_match = req.get("unit") is None or norm_unit(req.get("unit")) == norm_unit(v["moq_unit"])
    return {
        "unit_match": unit_match,
        "moq_ok": None if qty is None or not unit_match else v["moq"] <= qty,
        "lead_ok": None if dl is None else v["lead_time_days"] <= dl,
    }


# ----------------------------------------------------------------------------------------
# nodes
# ----------------------------------------------------------------------------------------
class Nodes:
    def __init__(self, deps: AgentDeps):
        self.d = deps

    async def parse(self, s: dict) -> dict:
        req, res = await self.d.llm.parse_requirement(s["raw_text"])
        await self.d.audit(s["request_id"], "agent:parse", "parsed",
                           {"requirement": req.model_dump(), "llm": res.usage("parse"),
                            "raw_llm_output": res.extra.get("raw")})
        return {"requirement": req.model_dump(), "clarify_rounds": 0, "llm_usage": [res.usage("parse")],
                "log": [f"parse: done via {res.source}"]}

    async def completeness(self, s: dict) -> dict:
        req = s["requirement"]
        missing, p_quote, usage = await find_missing(req, self.d.quotable, self.d.policy)
        if p_quote is not None or usage:
            await self.d.audit(s["request_id"], "agent:completeness", "quotable_checked",
                               {"p_quotable": p_quote, "threshold": self.d.policy.quotable_at, "llm": usage})
        return {"missing": missing, "p_quotable": p_quote, "llm_usage": usage,
                "log": [f"completeness: missing={missing} p_quotable={p_quote}"]}

    async def clarify(self, s: dict) -> dict:
        resume = interrupt({
            "type": "clarification",
            "missing": s["missing"],
            "questions": {f: QUESTIONS[f] for f in s["missing"]},
            "requirement": s["requirement"],
        })
        # Everything below runs once, after the user answers (safe place for side effects).
        answers = {k: v for k, v in (resume.get("answers") or {}).items() if k in QUESTIONS or k == "category"}
        usage = []
        if text := (resume.get("text") or "").strip():
            # Free-text answer: parse it with the same parser, and only fill fields still missing,
            # so a short reply like "200, by the 20th" can't overwrite what was already known.
            parsed, res = await self.d.llm.parse_requirement(text)
            usage.append(res.usage("clarify"))
            fill = set(s["missing"]) | {"category"} if not s["requirement"].get("category") else set(s["missing"])
            for f in fill:
                if f not in answers and getattr(parsed, f, None) not in (None, ""):
                    answers[f] = getattr(parsed, f)
        merged = {**s["requirement"], **answers}
        if "unit" in answers:
            merged["unit"] = norm_unit(str(answers["unit"]))
        if "delivery_location" in answers:
            merged["delivery_state"] = state_for(answers["delivery_location"]) or merged.get("delivery_state")
        req = SourcingRequirement(**merged)
        await self.d.audit(s["request_id"], "user", "clarified", {"answers": answers, "text": text or None})
        return {"requirement": req.model_dump(), "clarify_rounds": s.get("clarify_rounds", 0) + 1,
                "llm_usage": usage, "log": [f"clarify: got {sorted(answers)}"]}

    async def retrieve(self, s: dict) -> dict:
        cands = await self.d.retrieve(s["requirement"])
        # Nothing found: tell "nothing sells this" apart from "the search index isn't ready".
        index = await self.d.index_status() if not cands else None
        await self.d.audit(s["request_id"], "agent:retrieve", "retrieved",
                           {"n": len(cands), "ids": [c["id"] for c in cands], "index": index})
        return {"candidates": cands, "index": index, "log": [f"retrieve: {len(cands)} candidates"]}

    async def verify(self, s: dict) -> dict:
        req = s["requirement"]
        verified = [c | {"checks": code_checks(req, c)} for c in s["candidates"]]
        # Jev only sees candidates that survived the hard rules: no point judging a vendor
        # whose MOQ is too high, and it keeps the request small.
        survivors = [c for c in verified if c["checks"]["moq_ok"] is not False and c["checks"]["lead_ok"] is not False]
        outcome = await self.d.judge(req, survivors)
        usage = outcome.usage if outcome else []
        if outcome:
            for c in verified:
                c.update(outcome.by_vendor.get(c["id"], {}))
        jev = "off" if outcome is None else ("on" if outcome.ok else "failed -> code checks only")
        await self.d.audit(s["request_id"], "agent:verify", "verified",
                           {"jev": jev, "llm": usage,
                            "vendors": {c["id"]: {**c["checks"], "p_spec": c.get("p_spec"), "fit": c.get("fit")}
                                        for c in verified}})
        return {"verified": verified, "llm_usage": usage, "log": [f"verify: jev={jev}"]}

    async def rank(self, s: dict) -> dict:
        shortlist, dropped = rank_candidates(s["verified"], self.d.policy, self.d.top_n)
        await self.d.audit(s["request_id"], "agent:rank", "ranked",
                           {"policy": self.d.policy.__dict__,
                            "shortlist": [(c["id"], c["score"], c["review_flag"]) for c in shortlist],
                            "dropped": [(c["id"], c["drop_reason"]) for c in dropped]})
        return {"shortlist": shortlist, "dropped": dropped, "log": [f"rank: {len(shortlist)} shortlisted"]}

    async def report(self, s: dict) -> dict:
        vendors = [{
            "rank": i,
            "vendor_id": c["id"], "name": c["name"], "city": c["city"], "state": c["state"],
            "contact_name": c.get("contact_name"), "contact_email": c.get("contact_email"),
            "phone": c.get("phone"),
            "moq": f"{c['moq']} {c['moq_unit']}", "lead_time_days": c["lead_time_days"],
            "indicative_price": c.get("indicative_price"), "price_band": c.get("price_band"),
            "payment_terms": c.get("payment_terms"), "specs": c["specs"],
            "confidence": c["review_flag"],  # "shortlist" = confident match, "needs_review" = check the listing
            "match": {"p_spec": c.get("p_spec"), "fit": c.get("fit"), "similarity": c["similarity"],
                      "score": c["score"]},
            "checks": c["checks"],
        } for i, c in enumerate(s.get("shortlist", []), start=1)]
        near_misses = [{
            "vendor_id": c["id"], "name": c["name"], "city": c["city"], "reason": _reason(c),
            "moq": f"{c['moq']} {c['moq_unit']}", "lead_time_days": c["lead_time_days"],
            "p_spec": c.get("p_spec"),
        } for c in s.get("dropped", []) if c.get("drop_reason") != "below top_n"][:10]
        if vendors:
            message = f"{len(vendors)} vendor(s) can supply this."
        elif s.get("candidates"):
            message = "No vendor fully matches. See the near misses for the closest options."
        elif (ix := s.get("index")) and (not ix["vendors"] or ix["searchable"] < ix["vendors"]):
            message = (f"The vendor search isn't ready yet: {ix['searchable']} of {ix['vendors']} vendors are "
                       f"searchable with {ix['embedder']}. If the server just started, the vendors are still "
                       f"being loaded: try again in a few minutes. Otherwise check the server logs.")
        else:
            message = "No vendor in the database sells this to your delivery region."
        report = {"requirement": s["requirement"], "message": message, "vendors": vendors,
                  "near_misses": near_misses,
                  "summary": {"candidates": len(s.get("candidates", [])), "listed": len(vendors),
                              "needs_review": sum(v["confidence"] == "needs_review" for v in vendors),
                              "llm": _usage_totals(s.get("llm_usage", []))}}
        await self.d.audit(s["request_id"], "agent:report", "reported",
                           {"vendors": [v["vendor_id"] for v in vendors], "message": message})
        return {"report": report, "log": [f"report: {message}"]}


def _reason(c: dict) -> str:
    r = c.get("drop_reason", "")
    if r == "moq_ok":
        return f"minimum order {c['moq']} {c['moq_unit']} is above the requested quantity"
    if r == "lead_ok":
        return f"lead time {c['lead_time_days']} days is longer than the deadline"
    if r == "moq_ok, lead_ok":
        return "minimum order and lead time both too high"
    if r.startswith("low spec match"):
        return f"listing does not meet the stated specs (spec match {round(c['p_spec'] * 100)}%)"
    return r


def _usage_totals(usage: list[dict]) -> dict:
    return {
        "calls": len(usage),
        "input_tokens": sum(u["input_tokens"] for u in usage),
        "output_tokens": sum(u["output_tokens"] for u in usage),
        "latency_ms": sum(u["latency_ms"] for u in usage),
        "fallbacks": sum(1 for u in usage if u["error"]),
        "errors": [{"step": u["step"], "source": u["source"], "error": (u["error"] or "")[:300]}
                   for u in usage if u["error"]],
        "sources": sorted({u["source"] for u in usage}),
    }
