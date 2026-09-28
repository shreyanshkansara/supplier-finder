"""Semantic judgments with TypeSafe Jev.

Rule of the project: the LLM writes text, Jev makes judgments, code enforces rules.

Two judgments:
  judge_candidates   per vendor: does its listing satisfy the requested item + specs (Noul -> p_spec),
                     and how well does it fit overall (Score over 4 described levels -> fit 0..3).
                     All candidates go into ONE state and are judged in one request per batch,
                     so 25 vendors cost one round trip, not 25.
  check_quotable     is the requirement specific enough for a supplier to give a firm quote? (Noul)

What is deliberately NOT asked of Jev: MOQ, lead time, delivery region. Those are arithmetic
and lookups, already enforced in SQL and code before Jev sees a candidate.

Failure policy: judgments are an improvement, not a dependency. If Jev isn't configured the
functions return None; if a call fails they return an outcome carrying the error, and the
pipeline continues on code checks alone. Both cases are visible in the audit log and report.
"""

import logging
import time
from dataclasses import dataclass, field

from app.config import Settings, get_settings

log = logging.getLogger(__name__)

# The examples in these criteria are deliberately NOT taken from the evaluation set (no sheesham,
# plywood, foam...), so the eval measures how Jev generalises rather than whether it saw the answers.
SPEC_CRITERIA = {
    "true": "The listing offers the requested kind of item AND nothing in it contradicts a stated spec "
            "(material or species, size or thickness, density or grade, finish, treatment, mechanism, "
            "safety property...). Different wording for the same thing counts, e.g. a botanical or trade name "
            "for a common material, 'aluminum' for 'aluminium', 'fire-resistant' for 'fire-retardant'.",
    "false": "A different product, even if it shares words (e.g. marble-look tiles when marble is wanted, "
             "leather-effect vinyl when leather is wanted, a service when goods are wanted), OR a stated spec is "
             "missing from the listing or contradicted (e.g. 12 mm when 15 mm is needed, untreated when treated "
             "is needed, manual when motorised is needed), OR the listing is too vague to tell "
             "('all types available').",
}
FIT_CRITERIA = [
    "Wrong product, or the listing is too vague to say what is sold",
    "Right kind of product, but a key stated spec is missing or contradicted",
    "Right product; every key spec is covered, one minor detail is unclear",
    "Right product and every stated spec is explicitly covered",
]
QUOTABLE_INSTRUCTIONS = (
    "Do `requirement.item` and `requirement.specs` together say WHICH product or service is wanted, i.e. give "
    "at least one property that narrows it down beyond the general kind of goods and what it will be used for?")
QUOTABLE_CRITERIA = {
    "true": "Yes: at least one narrowing property, such as a specific type, variant or process (e.g. 'MDF' "
            "rather than just 'board', 'enamel paint' rather than just 'paint', 'laser cutting' rather than just "
            "'job work'), a size, thickness or length, a grade, "
            "density or count, a material or species, a finish or colour, or a property such as treated, "
            "waterproof or heat-resistant.",
    "false": "No: only the general kind of goods (e.g. 'paint', 'screws', 'timber', 'cloth', 'tiles'), "
             "optionally with what it is for (e.g. 'for the office chairs', 'for the showroom'), which does "
             "not tell a supplier which product to quote.",
}


@dataclass
class JudgeOutcome:
    """Per-vendor judgments plus the usage/errors of the calls that produced them."""

    by_vendor: dict[str, dict] = field(default_factory=dict)  # {vendor_id: {"p_spec", "fit", "fit_confidence"}}
    usage: list[dict] = field(default_factory=list)  # same shape as LLMResult.usage(), one per call

    @property
    def ok(self) -> bool:
        return bool(self.by_vendor) and not any(u["error"] for u in self.usage)


def _usage(step: str, model: str, t0: float, resp=None, error: str | None = None) -> dict:
    u = getattr(resp, "usage", None)
    return {"step": step, "source": f"jev:{model}", "input_tokens": getattr(u, "input_tokens", 0) or 0,
            "output_tokens": getattr(u, "output_tokens", 0) or 0,
            "latency_ms": int((time.perf_counter() - t0) * 1000), "error": error}


def _client(settings: Settings):
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

    return AsyncTypeSafeClient(api_key=settings.typesafe_api_key, model=settings.typesafe_model,
                               retry=RetryPolicy(max_retries=settings.llm_retry_attempts,
                                                 timeout=settings.llm_timeout_s))


def requirement_state(req: dict) -> dict:
    return {k: req.get(k) for k in ("item", "specs", "category") if req.get(k)}


def build_candidate_questions(n: int) -> dict:
    from typesafe_sdk import Noul, Score

    questions = {}
    for i in range(n):
        questions[f"spec_{i}"] = Noul(
            instructions=f"Does the listing `vendors[{i}]` (its `specs` and `categories`) offer what "
                         f"`requirement` asks for, meeting every spec stated in `requirement.item` and "
                         f"`requirement.specs`?",
            criteria=SPEC_CRITERIA)
        questions[f"fit_{i}"] = Score(
            instructions=f"How well does the listing `vendors[{i}]` match `requirement` as a whole?",
            criteria=FIT_CRITERIA)
    return questions


async def judge_candidates(requirement: dict, candidates: list[dict], *, settings: Settings | None = None,
                           client=None) -> JudgeOutcome | None:
    """Judge every candidate. None when Jev isn't configured (and no client is injected)."""
    s = settings or get_settings()
    if not candidates or (client is None and not s.typesafe_api_key):
        return None
    model = s.typesafe_model or "jev-latest"
    out = JudgeOutcome()
    req_state = requirement_state(requirement)
    size = max(1, s.jev_batch_size)
    try:
        # one client (one connection pool) for all batches of this request
        async with (client if client is not None else _client(s)) as ts:
            for start in range(0, len(candidates), size):
                batch = candidates[start : start + size]
                vendors = [{"id": c["id"], "categories": c["categories"], "specs": c["specs"]} for c in batch]
                t0 = time.perf_counter()
                try:
                    resp = await ts.system_one(state={"requirement": req_state, "vendors": vendors},
                                               questions=build_candidate_questions(len(batch)))
                    for i, v in enumerate(vendors):
                        fit = resp.scores[f"fit_{i}"]
                        out.by_vendor[v["id"]] = {
                            "p_spec": round(float(resp.nouls[f"spec_{i}"].noul), 4),
                            "fit": round(float(fit.score), 4),
                            "fit_confidence": round(float(fit.confidence), 4) if fit.confidence is not None else None,
                        }
                    out.usage.append(_usage("verify", model, t0, resp))
                except Exception as e:  # noqa: BLE001 - never let judgments break a run
                    log.warning("Jev judge_candidates batch failed: %s", e)
                    out.usage.append(_usage("verify", model, t0, error=f"{type(e).__name__}: {e}"))
    except Exception as e:  # noqa: BLE001 - e.g. client construction / auth
        log.warning("Jev client failed: %s", e)
        out.usage.append({"step": "verify", "source": f"jev:{model}", "input_tokens": 0, "output_tokens": 0,
                          "latency_ms": 0, "error": f"{type(e).__name__}: {e}"})
    if any(u["error"] for u in out.usage):
        # All-or-nothing: mixing judged and unjudged vendors would rank them on different scales.
        out.by_vendor = {}
    return out


async def check_quotable(requirement: dict, *, settings: Settings | None = None,
                         client=None) -> tuple[float | None, dict | None]:
    """Probability that the requirement is specific enough to quote. (None, None) if Jev is off."""
    s = settings or get_settings()
    if client is None and not s.typesafe_api_key:
        return None, None
    if not requirement.get("item"):
        return None, None  # the item itself is missing; nothing to judge
    from typesafe_sdk import Noul

    model = s.typesafe_model or "jev-latest"
    t0 = time.perf_counter()
    try:
        cm = client if client is not None else _client(s)
        async with cm as ts:
            resp = await ts.system_one(
                state={"requirement": requirement_state(requirement)},
                questions={"quotable": Noul(instructions=QUOTABLE_INSTRUCTIONS, criteria=QUOTABLE_CRITERIA)})
        return round(float(resp.nouls["quotable"].noul), 4), _usage("completeness", model, t0, resp)
    except Exception as e:  # noqa: BLE001
        log.warning("Jev check_quotable failed: %s", e)
        return None, _usage("completeness", model, t0, error=f"{type(e).__name__}: {e}")
