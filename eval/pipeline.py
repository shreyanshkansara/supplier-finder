"""Shared evaluation plumbing: runs the same retrieval, checks, judgments and ranking the app runs.

Nothing here re-implements app logic. It calls hybrid_search, code_checks, judge_candidates and
rank_candidates directly, so a number from the eval is a number about the app.

Jev judgments are cached in eval/results/jev_cache.json, keyed by (model, requirement, vendor
listing). Re-running the eval or the threshold tuner costs nothing once a set has been judged,
and changing thresholds or weights never needs new model calls.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.agent.nodes import code_checks
from app.config import Settings
from app.services.judgments import JudgeOutcome, judge_candidates, requirement_state
from app.services.retrieval import hybrid_search

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
K = 5
TOP_K = 25


def load_requests() -> list[dict]:
    return json.loads((EVAL_DIR / "seed_requests.json").read_text())


def precision_at_k(ids: list[str], gold: set[str], k: int = K) -> float:
    """Hits in the top k, divided by the most hits possible (min(k, #gold)).

    Plain precision@5 would cap a request with only 3 correct vendors at 0.6; this doesn't.
    """
    if not gold:
        raise ValueError("precision is undefined without gold vendors")
    return sum(i in gold for i in ids[:k]) / min(k, len(gold))


def survives(c: dict) -> bool:
    return c["checks"]["moq_ok"] is not False and c["checks"]["lead_ok"] is not False


@dataclass
class Prepared:
    """One request after parse -> retrieve -> code checks (everything before Jev)."""

    rid: str
    kind: str
    gold: set[str]
    requirement: dict
    candidates: list[dict]  # survivors of the code checks, with similarity + checks
    usage: list[dict] = field(default_factory=list)
    latency_ms: int = 0


class ParseCache:
    """Caches the model's RAW parse output per (parser, prompt version, text).

    The Gemini free tier allows only a few hundred requests a day, and run_eval, run_parse_eval and
    the tuner all parse the same 40 texts. Only raw model output is cached; the code-side
    post-processing is re-applied on every hit, so changing that code never needs new model calls.
    Changing the prompt or schema changes the version and invalidates the cache. Fallback (failed)
    parses are never cached. Delete the file to force fresh parses.
    """

    def __init__(self, path: Path = RESULTS_DIR / "parse_cache.json", enabled: bool = True):
        from app.services import llm as llm_module

        self.path, self.enabled = path, enabled
        self.version = hashlib.sha1((llm_module.PARSE_INSTRUCTIONS + json.dumps(
            llm_module.ParsedRequirement.model_json_schema(), sort_keys=True)).encode()).hexdigest()[:12]
        self.data: dict = json.loads(path.read_text()) if enabled and path.exists() else {}

    async def parse(self, llm, text: str):
        from app.services.llm import LLMResult, ParsedRequirement

        cacheable = self.enabled and hasattr(llm, "postprocess")
        key = f"{llm.name}|{self.version}|{text}"
        if cacheable and key in self.data:
            raw = ParsedRequirement(**self.data[key])
            return llm.postprocess(raw, text), LLMResult(source=f"{llm.name}(cached)")
        req, res = await llm.parse_requirement(text)
        if cacheable and not res.error and "raw" in res.extra:
            self.data[key] = res.extra["raw"]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=0, sort_keys=True))
        return req, res


async def prepare(session, embedder, llm, r: dict, parse_cache: "ParseCache | None" = None) -> Prepared:
    t0 = time.perf_counter()
    parsed, res = await (parse_cache.parse(llm, r["text"]) if parse_cache else llm.parse_requirement(r["text"]))
    req = parsed.model_dump()
    query = f"{req.get('item') or ''}. {req.get('specs') or ''}"
    found = await hybrid_search(session, embedder, query, category=req.get("category"),
                                delivery_state=req.get("delivery_state"), k=TOP_K)
    checked = [c | {"checks": code_checks(req, c)} for c in found]
    return Prepared(rid=r["id"], kind=r["kind"], gold=set(r.get("gold_vendor_ids") or []), requirement=req,
                    candidates=[c for c in checked if survives(c)], usage=[res.usage("parse")],
                    latency_ms=int((time.perf_counter() - t0) * 1000))


class JevCache:
    def __init__(self, path: Path = RESULTS_DIR / "jev_cache.json", enabled: bool = True):
        self.path, self.enabled = path, enabled
        self.data: dict = json.loads(path.read_text()) if enabled and path.exists() else {}

    @staticmethod
    def key(model: str, req: dict, cand: dict) -> str:
        raw = json.dumps([model, requirement_state(req), cand["id"], cand["specs"], cand["categories"]],
                         sort_keys=True)
        return hashlib.sha1(raw.encode()).hexdigest()

    def save(self) -> None:
        if self.enabled and self.data:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=0, sort_keys=True))


async def judge_cached(p: Prepared, settings: Settings, cache: JevCache, judge=judge_candidates) -> JudgeOutcome:
    """Judge p.candidates, only calling Jev for listings not already in the cache."""
    model = settings.typesafe_model or "jev-latest"
    keys = {c["id"]: JevCache.key(model, p.requirement, c) for c in p.candidates}
    todo = [c for c in p.candidates if keys[c["id"]] not in cache.data]
    out = JudgeOutcome(by_vendor={c["id"]: cache.data[keys[c["id"]]] for c in p.candidates
                                  if keys[c["id"]] in cache.data})
    if todo:
        fresh = await judge(p.requirement, todo, settings=settings)
        out.usage = fresh.usage if fresh else []
        if fresh and fresh.ok:
            for c in todo:
                cache.data[keys[c["id"]]] = fresh.by_vendor[c["id"]]
                out.by_vendor[c["id"]] = fresh.by_vendor[c["id"]]
        else:
            out.by_vendor = {}  # same all-or-nothing rule as the app
    return out


def with_judgments(candidates: list[dict], outcome: JudgeOutcome) -> list[dict]:
    return [c | outcome.by_vendor.get(c["id"], {}) for c in candidates]


def tokens(usage: list[dict]) -> int:
    return sum(u["input_tokens"] + u["output_tokens"] for u in usage)


def fallbacks(usage: list[dict]) -> int:
    return sum(1 for u in usage if u["error"])
