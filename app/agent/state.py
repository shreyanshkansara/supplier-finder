"""Graph state. Everything is plain JSON-able data so the Postgres checkpointer can store it."""

import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict, total=False):
    request_id: str
    raw_text: str

    requirement: dict  # SourcingRequirement.model_dump()
    missing: list[str]  # fields the completeness check says we still need
    p_quotable: float | None  # Jev: is the requirement specific enough to quote?
    clarify_rounds: int

    candidates: list[dict]  # from retrieval, each with `similarity`
    index: dict | None  # search-index status, only looked up when retrieval found nothing
    verified: list[dict]  # candidates + `checks` + optional Jev probabilities
    shortlist: list[dict]  # ranked, with `score` and `review_flag`
    dropped: list[dict]  # ruled out, with `drop_reason` (become the report's near misses)

    report: dict
    llm_usage: Annotated[list[dict], operator.add]  # one entry per LLM call: tokens, latency, fallback
    log: Annotated[list[str], operator.add]  # human-readable trace, appended by each node
