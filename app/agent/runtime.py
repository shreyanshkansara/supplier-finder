"""Wires the graph to real dependencies and a checkpointer, and runs/resumes threads.

thread_id == sourcing_requests.id, so a paused run can be found and resumed from the API,
and with the Postgres checkpointer it survives a server restart.
"""

import contextlib
import logging
from collections.abc import AsyncIterator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.nodes import AgentDeps, RankPolicy
from app.config import Settings
from app.models import SourcingRequest
from app.services.audit import DbAudit
from app.services.embeddings import Embedder
from app.services.judgments import check_quotable, judge_candidates
from app.services.llm import LLM
from app.services.retrieval import hybrid_search
from app.services.vendor_import import index_status

log = logging.getLogger(__name__)


def production_deps(settings: Settings, sessionmaker: async_sessionmaker[AsyncSession],
                    embedder: Embedder, llm: LLM) -> AgentDeps:
    async def retrieve(req: dict) -> list[dict]:
        query = f"{req.get('item') or ''}. {req.get('specs') or ''}"
        async with sessionmaker() as s:
            return await hybrid_search(s, embedder, query, category=req.get("category"),
                                       delivery_state=req.get("delivery_state"), k=settings.retrieval_top_k)

    async def status() -> dict:
        async with sessionmaker() as s:
            return await index_status(s, embedder.name)

    return AgentDeps(
        index_status=status,
        retrieve=retrieve,
        judge=judge_candidates,
        quotable=check_quotable,
        policy=policy_from(settings),
        audit=DbAudit(sessionmaker),
        llm=llm,
        top_n=settings.max_vendors,
        max_clarify_rounds=settings.max_clarify_rounds,
    )


def policy_from(settings: Settings) -> RankPolicy:
    return RankPolicy(w_similarity=settings.rank_w_similarity, w_spec=settings.rank_w_spec,
                      w_fit=settings.rank_w_fit, shortlist_at=settings.jev_shortlist_at,
                      review_at=settings.jev_review_at, quotable_at=settings.jev_quotable_at)


@contextlib.asynccontextmanager
async def open_checkpointer(settings: Settings) -> AsyncIterator:
    if settings.checkpointer == "memory":
        yield InMemorySaver()
        return
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(settings.checkpointer_url) as saver:
        await saver.setup()  # creates LangGraph's own checkpoint tables (idempotent)
        yield saver


def _config(request_id) -> dict:
    return {"configurable": {"thread_id": str(request_id)}}


async def advance(graph, row: SourcingRequest, payload) -> SourcingRequest:
    """Run the graph until it finishes or pauses, then mirror its state onto the DB row.

    payload is the initial state for a new run, or Command(resume=...) to continue a paused one.
    TODO: move to a background worker if runs get slow; the API would then return 202 + poll.
    """
    config = _config(row.id)
    try:
        await graph.ainvoke(payload, config)
    except Exception as e:  # noqa: BLE001
        log.exception("run %s failed", row.id)
        row.status, row.error, row.pending = "failed", str(e), None
        return row

    snap = await graph.aget_state(config)
    values = snap.values
    row.requirement = values.get("requirement")
    interrupts = [i.value for t in snap.tasks for i in t.interrupts]
    if interrupts:
        pending = interrupts[0]
        row.pending = pending
        row.status = "clarifying"
    else:
        row.pending = None
        row.status = "completed"
        row.report = values.get("report")
    return row


def start_payload(row: SourcingRequest) -> dict:
    return {"request_id": str(row.id), "raw_text": row.raw_text, "log": []}


def resume_payload(value) -> Command:
    return Command(resume=value)
