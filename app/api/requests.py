"""Sourcing requests: start a run, inspect it, and resume it at each human pause."""

import uuid

from fastapi import APIRouter, Depends, Query, Request
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.runtime import advance, resume_payload, start_payload
from app.auth import require_api_key
from app.db import get_session
from app.errors import AppError, Conflict, NotFound
from app.models import AuditLog, SourcingRequest
from app.schemas import ClarifyIn, RequestCreate, RequestOut, RequestSummary, SourcingRequirement

router = APIRouter(prefix="/requests", tags=["requests"], dependencies=[Depends(require_api_key)])


async def _load(session: AsyncSession, request_id: uuid.UUID, *, lock: bool = False) -> SourcingRequest:
    # FOR NO KEY UPDATE: blocks a concurrent resume of the same run, but still lets graph nodes
    # insert audit rows that reference this one (a plain FOR UPDATE would deadlock them).
    row = await session.get(SourcingRequest, request_id, with_for_update={"key_share": True} if lock else None)
    if row is None:
        raise NotFound(f"Request {request_id}")
    return row


@router.post("", response_model=RequestOut, status_code=201)
async def create_request(body: RequestCreate, request: Request, session: AsyncSession = Depends(get_session)):
    row = SourcingRequest(id=uuid.uuid4(), raw_text=body.text, status="running")
    session.add(row)
    await session.commit()  # the row must exist before nodes write audit entries against it
    await advance(request.app.state.graph, row, start_payload(row))
    await session.commit()
    await session.refresh(row)
    return row


@router.get("", response_model=list[RequestSummary])
async def list_requests(limit: int = Query(20, ge=1, le=100), session: AsyncSession = Depends(get_session)):
    """Most recent requests first (for the web page's history list)."""
    rows = (await session.execute(
        select(SourcingRequest).order_by(SourcingRequest.created_at.desc()).limit(limit))).scalars().all()
    return rows


@router.get("/{request_id}", response_model=RequestOut)
async def get_request(request_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    return await _load(session, request_id)


@router.post("/{request_id}/clarify", response_model=RequestOut)
async def clarify(request_id: uuid.UUID, body: ClarifyIn, request: Request,
                  session: AsyncSession = Depends(get_session)):
    row = await _load(session, request_id, lock=True)  # row lock: no double-resume
    if row.status != "clarifying":
        raise Conflict(f"Request is '{row.status}', not waiting for clarification")
    # Validate before resuming, so a typo gets a 422 instead of failing the paused run.
    try:
        SourcingRequirement(**{**(row.requirement or {}), **body.answers})
    except ValidationError as e:
        raise AppError(422, "validation_error", "Invalid clarification answers",
                       [{"loc": err["loc"], "msg": err["msg"]} for err in e.errors()]) from e
    await advance(request.app.state.graph, row, resume_payload(body.model_dump()))
    await session.commit()
    await session.refresh(row)
    return row


@router.get("/{request_id}/report")
async def report(request_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    """The result: vendors that can supply the requirement, with contact details, and the near misses."""
    row = await _load(session, request_id)
    if row.status != "completed" or row.report is None:
        raise Conflict(f"Report not ready: request is '{row.status}'")
    return row.report


@router.get("/{request_id}/audit")
async def audit_trail(request_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    await _load(session, request_id)
    rows = (await session.execute(
        select(AuditLog).where(AuditLog.request_id == request_id).order_by(AuditLog.id))).scalars().all()
    return [{"at": r.created_at, "actor": r.actor, "action": r.action, "detail": r.detail} for r in rows]

