"""Append-only audit trail: every agent decision and every answer from the buyer."""

import logging
import uuid
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import AuditLog

log = logging.getLogger(__name__)


class AuditSink(Protocol):
    async def __call__(self, request_id: str | None, actor: str, action: str, detail: dict | None = None) -> None: ...


class DbAudit:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]):
        self.sessionmaker = sessionmaker

    async def __call__(self, request_id, actor, action, detail=None) -> None:
        async with self.sessionmaker() as s:
            s.add(AuditLog(request_id=uuid.UUID(str(request_id)) if request_id else None, actor=actor, action=action, detail=detail))
            await s.commit()


class MemoryAudit:
    def __init__(self):
        self.events: list[dict] = []

    async def __call__(self, request_id, actor, action, detail=None) -> None:
        self.events.append({"request_id": request_id, "actor": actor, "action": action, "detail": detail})
