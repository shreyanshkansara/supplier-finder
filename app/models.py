"""SQLAlchemy 2.0 ORM models. Keep in sync with migrations/versions/."""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 1536


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Vendor(TimestampMixin, Base):
    __tablename__ = "vendors"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # e.g. V-0001
    name: Mapped[str] = mapped_column(String(200))
    city: Mapped[str] = mapped_column(String(100))
    state: Mapped[str] = mapped_column(String(100))
    regions_served: Mapped[list[str]] = mapped_column(ARRAY(String(100)))
    categories: Mapped[list[str]] = mapped_column(ARRAY(String(100)))
    specs: Mapped[str] = mapped_column(Text)
    moq: Mapped[int] = mapped_column(Integer)
    moq_unit: Mapped[str] = mapped_column(String(20))
    lead_time_days: Mapped[int] = mapped_column(Integer)
    price_band: Mapped[str] = mapped_column(String(20))
    indicative_price: Mapped[str | None] = mapped_column(String(100))
    payment_terms: Mapped[str | None] = mapped_column(String(100))
    gst_registered: Mapped[bool] = mapped_column(Boolean, default=True)
    rating_notes: Mapped[str | None] = mapped_column(Text)
    contact_name: Mapped[str | None] = mapped_column(String(100))
    contact_email: Mapped[str] = mapped_column(String(200))
    phone: Mapped[str | None] = mapped_column(String(30))

    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    # Which embedder produced `embedding`, so switching models never mixes vector spaces.
    embedding_model: Mapped[str | None] = mapped_column(String(100))

    __table_args__ = (
        Index("ix_vendors_categories", "categories", postgresql_using="gin"),
        Index("ix_vendors_regions_served", "regions_served", postgresql_using="gin"),
        Index(
            "ix_vendors_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class SourcingRequest(TimestampMixin, Base):
    """One user request = one LangGraph thread (thread_id == id)."""

    __tablename__ = "sourcing_requests"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    raw_text: Mapped[str] = mapped_column(Text)
    # running | clarifying | completed | failed
    status: Mapped[str] = mapped_column(String(30), default="running", index=True)
    requirement: Mapped[dict | None] = mapped_column(JSONB)
    pending: Mapped[dict | None] = mapped_column(JSONB)  # the clarifying questions while paused
    report: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(100), default="api")


class AuditLog(Base):
    """Every agent decision and every human action, append-only."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    request_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sourcing_requests.id", ondelete="CASCADE"), index=True
    )
    actor: Mapped[str] = mapped_column(String(100))  # node name, or "user:<who>"
    action: Mapped[str] = mapped_column(String(100))
    detail: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
