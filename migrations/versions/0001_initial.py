"""initial schema: vendors (pgvector), sourcing_requests, rfqs, audit_log

Revision ID: 0001
Revises:
Create Date: 2026-09-23
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql as pg

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _timestamps():
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "vendors",
        sa.Column("id", sa.String(16), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("city", sa.String(100), nullable=False),
        sa.Column("state", sa.String(100), nullable=False),
        sa.Column("regions_served", pg.ARRAY(sa.String(100)), nullable=False),
        sa.Column("categories", pg.ARRAY(sa.String(100)), nullable=False),
        sa.Column("specs", sa.Text, nullable=False),
        sa.Column("moq", sa.Integer, nullable=False),
        sa.Column("moq_unit", sa.String(20), nullable=False),
        sa.Column("lead_time_days", sa.Integer, nullable=False),
        sa.Column("price_band", sa.String(20), nullable=False),
        sa.Column("indicative_price", sa.String(100)),
        sa.Column("payment_terms", sa.String(100)),
        sa.Column("gst_registered", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("rating_notes", sa.Text),
        sa.Column("contact_name", sa.String(100)),
        sa.Column("contact_email", sa.String(200), nullable=False),
        sa.Column("phone", sa.String(30)),
        sa.Column("embedding", Vector(1536)),
        sa.Column("embedding_model", sa.String(100)),
        *_timestamps(),
    )
    op.create_index("ix_vendors_categories", "vendors", ["categories"], postgresql_using="gin")
    op.create_index("ix_vendors_regions_served", "vendors", ["regions_served"], postgresql_using="gin")
    op.create_index(
        "ix_vendors_embedding_hnsw",
        "vendors",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "sourcing_requests",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("raw_text", sa.Text, nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("requirement", pg.JSONB),
        sa.Column("pending", pg.JSONB),
        sa.Column("report", pg.JSONB),
        sa.Column("error", sa.Text),
        sa.Column("created_by", sa.String(100), nullable=False),
        *_timestamps(),
    )
    op.create_index("ix_sourcing_requests_status", "sourcing_requests", ["status"])

    op.create_table(
        "rfqs",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "request_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("sourcing_requests.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("vendor_id", sa.String(16), sa.ForeignKey("vendors.id"), nullable=False),
        sa.Column("subject", sa.String(300), nullable=False),
        sa.Column("body", sa.Text, nullable=False),
        sa.Column("line_items", pg.JSONB),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("idempotency_key", sa.String(64), nullable=False, unique=True),
        sa.Column("sent_at", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.UniqueConstraint("request_id", "vendor_id", name="uq_rfqs_request_vendor"),
    )

    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column(
            "request_id", pg.UUID(as_uuid=True), sa.ForeignKey("sourcing_requests.id", ondelete="CASCADE")
        ),
        sa.Column("actor", sa.String(100), nullable=False),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("detail", pg.JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_audit_log_request_id", "audit_log", ["request_id"])


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("rfqs")
    op.drop_table("sourcing_requests")
    op.drop_table("vendors")
