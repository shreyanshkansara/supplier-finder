"""drop rfqs: the service now returns a vendor list instead of sending RFQ emails

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-23
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("rfqs")
    # Runs that were paused waiting for RFQ approval can't be resumed any more (that step is gone).
    op.execute(
        "UPDATE sourcing_requests SET status = 'failed', pending = NULL, "
        "error = 'Stopped: the RFQ approval step was removed in v0.4. Submit the request again.' "
        "WHERE status = 'awaiting_approval'"
    )


def downgrade() -> None:
    op.create_table(
        "rfqs",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("request_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("sourcing_requests.id", ondelete="CASCADE"), nullable=False),
        sa.Column("vendor_id", sa.String(16), sa.ForeignKey("vendors.id"), nullable=False),
        sa.Column("subject", sa.String(300), nullable=False),
        sa.Column("body", sa.Text, nullable=False),
        sa.Column("line_items", pg.JSONB),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("idempotency_key", sa.String(64), nullable=False, unique=True),
        sa.Column("sent_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("request_id", "vendor_id", name="uq_rfqs_request_vendor"),
    )
