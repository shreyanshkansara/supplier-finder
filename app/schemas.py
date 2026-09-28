"""Pydantic v2 models: the typed contract between the API, the LLM and the graph."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ----------------------------------------------------------------------------------------
# Agent-facing models (the LLM fills these via structured outputs)
# ----------------------------------------------------------------------------------------
# Fields that must simply be filled in. `specs` is not here: whether the product is specific enough
# is decided in the completeness step (see app.agent.nodes.find_missing).
REQUIRED_FIELDS = ("item", "quantity", "unit", "delivery_location", "deadline_days")


class SourcingRequirement(BaseModel):
    """What the buyer needs. Anything the user didn't say stays None, never guessed."""

    item: str | None = Field(None, description="What to buy, e.g. 'waterproof plywood'")
    specs: str | None = Field(None, description="Specs needed to quote: size, thickness, density, finish...")
    quantity: float | None = Field(None, gt=0)
    unit: str | None = Field(None, description="cft, sheets, pcs, kg, m, L, sq ft")
    delivery_location: str | None = Field(None, description="City, e.g. 'Jodhpur'")
    delivery_state: str | None = Field(None, description="Indian state, e.g. 'Rajasthan'")
    deadline_days: int | None = Field(None, gt=0, description="Days from today")
    category: str | None = Field(None, description="One of the vendor categories, if clear")
    notes: str | None = None

    def missing_fields(self) -> list[str]:
        return [f for f in REQUIRED_FIELDS if getattr(self, f) in (None, "")]


# ----------------------------------------------------------------------------------------
# API models
# ----------------------------------------------------------------------------------------
class VendorOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    city: str
    state: str
    regions_served: list[str]
    categories: list[str]
    specs: str
    moq: int
    moq_unit: str
    lead_time_days: int
    price_band: str
    indicative_price: str | None
    payment_terms: str | None
    gst_registered: bool
    rating_notes: str | None
    contact_name: str | None
    contact_email: str
    phone: str | None


class VendorPage(BaseModel):
    items: list[VendorOut]
    total: int
    limit: int
    offset: int


class ImportResult(BaseModel):
    upserted: int
    embedded: int
    embedding_model: str


class RequestCreate(BaseModel):
    text: str = Field(..., min_length=3, max_length=4000, examples=[
        "200 sheets of 18 mm waterproof plywood delivered to Udaipur within 20 days"])


RunStatus = Literal["running", "clarifying", "completed", "failed"]


class RequestOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: RunStatus
    raw_text: str
    requirement: dict[str, Any] | None
    pending: dict[str, Any] | None = Field(
        None, description="While paused: the clarifying questions for the buyer")
    report: dict[str, Any] | None = Field(
        None, description="When completed: the vendors that can supply this, and the near misses")
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class RequestSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    raw_text: str
    status: RunStatus
    created_at: datetime


class ClarifyIn(BaseModel):
    """Answer the clarifying questions with structured fields, free text, or both.

    {"answers": {"quantity": 200, "deadline_days": 20}}
    {"text": "200 sheets, deliver to Udaipur within 20 days"}
    """

    answers: dict[str, Any] = Field(default_factory=dict)
    text: str | None = Field(None, max_length=2000)

    @model_validator(mode="after")
    def _one_of(self):
        if not self.answers and not (self.text and self.text.strip()):
            raise ValueError("provide `answers`, `text`, or both")
        return self


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: Any | None = None


class ErrorOut(BaseModel):
    error: ErrorBody
