import os

# Must be set before app modules read settings.
os.environ.setdefault("API_KEY", "test-key")
os.environ["CHECKPOINTER"] = "memory"
os.environ.pop("GEMINI_API_KEY", None)  # tests always use the offline embedder and rules parser
os.environ.pop("TYPESAFE_API_KEY", None)
if os.environ.get("TEST_DATABASE_URL"):
    os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]

import pytest  # noqa: E402

from app.agent.nodes import AgentDeps  # noqa: E402
from app.services.audit import MemoryAudit  # noqa: E402

API = {"X-API-Key": "test-key"}


def fake_vendor(vid: str, sim: float, moq: int = 50, lead: int = 10, unit: str = "sheets") -> dict:
    return {
        "id": vid, "name": f"Vendor {vid}", "city": "Jodhpur", "state": "Rajasthan",
        "regions_served": ["Rajasthan"], "categories": ["Panel boards"],
        "specs": "Waterproof plywood, 12, 16 and 18 mm", "moq": moq, "moq_unit": unit,
        "lead_time_days": lead, "price_band": "mid", "indicative_price": "₹3,000 per sheet",
        "payment_terms": "100% advance", "rating_notes": None, "contact_name": "Test Person",
        "contact_email": f"vendor.test+{vid.lower()}@example.com", "phone": "+91-00000-00001",
        "similarity": sim,
    }


@pytest.fixture
def fake_deps():
    candidates = [
        fake_vendor("V-1", 0.91),
        fake_vendor("V-2", 0.85, moq=500),  # fails MOQ for a 200-sheet order
        fake_vendor("V-3", 0.80, lead=40),  # fails lead time
        fake_vendor("V-4", 0.75),
    ]

    async def retrieve(req):
        return [dict(c) for c in candidates]

    # judge / quotable default to "Jev not configured": code checks only
    return AgentDeps(retrieve=retrieve, audit=MemoryAudit(), top_n=5)
