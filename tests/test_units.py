"""Pure unit tests: no database, no network."""

import pytest

from app.agent.nodes import code_checks, norm_unit, rule_based_parse
from app.schemas import SourcingRequirement
from app.services.embeddings import HashEmbedder
from app.services.geo import state_for
from app.services.vendor_import import ImportError_, parse_csv


def test_parse_complete_request():
    r = rule_based_parse("200 sheets of 18 mm waterproof plywood delivered to Udaipur within 20 days")
    assert r.quantity == 200 and r.unit == "sheets"
    assert r.delivery_location == "Udaipur" and r.delivery_state == "Rajasthan"
    assert r.deadline_days == 20
    assert r.missing_fields() == []


def test_parse_weeks_and_commas():
    r = rule_based_parse("1,500 pairs of soft-close channels to Jodhpur within 2 weeks")
    assert r.quantity == 1500 and r.unit == "pcs" and r.deadline_days == 14


@pytest.mark.parametrize("text,qty,unit", [
    ("2,000 soft-close concealed hinges, 35 mm cup, to Jodhpur within 10 days", 2000, "pcs"),
    ("2,000 five-ply corrugated boxes to Jodhpur within 1 week", 2000, "pcs"),
    ("1 job of CNC routing in Jodhpur within 10 days", 1, "jobs"),
])
def test_parse_counts_of_countable_items(text, qty, unit):
    r = rule_based_parse(text)
    assert (r.quantity, r.unit) == (qty, unit)


def test_vague_request_is_incomplete():
    missing = rule_based_parse("Need some good wood for the new bed range").missing_fields()
    assert {"quantity", "unit", "delivery_location", "deadline_days"} <= set(missing)


def test_requirement_rejects_nonsense_quantity():
    with pytest.raises(ValueError):
        SourcingRequirement(quantity=-5)


@pytest.mark.parametrize("raw,expected", [("Sheets", "sheets"), ("pairs", "pcs"), ("litres", "L"),
                                          ("sqft", "sq ft"), (None, None)])
def test_norm_unit(raw, expected):
    assert norm_unit(raw) == expected


def test_code_checks():
    v = {"moq": 100, "moq_unit": "sheets", "lead_time_days": 10}
    assert code_checks({"quantity": 200, "unit": "sheets", "deadline_days": 20}, v) == {
        "unit_match": True, "moq_ok": True, "lead_ok": True}
    assert code_checks({"quantity": 50, "unit": "sheets", "deadline_days": 5}, v)["moq_ok"] is False
    assert code_checks({"quantity": 50, "unit": "sheets", "deadline_days": 5}, v)["lead_ok"] is False
    # Different units: can't compare, so unknown rather than a false pass/fail
    assert code_checks({"quantity": 50, "unit": "kg", "deadline_days": None}, v)["moq_ok"] is None


async def test_hash_embedder_is_deterministic_and_lexical():
    e = HashEmbedder(1536)
    a, b, c = await e.embed(["waterproof plywood 18 mm", "18 mm waterproof plywood sheets", "PU foam 40 density"])
    assert a == (await e.embed(["waterproof plywood 18 mm"]))[0]
    dot = lambda x, y: sum(i * j for i, j in zip(x, y, strict=True))  # noqa: E731
    assert dot(a, b) > dot(a, c)


def test_state_lookup():
    assert state_for("jodhpur") == "Rajasthan" and state_for("Atlantis") is None


def test_parse_csv_roundtrip():
    rows = parse_csv(open("data/vendors.csv", "rb").read())
    assert len(rows) == 500
    assert isinstance(rows[0]["regions_served"], list) and isinstance(rows[0]["moq"], int)


def test_parse_csv_missing_columns():
    with pytest.raises(ImportError_):
        parse_csv(b"vendor_id,name\nV-1,x\n")


@pytest.mark.parametrize("raw,async_url,sync_url", [
    ("postgresql://u:p@ep-x.neon.tech/neondb?sslmode=require&channel_binding=require",
     "postgresql+asyncpg://u:p@ep-x.neon.tech/neondb?ssl=require",
     "postgresql://u:p@ep-x.neon.tech/neondb?sslmode=require"),
    ("postgres://u:p@h:5432/db", "postgresql+asyncpg://u:p@h:5432/db", "postgresql://u:p@h:5432/db"),
    ("postgresql+asyncpg://u:p@localhost/db", "postgresql+asyncpg://u:p@localhost/db", "postgresql://u:p@localhost/db"),
])
def test_hosted_database_urls_are_normalised(raw, async_url, sync_url):
    from app.config import Settings

    s = Settings(database_url=raw)
    assert (s.database_url, s.checkpointer_url) == (async_url, sync_url)


def test_precomputed_vectors_round_trip(tmp_path, monkeypatch):
    from app.services import vendor_import as vi

    monkeypatch.setattr(vi, "PRECOMPUTED_DIR", tmp_path)
    vec = [0.6, 0.8] + [0.0] * 6
    vi.save_precomputed("gemini:test:8", 8, [("abc", vec)])
    loaded = vi.load_precomputed("gemini:test:8")
    assert list(loaded) == ["abc"] and [round(x, 3) for x in loaded["abc"][:2]] == [0.6, 0.8]
    assert vi.load_precomputed("other") == {}


def test_shipped_gemini_vectors_cover_every_vendor():
    import json
    from pathlib import Path

    from app.services import vendor_import as vi

    known = vi.load_precomputed("gemini:gemini-embedding-2:1536")
    rows = json.loads(Path("data/vendors.json").read_text())
    missing = [r["vendor_id"] for r in rows if vi.text_hash(vi.embedding_text(r)) not in known]
    assert missing == []  # a fresh install is searchable without spending embedding quota
