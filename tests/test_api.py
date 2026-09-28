"""API tests. The `db` ones need Postgres + pgvector at TEST_DATABASE_URL (CI provides one)."""

import os
import subprocess
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

from tests.conftest import API

needs_db = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")


@pytest.fixture(scope="module")
def client():
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c


# ---------------------------------------------------------------- no database needed
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert r.headers["X-Request-ID"]


def test_auth_required_and_error_format(client):
    r = client.get("/vendors", headers={"X-Request-ID": "abc123"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"
    assert r.json()["error"]["request_id"] == "abc123"


def test_validation_error_format(client):
    r = client.post("/requests", json={"text": ""}, headers=API)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


# ---------------------------------------------------------------- database
@pytest.fixture(scope="module")
def db_client():
    env = {**os.environ, "DATABASE_URL": os.environ["TEST_DATABASE_URL"]}
    for cmd in (["downgrade", "base"], ["upgrade", "head"]):
        subprocess.run([sys.executable, "-m", "alembic", *cmd], check=True, env=env)
    from app.main import create_app

    with TestClient(create_app()) as c:
        with open("data/vendors.csv", "rb") as fh:
            r = c.post("/vendors/import", files={"file": ("vendors.csv", fh, "text/csv")}, headers=API)
        assert r.status_code == 201, r.text
        assert r.json()["upserted"] == 500
        yield c


@needs_db
@pytest.mark.db
def test_list_and_filter_vendors(db_client):
    r = db_client.get("/vendors", params={"category": "Panel boards", "serves": "Rajasthan", "limit": 5},
                      headers=API)
    body = r.json()
    assert r.status_code == 200 and len(body["items"]) == 5 and body["total"] > 5
    for v in body["items"]:
        assert "Panel boards" in v["categories"]
        assert {"Rajasthan", "Pan-India"} & set(v["regions_served"])
    assert db_client.get("/vendors/V-9999", headers=API).status_code == 404


@needs_db
@pytest.mark.db
def test_request_flow_clarify_then_vendor_list(db_client):
    r = db_client.post("/requests", json={"text": "Need waterproof plywood, 18 mm thick"}, headers=API)
    assert r.status_code == 201, r.text
    req = r.json()
    assert req["status"] == "clarifying" and "quantity" in req["pending"]["missing"] and req["report"] is None
    rid = req["id"]

    # the result isn't ready while the run waits for the buyer -> 409 in the standard error format
    r = db_client.get(f"/requests/{rid}/report", headers=API)
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"

    # a bad answer is rejected without breaking the paused run
    r = db_client.post(f"/requests/{rid}/clarify", headers=API, json={"answers": {"quantity": -3}})
    assert r.status_code == 422

    r = db_client.post(f"/requests/{rid}/clarify", headers=API, json={"answers": {
        "quantity": 200, "unit": "sheets", "delivery_location": "Udaipur", "deadline_days": 20}})
    req = r.json()
    assert req["status"] == "completed", req
    vendors = req["report"]["vendors"]
    assert 1 <= len(vendors) <= 10
    for v in vendors:
        assert v["contact_email"] and v["lead_time_days"] <= 20
        assert v["checks"]["moq_ok"] is not False

    # finished runs can't be resumed
    assert db_client.post(f"/requests/{rid}/clarify", json={"answers": {"quantity": 1}},
                          headers=API).status_code == 409

    assert db_client.get(f"/requests/{rid}/report", headers=API).json() == req["report"]
    audit = db_client.get(f"/requests/{rid}/audit", headers=API).json()
    actions = [a["action"] for a in audit]
    for a in ("parsed", "clarified", "retrieved", "verified", "ranked", "reported"):
        assert a in actions


@needs_db
@pytest.mark.db
def test_unknown_request_404(db_client):
    r = db_client.get(f"/requests/{uuid.uuid4()}", headers=API)
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


@needs_db
@pytest.mark.db
def test_clarify_with_free_text(db_client):
    rid = db_client.post("/requests", json={"text": "Need waterproof plywood, 18 mm thick"}, headers=API).json()["id"]
    r = db_client.post(f"/requests/{rid}/clarify", headers=API,
                       json={"text": "200 sheets, deliver to Udaipur within 20 days"})
    body = r.json()
    assert body["status"] == "completed", body
    assert body["requirement"]["quantity"] == 200 and body["requirement"]["delivery_state"] == "Rajasthan"
    assert body["requirement"]["item"] == "Need waterproof plywood, 18 mm thick"  # not overwritten
    assert db_client.post(f"/requests/{rid}/clarify", json={}, headers=API).status_code == 422


def test_web_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Supplier Finder" in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200


@needs_db
@pytest.mark.db
def test_request_history_lists_newest_first(db_client):
    db_client.post("/requests", json={"text": "Need some foam"}, headers=API)
    rows = db_client.get("/requests", params={"limit": 3}, headers=API).json()
    assert 1 <= len(rows) <= 3 and rows[0]["raw_text"] == "Need some foam"
    assert set(rows[0]) == {"id", "raw_text", "status", "created_at"}


@needs_db
@pytest.mark.db
def test_migration_0002_closes_runs_paused_for_approval(db_client):
    import psycopg

    from app.config import Settings

    url = Settings(database_url=os.environ["TEST_DATABASE_URL"]).checkpointer_url
    env = {**os.environ, "DATABASE_URL": os.environ["TEST_DATABASE_URL"]}
    subprocess.run([sys.executable, "-m", "alembic", "downgrade", "0001"], check=True, env=env)
    rid = uuid.uuid4()
    with psycopg.connect(url, autocommit=True) as c:
        c.execute("INSERT INTO sourcing_requests (id, raw_text, status, created_by) "
                  "VALUES (%s, 'old run', 'awaiting_approval', 'test')", (rid,))
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True, env=env)
    r = db_client.get(f"/requests/{rid}", headers=API).json()
    assert r["status"] == "failed" and "approval step was removed" in r["error"]
