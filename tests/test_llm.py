"""LLM + embedding tests with a mocked Gemini client: no key, no network, no cost."""

import json
from datetime import date
from types import SimpleNamespace

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent.graph import build_graph
from app.config import Settings
from app.services.embeddings import GeminiEmbedder, HashEmbedder, get_embedder, make_client
from app.services.llm import (
    GeminiLLM,
    ParsedRequirement,
    RuleBasedLLM,
    get_llm,
    to_requirement,
)

TODAY = date(2026, 9, 23)


class FakeClient:
    """Mimics `client.aio.models.generate_content(...)`.

    `reply` is a Pydantic object, an exception, None (empty/blocked response), or fn(kwargs) -> one of those.
    """

    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.aio = SimpleNamespace(models=self)

    async def generate_content(self, **kw):
        self.calls.append(kw)
        out = self.reply(kw) if callable(self.reply) else self.reply
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(
            text=out.model_dump_json() if out is not None else None,
            usage_metadata=SimpleNamespace(prompt_token_count=120, candidates_token_count=30, thoughts_token_count=10))


def schema_title(kw) -> str:
    return kw["config"].response_json_schema["title"]


def parsed(**kw) -> ParsedRequirement:
    base = dict(item=None, specs=None, quantity=None, unit=None, delivery_location=None,
                deadline_days=None, deadline_date=None, category=None, notes=None)
    return ParsedRequirement(**(base | kw))


# ---------------------------------------------------------------- schemas
SUPPORTED_KEYS = {"type", "properties", "required", "anyOf", "enum", "description", "title", "items", "format",
                  "minimum", "maximum", "minItems", "maxItems", "$ref", "$defs", "additionalProperties"}


def _keys(node, found):
    if isinstance(node, dict):
        for k, v in node.items():
            if k != "properties":
                found.add(k)
            _keys(v, found) if k != "properties" else [_keys(x, found) for x in v.values()]
    elif isinstance(node, list):
        for x in node:
            _keys(x, found)
    return found


def test_llm_schema_uses_only_keywords_gemini_supports():
    schema = ParsedRequirement.model_json_schema()
    assert set(schema["required"]) == set(schema["properties"])  # every field must be answered (maybe null)
    assert _keys(schema, set()) <= SUPPORTED_KEYS


# ---------------------------------------------------------------- parse
async def test_parse_cleans_up_model_output_in_code():
    fake = FakeClient(parsed(item="waterproof plywood", specs="18 mm", quantity=200, unit="sheets",
                             delivery_location="Udaipur", deadline_date="2026-10-13", category="Panel boards"))
    llm = GeminiLLM("test-model", client=fake, today=TODAY)
    req, res = await llm.parse_requirement("200 sheets of 18 mm waterproof ply to Udaipur by 13 Oct")

    assert req.deadline_days == 20  # date -> days computed in code, not by the model
    assert req.delivery_state == "Rajasthan"  # lookup table, not the model
    assert req.category == "Panel boards" and req.missing_fields() == []
    assert res.source == "gemini:test-model" and res.error is None
    assert (res.input_tokens, res.output_tokens) == (120, 40)  # thinking tokens count as output
    call = fake.calls[0]
    assert call["model"] == "test-model" and schema_title(call) == "ParsedRequirement"
    assert call["config"].response_mime_type == "application/json"
    assert "2026-09-23" in call["config"].system_instruction


async def test_parse_drops_nonsense_values():
    fake = FakeClient(parsed(item="foam", quantity=-5, deadline_date="not-a-date", unit="kg"))
    req, _ = await GeminiLLM("m", client=fake, today=TODAY).parse_requirement("foam")
    assert req.quantity is None and req.deadline_days is None and req.unit == "kg"


async def test_past_deadline_becomes_missing():
    req = to_requirement(parsed(item="x", deadline_date="2026-09-01"), TODAY)
    assert req.deadline_days is None


class _BadJSON:
    def model_dump_json(self):
        return '{"item": "x"}'  # missing required fields -> Pydantic rejects it


@pytest.mark.parametrize("reply", [RuntimeError("503 UNAVAILABLE"), None, _BadJSON()])
async def test_parse_falls_back_to_rules(reply):
    llm = GeminiLLM("m", client=FakeClient(reply), today=TODAY)
    req, res = await llm.parse_requirement("200 sheets of plywood to Jodhpur within 2 weeks")
    assert res.source == "rules(fallback)" and res.error
    assert req.quantity == 200 and req.deadline_days == 14


# ---------------------------------------------------------------- selection
def test_provider_selection():
    assert isinstance(get_llm(Settings(gemini_api_key=None)), RuleBasedLLM)
    assert isinstance(get_embedder(Settings(gemini_api_key=None)), HashEmbedder)
    with pytest.raises(RuntimeError, match="GEMINI_MODEL"):
        get_llm(Settings(gemini_api_key="test-key", gemini_model=None))
    s = Settings(gemini_api_key="test-key", gemini_model="some-flash-model")
    assert get_llm(s).name == "gemini:some-flash-model"
    assert get_embedder(s).name == "gemini:gemini-embedding-2:1536"


# ---------------------------------------------------------------- whole graph with a mocked model
async def test_graph_with_llm_free_text_clarify_and_category(fake_deps):
    def reply(kw):
        if kw["contents"].startswith("Need"):
            return parsed(item="waterproof plywood", specs="18 mm", category="Panel boards")
        return parsed(quantity=200, unit="sheets", delivery_location="Udaipur", deadline_days=20,
                      item="ignored because item is already known")

    seen = {}

    async def retrieve(req):
        seen.update(req)
        return await original(req)

    original, fake_deps.retrieve = fake_deps.retrieve, retrieve
    fake_deps.llm = GeminiLLM("m", client=FakeClient(reply), today=TODAY)
    graph = build_graph(fake_deps).compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "llm-1"}}

    await graph.ainvoke({"request_id": "55555555-5555-5555-5555-555555555555", "log": [],
                         "raw_text": "Need waterproof plywood, 18 mm"}, cfg)
    await graph.ainvoke(Command(resume={"text": "200 sheets, Udaipur, within 20 days"}), cfg)

    assert seen["category"] == "Panel boards" and seen["item"] == "waterproof plywood"
    assert seen["quantity"] == 200 and seen["delivery_state"] == "Rajasthan"
    snap = await graph.aget_state(cfg)
    report = snap.values["report"]
    assert [v["vendor_id"] for v in report["vendors"]] == ["V-1", "V-4"]
    llm = report["summary"]["llm"]
    assert llm["calls"] == 2 and llm["input_tokens"] == 240 and llm["fallbacks"] == 0  # parse + clarify


# ---------------------------------------------------------------- the real SDK, fake HTTP
def _real_client(handler):
    """The actual google-genai client; only the HTTP layer is mocked. Catches SDK or schema drift."""
    return make_client("test-key", retry_attempts=1,
                       httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_real_sdk_generate_sends_json_schema_and_parses_reply():
    sent = {}
    reply = {"item": "waterproof plywood", "specs": "18 mm", "quantity": 200, "unit": "sheets",
             "delivery_location": "Udaipur", "deadline_days": 20, "deadline_date": None,
             "category": "Panel boards", "notes": None}

    def handler(request):
        sent["url"], sent["body"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": json.dumps(reply)}], "role": "model"},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 30, "totalTokenCount": 80}})

    req, res = await GeminiLLM("test-model", client=_real_client(handler), today=TODAY).parse_requirement("200 sheets")

    assert sent["url"].endswith("/models/test-model:generateContent")
    gen = sent["body"]["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert gen["responseJsonSchema"]["title"] == "ParsedRequirement"
    assert "Rohida" in sent["body"]["systemInstruction"]["parts"][0]["text"]
    assert "tools" not in sent["body"]
    assert req.quantity == 200 and req.category == "Panel boards" and req.delivery_state == "Rajasthan"
    assert (res.input_tokens, res.output_tokens, res.error) == (50, 30, None)


async def test_real_sdk_embeddings_one_vector_per_text():
    sent = {}

    def handler(request):
        sent["url"], sent["body"] = str(request.url), json.loads(request.content)
        n = len(sent["body"]["requests"])
        return httpx.Response(200, json={"embeddings": [{"values": [3.0, 4.0] + [0.0] * 1534}] * n})

    emb = GeminiEmbedder("gemini-embedding-2", 1536, client=_real_client(handler))
    vectors = await emb.embed(["waterproof plywood", "PU foam"], kind="query")

    assert sent["url"].endswith("/models/gemini-embedding-2:batchEmbedContents")
    reqs = sent["body"]["requests"]
    assert len(reqs) == 2  # one request per text, not one merged multi-part content
    assert reqs[0]["content"]["parts"][0]["text"] == "task: search result | query: waterproof plywood"
    assert reqs[0]["outputDimensionality"] == 1536
    assert len(vectors) == 2 and vectors[0][:2] == [0.6, 0.8]  # L2-normalised in code

    await emb.embed(["some vendor specs"], kind="document")
    assert sent["body"]["requests"][0]["content"]["parts"][0]["text"] == "title: none | text: some vendor specs"


async def test_embedder_rejects_wrong_dimensions():
    def handler(request):
        return httpx.Response(200, json={"embeddings": [{"values": [1.0] * 768}]})

    with pytest.raises(ValueError, match="1536 dims"):
        await GeminiEmbedder("gemini-embedding-2", 1536, client=_real_client(handler)).embed(["x"])


def test_prompts_do_not_leak_the_evaluation_set():
    """Examples in prompts/schemas must not come from eval/seed_requests.json, or the eval would flatter us."""
    import json as _json
    from pathlib import Path

    from app.services import judgments, llm

    texts = [llm.PARSE_INSTRUCTIONS, _json.dumps(ParsedRequirement.model_json_schema()),
             _json.dumps(judgments.SPEC_CRITERIA),
             _json.dumps(judgments.FIT_CRITERIA), judgments.QUOTABLE_INSTRUCTIONS, _json.dumps(judgments.QUOTABLE_CRITERIA)]
    blob = " ".join(texts).lower()
    requests = _json.loads(Path("eval/seed_requests.json").read_text())
    vague = [r["text"].lower().rstrip(".") for r in requests if r["kind"] == "vague"]
    assert not [v for v in vague if v in blob]
    for word in ("sheesham", "plywood", "sissoo", "hinges", "good wood", "tarpaulin"):
        assert word not in blob, word


async def test_missing_unit_is_filled_from_the_text_when_quantities_agree():
    fake = FakeClient(parsed(item="stretch wrap film", specs="23 micron", quantity=600, unit=None))
    req, _ = await GeminiLLM("m", client=fake, today=TODAY).parse_requirement("600 rolls of 23 micron stretch wrap")
    assert req.unit == "pcs"
    fake = FakeClient(parsed(item="film", quantity=5, unit=None))  # rules read 600, model says 5: don't guess
    req, _ = await GeminiLLM("m", client=fake, today=TODAY).parse_requirement("600 rolls of film")
    assert req.unit is None
