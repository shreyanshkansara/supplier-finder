# Sourcing Agent

A backend service that helps a furniture manufacturer find suppliers. A buyer types what they need. The agent asks about anything missing, then returns **the list of vendors that can supply it**. Each vendor comes with contact details, terms, and why it matched. The list also shows the near misses and why each was ruled out. Contacting the vendors stays with the buyer.

The demo customer is **Rohida Furniture Works**, a fictional manufacturer in Jodhpur. All 500 vendors are synthetic (see [`data/`](data/)), and their contact details are placeholders.

> Design rule: **the LLM writes text, Jev makes judgments, and code enforces rules.**

![Vendor list for a hotel fabric request](docs/screenshots/2-results.png)

More screenshots: [clarifying questions](docs/screenshots/1-clarify.png), [no vendor fits](docs/screenshots/3-no-match.png), [mobile](docs/screenshots/4-mobile.png). Deploying: [docs/DEPLOY.md](docs/DEPLOY.md). Demo script: [docs/DEMO.md](docs/DEMO.md).

## Architecture

```
Web page (/) · Swagger (/docs)
        │
FastAPI ──► LangGraph workflow (state saved in Postgres)
        │     parse → completeness ─┬─► clarify (PAUSE) ─► completeness
        │                           └─► retrieve (SQL filters + pgvector)
        │     → verify (code checks + Jev) → rank → report (vendor list + near misses)
        │
Postgres + pgvector: vendors · sourcing_requests · audit_log · LangGraph checkpoints
```

- **Paused runs survive restarts.** Each request is a LangGraph thread whose `thread_id` is the request id. The Postgres checkpointer stores its state, so a request waiting for the buyer's answers can be resumed after the server restarts.
- **Hard constraints live in SQL or code, never in a model.** Category and delivery region are `WHERE` clauses, using GIN indexes on the array columns. MOQ and lead time are Python checks.
- **Results show their reasoning.** Every listed vendor carries its code checks and Jev's probabilities, and every near miss carries the reason it was ruled out.
- **Everything is audited.** Every agent decision and every answer from the buyer is written to `audit_log`.

## Quick start (Docker)

```bash
cp .env.example .env              # set API_KEY (and the model keys, if you have them)
docker compose up --build -d      # Postgres+pgvector + the API; migrates, then seeds the 500 vendors in the background
open http://localhost:8000        # the web page (Swagger is at /docs)
```

The page asks once for the `API_KEY`. Seeding takes a few seconds: the Gemini vectors for the demo vendors ship in `data/embeddings`, so it makes no API calls. `docker compose logs -f api` shows `seeded 500 vendors` when it's done.

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
docker compose up -d db
alembic upgrade head
python -m scripts.seed_vendors
uvicorn app.main:app --reload
```

## Adding your Gemini API key

The app runs without any keys. It then uses a rule-based parser, an offline hashing embedder and code-only checks. That's free and deterministic, which is why tests and CI use it. To switch on Gemini:

1. Create a key in [Google AI Studio](https://aistudio.google.com/apikey). In `.env`, set `GEMINI_API_KEY`, and set `GEMINI_MODEL` to a Flash or Flash-Lite model id copied from AI Studio. If the key is set without a model, the app refuses to start rather than guessing.
2. Run `python -m scripts.seed_vendors` again. Vendors get re-embedded with `gemini-embedding-2` at 1536 dimensions (4 batched calls for 400 vendors). Retrieval never mixes vectors from two embedders.
3. Run `python -m eval.run_parse_eval` and `python -m eval.run_eval`. Put the numbers in the Evaluation section below.
4. Set `TYPESAFE_API_KEY` to turn on Jev: spec and fit judgments in `verify`, and the "specific enough to quote?" check in `completeness`. Then tune the thresholds (see Evaluation).

**Free tier vs paid.** The free tier is enough to build and run evaluations. It has two catches:
- **Google may use your data.** Free-tier prompts may be used to improve Google's products. That's fine for the synthetic vendors, but switch to the paid tier before a real business's supplier data goes through the tool. A Google AI Pro subscription includes $10/month of Google Cloud credit through the Google Developer Program, which covers this project's paid usage many times over.
- **Rate limits.** One request makes 1–2 Gemini calls (the parse, plus one per plain-text clarification) and 1–2 Jev calls. Calls are paced to `GEMINI_RPM` per minute, and after a 429 the app waits as long as the API asks.

If a Gemini call still fails, it falls back to the rule-based parser. That covers a rate limit, a network error, a blocked response, or invalid JSON. The run keeps going, and the fallback is recorded in the audit log and the report (`summary.llm.fallbacks`). **Check that count after an evaluation run:** fallbacks mean you measured the rules, not Gemini.

## Using the API

Every endpoint except `/health` needs the `X-API-Key` header.

| Endpoint | What it does |
|---|---|
| `POST /requests` | Starts a run. Returns its id and status: `clarifying` (questions in `pending`) or `completed` (vendor list in `report`) |
| `GET /requests` | Recent requests, newest first (used by the page's history list) |
| `GET /requests/{id}` | Current status, plus `pending` (the questions) or `report` (the result) |
| `POST /requests/{id}/clarify` | Resumes a run. Answer with fields, `{"answers": {"quantity": 200, "deadline_days": 20}}`, or with plain text, `{"text": "200 sheets to Udaipur within 20 days"}`. Plain text only fills fields that are still missing |
| `GET /requests/{id}/report` | The result: `vendors`, ranked, each with contact details, MOQ, lead time, price, `confidence` (`shortlist` or `needs_review`) and the checks it passed. Also `near_misses` with the reason each was ruled out, and a one-line `message` |
| `GET /requests/{id}/audit` | The audit trail |
| `GET /vendors` | Filterable (`category`, `state`, `serves`, `q`) and paginated |
| `POST /vendors/import` | CSV upload in the same format as `data/vendors.csv`. Upserts by id and embeds |

```bash
H='X-API-Key: change-me'
curl -s -X POST localhost:8000/requests -H "$H" -H 'content-type: application/json' \
  -d '{"text": "Need 300 cft kiln-dried sheesham, 1.5 inch, moisture under 12%"}'
# -> status "clarifying", pending.missing = ["delivery_location", "deadline_days"]
curl -s -X POST localhost:8000/requests/<id>/clarify -H "$H" -H 'content-type: application/json' \
  -d '{"answers": {"delivery_location": "Jodhpur", "deadline_days": 21}}'
# -> status "completed", report.vendors = the vendors that can supply it, report.near_misses = the closest others
```

Errors always have the same shape: `{"error": {"code", "message", "request_id", "details"}}`. Every response carries an `X-Request-ID` header, which also appears in the logs.

## What's built and what's next

| Piece | Status |
|---|---|
| Schema, migrations, vendor import, hybrid retrieval | ✅ |
| API, auth, error format, request-id logging, audit log | ✅ |
| Graph with a clarification pause and the Postgres checkpointer | ✅ |
| Code checks (MOQ, lead time, units), ranking, confidence routing | ✅ |
| Result report: ranked vendor list with contact details and match evidence, plus near misses with reasons | ✅ |
| `parse`: Gemini JSON-schema output into `SourcingRequirement`. Code turns dates into days, normalises units and looks up the state; the LLM also picks the vendor category | ✅ Week 2 |
| Plain-text answers to clarifying questions | ✅ Week 2 |
| Token, latency and fallback tracking per LLM call (audit log + `report.summary.llm`) | ✅ Week 2 |
| Jev spec (Noul) and fit (Score) judgments for every candidate that passes the code checks, one request per 25 vendors, with described criteria for yes/no and each fit level | ✅ Week 3 |
| Jev "specific enough to quote?" check in `completeness`: "200 sheets of some plywood" goes back to the buyer even though no field is empty | ✅ Week 3 |
| `RankPolicy`: weights and thresholds come from settings, and the eval uses the same `rank_candidates` the app uses | ✅ Week 3 |
| If Jev fails, all of its judgments for that request are discarded and ranking uses code checks only, so judged and unjudged vendors are never mixed. The failure is logged in the audit trail and the report | ✅ Week 3 |
| Eval: 40 seed requests (26 clear, 8 vague, 6 tricky), a `full` config, no-match accuracy, tokens and latency, results written to `eval/results/` | ✅ Week 3 |
| Threshold tuner with cached Jev judgments (`eval/tune_thresholds.py`) | ✅ Week 3 (tuned: review 0.25, shortlist 0.35) |
| Test coverage in CI | ✅ Week 3 (97%) |
| Real-model evaluation, threshold tuning, free-tier pacing | ✅ Week 3 (see Results) |
| RFQ drafting and email sending | Removed by design: the product ends at the vendor list |
| Web page at `/`: request box, clarifying questions (fields or plain text), ranked vendor cards with contacts and evidence, near misses, request history. Plain HTML/CSS/JS with no build step, light and dark mode, works on mobile, all data inserted as text | ✅ Week 4 |
| Deployment: `render.yaml` for Render plus Neon Postgres, hosted DB URLs handled automatically (`sslmode`, `channel_binding`), `$PORT`-aware image, background seeding on first start | ✅ Week 4 (guide in [docs/DEPLOY.md](docs/DEPLOY.md)) |
| Demo video | Script in [docs/DEMO.md](docs/DEMO.md); you record it |

## Evaluation

```bash
python -m eval                         # both evaluations below
python -m eval.run_parse_eval          # parser + completeness
python -m eval.run_eval                # retrieval + ranking (needs the DB seeded)
python -m eval.tune_thresholds         # pick JEV_REVIEW_AT / JEV_SHORTLIST_AT (needs TYPESAFE_API_KEY)
```

The set has 40 requests in `eval/seed_requests.json`, generated with the vendor data: 26 clear, 8 vague and 6 tricky. The tricky ones are synonyms ("Dalbergia sissoo", "fire-retardant") and requests where no vendor fits (kiln-dried acacia, 30 mm plywood, 2-day delivery). **Never load `eval/` files into the app or a prompt.** Results are written to `eval/results/latest.{md,json}` and `parse_latest.{md,json}`.

`run_eval` adds one layer per configuration. `vector_only` is raw text into pgvector. `sql_vector` adds the category and state filters. `sql_vector_code` adds the MOQ and lead-time checks. The first three use hand-labelled category, state, quantity and deadline. `parsed_pipeline` is the real path: the app's own parser reads the raw text, and what it extracts drives the search and checks. `full` adds Jev and the `RankPolicy`, so it is the app end to end.

The metrics:
- **precision@5**: hits in the top 5, divided by min(5, number of correct vendors). A request with only 3 correct vendors can still score 1.0.
- **no-match**: for requests where nobody fits, the share where the list comes back empty. The right answer there is "no vendor fits", not five wrong vendors.
- **Full config only**: confident-match precision (share of vendors marked `shortlist` that are correct) and the share marked `needs_review`.
- **Cost**: model tokens and latency per request.

Jev judgments are cached in `eval/results/jev_cache.json`, so re-runs and threshold tuning cost nothing extra.

### Results

Measured on 23 September 2026 with `gemini-3.5-flash-lite` (request parser), `gemini-embedding-2` (1536 dimensions), and Jev (`jev-latest`), using the tuned thresholds. The run had no caches and no fallbacks. The full output is in `eval/results/`.

| Config | precision@5 | no-match |
|---|---|---|
| vector_only | 0.536 | 0.00 |
| sql_vector | 0.637 | 0.00 |
| sql_vector_code | 0.854 | 0.25 |
| parsed_pipeline (Gemini parser, no Jev) | 0.932 | 0.25 |
| **full** (Gemini + Jev + tuned policy) | **0.971** | **1.00** |

- **Retrieval:** the full pipeline finds 97% of the correct vendors that could fit in the top 5, up from 54% for plain vector search. Those vendors are ones with the right product and specs, an MOQ that fits, the lead time and the delivery region.
- **No-match:** in all 4 requests where nobody fits (kiln-dried acacia, 30 mm plywood, 2-day delivery, a 5-day plywood deadline), the full pipeline returns an empty list. Code checks alone catch only 1 of the 4.
- **Confidence labels:** 98.3% of vendors marked as confident matches are correct. 2–3% of listed vendors are marked `needs_review`, meaning the buyer should check that listing first.
- **Cost:** a median of about 5,800 model tokens per request (parse plus one Jev call over up to 25 vendors). Median latency is 1.9 s, and the mean is 3.5 s because of free-tier pacing.
- **Parser and completeness:** quantity, deadline, category and state accuracy are all 1.00 on 32 specific requests. All 8 vague requests are sent back for clarification, and no specific request is asked for details it already gave (0 of 32).

The thresholds come from `eval/tune_thresholds.py`: `JEV_REVIEW_AT=0.25` and `JEV_SHORTLIST_AT=0.35`. At those values 96.7% of genuinely matching vendors are kept, and at least 98% of confident matches are correct.

**Caveat:** the vendors and requests are synthetic and were generated together. The prompts and Jev criteria were written without any examples from the set, and a test enforces this. Even so, 40 requests is a small sample, so read these as evidence the design works, not as production accuracy.

**Without any keys** (offline hash embedder, rules parser): 0.465 / 0.588 / 0.842 / 0.789 precision@5 for the first four configs, with no-match stuck at 0.25.

### What changed while measuring

The first real run found problems the mocked tests couldn't:

- **Jev "specific enough to quote?" check:** the first question sent back 30 of 32 perfectly clear requests. It asked whether a supplier could give a *firm price*, which Jev rightly judged was almost never possible. The question was narrowed to "does it say *which* product?". That separated specific requests (0.74 or higher) from vague ones (0.35 or lower).
- **Specs field:** Gemini often puts the whole spec into `item` ("memory foam", "natural teak veneer"). `specs` is therefore no longer a required field; Jev decides on item and specs together.
- **Free-tier rate limit:** Flash-Lite allows 15 requests per minute. Retrying 429s burned that quota, so calls are now paced client-side and honour the API's own retry delay. A daily-quota error fails fast to the fallback instead of waiting.

## Resume bullets (from the numbers above)

- Built an async FastAPI + Postgres/pgvector supplier-discovery service that combines SQL filters with vector search, with a web front end and 70 pytest tests (96% coverage) in GitHub Actions CI and Docker Compose.
- Designed resumable API endpoints over a LangGraph workflow with a Postgres checkpointer, so runs paused for clarification survive server restarts.
- Built a LangGraph agent pipeline (parse, clarify, retrieve, verify, rank) using Gemini with Pydantic-validated structured output. Vague requests go back to the buyer: 8 of 8 caught, with 0 of 32 clear requests asked unnecessarily.
- Used TypeSafe Jev for calibrated spec and fit checks across all candidates in one call. This raised precision@5 from 0.54 (vector-only) to 0.97, and made the system return "no vendor fits" correctly in 4 of 4 impossible requests (vs 1 of 4 with code checks).
- Tuned confidence thresholds on a labelled 40-request set: 98% of vendors labelled as confident matches are correct, and about 2% are flagged for the buyer to check.

## Tests

```bash
pytest --cov                               # unit, graph, mocked-LLM/Jev and eval tests, no database needed
TEST_DATABASE_URL=postgresql+asyncpg://sourcing:sourcing@localhost:5432/sourcing_test pytest  # + API/DB tests
```

The model tests never call Gemini or TypeSafe. The Jev tests use the real TypeSafe SDK with only its HTTP transport mocked. They check the question and criteria payload, batching, the all-or-nothing failure rule, and quotability. The LLM tests never call Gemini. Most use a fake client. Three go through the real `google-genai` SDK with only the HTTP layer mocked. They check that requests carry the JSON schema, that replies parse, and that embeddings come back as one vector per text. The DB tests create the schema with Alembic, import the vendor CSV, and drive a request through clarification to the vendor list over HTTP. The eval harness is tested too: the metrics, the Jev cache, threshold selection, and an end-to-end run on the seed data. CI (GitHub Actions) runs ruff and the full suite, with coverage, against a `pgvector/pgvector:pg16` service. The page itself was checked in a headless browser: the clarify-to-results flow, light and dark mode, and a 390 px phone width with no sideways scrolling. 70 tests, 96% coverage.

## Layout

```
app/
  main.py            FastAPI app, lifespan (checkpointer, compiled graph, optional seeding), health checks
  static/            the web page (index.html, app.css, app.js)
  config.py          settings from env
  models.py          SQLAlchemy models        migrations/   Alembic
  schemas.py         Pydantic models (SourcingRequirement, API I/O)
  api/               vendors, requests routers
  agent/             graph.py (edges), nodes.py (the steps), runtime.py (run/resume, deps)
  catalog.py         fixed vocabularies: vendor categories, units
  services/          llm (Gemini + rules fallback), embeddings (Gemini + offline), retrieval, vendor_import,
                     judgments (Jev), ratelimit, audit, geo
scripts/seed_vendors.py
data/                vendors.csv/json + generate_vendors.py (seeded, reproducible)
eval/                seed requests + gold labels, pipeline.py (shared), run_eval.py, run_parse_eval.py,
                     tune_thresholds.py, results/
tests/
docs/                DEPLOY.md, DEMO.md, screenshots/
render.yaml          Render blueprint (free web service; database on Neon)
```
