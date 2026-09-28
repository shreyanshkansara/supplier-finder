# 2-minute demo script

Record your screen with the web page open (locally or on Render). Wake a sleeping Render app a minute before you start. Aim for about 2 minutes: one sentence per step, and let the screen do the talking.

| Time | Do | Say |
|---|---|---|
| 0:00 | Page open, empty | "A furniture maker in Jodhpur needs suppliers. Today that's calls, IndiaMART searches and WhatsApp. This finds the vendors who can actually supply it." |
| 0:10 | Type **"We need fabric for the sofas in the new hotel project"** and click Find suppliers | "A vague request, like people really write them." |
| 0:20 | The clarification form appears | "Gemini read it into typed fields and found the gaps. Jev, a small judgment model, decided 'fabric for the sofas' doesn't say *which* product, so it asks. The run is paused and saved in Postgres, so it survives a restart." |
| 0:35 | Type **"flame-retardant upholstery fabric, 600 metres, deliver to Jaipur within 4 weeks"** in the free-text box and click Continue | "I can answer in plain words." |
| 0:45 | The vendor list appears | "These vendors can supply it, ranked. Search is SQL filters plus pgvector. Minimum order and lead time are checked in code, not by a model. Jev checks every listing against the specs in one call." |
| 1:00 | Open **Why it matched** on #1, point at the contact details and the *Check listing* label on #8 | "Each vendor shows its contacts, terms and evidence. Jev is less sure about this one, so it's flagged for a human to check." |
| 1:10 | Scroll to **Near misses** | "The near misses say why they were ruled out: lead time too long, minimum order too high, or the listing doesn't meet the spec." |
| 1:25 | Click the **One nobody can supply** example (30 mm plywood) and Find suppliers | "Nobody stocks 30 mm. The right answer is 'no vendor fits', not five wrong ones." |
| 1:40 | Show the README results table (or `eval/results/latest.md`) | "I measured it on 40 labelled requests. Plain vector search puts 54% of the correct vendors in the top 5; the full pipeline gets 97%, and it says 'nobody fits' correctly in 4 of 4 impossible requests, where code checks alone manage 1. The thresholds were tuned on that set." |
| 1:55 | End on the page | "FastAPI, Postgres/pgvector, LangGraph, Gemini and Jev, with 70 tests in CI. The code is on GitHub." |

Tips:
- Run the demo once before recording. That also warms the database, and Gemini's pacing means the first request can take 10–20 seconds.
- Make the browser window about 1280 px wide so the list and the form fit.
- If you show the API, use Swagger at `/docs`: `POST /requests` then `GET /requests/{id}/report`.
