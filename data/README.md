# Vendor dataset: Rohida Furniture Works (synthetic)

500 made-up suppliers for a fictional furniture maker in Jodhpur. None of these vendors exist. Every contact email is a plus-address on a test inbox (`rfq.test+v0001@example.com` by default), and every phone number is deliberately invalid (`+91-00000-…`), so nothing can ever reach a real business.

## Regenerate

```bash
python data/generate_vendors.py --n 500 --seed 42 --email-base you.test.inbox@gmail.com --out data --eval-out eval
```

The same seed always produces identical files. Set `--email-base` to an inbox you own before you try live sending.

## Files

| File | Load into the app? | What it is |
|---|---|---|
| `vendors.csv` / `vendors.json` | **Yes** | The vendor table: id, name, city, state, regions served, categories, free-text specs, MOQ + unit, lead time, price band, indicative price, payment terms, GST flag, rating notes, contact. In the CSV, list fields are separated by ` \| `. |
| `eval/vendor_truth.jsonl` | **No** | Structured attributes behind each vendor's free text (species, seasoning, thickness, density, soft-close…). Used only to compute gold labels. |
| `eval/hard_cases.csv` | **No** | Which vendors were planted as hard cases, what they target and why they should fail. |
| `eval/seed_requests.json` | **No** (eval only) | 12 seed sourcing requests with their gold matching vendor ids. |

Keep the three evaluation files out of the database and out of every prompt. If the agent can see them, the evaluation is meaningless.

## What's inside

| Planted case | Count | Purpose |
|---|---|---|
| `synonym` | 14 | "Dalbergia sissoo" instead of sheesham, "chipboard" instead of particle board. These should still match. |
| `near_miss` | 22 | Every constraint passes but the spec is slightly wrong: air-dried timber, 32-density foam, no 18 mm, not soft-close, not flame-retardant. |
| `wrong_constraint:*` | 15 | A perfect product that fails on MOQ, lead time, or delivery region. Code checks should catch these. |
| `keyword_trap` | 8 | The right words for the wrong thing: sheesham-look laminate, waterproof tarpaulins, fire-safety blankets. |
| `vague` | 8 | "All types of wood available." |
| `duplicate` | 6 | The same business listed twice under a slightly different name. |

That's 73 of the 500 vendors (15%). Every clear request also gets a few guaranteed genuine matches: 5 each for R01–R07 and 3 each for R13–R31. Together with the matches that occur naturally in the data, every clear request has 3–11 correct vendors.

Gold matching rule: a vendor sells a product that meets the spec, its MOQ ≤ the requested quantity, its lead time ≤ the deadline, and it delivers to Rajasthan (or Pan-India).

There are 40 seed requests: 26 clear, 8 vague and 6 tricky.

| Request | Type | Gold vendors |
|---|---|---|
| R01–R07, R13–R31 | Clear, across all 12 categories | 3–11 each |
| R08–R10, R32–R36 | Vague | 0. The agent should ask clarifying questions |
| R37, R38 | Tricky (synonyms: "Dalbergia sissoo", "fire-retardant") | 6–7 |
| R11 | Tricky | 0. All acacia stock is air-dried |
| R12 | Tricky | 0. Waterproof plywood always needs 7+ days |
| R39 | Tricky | 0. Nobody stocks 30 mm waterproof plywood |
| R40 | Tricky | 0. Nobody delivers memory foam in 2 days |

Each request records its category and delivery state, so the evaluation can compare hand-labelled filters with what the parser extracts. The gold ids are recomputed on every run. To add requests, add them to `SEED_REQUESTS` with a `match` rule (and to `CAT_OF`/`NEW_PERFECT` if they need planted matches).
