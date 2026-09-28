Run 2026-09-22T22:17:57+00:00 · embedder `gemini:gemini-embedding-2:1536` · parser `gemini:gemini-3.5-flash-lite` · Jev `jev-latest`
28 requests with gold vendors, 4 with none.

| Config | precision@5 | no-match |
|---|---|---|
| vector_only | 0.536 | 0.00 |
| sql_vector | 0.637 | 0.00 |
| sql_vector_code | 0.854 | 0.25 |
| parsed_pipeline | 0.932 | 0.25 |
| full | 0.971 | 1.00 |

Full pipeline: auto-shortlist precision 0.983, review load 0.025, Jev failures 0.

Per request (parsed/full pipeline): 5661.219 model tokens, 3495.406 ms. Fallbacks: 0.
