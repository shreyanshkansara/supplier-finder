"""Parser + completeness evaluation over the 40 seed requests.

    python -m eval.run_parse_eval

Does the parse step extract the right fields, and does the completeness step send back the
right requests? Uses the app's own parser (Gemini if configured, else rules) and, when
TYPESAFE_API_KEY is set, the Jev "specific enough to quote?" check with the app's threshold.

Metrics:
  field accuracy     quantity / deadline_days / category / delivery_state on non-vague requests
  vague caught       vague requests that the app would send back for clarification
  false clarify      specific requests the app would bother the buyer about unnecessarily
                     (a missing deadline counts as correct when the request really has none)
No database needed. Writes eval/results/parse_latest.{json,md}.
"""

import asyncio
import json
import time

from app.agent.nodes import find_missing
from app.agent.runtime import policy_from
from app.config import get_settings
from app.services.judgments import check_quotable
from app.services.llm import get_llm
from eval.pipeline import RESULTS_DIR, ParseCache, load_requests

FIELDS = ("quantity", "deadline_days", "category", "delivery_state")


def expected(r: dict) -> dict:
    return {"quantity": r.get("qty"), "deadline_days": r.get("deadline_days"),
            "category": r.get("category"), "delivery_state": r.get("delivery_state")}


def field_ok(f: str, got, want) -> bool:
    if f == "quantity":
        return got is not None and want is not None and float(got) == float(want)
    return got == want


async def missing_for(req, settings) -> tuple[list[str], float | None, list[dict]]:
    """Exactly the completeness node's logic (find_missing), with the real Jev check when configured."""

    async def quotable(r):
        return await check_quotable(r, settings=settings)

    return await find_missing(req.model_dump(), quotable, policy_from(settings))


async def run(use_cache: bool = True) -> dict:
    settings = get_settings()
    llm = get_llm()
    parse_cache = ParseCache(enabled=use_cache)
    requests = load_requests()
    specific = [r for r in requests if r["kind"] != "vague"]
    vague = [r for r in requests if r["kind"] == "vague"]
    hits = {f: 0 for f in FIELDS}
    rows, tokens, fallbacks, t0 = [], 0, 0, time.perf_counter()

    async def parse(text):
        nonlocal tokens, fallbacks
        req, res = await parse_cache.parse(llm, text)
        missing, p, usage = await missing_for(req, settings)
        for u in [res.usage("parse"), *usage]:
            tokens += u["input_tokens"] + u["output_tokens"]
            fallbacks += bool(u["error"])
        return req, missing, p

    false_clarify = 0
    for r in specific:
        req, missing, p = await parse(r["text"])
        got, exp = req.model_dump(), expected(r)
        ok = {f: field_ok(f, got[f], exp[f]) for f in FIELDS}
        for f in FIELDS:
            hits[f] += ok[f]
        allowed = {"deadline_days"} if r.get("deadline_days") is None else set()
        unneeded = sorted(set(missing) - allowed)
        false_clarify += bool(unneeded)
        rows.append({"id": r["id"], "misses": {f: {"got": got[f], "want": exp[f]} for f in FIELDS if not ok[f]},
                     "asked_unnecessarily": unneeded, "p_quotable": p})

    caught = []
    for r in vague:
        req, missing, p = await parse(r["text"])
        caught.append({"id": r["id"], "caught": bool(missing), "missing": missing, "p_quotable": p})

    n = len(specific)
    return {
        "parser": llm.name, "jev": "on" if settings.typesafe_api_key else "off",
        "field_accuracy": {f: round(hits[f] / n, 3) for f in FIELDS},
        "vague_caught": f"{sum(c['caught'] for c in caught)}/{len(vague)}",
        "false_clarify": f"{false_clarify}/{n}",
        "tokens": tokens, "fallbacks": fallbacks, "seconds": round(time.perf_counter() - t0, 1),
        "rows": rows, "vague": caught,
    }


def to_markdown(res: dict) -> str:
    lines = [f"Parser `{res['parser']}` · Jev quotability check {res['jev']}", "",
             "| Field | accuracy |", "|---|---|"]
    lines += [f"| {f} | {v:.2f} |" for f, v in res["field_accuracy"].items()]
    lines += ["", f"Vague requests caught: {res['vague_caught']} · specific requests asked unnecessarily: "
                  f"{res['false_clarify']} · tokens: {res['tokens']} · fallbacks: {res['fallbacks']}"]
    misses = [r for r in res["rows"] if r["misses"] or r["asked_unnecessarily"]]
    if misses:
        lines += ["", "Misses:"]
        for r in misses:
            parts = [f"{f} got {m['got']!r} want {m['want']!r}" for f, m in r["misses"].items()]
            if r["asked_unnecessarily"]:
                parts.append(f"asked for {r['asked_unnecessarily']}")
            lines.append(f"- {r['id']}: " + "; ".join(parts))
    return "\n".join(lines) + "\n"


def main():
    res = asyncio.run(run())
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "parse_latest.json").write_text(json.dumps(res, indent=2, default=str))
    md = to_markdown(res)
    (RESULTS_DIR / "parse_latest.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
