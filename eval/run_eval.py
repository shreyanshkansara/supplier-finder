"""Retrieval + ranking evaluation over the 40 seed requests.

    python -m eval.run_eval [--no-cache]

Configurations (each adds one layer to the previous):
  vector_only      pgvector similarity over all vendors, raw request text
  sql_vector       + hand-labelled category and delivery-state filters in SQL
  sql_vector_code  + MOQ / lead-time code checks (hand-labelled quantity and deadline)
  parsed_pipeline  the real path without Jev: the app's parser (Gemini or rules) reads the raw text,
                   and ITS item/specs/category/state/quantity/deadline drive search and checks
  full             parsed_pipeline + Jev spec/fit judgments + the RankPolicy (the app, end to end)

Metrics:
  precision@5      over requests that have gold vendors: hits in the top 5 / min(5, #gold)
  no-match         over tricky requests with NO correct vendor: share where the list comes back empty
                   (the right answer is "nobody fits", not five wrong vendors)
  full only        auto-shortlist precision (gold share of vendors flagged "shortlist"),
                   review load (share of the top 5 flagged "needs_review")
  cost/latency     model tokens and wall-clock per request for the parsed/full pipelines

Writes eval/results/latest.json and eval/results/latest.md. Gold labels come from
eval/seed_requests.json; never load eval files into the app or a prompt.
"""

import argparse
import asyncio
import json
import statistics
import time
from datetime import UTC, datetime

from app.agent.nodes import code_checks, rank_candidates
from app.agent.runtime import policy_from
from app.config import get_settings
from app.db import dispose_engine, get_sessionmaker
from app.services.embeddings import get_embedder
from app.services.llm import get_llm
from app.services.retrieval import hybrid_search
from eval.pipeline import (
    RESULTS_DIR,
    TOP_K,
    JevCache,
    K,
    ParseCache,
    fallbacks,
    judge_cached,
    load_requests,
    precision_at_k,
    prepare,
    survives,
    tokens,
    with_judgments,
)

CONFIGS = ("vector_only", "sql_vector", "sql_vector_code", "parsed_pipeline", "full")


def mean(xs):
    return round(statistics.mean(xs), 3) if xs else None


async def run(use_cache: bool = True) -> dict:
    settings = get_settings()
    embedder, llm, policy = get_embedder(), get_llm(), policy_from(settings)
    jev_on = bool(settings.typesafe_api_key)
    cache = JevCache(enabled=use_cache)
    parse_cache = ParseCache(enabled=use_cache)
    requests = load_requests()
    per_request, t_start = [], time.perf_counter()

    async with get_sessionmaker()() as s:
        for r in requests:
            if r["kind"] == "vague":
                continue  # vague requests are scored by run_parse_eval (they should be clarified, not searched)
            gold = set(r["gold_vendor_ids"])
            row = {"id": r["id"], "kind": r["kind"], "gold": len(gold), "lists": {}}

            vec = await hybrid_search(s, embedder, r["text"], use_filters=False, k=TOP_K)
            hyb = await hybrid_search(s, embedder, r["text"], category=r["category"],
                                      delivery_state=r["delivery_state"], k=TOP_K)
            labelled = {"quantity": r["qty"], "deadline_days": r["deadline_days"], "unit": None}
            coded = [c for c in hyb if survives(c | {"checks": code_checks(labelled, c)})]
            row["lists"]["vector_only"] = [c["id"] for c in vec]
            row["lists"]["sql_vector"] = [c["id"] for c in hyb]
            row["lists"]["sql_vector_code"] = [c["id"] for c in coded]

            p = await prepare(s, embedder, llm, r, parse_cache)
            shortlist, _ = rank_candidates(p.candidates, policy, K)
            row["lists"]["parsed_pipeline"] = [c["id"] for c in shortlist]
            usage, latency = list(p.usage), p.latency_ms

            if jev_on:
                t0 = time.perf_counter()
                outcome = await judge_cached(p, settings, cache)
                usage += outcome.usage
                latency += int((time.perf_counter() - t0) * 1000)
                full, _ = rank_candidates(with_judgments(p.candidates, outcome), policy, K)
                row["lists"]["full"] = [c["id"] for c in full]
                row["flags"] = {c["id"]: c["review_flag"] for c in full}
                row["jev_ok"] = outcome.ok or not p.candidates

            row.update(tokens=tokens(usage), fallbacks=fallbacks(usage), latency_ms=latency,
                       parsed=p.requirement)
            if gold:
                row["precision"] = {c: round(precision_at_k(ids, gold), 3) for c, ids in row["lists"].items()}
            else:
                row["empty"] = {c: len(ids) == 0 for c, ids in row["lists"].items()}
            row["gold_ids"] = sorted(gold)
            per_request.append(row)

    cache.save()
    await dispose_engine()
    return summarise(per_request, embedder.name, llm.name, settings, jev_on, time.perf_counter() - t_start)


def summarise(rows, embedder_name, llm_name, settings, jev_on, seconds) -> dict:
    with_gold = [r for r in rows if r["gold"]]
    no_match = [r for r in rows if not r["gold"]]
    summary = {}
    for c in CONFIGS:
        if c == "full" and not jev_on:
            summary[c] = None
            continue
        summary[c] = {
            f"precision@{K}": mean([r["precision"][c] for r in with_gold]),
            "no_match": mean([1.0 if r["empty"][c] else 0.0 for r in no_match]),
        }
    if jev_on:
        auto, review = [], []
        for r in with_gold:
            gold = set(r["gold_ids"])
            flags = r.get("flags", {})
            auto += [vid in gold for vid, f in flags.items() if f == "shortlist"]
            review += [f == "needs_review" for f in flags.values()]
        summary["full"] |= {"auto_shortlist_precision": mean([float(x) for x in auto]),
                            "review_load": mean([float(x) for x in review]),
                            "jev_failures": sum(1 for r in rows if not r.get("jev_ok", True))}
    result = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedder": embedder_name, "parser": llm_name,
        "jev": (settings.typesafe_model or "jev-latest") if jev_on else "off",
        "policy": policy_from(settings).__dict__,
        "requests": {"with_gold": len(with_gold), "no_match": len(no_match)},
        "summary": summary,
        "per_request_avg": {"tokens": mean([r["tokens"] for r in rows]),
                            "latency_ms": mean([r["latency_ms"] for r in rows]),
                            "fallbacks_total": sum(r["fallbacks"] for r in rows)},
        "seconds": round(seconds, 1),
        "rows": rows,
    }
    return result


def to_markdown(res: dict) -> str:
    lines = [
        f"Run {res['run_at']} · embedder `{res['embedder']}` · parser `{res['parser']}` · Jev `{res['jev']}`",
        f"{res['requests']['with_gold']} requests with gold vendors, {res['requests']['no_match']} with none.",
        "",
        f"| Config | precision@{K} | no-match |",
        "|---|---|---|",
    ]
    for c in CONFIGS:
        m = res["summary"][c]
        if m is None:
            lines.append(f"| {c} | n/a (TYPESAFE_API_KEY not set) | n/a |")
        else:
            lines.append(f"| {c} | {m[f'precision@{K}']:.3f} | {m['no_match']:.2f} |")
    full = res["summary"]["full"]
    if full:
        auto = full["auto_shortlist_precision"]
        lines += ["", f"Full pipeline: auto-shortlist precision "
                      f"{'n/a (nothing auto-shortlisted)' if auto is None else auto}, "
                      f"review load {full['review_load']}, Jev failures {full['jev_failures']}."]
    a = res["per_request_avg"]
    lines += ["", f"Per request (parsed/full pipeline): {a['tokens']} model tokens, {a['latency_ms']} ms. "
                  f"Fallbacks: {a['fallbacks_total']}" + (" ⚠ some numbers measure the rule-based fallback, not the model"
                                                         if a["fallbacks_total"] else "") + "."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true", help="ignore cached Jev judgments")
    res = asyncio.run(run(use_cache=not ap.parse_args().no_cache))
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "latest.json").write_text(json.dumps(res, indent=2, default=str))
    md = to_markdown(res)
    (RESULTS_DIR / "latest.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
