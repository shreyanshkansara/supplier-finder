"""Tune the Jev routing thresholds on the labelled seed requests.

    python -m eval.tune_thresholds [--min-gold-kept 0.95] [--min-auto-precision 0.98]

Jev is called at most once per (requirement, vendor listing): judgments are cached in
eval/results/jev_cache.json (shared with run_eval), so the grid search itself is free.

How the thresholds are chosen (explicit policy, not magic):
  review_at     below it a vendor is dropped. Pick the value that maximises
                precision@5 + 0.5 * no-match accuracy, among values that still keep at least
                --min-gold-kept of the genuinely matching vendors (dropping real matches is worse
                than showing an extra one to the buyer). Ties go to the value that keeps more real
                matches, then to the stricter cut.
  shortlist_at  at or above it a vendor is auto-shortlisted; between the two it is flagged
                "needs_review". Pick the LOWEST value whose auto-shortlisted vendors are correct at
                least --min-auto-precision of the time, so review load is as small as it can be
                without the buyer having to double-check auto-accepted vendors.

Prints the grid's best rows and the .env lines to use; writes eval/results/thresholds.json.
With only ~30 labelled requests, treat the result as a starting point and re-tune as data grows.
"""

import argparse
import asyncio
import json
import statistics
from dataclasses import replace

from app.agent.nodes import RankPolicy, rank_candidates
from app.agent.runtime import policy_from
from app.config import get_settings
from app.db import dispose_engine, get_sessionmaker
from app.services.embeddings import get_embedder
from app.services.llm import get_llm
from eval.pipeline import (
    RESULTS_DIR,
    JevCache,
    K,
    ParseCache,
    judge_cached,
    load_requests,
    precision_at_k,
    prepare,
    with_judgments,
)

GRID = [round(x * 0.05, 2) for x in range(0, 20)]  # 0.00 .. 0.95


def evaluate_policy(rows: list[dict], policy: RankPolicy) -> dict:
    """rows: [{"gold": set, "candidates": [judged, code-checked candidates]}]."""
    prec, empty, kept, total_gold, auto, review = [], [], 0, 0, [], []
    for r in rows:
        shortlist, dropped = rank_candidates(r["candidates"], policy, K)
        ids = [c["id"] for c in shortlist]
        if r["gold"]:
            prec.append(precision_at_k(ids, r["gold"]))
            gold_here = [c for c in r["candidates"] if c["id"] in r["gold"]]
            total_gold += len(gold_here)
            kept += sum(1 for c in gold_here if not (c.get("p_spec") is not None and c["p_spec"] < policy.review_at))
            auto += [c["id"] in r["gold"] for c in shortlist if c["review_flag"] == "shortlist"]
            review += [c["review_flag"] == "needs_review" for c in shortlist]
        else:
            empty.append(1.0 if not ids else 0.0)
    avg = lambda xs: statistics.mean(xs) if xs else 0.0  # noqa: E731
    return {"precision@5": round(avg(prec), 4), "no_match": round(avg(empty), 4),
            "gold_kept": round(kept / total_gold, 4) if total_gold else 1.0,
            "auto_precision": round(avg([float(a) for a in auto]), 4) if auto else None,
            "review_load": round(avg([float(x) for x in review]), 4)}


def choose_thresholds(rows: list[dict], base: RankPolicy, min_gold_kept: float = 0.95,
                      min_auto_precision: float = 0.9) -> dict:
    # 1) review_at: best retrieval quality that still keeps the real matches
    candidates = []
    for review_at in GRID:
        m = evaluate_policy(rows, replace(base, review_at=review_at, shortlist_at=max(review_at, base.shortlist_at)))
        if m["gold_kept"] >= min_gold_kept:
            candidates.append((m["precision@5"] + 0.5 * m["no_match"], review_at, m))
    if not candidates:
        review_at, why = 0.0, f"no threshold keeps {min_gold_kept:.0%} of real matches; dropping nothing"
    else:
        # ties: keep more real matches first, then prefer the stricter cut
        best = max(candidates, key=lambda t: (t[0], t[2]["gold_kept"], t[1]))
        review_at, why = best[1], "best precision@5 + 0.5*no-match with gold kept"
    # 2) shortlist_at: lowest value whose auto-accepts are trustworthy
    shortlist_at, grid = 1.0, []
    for s_at in [x for x in GRID if x >= review_at] + [1.0]:
        m = evaluate_policy(rows, replace(base, review_at=review_at, shortlist_at=s_at))
        grid.append({"shortlist_at": s_at, **m})
        if m["auto_precision"] is not None and m["auto_precision"] >= min_auto_precision and shortlist_at == 1.0:
            shortlist_at = s_at
    final = evaluate_policy(rows, replace(base, review_at=review_at, shortlist_at=shortlist_at))
    return {"review_at": review_at, "shortlist_at": shortlist_at, "metrics": final, "why": why,
            "review_grid": [{"review_at": ra, **m} for _, ra, m in sorted(candidates, key=lambda t: -t[0])[:8]],
            "shortlist_grid": grid}


async def collect_rows(use_cache: bool = True) -> tuple[list[dict], dict]:
    settings = get_settings()
    embedder, llm = get_embedder(), get_llm()
    cache = JevCache(enabled=use_cache)
    parse_cache = ParseCache(enabled=use_cache)
    rows, failures = [], 0
    async with get_sessionmaker()() as s:
        for r in load_requests():
            if r["kind"] == "vague":
                continue
            p = await prepare(s, embedder, llm, r, parse_cache)
            outcome = await judge_cached(p, settings, cache)
            if p.candidates and not outcome.by_vendor:
                failures += 1
            rows.append({"id": r["id"], "gold": p.gold, "candidates": with_judgments(p.candidates, outcome)})
    cache.save()
    await dispose_engine()
    return rows, {"embedder": embedder.name, "parser": llm.name, "jev_failures": failures}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gold-kept", type=float, default=0.95)
    ap.add_argument("--min-auto-precision", type=float, default=0.98)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    settings = get_settings()
    if not settings.typesafe_api_key and not (RESULTS_DIR / "jev_cache.json").exists():
        raise SystemExit("Set TYPESAFE_API_KEY first: tuning needs Jev judgments (they are cached after one run).")

    rows, meta = asyncio.run(collect_rows(use_cache=not a.no_cache))
    judged = sum(1 for r in rows for c in r["candidates"] if c.get("p_spec") is not None)
    if not judged:
        raise SystemExit("No Jev judgments available (check TYPESAFE_API_KEY and the jev_failures count).")
    result = choose_thresholds(rows, policy_from(settings), a.min_gold_kept, a.min_auto_precision)
    result |= meta | {"judged_candidates": judged, "min_gold_kept": a.min_gold_kept,
                      "min_auto_precision": a.min_auto_precision}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "thresholds.json").write_text(json.dumps(result, indent=2))

    print(f"\n{judged} judged candidates over {len(rows)} requests (Jev failures: {meta['jev_failures']})\n")
    print(f"{'review_at':>9} {'p@5':>6} {'no-match':>8} {'gold kept':>9}")
    for g in result["review_grid"]:
        print(f"{g['review_at']:>9.2f} {g['precision@5']:>6.3f} {g['no_match']:>8.2f} {g['gold_kept']:>9.2f}")
    m = result["metrics"]
    print(f"\nChosen: review_at={result['review_at']} ({result['why']}), shortlist_at={result['shortlist_at']}")
    print(f"-> precision@5 {m['precision@5']}, no-match {m['no_match']}, gold kept {m['gold_kept']}, "
          f"auto-shortlist precision {m['auto_precision']}, review load {m['review_load']}\n")
    print("Put these in .env:\n")
    print(f"JEV_REVIEW_AT={result['review_at']}\nJEV_SHORTLIST_AT={result['shortlist_at']}")


if __name__ == "__main__":
    main()
