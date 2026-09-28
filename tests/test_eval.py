"""The evaluation harness itself: metrics, caching, threshold selection, and an end-to-end run."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.agent.nodes import RankPolicy
from app.config import Settings
from app.services.judgments import JudgeOutcome
from eval.pipeline import JevCache, Prepared, judge_cached, precision_at_k
from eval.tune_thresholds import choose_thresholds, evaluate_policy


def test_precision_at_k_is_normalised_by_available_gold():
    assert precision_at_k(["a", "b", "x", "y", "z"], {"a", "b", "c"}) == 2 / 3
    assert precision_at_k(["a", "b", "c", "d", "e", "f"], {"a", "b", "c", "d", "e", "f", "g"}) == 1.0
    with pytest.raises(ValueError):
        precision_at_k(["a"], set())


def cand(vid, p_spec, sim=0.5):
    return {"id": vid, "similarity": sim, "p_spec": p_spec, "fit": 3 * p_spec,
            "checks": {"unit_match": True, "moq_ok": True, "lead_ok": True}}


# Gold vendors get high p_spec, near misses middling, wrong products low; one request has no match.
ROWS = [
    {"gold": {"g1", "g2"}, "candidates": [cand("g1", 0.95), cand("g2", 0.8), cand("n1", 0.5, 0.9),
                                          cand("w1", 0.1, 0.99), cand("w2", 0.05, 0.98)]},
    {"gold": {"g3"}, "candidates": [cand("g3", 0.9), cand("n2", 0.45, 0.95), cand("w3", 0.2, 0.97)]},
    {"gold": set(), "candidates": [cand("n3", 0.3), cand("w4", 0.1)]},
]


def test_evaluate_policy_metrics():
    m = evaluate_policy(ROWS, RankPolicy(review_at=0.4, shortlist_at=0.7))
    assert m["precision@5"] == 1.0 and m["gold_kept"] == 1.0
    assert m["no_match"] == 1.0  # both no-match candidates are below 0.4 -> empty list
    assert m["auto_precision"] == 1.0 and m["review_load"] == 2 / 5  # n1, n2 flagged for review


def test_choose_thresholds_picks_a_safe_review_cut_and_lowest_trustworthy_shortlist_cut():
    res = choose_thresholds(ROWS, RankPolicy(), min_gold_kept=1.0, min_auto_precision=1.0)
    assert 0.3 < res["review_at"] <= 0.8  # drops the wrong products, keeps g2 (0.8)
    assert res["metrics"]["gold_kept"] == 1.0 and res["metrics"]["no_match"] == 1.0
    assert res["review_at"] <= res["shortlist_at"] <= 0.8
    assert res["metrics"]["auto_precision"] == 1.0


async def test_judge_cached_only_calls_jev_for_new_listings(tmp_path: Path):
    calls = []

    async def judge(req, cands, settings=None):
        calls.append([c["id"] for c in cands])
        return JudgeOutcome(by_vendor={c["id"]: {"p_spec": 0.9, "fit": 3.0, "fit_confidence": 1.0} for c in cands},
                            usage=[{"step": "verify", "source": "jev:t", "input_tokens": 1, "output_tokens": 1,
                                    "latency_ms": 1, "error": None}])

    cands = [{"id": f"V-{i}", "specs": "s", "categories": ["c"]} for i in range(3)]
    p = Prepared(rid="R", kind="clear", gold=set(), requirement={"item": "x", "specs": "y"}, candidates=cands)
    cache = JevCache(tmp_path / "cache.json")
    s = Settings(typesafe_api_key=None)
    out1 = await judge_cached(p, s, cache, judge=judge)
    cache.save()
    p.candidates = cands + [{"id": "V-9", "specs": "s", "categories": ["c"]}]
    out2 = await judge_cached(p, s, JevCache(tmp_path / "cache.json"), judge=judge)
    assert calls == [["V-0", "V-1", "V-2"], ["V-9"]]  # second run only judges the new vendor
    assert set(out1.by_vendor) == {"V-0", "V-1", "V-2"} and len(out2.by_vendor) == 4


@pytest.mark.db
@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
async def test_eval_runs_end_to_end_on_the_seed_data():
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True,
                   env={**os.environ, "DATABASE_URL": os.environ["TEST_DATABASE_URL"]})
    from eval import run_eval, run_parse_eval
    from scripts.seed_vendors import main as seed

    await seed(Path("data/vendors.json"))
    res = await run_eval.run(use_cache=False)
    s = res["summary"]
    assert res["requests"] == {"with_gold": 28, "no_match": 4}
    assert s["full"] is None  # Jev not configured in tests
    # each layer should help: filters beat raw vectors, code checks beat filters
    assert s["vector_only"]["precision@5"] < s["sql_vector"]["precision@5"] < s["sql_vector_code"]["precision@5"]
    assert "| sql_vector_code |" in run_eval.to_markdown(res)

    parse = await run_parse_eval.run()
    assert parse["vague_caught"] == "8/8" and parse["field_accuracy"]["delivery_state"] == 1.0


@pytest.mark.db
@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
async def test_seed_if_needed_is_idempotent_and_reseeds_for_a_new_embedder():
    from app.db import dispose_engine, get_sessionmaker
    from app.services.embeddings import HashEmbedder
    from app.services.vendor_import import seed_if_needed

    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True,
                   env={**os.environ, "DATABASE_URL": os.environ["TEST_DATABASE_URL"]})
    sm = get_sessionmaker()
    await seed_if_needed(sm, HashEmbedder(1536))  # make sure it's seeded with this embedder
    assert await seed_if_needed(sm, HashEmbedder(1536)) == 0  # second call: nothing to do

    other = HashEmbedder(1536)
    other.name = "other-embedder"
    assert await seed_if_needed(sm, other) == 500  # different embedder: re-embed everything
    assert await seed_if_needed(sm, HashEmbedder(1536)) == 500  # and back, for the other tests
    await dispose_engine()


@pytest.mark.db
@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
async def test_seeding_keeps_progress_when_embedding_fails_midway():
    import json
    from pathlib import Path

    from sqlalchemy import text

    from app.db import dispose_engine, get_sessionmaker
    from app.services.embeddings import HashEmbedder
    from app.services.vendor_import import index_status, seed_if_needed

    class Flaky(HashEmbedder):
        def __init__(self, fail_after):
            super().__init__(1536)
            self.name, self.calls, self.fail_after = "flaky-embedder", 0, fail_after

        async def embed(self, texts, kind="document"):
            self.calls += 1
            if self.calls > self.fail_after:
                raise RuntimeError("429 RESOURCE_EXHAUSTED ... PerDay")
            return await super().embed(texts, kind)

    sm = get_sessionmaker()
    with pytest.raises(RuntimeError):
        await seed_if_needed(sm, Flaky(fail_after=3))  # 3 batches of 50 succeed, then the "daily quota" hits
    async with sm() as s:
        assert (await index_status(s, "flaky-embedder"))["searchable"] == 150
    again = Flaky(fail_after=100)
    await seed_if_needed(sm, again)
    async with sm() as s:
        assert (await index_status(s, "flaky-embedder"))["searchable"] == 500
    rows = len(json.loads(Path("data/vendors.json").read_text()))
    assert again.calls == (rows - 150) // 50  # only what was still missing
    async with sm() as s:  # restore the hash vectors the other DB tests use
        await s.execute(text("UPDATE vendors SET embedding_model = NULL"))
        await s.commit()
    await seed_if_needed(sm, HashEmbedder(1536))
    await dispose_engine()
