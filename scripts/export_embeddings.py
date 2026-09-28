"""Save the vendors' current vectors to data/embeddings/<embedder>.f16.gz.

    python -m scripts.export_embeddings

Seeding then reuses them instead of calling the embedding API again: a fresh install or a new
deployment is searchable in seconds and spends none of the free-tier embedding quota. Vectors
are keyed by a hash of the exact embedded text, so edited vendors are still re-embedded.
"""

import asyncio

from sqlalchemy import select

from app.db import dispose_engine, get_sessionmaker
from app.models import Vendor
from app.services.embeddings import get_embedder
from app.services.vendor_import import embedding_text, save_precomputed, text_hash


async def main() -> None:
    embedder = get_embedder()
    async with get_sessionmaker()() as s:
        rows = (await s.execute(select(Vendor).where(Vendor.embedding_model == embedder.name)
                                .order_by(Vendor.id))).scalars().all()
        items = [(text_hash(embedding_text({"categories": v.categories, "specs": v.specs})),
                  [float(x) for x in v.embedding]) for v in rows]
    await dispose_engine()
    if not items:
        raise SystemExit(f"No vendors embedded with {embedder.name}; run scripts.seed_vendors first.")
    path = save_precomputed(embedder.name, len(items[0][1]), items)
    print(f"Saved {len(items)} vectors for {embedder.name} to {path} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    asyncio.run(main())
