"""Load data/vendors.json into Postgres and embed every vendor's specs.

    python -m scripts.seed_vendors                # uses DATABASE_URL and the configured embedder
    python -m scripts.seed_vendors --file data/vendors.json

Safe to re-run: vendors are upserted by id. Re-run after switching embedders (e.g. adding a
GEMINI_API_KEY) so every vector comes from the same model. 400 vendors = 4 batched embedding calls.
"""

import argparse
import asyncio
import json
from pathlib import Path

from app.db import dispose_engine, get_sessionmaker
from app.services.embeddings import get_embedder
from app.services.vendor_import import upsert_vendors


async def main(path: Path) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    embedder = get_embedder()
    async with get_sessionmaker()() as session:
        n = await upsert_vendors(session, embedder, rows)
    await dispose_engine()
    print(f"Seeded {n} vendors from {path} (embedder: {embedder.name})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=Path("data/vendors.json"))
    asyncio.run(main(ap.parse_args().file))
