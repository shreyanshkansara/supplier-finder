"""Load vendors from CSV/JSON rows, upsert them, and (re)compute embeddings."""

import csv
import gzip
import hashlib
import io
import json
import logging
import struct
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Vendor
from app.services.embeddings import Embedder, l2_normalize

log = logging.getLogger(__name__)

LIST_FIELDS = ("regions_served", "categories")
INT_FIELDS = ("moq", "lead_time_days")
REQUIRED = ("vendor_id", "name", "city", "state", "regions_served", "categories", "specs", "moq",
            "moq_unit", "lead_time_days", "price_band", "contact_email")


class ImportError_(ValueError):
    pass


def parse_csv(content: bytes | str) -> list[dict]:
    text = content.decode("utf-8-sig") if isinstance(content, bytes) else content
    reader = csv.DictReader(io.StringIO(text))
    missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
    if missing:
        raise ImportError_(f"CSV is missing columns: {', '.join(missing)}")
    rows = []
    for i, r in enumerate(reader, start=2):
        try:
            row = dict(r)
            for f in LIST_FIELDS:
                row[f] = [x.strip() for x in (row.get(f) or "").split("|") if x.strip()]
            for f in INT_FIELDS:
                row[f] = int(row[f])
            row["gst_registered"] = str(row.get("gst_registered", "true")).strip().lower() in ("true", "1", "yes")
            rows.append(row)
        except (ValueError, KeyError) as e:
            raise ImportError_(f"Row {i}: {e}") from e
    return rows


def _to_record(row: dict) -> dict:
    return {
        "id": row["vendor_id"], "name": row["name"], "city": row["city"], "state": row["state"],
        "regions_served": row["regions_served"], "categories": row["categories"], "specs": row["specs"],
        "moq": int(row["moq"]), "moq_unit": row["moq_unit"], "lead_time_days": int(row["lead_time_days"]),
        "price_band": row["price_band"], "indicative_price": row.get("indicative_price"),
        "payment_terms": row.get("payment_terms"), "gst_registered": bool(row.get("gst_registered", True)),
        "rating_notes": row.get("rating_notes"), "contact_name": row.get("contact_name"),
        "contact_email": row["contact_email"], "phone": row.get("phone"),
    }


def embedding_text(rec: dict) -> str:
    """What gets embedded: categories + specs (not name/contact, which add noise)."""
    return f"{' / '.join(rec['categories'])}. {rec['specs']}"


PRECOMPUTED_DIR = Path(__file__).resolve().parents[2] / "data" / "embeddings"


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _precomputed_path(embedder_name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-." else "_" for ch in embedder_name)
    return PRECOMPUTED_DIR / f"{safe}.f16.gz"


def load_precomputed(embedder_name: str) -> dict[str, list[float]]:
    """{text_hash: vector} saved by scripts/export_embeddings.py, if a file exists for this embedder.

    Keyed by the hash of the exact text that was embedded, so an edited vendor is re-embedded
    and never gets a stale vector. Stored as float16 (half the size; cosine ranking unchanged).
    """
    path = _precomputed_path(embedder_name)
    if not path.exists():
        return {}
    raw = gzip.decompress(path.read_bytes())
    header_len = struct.unpack("<I", raw[:4])[0]
    header = json.loads(raw[4 : 4 + header_len])
    dim, body = header["dim"], raw[4 + header_len :]
    out = {}
    for i, h in enumerate(header["hashes"]):
        vec = struct.unpack(f"<{dim}e", body[i * dim * 2 : (i + 1) * dim * 2])
        out[h] = l2_normalize(list(vec))
    return out


def save_precomputed(embedder_name: str, dim: int, items: list[tuple[str, list[float]]]) -> Path:
    header = json.dumps({"embedder": embedder_name, "dim": dim, "hashes": [h for h, _ in items]}).encode()
    body = b"".join(struct.pack(f"<{dim}e", *vec) for _, vec in items)
    path = _precomputed_path(embedder_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(struct.pack("<I", len(header)) + header + body, mtime=0))
    return path


async def _write(session: AsyncSession, records: list[dict]) -> None:
    for i in range(0, len(records), 200):
        chunk = records[i : i + 200]
        stmt = insert(Vendor).values(chunk)
        update_cols = {c: stmt.excluded[c] for c in chunk[0] if c != "id"}
        await session.execute(stmt.on_conflict_do_update(index_elements=[Vendor.id], set_=update_cols))
    await session.commit()


async def upsert_vendors(session: AsyncSession, embedder: Embedder, rows: list[dict], batch: int = 50) -> int:
    """Upsert vendors with vectors. Uses precomputed vectors where the text matches, embeds the rest
    in batches and commits after each one, so a failure (e.g. a daily quota) keeps the progress made
    and the next run only embeds what is still missing."""
    records = [_to_record(r) for r in rows]
    known = load_precomputed(embedder.name)
    # Vendors already in the DB with a vector from this embedder for the same text: keep that vector.
    existing = await session.execute(
        select(Vendor.id, Vendor.categories, Vendor.specs)
        .where(Vendor.id.in_([r["id"] for r in records]), Vendor.embedding_model == embedder.name))
    done = {vid: text_hash(embedding_text({"categories": cats, "specs": specs})) for vid, cats, specs in existing}
    kept, fresh, todo = [], [], []
    for rec in records:
        h = text_hash(embedding_text(rec))
        if done.get(rec["id"]) == h:
            kept.append(rec)  # row fields updated, vector untouched
        elif (vec := known.get(h)) is not None:
            rec["embedding"], rec["embedding_model"] = vec, embedder.name
            fresh.append(rec)
        else:
            rec["embedding"], rec["embedding_model"] = None, None
            todo.append(rec)
    log.info("%d vendors already searchable, %d from precomputed %s vectors, %d to embed",
             len(kept), len(fresh), embedder.name, len(todo))
    # Rows first (so the vendor list/filters work at once), then vectors batch by batch.
    if kept:
        await _write(session, kept)
    if fresh or todo:
        await _write(session, fresh + todo)
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        vectors = await embedder.embed([embedding_text(r) for r in chunk], kind="document")
        for rec, vec in zip(chunk, vectors, strict=True):
            rec["embedding"], rec["embedding_model"] = vec, embedder.name
        await _write(session, chunk)
        log.info("embedded %d/%d vendors with %s", min(i + batch, len(todo)), len(todo), embedder.name)
    return len(records)


async def index_status(session: AsyncSession, embedder_name: str) -> dict:
    total = (await session.execute(select(func.count()).select_from(Vendor))).scalar_one()
    ready = (await session.execute(select(func.count()).select_from(Vendor)
                                   .where(Vendor.embedding_model == embedder_name))).scalar_one()
    return {"vendors": total, "searchable": ready, "embedder": embedder_name}


DEFAULT_SEED_FILE = Path(__file__).resolve().parents[2] / "data" / "vendors.json"


async def seed_if_needed(sessionmaker, embedder: Embedder, path: Path = DEFAULT_SEED_FILE) -> int:
    """Load `path` unless every vendor already has a vector from THIS embedder. Returns rows loaded.

    Safe to call on every start: after the first successful run it only does two COUNT queries.
    Switching embedders (e.g. adding a Gemini key) makes it re-embed once.
    """
    async with sessionmaker() as s:
        st = await index_status(s, embedder.name)
        if st["vendors"] and st["searchable"] == st["vendors"]:
            log.info("vendors already seeded for %s (%d rows)", embedder.name, st["vendors"])
            return 0
        rows = json.loads(path.read_text(encoding="utf-8"))
        log.info("seeding %d vendors with %s ...", len(rows), embedder.name)
        n = await upsert_vendors(s, embedder, rows)
        log.info("seeded %d vendors", n)
        return n
