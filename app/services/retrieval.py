"""Hybrid retrieval: hard constraints in SQL WHERE, then pgvector similarity on the specs.

Geography and category are facts, not judgments, so they're plain SQL filters. No agent needed.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Vendor
from app.services.embeddings import Embedder


def vendor_to_dict(v: Vendor) -> dict:
    return {
        "id": v.id, "name": v.name, "city": v.city, "state": v.state,
        "regions_served": list(v.regions_served), "categories": list(v.categories),
        "specs": v.specs, "moq": v.moq, "moq_unit": v.moq_unit, "lead_time_days": v.lead_time_days,
        "price_band": v.price_band, "indicative_price": v.indicative_price,
        "payment_terms": v.payment_terms, "rating_notes": v.rating_notes,
        "contact_name": v.contact_name, "contact_email": v.contact_email, "phone": v.phone,
    }


async def hybrid_search(
    session: AsyncSession,
    embedder: Embedder,
    query_text: str,
    *,
    category: str | None = None,
    delivery_state: str | None = None,
    k: int = 25,
    use_filters: bool = True,
) -> list[dict]:
    """Return up to k vendor dicts with a `similarity` in [-1, 1], best first.

    use_filters=False gives the vector-only baseline for the evaluation.
    """
    qvec = (await embedder.embed([query_text], kind="query"))[0]
    distance = Vendor.embedding.cosine_distance(qvec).label("distance")
    stmt = (
        select(Vendor, distance)
        .where(Vendor.embedding.is_not(None), Vendor.embedding_model == embedder.name)
        .order_by(distance)
        .limit(k)
    )
    if use_filters:
        if category:
            stmt = stmt.where(Vendor.categories.contains([category]))  # uses the GIN index
        if delivery_state:
            stmt = stmt.where(Vendor.regions_served.overlap([delivery_state, "Pan-India"]))

    rows = (await session.execute(stmt)).all()
    return [vendor_to_dict(v) | {"similarity": round(1 - float(d), 4)} for v, d in rows]
