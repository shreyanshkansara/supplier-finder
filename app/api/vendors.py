from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_api_key
from app.db import get_session
from app.errors import AppError
from app.models import Vendor
from app.schemas import ImportResult, VendorOut, VendorPage
from app.services.embeddings import get_embedder
from app.services.vendor_import import ImportError_, parse_csv, upsert_vendors

router = APIRouter(prefix="/vendors", tags=["vendors"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=VendorPage)
async def list_vendors(
    category: str | None = None,
    state: str | None = Query(None, description="Vendor's home state"),
    serves: str | None = Query(None, description="Delivers to this state (Pan-India counts)"),
    q: str | None = Query(None, description="Text search in name and specs"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    conds = []
    if category:
        conds.append(Vendor.categories.contains([category]))
    if state:
        conds.append(Vendor.state == state)
    if serves:
        conds.append(Vendor.regions_served.overlap([serves, "Pan-India"]))
    if q:
        conds.append(or_(Vendor.name.ilike(f"%{q}%"), Vendor.specs.ilike(f"%{q}%")))

    total = (await session.execute(select(func.count()).select_from(Vendor).where(*conds))).scalar_one()
    rows = (await session.execute(
        select(Vendor).where(*conds).order_by(Vendor.id).limit(limit).offset(offset))).scalars().all()
    return VendorPage(items=[VendorOut.model_validate(v) for v in rows], total=total, limit=limit, offset=offset)


@router.get("/{vendor_id}", response_model=VendorOut)
async def get_vendor(vendor_id: str, session: AsyncSession = Depends(get_session)):
    v = await session.get(Vendor, vendor_id)
    if v is None:
        raise AppError(404, "not_found", f"Vendor {vendor_id} not found")
    return VendorOut.model_validate(v)


@router.post("/import", response_model=ImportResult, status_code=201)
async def import_vendors(file: UploadFile = File(..., description="vendors.csv"),
                         session: AsyncSession = Depends(get_session)):
    try:
        rows = parse_csv(await file.read())
    except ImportError_ as e:
        raise AppError(422, "invalid_csv", str(e)) from e
    embedder = get_embedder()
    n = await upsert_vendors(session, embedder, rows)
    return ImportResult(upserted=n, embedded=n, embedding_model=embedder.name)
