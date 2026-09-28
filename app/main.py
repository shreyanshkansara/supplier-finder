import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import build_graph
from app.agent.runtime import open_checkpointer, production_deps
from app.api import requests as requests_api
from app.api import vendors as vendors_api
from app.config import get_settings
from app.db import dispose_engine, get_session, get_sessionmaker
from app.errors import register_error_handlers
from app.middleware import RequestIdMiddleware, configure_logging
from app.services.embeddings import get_embedder
from app.services.llm import get_llm
from app.services.vendor_import import index_status, seed_if_needed

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    async with open_checkpointer(settings) as saver:
        llm = get_llm(settings)  # fails fast if GEMINI_API_KEY is set without GEMINI_MODEL
        embedder = get_embedder()
        app.state.embedder_name = embedder.name
        deps = production_deps(settings, get_sessionmaker(), embedder, llm)
        app.state.graph = build_graph(deps).compile(checkpointer=saver)
        log.info("graph ready (llm=%s, checkpointer=%s)", llm.name, settings.checkpointer)
        seeding = None
        if settings.seed_on_start:
            # background, so the server starts listening immediately (hosts time out slow boots)
            seeding = asyncio.create_task(_seed(get_sessionmaker(), embedder))
        yield
        if seeding and not seeding.done():
            seeding.cancel()
    await dispose_engine()


async def _seed(sessionmaker, embedder) -> None:
    try:
        await seed_if_needed(sessionmaker, embedder)
    except Exception:  # noqa: BLE001 - a failed seed must not take the API down
        log.exception("seeding vendors failed")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name, version="0.4.0", lifespan=lifespan)
    app.add_middleware(RequestIdMiddleware)
    register_error_handlers(app)
    app.include_router(vendors_api.router)
    app.include_router(requests_api.router)

    @app.get("/", include_in_schema=False)
    async def index():
        """The web page: type a request, answer questions, see the vendors."""
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health", tags=["ops"])
    async def health():
        return {"status": "ok"}

    @app.get("/health/ready", tags=["ops"])
    async def ready(request: Request, session: AsyncSession = Depends(get_session)):
        """Database reachable, plus how many vendors the current embedder can search."""
        await session.execute(text("SELECT 1"))
        index = await index_status(session, request.app.state.embedder_name)
        ok = index["vendors"] > 0 and index["searchable"] == index["vendors"]
        return {"status": "ready" if ok else "loading", "index": index}

    return app


app = create_app()
