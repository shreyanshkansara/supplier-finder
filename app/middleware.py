"""Request-id propagation and request logging."""

import logging
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
log = logging.getLogger("app.request")


def get_request_id() -> str | None:
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = _request_id.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            ms = (time.perf_counter() - start) * 1000
            log.info("%s %s %.0fms", request.method, request.url.path, ms)
            _request_id.reset(token)
        response.headers["X-Request-ID"] = rid
        return response
