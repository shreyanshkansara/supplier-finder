"""One error format for every failure: {"error": {"code", "message", "request_id", "details"}}."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.middleware import get_request_id

log = logging.getLogger(__name__)


class AppError(Exception):
    def __init__(self, status_code: int, code: str, message: str, details=None):
        self.status_code, self.code, self.message, self.details = status_code, code, message, details
        super().__init__(message)


class NotFound(AppError):
    def __init__(self, what: str):
        super().__init__(404, "not_found", f"{what} not found")


class Conflict(AppError):
    def __init__(self, message: str, details=None):
        super().__init__(409, "conflict", message, details)


def _body(code: str, message: str, details=None) -> dict:
    return {"error": {"code": code, "message": message, "request_id": get_request_id(), "details": details}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return JSONResponse(_body(exc.code, exc.message, exc.details), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        details = [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(_body("validation_error", "Invalid request", details), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        code = {401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed"}.get(
            exc.status_code, "http_error")
        return JSONResponse(_body(code, str(exc.detail)), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(_body("internal_error", "Something went wrong"), status_code=500)
