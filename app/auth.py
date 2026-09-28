"""Simple API-key auth. Swap for JWT later without touching the routers."""

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    key: str | None = Depends(api_key_header), settings: Settings = Depends(get_settings)
) -> str:
    if not key or not secrets.compare_digest(key, settings.api_key):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid X-API-Key")
    return "api-key-user"
