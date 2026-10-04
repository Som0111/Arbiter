import secrets

from fastapi import Header, HTTPException

from arbiter.config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().arbiter_api_key
    if not x_api_key or not expected or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing API key")
