import logging
import os

from fastapi import Header, HTTPException

log = logging.getLogger(__name__)

PREFIX = "ARBITER_KEY_"


def make_require_api_key(tenants: dict):
    """Build the key->tenant map from ARBITER_KEY_<TENANT_ID_UPPER> env vars.

    Raises RuntimeError if an env var names a tenant missing from tenants.yaml.
    Returns a FastAPI dependency resolving X-API-Key to a tenant_id (401 otherwise).
    """
    by_upper = {t.upper(): t for t in tenants}
    key_map: dict[str, str] = {}
    for env_var, value in os.environ.items():
        if not env_var.startswith(PREFIX) or not value:
            continue
        tenant_id = by_upper.get(env_var[len(PREFIX):].upper())
        if tenant_id is None:
            raise RuntimeError(
                f"Env var {env_var!r} maps to a tenant that is not defined in "
                "tenants.yaml. Remove the env var or add the tenant."
            )
        key_map[value] = tenant_id

    if not key_map:
        log.warning(
            "No ARBITER_KEY_* env vars found - all authenticated requests will get 401. "
            "Run scripts/generate_keys.py and set the vars."
        )

    def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="missing X-API-Key header")
        tenant_id = key_map.get(x_api_key)
        if tenant_id is None:
            raise HTTPException(status_code=401, detail="invalid API key")
        return tenant_id

    return require_api_key
