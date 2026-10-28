from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str
    gemini_api_key: str
    logtail_source_token: str = ""
    redis_url: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()


VALID_TIERS = {"simple", "standard", "complex"}


class TenantConfig(BaseModel):
    daily_budget_usd: float
    rate_limit_rpm: int
    tier_cap: str

    @field_validator("tier_cap")
    @classmethod
    def valid_tier(cls, v: str) -> str:
        if v not in VALID_TIERS:
            raise ValueError(f"tier_cap must be one of {sorted(VALID_TIERS)}, got '{v}'")
        return v


def load_tenants() -> dict:
    """tenants.yaml as {tenant_id: {...}}; raises ValueError at startup if any entry is malformed."""
    with open(ROOT / "config" / "tenants.yaml") as f:
        raw = yaml.safe_load(f)["tenants"]
    validated = {}
    for tid, cfg in raw.items():
        try:
            validated[tid] = TenantConfig(**cfg).model_dump()
        except Exception as exc:
            raise ValueError(f"Invalid tenant config for '{tid}': {exc}") from exc
    return validated
