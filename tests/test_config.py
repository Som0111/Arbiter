import pytest
import yaml

from arbiter import config


def load(tmp_path, monkeypatch, tenants):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "tenants.yaml").write_text(yaml.dump({"tenants": tenants}))
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return config.load_tenants()


GOOD = {"daily_budget_usd": 1.0, "rate_limit_rpm": 10, "tier_cap": "standard"}


def test_valid_tenants_load(tmp_path, monkeypatch):
    assert load(tmp_path, monkeypatch, {"a": GOOD}) == {"a": GOOD}


@pytest.mark.parametrize("bad", [
    {**GOOD, "tier_cap": "premium"},
    {**GOOD, "rate_limit_rpm": "lots"},
    {"daily_budget_usd": 1.0, "tier_cap": "simple"},  # missing rate_limit_rpm
])
def test_malformed_tenant_fails_at_load_naming_the_tenant(tmp_path, monkeypatch, bad):
    with pytest.raises(ValueError, match="Invalid tenant config for 'a'"):
        load(tmp_path, monkeypatch, {"a": bad})


def test_shipped_tenants_yaml_is_valid():
    assert {"tenant_a", "tenant_b", "tenant_test"} <= set(config.load_tenants())
