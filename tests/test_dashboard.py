import importlib.util
import random
from datetime import UTC, datetime
from pathlib import Path

from streamlit.testing.v1 import AppTest

from arbiter.logger import write_log

ROOT = Path(__file__).resolve().parent.parent


def _load_sample_script():
    spec = importlib.util.spec_from_file_location(
        "generate_sample_logs", ROOT / "scripts" / "generate_sample_logs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_dashboard(log_path, monkeypatch):
    monkeypatch.setenv("LOG_PATH", str(log_path))
    at = AppTest.from_file(str(ROOT / "ui" / "dashboard.py"), default_timeout=60)
    at.run()
    return at


def test_empty_log_shows_no_data_message(tmp_path, monkeypatch):
    at = run_dashboard(tmp_path / "none.jsonl", monkeypatch)
    assert not at.exception
    assert "No data yet" in at.info[0].value


def test_dashboard_renders_with_sample_logs(tmp_path, monkeypatch):
    make_entry = _load_sample_script().make_entry
    path = tmp_path / "requests.jsonl"
    rng, now = random.Random(1), datetime.now(UTC)
    for _ in range(20):
        write_log(make_entry(rng, now), str(path))
    at = run_dashboard(path, monkeypatch)
    assert not at.exception
    assert at.metric[0].value == "20"
    assert len(at.metric) == 4
