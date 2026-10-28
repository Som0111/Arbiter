import logging
from dataclasses import asdict
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from arbiter import logger
from arbiter.logger import RequestLog, read_logs, write_log

TOKEN = "tok_secret_123"


class InlinePool:
    def submit(self, fn, *a):
        fn(*a)


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "requests.jsonl"))
    monkeypatch.setattr(logger, "_pool", InlinePool())
    monkeypatch.setattr(logger, "_logtail_client", None)


def set_token(monkeypatch, token):
    monkeypatch.setattr("arbiter.config.get_settings",
                        lambda: SimpleNamespace(logtail_source_token=token))


def entry():
    return RequestLog("r1", "tenant_a", "simple", "m", 1, 1, 0.01, 10.0, False, False,
                      True, None, datetime.now(UTC).isoformat())


def mock_client(monkeypatch, handler):
    client = httpx.Client(base_url="https://in.logs.betterstack.com",
                          transport=httpx.MockTransport(handler))
    monkeypatch.setattr(logger, "_logtail_client", client)


def test_ships_payload_when_token_set(monkeypatch):
    sent = []
    mock_client(monkeypatch, lambda req: sent.append(req) or httpx.Response(202))
    set_token(monkeypatch, TOKEN)
    e = entry()
    write_log(e)
    assert len(sent) == 1 and sent[0].method == "POST"
    assert sent[0].read() == httpx.Request("POST", "/", json=asdict(e)).read()
    assert len(read_logs()) == 1  # local file still written


def test_no_request_without_token(monkeypatch):
    shipper = MagicMock()
    monkeypatch.setattr(logger, "_ship_to_logtail", shipper)
    set_token(monkeypatch, "")
    write_log(entry())
    shipper.assert_not_called()
    assert len(read_logs()) == 1


@pytest.mark.parametrize("handler", [
    lambda req: httpx.Response(500),
    lambda req: (_ for _ in ()).throw(httpx.ConnectTimeout("boom")),
])
def test_http_failure_does_not_raise(monkeypatch, handler):
    mock_client(monkeypatch, handler)
    set_token(monkeypatch, TOKEN)
    write_log(entry())
    assert len(read_logs()) == 1


def test_local_write_failure_does_not_raise_and_still_ships(monkeypatch, tmp_path):
    shipper = MagicMock()
    monkeypatch.setattr(logger, "_ship_to_logtail", shipper)
    set_token(monkeypatch, TOKEN)
    write_log(entry(), path=str(tmp_path))  # a directory: open() fails
    shipper.assert_called_once()


def test_token_never_logged(monkeypatch, caplog):
    mock_client(monkeypatch, lambda req: httpx.Response(500))
    set_token(monkeypatch, TOKEN)
    with caplog.at_level(logging.DEBUG):
        write_log(entry())
    assert TOKEN not in caplog.text
