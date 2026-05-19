# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Phase 0.5 — peer_agents Hermes plugin.

These tests import the plugin module directly from the bundled template
directory (which is not on sys.path in normal use) so we can validate:

  * overlay → peer_agents extraction (and skip when absent / malformed),
  * lazy hostname resolution from SLURM_MASTER_NODE_HET_GROUP_<N>,
  * tool registration (handler signature, schema shape, no-op when empty),
  * HTTP call shape against a stub server.

The plugin's runtime entry point ``register(ctx)`` matches the Hermes
``ctx.register_tool(...)`` interface; we drive it with a tiny fake context
that captures registered tools.
"""

from __future__ import annotations

import http.server
import importlib.util
import json
import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Import the plugin from the template directory (not on sys.path normally).
# ---------------------------------------------------------------------------


_PLUGIN_PATH = (
    Path(__file__).resolve().parent.parent / "hermes_home_template" / "plugins" / "peer_agents" / "__init__.py"
)


def _load_plugin():
    spec = importlib.util.spec_from_file_location("_test_peer_agents_plugin", _PLUGIN_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plugin = _load_plugin()


# ---------------------------------------------------------------------------
# Fake registration context (mirrors hermes_cli.plugins.PluginContext surface).
# ---------------------------------------------------------------------------


class _FakeCtx:
    def __init__(self) -> None:
        self.registered: List[Dict[str, Any]] = []

    def register_tool(self, **kw):
        self.registered.append(kw)


# ---------------------------------------------------------------------------
# Overlay loading
# ---------------------------------------------------------------------------


def _write_overlay(home: Path, payload: dict) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    path = home / "hermes_agent_overlay.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_peer_agents_normalizes_entries(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "h"
    _write_overlay(
        home,
        {
            "peer_agents": {
                "good": {"het_group": 1, "port": 9000},
                "good_with_host": {"het_group": 0, "port": 9001, "host": "fixed.example"},
                "missing_port": {"het_group": 2},  # dropped
                "not_a_dict": "lol",  # dropped
            }
        },
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    peers = plugin._load_peer_agents()
    assert set(peers) == {"good", "good_with_host"}
    assert peers["good"]["het_group"] == 1
    assert peers["good"]["port"] == 9000
    assert peers["good"]["host"] is None
    assert peers["good_with_host"]["host"] == "fixed.example"


def test_load_peer_agents_returns_empty_when_no_hermes_home(monkeypatch) -> None:
    monkeypatch.delenv("HERMES_HOME", raising=False)
    # If hermes_constants is unavailable AND no env var is set, we get an
    # empty dict (no tools register, no error).
    assert plugin._load_peer_agents() == {}


def test_load_peer_agents_returns_empty_when_overlay_missing(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "h"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert plugin._load_peer_agents() == {}


def test_load_peer_agents_returns_empty_when_overlay_malformed(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "h"
    home.mkdir()
    (home / "hermes_agent_overlay.json").write_text("{ not json")
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert plugin._load_peer_agents() == {}


# ---------------------------------------------------------------------------
# Lazy hostname resolution
# ---------------------------------------------------------------------------


def test_resolve_host_uses_slurm_env_var(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_MASTER_NODE_HET_GROUP_3", "node-x42")
    host = plugin._resolve_host({"het_group": 3, "port": 9000, "host": None})
    assert host == "node-x42"


def test_resolve_host_falls_back_to_localhost(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_MASTER_NODE_HET_GROUP_7", raising=False)
    host = plugin._resolve_host({"het_group": 7, "port": 9000, "host": None})
    assert host == "localhost"


def test_resolve_host_explicit_override_wins(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_MASTER_NODE_HET_GROUP_3", "ignored")
    host = plugin._resolve_host({"het_group": 3, "port": 9000, "host": "explicit.example"})
    assert host == "explicit.example"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_noop_when_no_peers(monkeypatch) -> None:
    monkeypatch.delenv("HERMES_HOME", raising=False)
    ctx = _FakeCtx()
    plugin.register(ctx)
    assert ctx.registered == []


def test_register_creates_one_tool_per_peer(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "orch_home"
    _write_overlay(
        home,
        {
            "peer_agents": {
                "analyst": {"het_group": 0, "port": 9001},
                "coder": {"het_group": 1, "port": 9002},
            }
        },
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    ctx = _FakeCtx()
    plugin.register(ctx)

    names = sorted(r["name"] for r in ctx.registered)
    assert names == ["call_analyst", "call_coder"]
    # Schemas advertise the LLM-facing 'task' string parameter.
    for r in ctx.registered:
        schema = r["schema"]
        assert schema["parameters"]["required"] == ["task"]
        assert schema["parameters"]["properties"]["task"]["type"] == "string"
        # Every registered tool lives under the peer_agents toolset so
        # operators can disable the whole peer-call surface in one place.
        assert r["toolset"] == "peer_agents"


def test_handler_rejects_missing_task(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "h"
    _write_overlay(home, {"peer_agents": {"x": {"het_group": 0, "port": 9000}}})
    monkeypatch.setenv("HERMES_HOME", str(home))
    ctx = _FakeCtx()
    plugin.register(ctx)
    handler = ctx.registered[0]["handler"]
    resp = json.loads(handler({}))
    assert resp == {"success": False, "error": "call_x requires non-empty 'task' string"}


def test_handler_rejects_empty_task(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "h"
    _write_overlay(home, {"peer_agents": {"x": {"het_group": 0, "port": 9000}}})
    monkeypatch.setenv("HERMES_HOME", str(home))
    ctx = _FakeCtx()
    plugin.register(ctx)
    handler = ctx.registered[0]["handler"]
    resp = json.loads(handler({"task": ""}))
    assert resp["success"] is False


# ---------------------------------------------------------------------------
# End-to-end: real HTTP server playing the role of a peer's /task endpoint.
# ---------------------------------------------------------------------------


class _FakeTaskHandler(http.server.BaseHTTPRequestHandler):
    # Per-class capture of the last request body so tests can assert what
    # the plugin POSTed to a peer.
    last_body: dict = {}
    reply_payload: dict = {"generation": "the answer is 42", "error": ""}
    reply_status: int = 200

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            _FakeTaskHandler.last_body = json.loads(raw)
        except json.JSONDecodeError:
            _FakeTaskHandler.last_body = {"_raw": raw.decode("utf-8", errors="replace")}
        self.send_response(_FakeTaskHandler.reply_status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(_FakeTaskHandler.reply_payload).encode("utf-8"))

    def log_message(self, format, *args):  # noqa: A002 - shadow stdlib name to silence logs
        return  # quiet — pytest captures stderr otherwise


@pytest.fixture
def fake_peer_server():
    """Spin up a localhost HTTP server playing the role of a peer's /task."""
    _FakeTaskHandler.last_body = {}
    _FakeTaskHandler.reply_payload = {"generation": "the answer is 42", "error": ""}
    _FakeTaskHandler.reply_status = 200
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeTaskHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_handler_posts_to_peer_and_returns_generation(tmp_path: Path, monkeypatch, fake_peer_server: int) -> None:
    home = tmp_path / "h"
    _write_overlay(
        home,
        {"peer_agents": {"echo": {"het_group": 99, "port": fake_peer_server}}},
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("SLURM_MASTER_NODE_HET_GROUP_99", "127.0.0.1")

    ctx = _FakeCtx()
    plugin.register(ctx)
    handler = ctx.registered[0]["handler"]
    resp = json.loads(handler({"task": "what is 6 * 7?"}))

    assert resp["success"] is True
    assert resp["result"] == "the answer is 42"
    # The peer received the orchestrator's task verbatim, wrapped in the
    # CallAgentTool /task contract.
    body = _FakeTaskHandler.last_body
    assert isinstance(body.get("task_id"), str) and body["task_id"]
    assert body["messages"] == [{"role": "user", "content": "what is 6 * 7?"}]


def test_handler_surfaces_peer_error(tmp_path: Path, monkeypatch, fake_peer_server: int) -> None:
    home = tmp_path / "h"
    _write_overlay(
        home,
        {"peer_agents": {"flaky": {"het_group": 99, "port": fake_peer_server}}},
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("SLURM_MASTER_NODE_HET_GROUP_99", "127.0.0.1")
    _FakeTaskHandler.reply_payload = {"generation": "", "error": "model server timed out"}

    ctx = _FakeCtx()
    plugin.register(ctx)
    handler = ctx.registered[0]["handler"]
    resp = json.loads(handler({"task": "anything"}))
    assert resp == {"success": False, "error": "model server timed out"}


def test_handler_handles_unreachable_peer(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "h"
    # Point at a port nothing's listening on.
    _write_overlay(
        home,
        {"peer_agents": {"dead": {"het_group": 0, "port": 1}}},
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("SLURM_MASTER_NODE_HET_GROUP_0", "127.0.0.1")

    ctx = _FakeCtx()
    plugin.register(ctx)
    handler = ctx.registered[0]["handler"]
    resp = json.loads(handler({"task": "anything"}))
    assert resp["success"] is False
    assert "could not reach dead" in resp["error"]
