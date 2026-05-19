# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Phase 0.5 — overlay JSON merging in HermesAgent.model_post_init.

The ns hermes_agent_rollouts pipeline writes
``<HERMES_HOME>/hermes_agent_overlay.json`` per agent.  At Hermes startup,
``HermesAgent.model_post_init`` reads that file and field-overrides the
``HermesAgentConfig`` so the manifest does not have to round-trip through
Hydra command-line args (where peer URLs and JSON dicts would need
brittle quoting).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent.app import (
    HermesAgent,
    HermesAgentConfig,
    ModelServerRef,
    ResourcesServerRef,
)


def _cfg(**kwargs) -> HermesAgentConfig:
    return HermesAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="hermes_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="dummy"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # Same env-isolation as test_app.py — the agent ctor mutates os.environ.
    for var in ("HERMES_HOME", "TERMINAL_ENV", "TERMINAL_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)
    yield


# ---------------------------------------------------------------------------
# Overlay merging
# ---------------------------------------------------------------------------


def _write_overlay(home: Path, payload: dict) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    path = home / "hermes_agent_overlay.json"
    path.write_text(json.dumps(payload))
    return path


def test_overlay_merge_sets_known_fields(tmp_path: Path) -> None:
    home = tmp_path / "agent_home"
    _write_overlay(
        home,
        {
            "agent_name": "overlay-named",
            "trace_dir": "/results/traces/overlay-named",
            "persist_memory": True,
            "save_trajectories": True,
            "peer_agents": {
                "analyst": {"het_group": 1, "port": 9001},
            },
        },
    )

    agent = HermesAgent(
        config=_cfg(hermes_home=str(home)),
        server_client=MagicMock(spec=ServerClient),
    )

    assert agent.config.agent_name == "overlay-named"
    assert agent.config.trace_dir == "/results/traces/overlay-named"
    assert agent.config.persist_memory is True
    assert agent.config.save_trajectories is True
    # peer_agents survives verbatim as a dict.
    assert agent.config.peer_agents == {"analyst": {"het_group": 1, "port": 9001}}


def test_overlay_absent_leaves_config_unchanged(tmp_path: Path) -> None:
    """No overlay file → config defaults stay in effect (back-compat)."""
    home = tmp_path / "no_overlay_home"
    home.mkdir()
    agent = HermesAgent(
        config=_cfg(hermes_home=str(home), agent_name="from-yaml"),
        server_client=MagicMock(spec=ServerClient),
    )
    assert agent.config.agent_name == "from-yaml"
    assert agent.config.peer_agents is None
    assert agent.config.persist_memory is False  # default


def test_overlay_unknown_keys_are_ignored(tmp_path: Path, caplog) -> None:
    home = tmp_path / "weird_overlay"
    _write_overlay(home, {"agent_name": "ok", "moedl": "/m/typo"})
    # Should not raise — unknown keys are warned and skipped so a forward-
    # compatible pipeline overlay can never break an older agent.
    agent = HermesAgent(
        config=_cfg(hermes_home=str(home)),
        server_client=MagicMock(spec=ServerClient),
    )
    assert agent.config.agent_name == "ok"
    # Phase 0.5 default for new field stays None.
    assert agent.config.peer_agents is None


def test_overlay_malformed_json_is_warned_not_raised(tmp_path: Path) -> None:
    home = tmp_path / "bad_overlay"
    home.mkdir()
    (home / "hermes_agent_overlay.json").write_text("{not json")
    # Constructor must succeed even when the overlay is unreadable; the
    # agent falls back to whatever the YAML/Hydra config provided.
    agent = HermesAgent(
        config=_cfg(hermes_home=str(home), agent_name="fallback"),
        server_client=MagicMock(spec=ServerClient),
    )
    assert agent.config.agent_name == "fallback"


def test_overlay_only_consulted_when_hermes_home_set(tmp_path: Path) -> None:
    """If config.hermes_home is None we don't look for an overlay anywhere —
    avoids accidentally picking up an overlay from the user's ~/.hermes."""
    # Write an overlay at ~/.hermes/hermes_agent_overlay.json equivalent —
    # by *not* setting config.hermes_home we should ignore it entirely.
    # We can't safely mutate ~/.hermes in a test, so just assert the agent
    # constructs cleanly with no hermes_home and no overlay-derived fields.
    agent = HermesAgent(
        config=_cfg(hermes_home=None, agent_name="no-home"),
        server_client=MagicMock(spec=ServerClient),
    )
    assert agent.config.agent_name == "no-home"
    assert agent.config.peer_agents is None


def test_overlay_not_a_dict_is_warned_not_raised(tmp_path: Path) -> None:
    home = tmp_path / "list_overlay"
    home.mkdir()
    (home / "hermes_agent_overlay.json").write_text(json.dumps(["accidentally", "a", "list"]))
    agent = HermesAgent(
        config=_cfg(hermes_home=str(home), agent_name="still ok"),
        server_client=MagicMock(spec=ServerClient),
    )
    assert agent.config.agent_name == "still ok"
