# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessageForTraining,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent.app import (
    HermesAgent,
    HermesAgentConfig,
    ModelServerRef,
    ResourcesServerRef,
    _split_input_to_user_and_history,
    _trajectory_to_output_items,
)


def _config(**kwargs) -> HermesAgentConfig:
    return HermesAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        model_server=ModelServerRef(type="responses_api_models", name=""),
        **kwargs,
    )


class TestSanity:
    def test_construct(self) -> None:
        HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

    def test_concurrency_semaphore_initialized(self) -> None:
        agent = HermesAgent(config=_config(concurrency=4), server_client=MagicMock(spec=ServerClient))
        assert agent.sem._value == 4


class TestSplitInputToUserAndHistory:
    def test_user_only(self) -> None:
        items = [NeMoGymEasyInputMessage(role="user", content="hi")]
        user, history, system = _split_input_to_user_and_history(items)
        assert user == "hi"
        assert history == []
        assert system is None

    def test_system_plus_user(self) -> None:
        items = [
            NeMoGymEasyInputMessage(role="system", content="be helpful"),
            NeMoGymEasyInputMessage(role="user", content="hi"),
        ]
        user, history, system = _split_input_to_user_and_history(items)
        assert user == "hi"
        assert history == []
        assert system == "be helpful"

    def test_history_then_user(self) -> None:
        items = [
            NeMoGymEasyInputMessage(role="user", content="first"),
            NeMoGymEasyInputMessage(role="assistant", content="reply"),
            NeMoGymEasyInputMessage(role="user", content="follow-up"),
        ]
        user, history, system = _split_input_to_user_and_history(items)
        assert user == "follow-up"
        assert history == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        assert system is None

    def test_resumed_ends_on_assistant(self) -> None:
        items = [
            NeMoGymEasyInputMessage(role="user", content="q"),
            NeMoGymEasyInputMessage(role="assistant", content="a"),
        ]
        user, history, system = _split_input_to_user_and_history(items)
        assert user == ""
        assert history == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]

    def test_dict_inputs(self) -> None:
        items = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "ok"}]
        user, history, system = _split_input_to_user_and_history(items)
        assert user == "ok"
        assert history == []
        assert system == "be brief"


class TestTrajectoryToOutputItems:
    def test_empty(self) -> None:
        assert _trajectory_to_output_items([], 0) == []

    def test_drops_input_prefix(self) -> None:
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        out = _trajectory_to_output_items(msgs, 1)
        assert len(out) == 1
        assert isinstance(out[0], NeMoGymResponseOutputMessageForTraining)

    def test_assistant_with_tokens(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": "answer",
                "prompt_token_ids": [1, 2],
                "generation_token_ids": [3, 4],
                "generation_log_probs": [0.0, -0.1],
            }
        ]
        out = _trajectory_to_output_items(msgs, 0)
        assert len(out) == 1
        assert isinstance(out[0], NeMoGymResponseOutputMessageForTraining)
        assert out[0].generation_token_ids == [3, 4]
        assert out[0].prompt_token_ids == [1, 2]

    def test_assistant_with_tool_call_and_tool_result(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "file.txt\n"},
        ]
        out = _trajectory_to_output_items(msgs, 0)
        assert len(out) == 3
        assert isinstance(out[0], NeMoGymResponseOutputMessageForTraining)
        assert isinstance(out[1], NeMoGymResponseFunctionToolCall)
        assert out[1].name == "terminal"
        assert out[1].arguments == '{"cmd":"ls"}'
        assert isinstance(out[2], NeMoGymFunctionCallOutput)
        assert out[2].call_id == "c1"
        assert out[2].output == "file.txt\n"

    def test_skips_non_dict_items(self) -> None:
        msgs = [None, "string", {"role": "assistant", "content": "ok"}]
        out = _trajectory_to_output_items(msgs, 0)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# Phase 0 additions
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_global_state(monkeypatch):
    """``HermesAgent.model_post_init`` mutates ``os.environ`` and the
    trace_writer module keeps a process-wide writer cache.  Reset both
    around every test so we don't leak state across cases."""
    # Drop any HERMES_HOME / TERMINAL_* set by earlier tests in the same
    # process before this test runs; restore on teardown via monkeypatch.
    for var in ("HERMES_HOME", "TERMINAL_ENV", "TERMINAL_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)
    try:
        from responses_api_agents.hermes_agent import trace_writer

        trace_writer._reset_writers_for_tests()
    except Exception:  # noqa: BLE001 - module not importable in some envs
        pass
    yield
    try:
        from responses_api_agents.hermes_agent import trace_writer

        trace_writer._reset_writers_for_tests()
    except Exception:  # noqa: BLE001
        pass


class _FakeAIAgent:
    """Capture AIAgent kwargs without invoking any LLM.

    ``HermesAgent.responses`` does ``from run_agent import AIAgent`` inside
    the request handler, so we install a stub module ``run_agent`` with this
    class.  ``run_conversation`` returns an empty messages list so the
    pad-empty-assistant branch fires deterministically.
    """

    instances: list["_FakeAIAgent"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.compression_enabled = None  # set by responses() after init
        _FakeAIAgent.instances.append(self)

    def _build_api_kwargs(self, api_messages):
        return {}

    def run_conversation(self, user_message, system_message, history):
        return {"messages": [], "error": None}


@pytest.fixture
def fake_aiagent(monkeypatch):
    _FakeAIAgent.instances.clear()
    # The lazy ``from run_agent import AIAgent`` inside responses() resolves
    # against ``sys.modules["run_agent"].AIAgent`` — stub the module wholesale
    # so we don't need hermes-agent installed at test time.
    fake_mod = types.ModuleType("run_agent")
    fake_mod.AIAgent = _FakeAIAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_mod)
    yield _FakeAIAgent
    _FakeAIAgent.instances.clear()


def _fake_request():
    class _State:
        pass

    req = MagicMock()
    req.state = _State()
    req.cookies = {}
    return req


def _make_agent(**cfg_kwargs):
    """HermesAgent with ``_resolve_model_base_url`` stubbed (avoid touching
    server_client.global_config_dict in unit tests).

    ``HermesAgent`` is a pydantic model, so we bypass field validation with
    ``object.__setattr__`` to replace the method.
    """
    agent = HermesAgent(config=_config(**cfg_kwargs), server_client=MagicMock(spec=ServerClient))
    object.__setattr__(agent, "_resolve_model_base_url", lambda: "http://test/v1")
    return agent


def _input_body() -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[NeMoGymEasyInputMessage(role="user", content="hello")],
    )


# T0.1 -----------------------------------------------------------------------


class TestT01DefaultFlagsAreFalse:
    """Backward-compat guard: a no-arg HermesAgentConfig() must not enable
    any persistence or trace behavior — existing benchmarks depend on this."""

    def test_persistence_flags_default_false(self) -> None:
        cfg = _config()
        assert cfg.hermes_home is None
        assert cfg.persist_memory is False
        assert cfg.persist_skills is False
        assert cfg.persist_session is False
        assert cfg.save_trajectories is False
        assert cfg.enable_compression is False
        assert cfg.trace_dir is None
        assert cfg.agent_name is None


# T0.2 -----------------------------------------------------------------------


class TestT02HermesHomeIsolation:
    """Construction-time HERMES_HOME override.

    The ContextVar branch is best-effort: if hermes-agent is not on the path
    (unit-test environments), only the env var is asserted.  When hermes
    *is* available the override should be visible in the same thread.
    """

    def test_sets_env_var(self, monkeypatch, tmp_path) -> None:
        # Clean baseline so we can detect the write.
        monkeypatch.delenv("HERMES_HOME", raising=False)
        home = tmp_path / "agent-a"
        home.mkdir()
        _make_agent(hermes_home=str(home))
        assert os.environ["HERMES_HOME"] == str(home)

    def test_unset_hermes_home_does_not_clobber_env(self, monkeypatch) -> None:
        """If config.hermes_home is None we leave any pre-existing
        HERMES_HOME alone (the launching script may have set one)."""
        monkeypatch.setenv("HERMES_HOME", "/pre-existing/home")
        _make_agent()  # hermes_home defaults to None
        assert os.environ["HERMES_HOME"] == "/pre-existing/home"

    def test_contextvar_override_visible_when_available(self, monkeypatch, tmp_path) -> None:
        try:
            from hermes_constants import get_hermes_home_override
        except ImportError:
            pytest.skip("hermes-agent not importable in this environment")
        monkeypatch.delenv("HERMES_HOME", raising=False)
        home = tmp_path / "agent-x"
        home.mkdir()
        _make_agent(hermes_home=str(home))
        assert get_hermes_home_override() == str(home)


# T0.7 / T0.8 ----------------------------------------------------------------


class TestT07T08AIAgentKwargsFromFlags:
    """Assert HermesAgent.responses constructs AIAgent with kwargs that
    match the configured persistence flags."""

    def test_t07_responses_disables_persistence_by_default(self, fake_aiagent) -> None:
        agent = _make_agent()
        asyncio.run(agent.responses(_fake_request(), _input_body()))
        assert len(fake_aiagent.instances) == 1
        kw = fake_aiagent.instances[0].kwargs
        assert kw["skip_memory"] is True  # persist_memory=False -> skip_memory=True
        assert kw["skip_context_files"] is True
        # ``persist_session`` is intentionally no longer forwarded to
        # AIAgent — NousResearch/hermes-agent dropped that constructor
        # kwarg; the agent governs session persistence via skip_memory
        # and its own _persist_session method.  We keep the config
        # field for callers (test_t07 still verifies the config default).
        assert "persist_session" not in kw
        assert kw["save_trajectories"] is False
        # Compression is set AFTER init; check the post-init attribute.
        assert fake_aiagent.instances[0].compression_enabled is False

    def test_t08_responses_enables_persistence_when_flags_true(self, fake_aiagent, tmp_path) -> None:
        home = tmp_path / "agent-with-memory"
        home.mkdir()
        agent = _make_agent(
            hermes_home=str(home),
            persist_memory=True,
            persist_session=True,
            save_trajectories=True,
            enable_compression=True,
        )
        asyncio.run(agent.responses(_fake_request(), _input_body()))
        assert len(fake_aiagent.instances) == 1
        kw = fake_aiagent.instances[0].kwargs
        assert kw["skip_memory"] is False
        assert kw["skip_context_files"] is False
        # See test_t07 — persist_session no longer flows through.
        assert "persist_session" not in kw
        assert kw["save_trajectories"] is True
        assert fake_aiagent.instances[0].compression_enabled is True

    def test_session_id_is_propagated_to_aiagent(self, fake_aiagent) -> None:
        agent = _make_agent()
        asyncio.run(agent.responses(_fake_request(), _input_body()))
        kw = fake_aiagent.instances[0].kwargs
        # Auto-generated session_id starts with "sess_" (uuid-derived).
        assert isinstance(kw["session_id"], str)
        assert kw["session_id"].startswith("sess_")

    def test_session_id_uses_request_state_task_id(self, fake_aiagent) -> None:
        agent = _make_agent()
        req = _fake_request()
        req.state.task_id = "task-from-orchestrator"
        asyncio.run(agent.responses(req, _input_body()))
        kw = fake_aiagent.instances[0].kwargs
        assert kw["session_id"] == "task-from-orchestrator"

    def test_trace_dir_writes_session_brackets(self, fake_aiagent, tmp_path) -> None:
        """When trace_dir is set, session_start/session_end fence the file
        even without any AIAgent callbacks firing."""
        agent = _make_agent(trace_dir=str(tmp_path), agent_name="alice")
        req = _fake_request()
        req.state.task_id = "trace-task"
        asyncio.run(agent.responses(req, _input_body()))
        log = (tmp_path / "trace-task.jsonl").read_text().splitlines()
        import json as _json

        kinds = [_json.loads(line)["kind"] for line in log]
        assert kinds[0] == "session_start"
        assert kinds[-1] == "session_end"
