# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""T0.3 / T0.4 — /task adapter for CallAgentTool compatibility.

The worker contract is documented at
``nemo_skills.mcp.servers.agent_tool.CallAgentTool._http_call``:

    POST {url}/task
    body: {"task_id": str, "messages": [{"role": "user", "content": str}, ...]}
    reply: {"generation": str, "error": str?}

These tests stub out ``HermesAgent.responses`` so we exercise the adapter
plumbing without booting an LLM.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent.app import (
    HermesAgent,
    HermesAgentConfig,
    ModelServerRef,
    ResourcesServerRef,
    TaskRequest,
    TaskResponse,
)


def _config(**kwargs) -> HermesAgentConfig:
    return HermesAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="hermes_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="dummy"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        **kwargs,
    )


def _fake_request(*, task_id_on_state: bool = False):
    """Minimal Request stand-in: only ``.state`` and ``.cookies`` are touched."""

    class _State:
        pass

    req = MagicMock()
    req.state = _State()
    if task_id_on_state:
        req.state.task_id = "preset"
    req.cookies = {}
    return req


def _mk_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0,
        model="m",
        object="response",
        output=[
            NeMoGymResponseOutputMessageForTraining(
                id="msg-0",
                content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                role="assistant",
                status="completed",
                type="message",
                prompt_token_ids=[1, 2],
                generation_token_ids=[3, 4],
                generation_log_probs=[0.0, 0.0],
            )
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        usage=NeMoGymResponseUsage(
            input_tokens=0,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
            output_tokens=0,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=0,
        ),
    )


# ---------------------------------------------------------------------------
# T0.3
# ---------------------------------------------------------------------------


def test_task_endpoint_translates_payload_and_returns_generation() -> None:
    """A standard CallAgentTool payload yields the last assistant text."""
    agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

    captured = {}

    async def fake_responses(request, body):
        captured["body"] = body
        captured["task_id_on_state"] = getattr(request.state, "task_id", None)
        return _mk_response("the answer is 42")

    # HermesAgent is a pydantic model — bypass field validation to override
    # an instance method for the test.  Same trick is used elsewhere in the
    # NeMo-Gym test suite to stub server methods.
    object.__setattr__(agent, "responses", fake_responses)

    body = TaskRequest(
        task_id="task-abc",
        messages=[{"role": "user", "content": "what is the answer?"}],
    )
    resp = asyncio.run(agent.task(_fake_request(), body))
    # The CallAgentTool wire format is what we assert against.
    assert isinstance(resp, TaskResponse)
    assert resp.generation == "the answer is 42"
    assert resp.error == ""

    # The body forwarded to responses() carries the messages as gym input items.
    forwarded = captured["body"]
    assert hasattr(forwarded, "input")
    assert len(forwarded.input) == 1
    item = forwarded.input[0]
    role = getattr(item, "role", None) or item.get("role")
    content = getattr(item, "content", None) or item.get("content")
    assert role == "user"
    assert content == "what is the answer?"

    # task_id is stashed on request.state so traces key by it.
    assert captured["task_id_on_state"] == "task-abc"


def test_task_endpoint_propagates_multiple_messages() -> None:
    agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

    captured = {}

    async def fake_responses(request, body):
        captured["body"] = body
        return _mk_response("ok")

    # HermesAgent is a pydantic model — bypass field validation to override
    # an instance method for the test.  Same trick is used elsewhere in the
    # NeMo-Gym test suite to stub server methods.
    object.__setattr__(agent, "responses", fake_responses)

    body = TaskRequest(
        task_id="t",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hi back"},
            {"role": "user", "content": "follow-up"},
        ],
    )
    resp = asyncio.run(agent.task(_fake_request(), body))
    assert resp.generation == "ok"
    assert len(captured["body"].input) == 3


def test_task_endpoint_returns_error_on_underlying_failure() -> None:
    agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

    async def boom(request, body):
        raise RuntimeError("model server down")

    object.__setattr__(agent, "responses", boom)

    body = TaskRequest(task_id="t", messages=[{"role": "user", "content": "hi"}])
    resp = asyncio.run(agent.task(_fake_request(), body))
    assert resp.generation == ""
    assert "RuntimeError" in resp.error
    assert "model server down" in resp.error


# ---------------------------------------------------------------------------
# T0.4 — /task is a peer-to-peer endpoint and MUST NOT touch the
# resources_server.  Otherwise multi-agent runs would double-verify and
# corrupt session state.
# ---------------------------------------------------------------------------


def test_task_endpoint_does_not_call_resources_server() -> None:
    sc = MagicMock(spec=ServerClient)
    sc.post = AsyncMock()  # would raise if called
    agent = HermesAgent(config=_config(), server_client=sc)

    async def fake_responses(request, body):
        return _mk_response("done")

    # HermesAgent is a pydantic model — bypass field validation to override
    # an instance method for the test.  Same trick is used elsewhere in the
    # NeMo-Gym test suite to stub server methods.
    object.__setattr__(agent, "responses", fake_responses)

    body = TaskRequest(task_id="t", messages=[{"role": "user", "content": "hi"}])
    resp = asyncio.run(agent.task(_fake_request(), body))
    assert resp.error == ""
    # The resources_server / verify path is owned by the orchestrator's /run,
    # not by peer-agent /task — no upstream HTTP must be issued from /task.
    sc.post.assert_not_called()


def test_setup_webserver_registers_task_route() -> None:
    agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
    app = agent.setup_webserver()
    routes = {(r.path, tuple(sorted(r.methods))) for r in app.routes if hasattr(r, "methods")}
    assert ("/task", ("POST",)) in routes
    # Base class routes still present (regression guard).
    assert ("/v1/responses", ("POST",)) in routes
    assert ("/run", ("POST",)) in routes


@pytest.mark.parametrize("task_id", [None, ""])
def test_task_endpoint_without_task_id_still_works(task_id) -> None:
    agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

    async def fake_responses(request, body):
        return _mk_response("ok")

    # HermesAgent is a pydantic model — bypass field validation to override
    # an instance method for the test.  Same trick is used elsewhere in the
    # NeMo-Gym test suite to stub server methods.
    object.__setattr__(agent, "responses", fake_responses)

    body = TaskRequest(task_id=task_id, messages=[{"role": "user", "content": "hi"}])
    resp = asyncio.run(agent.task(_fake_request(), body))
    assert resp.generation == "ok"
    assert resp.error == ""
