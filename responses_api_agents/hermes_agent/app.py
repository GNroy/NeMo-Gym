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

import asyncio
import json
import logging
import os
import sys
from asyncio import Semaphore
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import model_tools  # noqa: F401  # fail-fast if hermes-agent isn't installed
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


def _trajectory_to_output_items(messages, n_input):
    output_items = []
    for item in messages[n_input:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "") or ""
        if isinstance(content, list):
            content = "".join(c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "") for c in content)
        if role == "assistant":
            output_items.append(
                NeMoGymResponseOutputMessageForTraining(
                    id=f"msg-{len(output_items)}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=content, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                    prompt_token_ids=item.get("prompt_token_ids") or [],
                    generation_token_ids=item.get("generation_token_ids") or [],
                    generation_log_probs=item.get("generation_log_probs") or [],
                )
            )
            for tc in item.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if not fn:
                    continue
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=fn.get("arguments", ""),
                        call_id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        type="function_call",
                        id=tc.get("id"),
                        status="completed",
                    )
                )
        elif role == "tool":
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=item.get("tool_call_id", ""),
                    output=content,
                    status="completed",
                )
            )
    return output_items


LOG = logging.getLogger(__name__)


# if ray close sys.stderr mid-request, write to the original fd
class _SafeStderrHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = sys.__stderr__
            if stream is None:
                return
            stream.write(msg + "\n")
            stream.flush()
        except Exception:
            pass


if not LOG.handlers:
    LOG.addHandler(_SafeStderrHandler(level=logging.WARNING))


def _split_input_to_user_and_history(input_items) -> tuple[str, list[dict], Optional[str]]:
    items = list(input_items)
    system_message: Optional[str] = None
    if items:
        first = items[0]
        first_role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        first_content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
        if first_role == "system":
            if isinstance(first_content, list):
                first_content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in first_content
                )
            system_message = first_content or ""
            items = items[1:]

    user_message = ""
    history: list[dict] = []
    for idx, item in enumerate(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        if isinstance(content, list):
            content = "".join((p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content)
        content = content or ""
        if idx == len(items) - 1 and role == "user":
            user_message = content
        else:
            history.append({"role": role, "content": content})
    return user_message, history, system_message


class HermesAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    concurrency: int = 32
    max_turns: int = 30
    enabled_toolsets: Optional[list[str]] = None
    disabled_toolsets: Optional[list[str]] = None
    temperature: float = 1.0
    terminal_backend: str = "local"
    terminal_timeout: int = 60
    system_prompt: Optional[str] = None
    # --- Phase 0 additions --------------------------------------------------
    # Each agent's HERMES_HOME (per-run snapshot of memory/skills/sessions).
    # When set, ``model_post_init`` exports it via both env var (for hermes
    # subprocesses) and the ContextVar override (for in-process callers).
    # Defaults to ``None`` so existing benchmarks keep the historic
    # ``~/.hermes`` fallback behavior.
    hermes_home: Optional[str] = None
    # Persistence flags forwarded to ``AIAgent``.  Defaults preserve the
    # original benchmark behavior (everything off — sessions are stateless).
    # The new pipeline turns these on when ``hermes_home`` points at a
    # job-local snapshot.
    persist_memory: bool = False
    persist_skills: bool = False
    persist_session: bool = False
    save_trajectories: bool = False
    enable_compression: bool = False
    # JSONL trace sink.  When set, ``trace_writer.build_callbacks`` is wired
    # into ``AIAgent`` for every conversation; events land in
    # ``<trace_dir>/<session_id>.jsonl``.  ``agent_name`` defaults to
    # ``config.name`` so multi-agent fleets get distinct trace identities.
    trace_dir: Optional[str] = None
    agent_name: Optional[str] = None
    # --- Phase 0.5 additions ------------------------------------------------
    # Peer-agent map (orchestrator-facing).  Keys are logical agent names,
    # values are ``{het_group: int, port: int}`` entries.  At runtime, the
    # bundled ``peer_agents`` Hermes plugin (see
    # ``hermes_home_template/plugins/peer_agents/``) registers one
    # ``call_<name>`` tool per entry; URLs are resolved lazily from
    # ``SLURM_MASTER_NODE_HET_GROUP_N`` env vars so the same overlay JSON
    # works regardless of which node SLURM picks for each het-group.
    peer_agents: Optional[Dict[str, Dict[str, Any]]] = None


class HermesAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class HermesAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class _TaskMessage(BaseModel):
    role: str
    content: str


class TaskRequest(BaseModel):
    """Payload accepted by the ``/task`` adapter endpoint.

    Matches ``nemo_skills.mcp.servers.agent_tool.CallAgentTool._http_call``:
    ``{"task_id": str, "messages": [{"role": "user", "content": str}, ...]}``.
    The worker runs one conversation and returns the last assistant text.
    """

    model_config = ConfigDict(extra="allow")
    task_id: Optional[str] = None
    messages: List[_TaskMessage]


class TaskResponse(BaseModel):
    """Reply shape consumed by ``CallAgentTool``.

    ``generation`` is the worker's final assistant text (may be empty).
    ``error`` is non-empty only on failure; ``CallAgentTool`` propagates
    it to the orchestrator as a tool error.
    """

    generation: str = ""
    error: str = ""


def _extract_last_assistant_text(resp: "NeMoGymResponse") -> str:
    """Return the text of the last assistant message in a NeMoGymResponse.

    Walks the response's output items in order; the last assistant
    ``output_text`` content wins.  Returns empty string if no assistant
    message was emitted (matches ``HermesAgent.responses`` padding behavior).
    """
    last = ""
    for item in resp.output or []:
        if getattr(item, "type", None) != "message":
            continue
        if getattr(item, "role", None) != "assistant":
            continue
        for chunk in getattr(item, "content", None) or []:
            text = getattr(chunk, "text", None)
            if text:
                last = text
    return last


class HermesAgent(SimpleResponsesAPIAgent):
    config: HermesAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        # hermes-agent reads these from env (cli.py / batch_runner.py); env vars are
        # process-global, so multiple HermesAgent instances in one process share them
        os.environ["TERMINAL_ENV"] = self.config.terminal_backend
        os.environ["TERMINAL_TIMEOUT"] = str(self.config.terminal_timeout)

        # HERMES_HOME isolation.  Setting the env var covers any hermes-agent
        # subprocess (terminal backends, MCP stdio servers, cron workers).
        # The ContextVar override additionally scopes get_hermes_home() within
        # this process — defense in depth so a future multi-agent-in-one-process
        # mode doesn't silently collide on the global env.
        if self.config.hermes_home:
            os.environ["HERMES_HOME"] = self.config.hermes_home
            try:
                from hermes_constants import set_hermes_home_override

                set_hermes_home_override(self.config.hermes_home)
            except ImportError:
                # hermes-agent not on path (e.g. unit tests using only this
                # module's helpers) — the env var still routes subprocesses.
                LOG.debug("hermes_constants not importable; relying on HERMES_HOME env var only")

            # Phase 0.5: merge the pipeline-written overlay JSON.
            # The ns hermes_agent_rollouts pipeline writes
            # ``<hermes_home>/hermes_agent_overlay.json`` from its manifest —
            # carrying agent_name, trace_dir, persistence flags, and the
            # orchestrator's peer_agents map.  We apply it as field-level
            # overrides on top of the YAML config so Hydra config_paths and
            # the manifest can coexist without one having to mirror the other.
            self._apply_overlay_file(Path(self.config.hermes_home) / "hermes_agent_overlay.json")

    def _apply_overlay_file(self, overlay_path: "Path") -> None:
        """Merge JSON overlay fields into ``self.config``.

        No-op if the file does not exist (the field was set via the existing
        Hydra path, or no overlay is in use).  Unknown keys are ignored with
        a warning to keep the contract additive — adding a new overlay key
        from the pipeline never breaks an older agent build.
        """
        if not overlay_path.is_file():
            return
        try:
            data = json.loads(overlay_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("hermes_agent: failed to read overlay %s: %s", overlay_path, exc)
            return
        if not isinstance(data, dict):
            LOG.warning("hermes_agent: overlay %s is not a JSON object; ignoring", overlay_path)
            return

        known = set(HermesAgentConfig.model_fields)
        for key, value in data.items():
            if key not in known:
                LOG.warning("hermes_agent: overlay key %r is not a known field; ignoring", key)
                continue
            # Pydantic v2 models forbid attribute assignment unless
            # ``validate_assignment`` is set; bypass via object.__setattr__ —
            # safe because the overlay only writes recognised fields and the
            # values are forwarded verbatim to AIAgent / trace_writer downstream.
            object.__setattr__(self.config, key, value)
            LOG.debug("hermes_agent: overlay set config.%s", key)

    def _resolve_model_base_url(self) -> str:
        # aiagent builds its own openai client; resolve policy_model url
        model_server_cfg = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.model_server.name,
        )
        base = self.server_client._build_server_base_url(model_server_cfg)
        return f"{base}/v1"

    def setup_webserver(self) -> FastAPI:
        """Extend the base server with a ``/task`` adapter for CallAgentTool.

        ``CallAgentTool._http_call`` (nemo_skills.mcp.servers.agent_tool) POSTs
        ``{"task_id", "messages": [{"role": "user", "content": ...}]}`` to
        ``{url}/task`` and expects ``{"generation": ..., "error": ...}``.
        We translate that to the existing ``responses()`` flow and extract the
        last assistant text — no resources_server seed/verify happens here;
        the worker is a peer agent, not a self-contained rollout.
        """
        app = super().setup_webserver()
        app.post("/task")(self.task)
        return app

    def _resolve_session_id(self, request: Request) -> str:
        """Pick a deterministic session_id for trace files.

        Order of precedence:
          1. ``request.state.task_id`` (set by ``/task`` from ``CallAgentTool``)
          2. Freshly minted UUID — guarantees a unique trace file per call.

        The ``isinstance(..., str)`` check is deliberate: ``MagicMock``-backed
        ``Request`` objects in unit tests resolve every attribute to a
        ``MagicMock``, which would otherwise pass a naive truthy check and
        emit unreadable trace filenames.
        """
        state = getattr(request, "state", None)
        task_id = getattr(state, "task_id", None) if state is not None else None
        if isinstance(task_id, str) and task_id:
            return task_id
        return f"sess_{uuid4().hex}"

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        from run_agent import AIAgent  # from hermes-agent on path

        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, history, input_system = _split_input_to_user_and_history(body.input)
        system_message = self.config.system_prompt or input_system

        base_url = self._resolve_model_base_url()
        model_name = str(self.config.model_server.name)

        session_id = self._resolve_session_id(request)
        agent_name = self.config.agent_name or self.config.name or "hermes_agent"

        # Wire up optional JSONL trace callbacks before constructing AIAgent.
        # Importing lazily so a missing trace_dir keeps this module importable
        # in environments that don't ship the trace_writer (unit tests of the
        # helper functions, schema validators, etc.).
        agent_kwargs: Dict[str, Any] = {}
        if self.config.trace_dir:
            from responses_api_agents.hermes_agent.trace_writer import (
                build_callbacks,
                emit_event,
            )

            agent_kwargs.update(build_callbacks(self.config.trace_dir, agent_name, session_id))
            emit_event(
                self.config.trace_dir,
                agent_name,
                session_id,
                "session_start",
                {
                    "model": model_name,
                    "max_turns": self.config.max_turns,
                    "history_len": len(history),
                    "user_message_preview": (user_message or "")[:200],
                },
            )
        else:
            emit_event = None  # type: ignore[assignment]

        agent = AIAgent(
            base_url=base_url,
            api_key="gym",  # pragma: allowlist secret
            model=model_name,
            use_streaming=False,
            temperature=self.config.temperature,
            insert_reasoning=True,
            max_iterations=self.config.max_turns,
            enabled_toolsets=self.config.enabled_toolsets,
            disabled_toolsets=self.config.disabled_toolsets,
            quiet_mode=True,
            # Persistence flags — defaults preserve original benchmark
            # behavior; the new pipeline turns these on per agent.
            skip_context_files=not self.config.persist_memory,
            skip_memory=not self.config.persist_memory,
            persist_session=self.config.persist_session,
            save_trajectories=self.config.save_trajectories,
            session_id=session_id,
            **agent_kwargs,
        )
        # Context compression mutates trajectories non-monotonically; keep it
        # off unless explicitly enabled — RL training needs faithful traces.
        agent.compression_enabled = self.config.enable_compression

        _original_build_api_kwargs = agent._build_api_kwargs

        def _patched_build_api_kwargs(api_messages):
            kw = _original_build_api_kwargs(api_messages)
            ctk = kw.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})
            ctk.setdefault("enable_thinking", True)
            ctk["truncate_history_thinking"] = False
            return kw

        agent._build_api_kwargs = _patched_build_api_kwargs

        try:
            result = await asyncio.to_thread(
                agent.run_conversation,
                user_message,
                system_message,
                history,
            )
        finally:
            if emit_event is not None:
                emit_event(
                    self.config.trace_dir,
                    agent_name,
                    session_id,
                    "session_end",
                    {},
                )

        messages = result.get("messages") or []
        # aiagent omits system from returned messages
        n_input = len(history) + 1

        output_items = _trajectory_to_output_items(messages, n_input)

        has_assistant_message = any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        )
        if not has_assistant_message:
            LOG.warning(
                "Hermes agent ended without an assistant message. Padding empty assistant message. This should not happen often, investigate: error=%r",
                result.get("error"),
            )
            last_valid = next(
                (
                    m
                    for m in reversed(messages)
                    if isinstance(m, dict) and m.get("role") == "assistant" and m.get("generation_token_ids")
                ),
                None,
            )
            pti = last_valid["prompt_token_ids"] if last_valid else [0]
            gti = last_valid["generation_token_ids"] if last_valid else [0]
            glp = (last_valid.get("generation_log_probs") if last_valid else None) or [0.0]
            output_items.append(
                NeMoGymResponseOutputMessageForTraining(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text=result.get("error") or "", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                    prompt_token_ids=pti,
                    generation_token_ids=gti,
                    generation_log_probs=glp,
                )
            )

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            output=output_items,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=0,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=0,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=0,
            ),
        )

    async def run(self, request: Request, body: HermesAgentRunRequest) -> HermesAgentVerifyResponse:
        async with self.sem:
            cookies = request.cookies

            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies

            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            gym_resp = NeMoGymResponse.model_validate(agent_resp_json)
            turns = sum(
                1
                for item in gym_resp.output
                if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            )
            last = gym_resp.output[-1] if gym_resp.output else None
            naturally = getattr(last, "type", None) == "message" and getattr(last, "role", None) == "assistant"

            return HermesAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )

    async def task(self, request: Request, body: TaskRequest) -> TaskResponse:
        """Peer-to-peer agent endpoint consumed by ``CallAgentTool``.

        Translates a ``{task_id, messages}`` payload into the existing
        ``responses`` flow.  Does **not** call the resources_server — workers
        are peer agents whose verification is owned by the orchestrator.
        Any exception during the underlying conversation is converted into a
        non-empty ``error`` field so the caller's tool layer can recover.
        """
        if body.task_id:
            # Stash on request.state so ``responses`` can use it as session_id
            # (trace files key by session, so the same task across agents
            # remains joinable).
            request.state.task_id = body.task_id

        create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role=m.role, content=m.content) for m in body.messages],
        )

        async with self.sem:
            try:
                resp = await self.responses(request, create_params)
            except Exception as exc:  # noqa: BLE001 - surface any failure to caller
                LOG.exception("HermesAgent.task failed: %s", exc)
                return TaskResponse(generation="", error=f"{type(exc).__name__}: {exc}")

        return TaskResponse(generation=_extract_last_assistant_text(resp), error="")


if __name__ == "__main__":
    HermesAgent.run_webserver()
