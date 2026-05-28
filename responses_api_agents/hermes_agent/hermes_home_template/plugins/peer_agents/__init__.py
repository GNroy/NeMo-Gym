# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""``peer_agents`` Hermes plugin — register ``call_<name>`` peer-call tools.

Source of truth: ``<HERMES_HOME>/hermes_agent_overlay.json`` written by
``ns hermes_agent_rollouts``.  Its ``peer_agents`` key (orchestrator
overlay only in Phase 1) is a mapping::

    {
        "analyst": {"het_group": 0, "port": 7777},
        "coder":   {"het_group": 1, "port": 7777}
    }

For each peer, we register a tool ``call_<name>`` that the LLM invokes
with a single ``task`` string.  At dispatch time the handler resolves
``host`` from ``SLURM_MASTER_NODE_HET_GROUP_<het_group>`` (falling back
to ``localhost`` for unit tests / local mode) and POSTs::

    POST http://{host}:{port}/task
    body: {"task_id": "<uuid>", "messages": [{"role": "user", "content": task}]}
    reply: {"generation": str, "error": str}

The HermesAgent at the peer side returns the last assistant text via the
``/task`` adapter added in Phase 0.

Why lazy hostname resolution?  SLURM_MASTER_NODE_HET_GROUP_N is only set
once NeMo-Run launches the job; in unit-test mode the env var is absent
and we transparently fall back to ``localhost``.  Resolving inside the
handler also lets the same plugin work across job restarts where SLURM
may pick a different host on re-queue.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# Per-call HTTP timeout for the peer ``/task`` POST.  Generous — workers
# may take many turns to answer; the orchestrator's own context-window
# pressure is the real limiter.
_DEFAULT_TIMEOUT_S = 600.0


# Session id used for the orchestrator-side delegation trace.  Per-call
# session_id is not surfaced to plugin handlers by hermes-agent, so we use a
# single shared file per orchestrator process; the fleet renderer joins
# delegation events with worker session events by ticket_id (= task_id).
_DELEGATION_SESSION_ID = "delegations"


# ---------------------------------------------------------------------------
# Overlay loader
# ---------------------------------------------------------------------------


def _load_overlay() -> Dict[str, Any]:
    """Return the parsed overlay JSON dict (or ``{}`` if absent/malformed)."""
    hermes_home = os.environ.get("HERMES_HOME", "").strip()
    if not hermes_home:
        try:
            # Fall back to hermes_constants when running inside a Hermes
            # process that uses the ContextVar override.
            from hermes_constants import get_hermes_home  # type: ignore

            hermes_home = str(get_hermes_home())
        except Exception:
            return {}

    overlay_path = os.path.join(hermes_home, "hermes_agent_overlay.json")
    if not os.path.isfile(overlay_path):
        return {}
    try:
        with open(overlay_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("peer_agents: could not read overlay %s: %s", overlay_path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _load_peer_agents() -> Dict[str, Dict[str, Any]]:
    """Read the peer-agents map from the active HERMES_HOME's overlay JSON.

    Returns an empty dict if the file is absent, malformed, or carries no
    ``peer_agents`` key.  An empty dict means *register no tools* — the
    Hermes plugin loader treats that as a soft no-op.
    """
    data = _load_overlay()
    peers = data.get("peer_agents") if isinstance(data, dict) else None
    if not isinstance(peers, dict):
        return {}
    # Normalize entries — strip anything that doesn't look like
    # ``{het_group, port}`` so a typo in the manifest doesn't register a
    # broken tool whose first invocation silently dies.
    normalized: Dict[str, Dict[str, Any]] = {}
    for name, entry in peers.items():
        if not isinstance(entry, dict):
            logger.warning("peer_agents: entry %r is not a mapping; ignoring", name)
            continue
        if "het_group" not in entry or "port" not in entry:
            logger.warning("peer_agents: entry %r missing het_group/port; ignoring", name)
            continue
        normalized[name] = {
            "het_group": int(entry["het_group"]),
            "port": int(entry["port"]),
            # Optional caller-supplied override; falls back to the
            # SLURM env var at handler time when absent.
            "host": entry.get("host"),
            "timeout_s": float(entry.get("timeout_s", _DEFAULT_TIMEOUT_S)),
        }
    return normalized


# ---------------------------------------------------------------------------
# URL resolution + HTTP call
# ---------------------------------------------------------------------------


def _resolve_host(peer: Dict[str, Any]) -> str:
    """Resolve a peer's hostname from the het-group env var (lazy)."""
    if peer.get("host"):
        return str(peer["host"])
    env_key = f"SLURM_MASTER_NODE_HET_GROUP_{peer['het_group']}"
    host = os.environ.get(env_key, "").strip()
    return host or "localhost"


def _emit_delegation(kind_payload: Dict[str, Any]) -> None:
    """Best-effort delegation-event emit into the orchestrator's trace tree.

    Skips silently when the overlay does not declare ``trace_dir`` /
    ``agent_name`` (e.g. unit tests, or a single-agent benchmark run).
    Tracing must never take down a delegation, so any IO/Import error is
    swallowed with a debug log line.
    """
    try:
        overlay = _load_overlay()
        trace_dir = overlay.get("trace_dir")
        agent_name = overlay.get("agent_name") or "orchestrator"
        if not trace_dir:
            return
        from responses_api_agents.hermes_agent.trace_writer import emit_event

        emit_event(
            trace_dir,
            agent_name,
            _DELEGATION_SESSION_ID,
            "delegation",
            kind_payload,
        )
    except Exception as exc:  # noqa: BLE001 - tracing is best-effort
        logger.debug("peer_agents: delegation emit failed: %s", exc)


def _call_peer(
    peer_name: str,
    peer: Dict[str, Any],
    task: str,
    *,
    ticket_id: Optional[str] = None,
) -> Dict[str, Any]:
    """POST a task to a peer's ``/task`` endpoint, return ``{generation, error}``."""
    host = _resolve_host(peer)
    url = f"http://{host}:{peer['port']}/task"
    # Reuse the supplied ticket_id as the wire task_id so the worker's
    # ``session_id`` (from HermesAgent._resolve_session_id) matches the
    # ticket_id this orchestrator emits in its delegation event — that
    # equality is the join key for the fleet trace renderer.
    task_id = ticket_id or uuid.uuid4().hex
    payload = {
        "task_id": task_id,
        "messages": [{"role": "user", "content": task}],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=peer["timeout_s"]) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return {"generation": "", "error": f"HTTP {exc.code} from {peer_name}: {exc.reason}"}
    except urllib.error.URLError as exc:
        return {"generation": "", "error": f"could not reach {peer_name} at {url}: {exc.reason}"}
    except Exception as exc:  # noqa: BLE001 - any failure becomes a tool error
        return {"generation": "", "error": f"{type(exc).__name__}: {exc}"}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"generation": "", "error": f"non-JSON reply from {peer_name}: {raw[:200]}"}
    if not isinstance(parsed, dict):
        return {"generation": "", "error": f"unexpected reply shape from {peer_name}"}
    return {
        "generation": str(parsed.get("generation", "") or ""),
        "error": str(parsed.get("error", "") or ""),
    }


# ---------------------------------------------------------------------------
# Tool handler factory
# ---------------------------------------------------------------------------


def _make_call_tool_handler(peer_name: str, peer: Dict[str, Any]):
    """Return a Hermes tool handler (``args, **kw`` → JSON string) for one peer."""

    def handler(args: Dict[str, Any], **_kw: Any) -> str:
        task = args.get("task") if isinstance(args, dict) else None
        if not isinstance(task, str) or not task:
            return json.dumps({"success": False, "error": f"call_{peer_name} requires non-empty 'task' string"})

        ticket_id = uuid.uuid4().hex
        task_preview = task[:200].replace("\n", " ")
        _emit_delegation(
            {
                "mode": "blocking",
                "worker": peer_name,
                "ticket_id": ticket_id,
                "task_preview": task_preview,
            }
        )

        result = _call_peer(peer_name, peer, task, ticket_id=ticket_id)

        # Pair the dispatch with a result event so the renderer can show
        # both the call and its outcome in line with the orchestrator's
        # other tool calls.  ``ok=False`` carries the peer's error message.
        result_text = result["generation"] if not result["error"] else result["error"]
        _emit_delegation(
            {
                "mode": "result",
                "worker": peer_name,
                "ticket_id": ticket_id,
                "ok": not result["error"],
                "result_preview": (result_text or "")[:200],
            }
        )

        # Hermes tools must return a JSON-encoded string; wrap into a
        # success/result/error envelope to match the in-tree tool convention.
        if result["error"]:
            return json.dumps({"success": False, "error": result["error"]})
        return json.dumps({"success": True, "result": result["generation"]})

    return handler


def _tool_schema_for(peer_name: str) -> Dict[str, Any]:
    """Return the OpenAI-style schema the LLM sees for ``call_<peer_name>``."""
    return {
        "name": f"call_{peer_name}",
        "description": (
            f"Delegate a task to the '{peer_name}' peer agent and return its "
            f"final assistant text.  Pass the full task description as the "
            f"'task' argument; the peer will return one synchronous answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "The complete task description for the peer.  Be specific — "
                        "include all context the peer needs; the peer cannot read "
                        "this conversation's earlier turns."
                    ),
                }
            },
            "required": ["task"],
            "additionalProperties": False,
        },
    }


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Hermes plugin loader entry point.

    Called once at Hermes startup with a registration context that exposes
    ``register_tool(...)``.  We enumerate the peer_agents map and add one
    ``call_<peer>`` tool per entry, all under the ``peer_agents`` toolset.

    No-ops cleanly when:
      * HERMES_HOME is unset (running outside the per-agent snapshot)
      * the overlay JSON is missing (single-agent benchmark mode)
      * ``peer_agents`` is empty (worker process, not the orchestrator)
    """
    peers = _load_peer_agents()
    if not peers:
        logger.debug("peer_agents: no peers configured — no tools registered")
        return
    for peer_name, peer in peers.items():
        ctx.register_tool(
            name=f"call_{peer_name}",
            toolset="peer_agents",
            schema=_tool_schema_for(peer_name),
            handler=_make_call_tool_handler(peer_name, peer),
            description=f"Delegate a task to the '{peer_name}' peer agent.",
        )
    logger.info("peer_agents: registered %d peer tool(s): %s", len(peers), sorted(peers))
