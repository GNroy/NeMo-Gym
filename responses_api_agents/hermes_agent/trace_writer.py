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

"""Thread-safe JSONL trace writer for Hermes AIAgent callbacks.

Hermes' ``AIAgent.run_conversation`` runs synchronously in a worker thread
(see ``HermesAgent.responses`` -> ``asyncio.to_thread``), and the agent fires
its observer callbacks from inside that thread.  ``build_callbacks`` returns
a dict of callbacks that conform to the loosely-typed callback signatures
in ``hermes-agent/agent/conversation_loop.py`` and ``tool_executor.py`` —
all use ``(*args, **kwargs)`` so we stay forward-compatible across hermes
pins.

One file per ``(agent_name, session_id)`` pair under ``trace_dir``.
Concurrent writes from one process are serialized by a per-file
``threading.Lock``; cross-process writes are serialized by OS-level append
semantics (each line is one short ``write()``).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional


LOG = logging.getLogger(__name__)

# All trace_writer instances in one process share this registry so callbacks
# for the same (trace_dir, session_id) write to the same locked file handle.
_WRITER_LOCK = threading.Lock()
_WRITERS: Dict[str, "_TraceFile"] = {}


class _TraceFile:
    """One append-only JSONL file with a per-file lock."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def write_event(self, event: Dict[str, Any]) -> None:
        line = json.dumps(event, default=_json_default, ensure_ascii=False)
        # One lock-guarded write per event keeps lines from interleaving and
        # keeps the file readable even if the process is killed mid-run.
        with self.lock:
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError as exc:
                # Logging here is best-effort — a broken trace sink must not
                # take down the rollout.
                LOG.warning("trace_writer: failed to append to %s: %s", self.path, exc)


def _json_default(obj: Any) -> Any:
    """Serialize objects the json module doesn't natively understand.

    Pydantic models, bytes, sets, anything else with ``__dict__`` — all
    coerced rather than raising, because traces are best-effort observability.
    """
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return repr(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return repr(obj)


def _truncate(value: Any, limit: int = 2000) -> Any:
    """Truncate string-ish payloads so traces stay bounded."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"...[+{len(value) - limit} chars]"
    if isinstance(value, dict):
        return {k: _truncate(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v, limit) for v in value]
    return value


def _get_writer(trace_dir: Path, session_id: str) -> _TraceFile:
    key = f"{trace_dir}|{session_id}"
    with _WRITER_LOCK:
        writer = _WRITERS.get(key)
        if writer is None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            writer = _TraceFile(trace_dir / f"{session_id}.jsonl")
            _WRITERS[key] = writer
        return writer


def _reset_writers_for_tests() -> None:
    """Test-only: clear the process-wide writer cache."""
    with _WRITER_LOCK:
        _WRITERS.clear()


def build_callbacks(
    trace_dir: str | Path,
    agent_name: str,
    session_id: str,
    *,
    payload_char_limit: int = 2000,
) -> Dict[str, Callable[..., None]]:
    """Build a dict of AIAgent callbacks that emit JSONL events.

    The returned dict is passed straight into ``AIAgent(...)`` as keyword
    arguments; AIAgent will call only the callbacks it knows about.  Every
    callback uses ``(*args, **kwargs)`` so future hermes-agent versions can
    add positional/keyword args without breaking us.

    Events have shape::

        {
            "ts": <float monotonic-ish epoch seconds>,
            "agent": <agent_name>,
            "session_id": <session_id>,
            "kind": <event kind>,
            "payload": {...}
        }
    """
    trace_dir = Path(trace_dir)
    writer = _get_writer(trace_dir, session_id)

    def _emit(kind: str, payload: Dict[str, Any]) -> None:
        event = {
            "ts": time.time(),
            "agent": agent_name,
            "session_id": session_id,
            "kind": kind,
            "payload": _truncate(payload, payload_char_limit),
        }
        writer.write_event(event)

    def tool_start_callback(*args: Any, **kwargs: Any) -> None:
        # Observed signature: (tc_id, name, args)
        tc_id = args[0] if len(args) > 0 else kwargs.get("tc_id")
        name = args[1] if len(args) > 1 else kwargs.get("name")
        tool_args = args[2] if len(args) > 2 else kwargs.get("args")
        _emit("tool_start", {"tc_id": tc_id, "tool_name": name, "args": tool_args})

    def tool_complete_callback(*args: Any, **kwargs: Any) -> None:
        # Observed signature: (tc_id, name, args, result)
        tc_id = args[0] if len(args) > 0 else kwargs.get("tc_id")
        name = args[1] if len(args) > 1 else kwargs.get("name")
        tool_args = args[2] if len(args) > 2 else kwargs.get("args")
        result = args[3] if len(args) > 3 else kwargs.get("result")
        _emit(
            "tool_complete",
            {"tc_id": tc_id, "tool_name": name, "args": tool_args, "result": result},
        )

    def tool_progress_callback(*args: Any, **kwargs: Any) -> None:
        # Observed signatures vary: (event_type, name, preview, args) or
        # (event_type, name) or (event_type, preview).  Record everything.
        _emit(
            "tool_progress",
            {"args": list(args), "kwargs": {k: v for k, v in kwargs.items()}},
        )

    def step_callback(*args: Any, **kwargs: Any) -> None:
        # Observed signature: (api_call_count, prev_tools)
        api_call_count = args[0] if len(args) > 0 else kwargs.get("api_call_count")
        prev_tools = args[1] if len(args) > 1 else kwargs.get("prev_tools")
        _emit(
            "step",
            {"api_call_count": api_call_count, "prev_tools": prev_tools},
        )

    def thinking_callback(*args: Any, **kwargs: Any) -> None:
        # Observed signature: (msg,)
        msg = args[0] if len(args) > 0 else kwargs.get("msg")
        if not msg:
            return  # AIAgent emits empty strings to clear the spinner — skip
        _emit("thinking", {"msg": msg})

    def reasoning_callback(*args: Any, **kwargs: Any) -> None:
        _emit("reasoning", {"args": list(args), "kwargs": dict(kwargs)})

    def interim_assistant_callback(*args: Any, **kwargs: Any) -> None:
        _emit("assistant_interim", {"args": list(args), "kwargs": dict(kwargs)})

    def status_callback(*args: Any, **kwargs: Any) -> None:
        # Observed signature: (severity, message)
        severity = args[0] if len(args) > 0 else kwargs.get("severity")
        message = args[1] if len(args) > 1 else kwargs.get("message")
        _emit("status", {"severity": severity, "message": message})

    return {
        "tool_start_callback": tool_start_callback,
        "tool_complete_callback": tool_complete_callback,
        "tool_progress_callback": tool_progress_callback,
        "step_callback": step_callback,
        "thinking_callback": thinking_callback,
        "reasoning_callback": reasoning_callback,
        "interim_assistant_callback": interim_assistant_callback,
        "status_callback": status_callback,
    }


def emit_event(
    trace_dir: str | Path,
    agent_name: str,
    session_id: str,
    kind: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    payload_char_limit: int = 2000,
) -> None:
    """Direct emission helper for non-AIAgent events (e.g. session start/end).

    Used by ``HermesAgent.responses`` to bracket each conversation with
    ``session_start``/``session_end`` rows so the trace file is self-contained
    even when no callbacks fired (e.g. the agent ended on iteration 0).
    """
    writer = _get_writer(Path(trace_dir), session_id)
    event = {
        "ts": time.time(),
        "agent": agent_name,
        "session_id": session_id,
        "kind": kind,
        "payload": _truncate(payload or {}, payload_char_limit),
    }
    writer.write_event(event)
