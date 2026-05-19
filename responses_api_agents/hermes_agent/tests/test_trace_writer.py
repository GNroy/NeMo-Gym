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
"""T0.5 / T0.6 — JSONL trace writer correctness and concurrency.

These tests run with stdlib only (no hermes-agent required) because they
exercise the trace writer in isolation from ``AIAgent``.
"""

import json
import threading
from pathlib import Path

import pytest

from responses_api_agents.hermes_agent import trace_writer


@pytest.fixture(autouse=True)
def _reset_writer_cache():
    # Each test uses its own tmp_path; clear the process-wide writer cache
    # so files keyed by path don't survive across tests in the same process.
    trace_writer._reset_writers_for_tests()
    yield
    trace_writer._reset_writers_for_tests()


# ---------------------------------------------------------------------------
# T0.5
# ---------------------------------------------------------------------------


def test_trace_writer_appends_one_event_per_line(tmp_path: Path) -> None:
    cbs = trace_writer.build_callbacks(tmp_path, agent_name="alice", session_id="s1")
    cbs["tool_start_callback"]("tc-1", "terminal", {"cmd": "ls"})
    cbs["tool_complete_callback"]("tc-1", "terminal", {"cmd": "ls"}, "file.txt\n")
    cbs["step_callback"](3, ["terminal"])
    cbs["thinking_callback"]("solving...")
    # Empty-string thinking events are dropped (AIAgent uses "" to clear the
    # spinner — would otherwise spam the trace).
    cbs["thinking_callback"]("")
    cbs["status_callback"]("warn", "rate limit")

    log_path = tmp_path / "s1.jsonl"
    assert log_path.is_file()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5, lines

    events = [json.loads(line) for line in lines]
    kinds = [e["kind"] for e in events]
    assert kinds == ["tool_start", "tool_complete", "step", "thinking", "status"]
    # Every event carries the required identity fields.
    for e in events:
        assert e["agent"] == "alice"
        assert e["session_id"] == "s1"
        assert isinstance(e["ts"], float) and e["ts"] > 0
        assert "payload" in e


def test_trace_writer_routes_separate_sessions_to_separate_files(tmp_path: Path) -> None:
    cbs_a = trace_writer.build_callbacks(tmp_path, agent_name="a", session_id="alpha")
    cbs_b = trace_writer.build_callbacks(tmp_path, agent_name="a", session_id="beta")
    cbs_a["thinking_callback"]("from alpha")
    cbs_b["thinking_callback"]("from beta")
    assert (tmp_path / "alpha.jsonl").read_text().count("from alpha") == 1
    assert (tmp_path / "beta.jsonl").read_text().count("from beta") == 1
    # And there's no cross-contamination.
    assert "from alpha" not in (tmp_path / "beta.jsonl").read_text()
    assert "from beta" not in (tmp_path / "alpha.jsonl").read_text()


def test_emit_event_direct_helper(tmp_path: Path) -> None:
    """``emit_event`` is used by HermesAgent.responses to bracket the run."""
    trace_writer.emit_event(tmp_path, "agent_x", "s-direct", "session_start", {"model": "test"})
    trace_writer.emit_event(tmp_path, "agent_x", "s-direct", "session_end", {})
    log = (tmp_path / "s-direct.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 2
    assert json.loads(log[0])["kind"] == "session_start"
    assert json.loads(log[1])["kind"] == "session_end"


def test_payload_is_truncated(tmp_path: Path) -> None:
    cbs = trace_writer.build_callbacks(tmp_path, agent_name="a", session_id="trunc", payload_char_limit=50)
    long_arg = "x" * 500
    cbs["tool_complete_callback"]("tc", "echo", {}, long_arg)
    event = json.loads((tmp_path / "trunc.jsonl").read_text().splitlines()[0])
    result_repr = event["payload"]["result"]
    assert isinstance(result_repr, str)
    assert len(result_repr) < len(long_arg)
    assert "...[+" in result_repr


def test_non_serializable_payload_doesnt_crash(tmp_path: Path) -> None:
    """A callback firing with an exotic object must not bring down the run."""

    class _Weird:
        def __init__(self) -> None:
            self.cmd = "ls"

    cbs = trace_writer.build_callbacks(tmp_path, agent_name="a", session_id="weird")
    cbs["tool_start_callback"]("tc", "terminal", _Weird())
    event = json.loads((tmp_path / "weird.jsonl").read_text().splitlines()[0])
    # The fallback either gave us a dict via __dict__ or a repr — both fine,
    # what matters is the line is well-formed JSON.
    assert event["kind"] == "tool_start"


# ---------------------------------------------------------------------------
# T0.6
# ---------------------------------------------------------------------------


def test_trace_writer_handles_concurrent_writes(tmp_path: Path) -> None:
    """50 threads emit one event each — file must contain 50 well-formed lines."""
    cbs = trace_writer.build_callbacks(tmp_path, agent_name="a", session_id="hot")

    n = 50

    def emit(i: int) -> None:
        cbs["tool_start_callback"](f"tc-{i}", "terminal", {"i": i})

    threads = [threading.Thread(target=emit, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    raw = (tmp_path / "hot.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw) == n, f"expected {n} lines, got {len(raw)}"
    # All lines must be parseable — no torn writes.
    parsed = [json.loads(line) for line in raw]
    indices = sorted(p["payload"]["args"]["i"] for p in parsed)
    assert indices == list(range(n))
