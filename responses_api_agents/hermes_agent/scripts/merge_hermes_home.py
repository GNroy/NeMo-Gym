# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Curator step: promote per-run HERMES_HOME learnings to the template.

Runnable standalone for offline audits or invoked between A/B/C rollout
passes to promote learnings from per-run HERMES_HOME into the template
that the next pass starts from.  Only allowlisted paths are copied back —
everything else (logs, sessions DB, kanban DB, .env, seed skills) stays
per-run.

Usage::

    python responses_api_agents/hermes_agent/scripts/merge_hermes_home.py \
        --template /path/to/hermes_template \
        --audit    /path/to/merge_audit.json \
        -- /job_dir/agents/orch/hermes_home /job_dir/agents/analyst/hermes_home
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

LOG = logging.getLogger("merge_hermes_home")


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------
# Each entry is (relative-path-from-hermes-home, kind).  ``kind`` is:
#   - "file"  → single file; copy if newer (or always when dry-run target file
#               is missing).
#   - "tree"  → directory; copy contents (relative paths preserved).
#
# Adding to this list expands what survives across runs — touch with care.

ALLOWLIST: Tuple[Tuple[str, str], ...] = (
    ("MEMORY.md", "file"),
    ("USER.md", "file"),
    ("skills/user", "tree"),
)


@dataclass
class FileAudit:
    rel_path: str
    src: str
    dst: str
    # "copied" | "unchanged" | "missing-src" | "would-copy" (dry-run).  Set
    # by ``merge()`` after deciding the disposition; defaults to "" so we
    # can construct then fill in.
    action: str = ""
    before_sha256: str = ""
    after_sha256: str = ""
    size_delta: int = 0


@dataclass
class MergeResult:
    template: str
    sources: List[str]
    dry_run: bool
    files: List[FileAudit] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "template": self.template,
            "sources": self.sources,
            "dry_run": self.dry_run,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "files": [vars(f) for f in self.files],
        }


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_allowlisted(src_home: Path) -> Iterable[Tuple[Path, str]]:
    """Yield ``(absolute_src_path, relative_path_inside_home)`` for every
    file the allowlist authorizes to flow back.

    Directories are expanded recursively.  Missing entries are silently
    skipped — they show up in the audit as ``missing-src``.
    """
    for rel, kind in ALLOWLIST:
        rel_path = src_home / rel
        if kind == "file":
            yield rel_path, rel
        elif kind == "tree":
            if not rel_path.is_dir():
                continue
            for f in sorted(rel_path.rglob("*")):
                if f.is_file():
                    # ``.relative_to(src_home)`` keeps the same subtree shape
                    # under the template so user skills land at the matching
                    # path verbatim.
                    yield f, str(f.relative_to(src_home))
        else:
            raise ValueError(f"unknown allowlist kind: {kind!r} for {rel!r}")


def merge(
    sources: List[Path],
    template: Path,
    dry_run: bool = False,
) -> MergeResult:
    """Merge one or more per-agent HERMES_HOME dirs into ``template``.

    Conflict resolution: last source wins.  ``sources`` is processed in
    order; if two agents' MEMORY.md disagree, the later one overwrites
    the earlier.  Per-agent skills/user/ entries land at their own subpath
    so they never collide.
    """
    result = MergeResult(template=str(template), sources=[str(s) for s in sources], dry_run=dry_run)
    template.mkdir(parents=True, exist_ok=True)

    for src in sources:
        src = Path(src)
        if not src.is_dir():
            LOG.warning("source HERMES_HOME not found: %s — skipping", src)
            result.files.append(
                FileAudit(rel_path=str(src), src=str(src), dst=str(template), action="missing-src")
            )
            continue
        for src_file, rel in _iter_allowlisted(src):
            dst_file = template / rel
            entry = FileAudit(rel_path=rel, src=str(src_file), dst=str(dst_file))
            if not src_file.is_file():
                entry.action = "missing-src"
                result.files.append(entry)
                continue

            entry.before_sha256 = _sha256(dst_file)
            entry.after_sha256 = _sha256(src_file)

            if entry.before_sha256 == entry.after_sha256:
                entry.action = "unchanged"
                result.files.append(entry)
                continue

            entry.size_delta = (
                src_file.stat().st_size - (dst_file.stat().st_size if dst_file.exists() else 0)
            )
            entry.action = "would-copy" if dry_run else "copied"
            if not dry_run:
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
            result.files.append(entry)

    result.finished_at = time.time()
    return result


def write_audit(result: MergeResult, audit_path: Path) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge per-agent HERMES_HOME snapshots back into the template.",
    )
    p.add_argument("--template", required=True, help="Path to the template HERMES_HOME.")
    p.add_argument(
        "--audit",
        required=True,
        help="Where to write the merge audit JSON (per-file before/after sha256, action).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing to the template.",
    )
    p.add_argument(
        "sources",
        nargs="+",
        help="One or more per-agent HERMES_HOME paths to merge back.",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format="%(message)s")
    args = _build_parser().parse_args(argv)

    sources = [Path(s) for s in args.sources]
    template = Path(args.template)
    audit_path = Path(args.audit)

    result = merge(sources=sources, template=template, dry_run=args.dry_run)
    write_audit(result, audit_path)

    n_copied = sum(1 for f in result.files if f.action == "copied")
    n_would = sum(1 for f in result.files if f.action == "would-copy")
    n_unchanged = sum(1 for f in result.files if f.action == "unchanged")
    n_missing = sum(1 for f in result.files if f.action == "missing-src")
    LOG.info(
        "merge: %d copied, %d would-copy (dry-run), %d unchanged, %d missing-src",
        n_copied,
        n_would,
        n_unchanged,
        n_missing,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
