---
name: sci-research-cite
description: "Cite sources whenever asserting an empirical or domain-specific fact."
version: 1.0.0
author: NeMo-Skills
license: Apache-2.0
metadata:
  hermes:
    tags: [research, science, citation, provenance]
    category: research
---

# Cite sources

When you state a non-self-evident fact — a constant, a published result, a
benchmark number, a paper claim — record where you got it.

## Rules

1. **Web sources**: use `web_search` / `web_extract` and quote the URL plus
   the publication date (if visible) in the note.
2. **Tool outputs**: when `execute_code` or an MCP tool returns a value
   you'll cite later, paste the exact tool call (name + arguments preview)
   alongside the result.
3. **Prior turns**: if you reuse a number from earlier in the conversation,
   reference the turn index ("from [COMPUTE] in step 3") rather than
   restating without attribution.
4. **Uncertain or contested facts**: prefer hedged phrasing ("commonly
   reported as …") and cite at least two independent sources.

A claim without a source is a claim that did not happen.  If you cannot
cite it, recompute it.
