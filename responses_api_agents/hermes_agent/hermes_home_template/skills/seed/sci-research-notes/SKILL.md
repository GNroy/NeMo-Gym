---
name: sci-research-notes
description: "Take rigorous, step-tagged research notes while solving a scientific task."
version: 1.0.0
author: NeMo-Skills
license: Apache-2.0
metadata:
  hermes:
    tags: [research, science, notes, methodology]
    category: research
---

# Scientific research notes

Before producing a final answer, structure intermediate work as a sequence
of explicit steps with these tags.  This keeps multi-agent traces auditable
and lets later runs reuse partial work.

- **[PARSE]** Restate the problem in your own words; list the knowns,
  unknowns, and units.
- **[PRINCIPLES]** Name the laws, theorems, or empirical relationships you
  intend to apply.  Cite them precisely.
- **[SETUP]** Write the equations, free-body diagram, dataset slice, or
  experimental protocol you will compute against.
- **[COMPUTE]** Carry out the calculation or measurement.  Show units at
  every step.  If you use a tool, name the tool and the exact inputs.
- **[VERIFY]** Sanity-check the result: dimensional analysis, limit cases,
  alternative method, or independent source.  Flag any disagreement.
- **[ANSWER]** State the final answer with units and the confidence level.

When delegating to another agent, include the relevant [PARSE], [PRINCIPLES],
and [SETUP] sections in the delegation message so the worker doesn't redo
the framing.
