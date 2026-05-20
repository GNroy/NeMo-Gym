# passthrough

No-op verifier for benchmarks whose answers are graded **outside**
NeMo-Gym — e.g. when the judge is an offline LLM (gpt-oss-120b,
o3-mini, etc.) running through `ns eval`.

`verify()` returns `reward=0.0` and echoes the request payload
verbatim so the downstream grader has both the model's `response` and
the `verifier_metadata` (e.g. `expected_answer`) to work with.

Typical pairing: this server + the `hermes_agent` agent server +
`ns hermes_agent_rollouts`, with the offline judge step running
separately on the output rollouts file.

## Smoke

```bash
ng_run "+config_paths=[\
resources_servers/passthrough/configs/passthrough.yaml,\
responses_api_agents/hermes_agent/configs/hermes_agent.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml]" \
  +hermes_agent.responses_api_agents.hermes_agent.resources_server.name=passthrough
```

Then collect rollouts against the example data:

```bash
ng_collect_rollouts \
  +agent_name=hermes_agent \
  +input_jsonl_fpath=resources_servers/passthrough/data/example.jsonl \
  +output_jsonl_fpath=results/passthrough_rollouts.jsonl \
  +num_repeats=1
```
