# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Passthrough resources server — verify() is a no-op for offline scoring.

Use this when the model's correctness is judged later (offline LLM judge,
human review, etc.) and ``ng_collect_rollouts`` only needs to round-trip
the generation through ``verify()`` without grading it.  Every ``verify``
call returns ``reward=0.0`` and echoes the request payload verbatim, so
the offline grader downstream has the full ``response`` + ``verifier_metadata``
to work with.

Pair this with the Hermes integration in
``responses_api_agents/hermes_agent`` when running multi-agent rollouts
against benchmarks whose judge lives outside NeMo-Gym (e.g. the
NeMo-Skills ``frontierscience-olympiad`` judge with ``gpt-oss-120b``).
"""

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class PassthroughResourcesServerConfig(BaseResourcesServerConfig):
    pass


class PassthroughResourcesServer(SimpleResourcesServer):
    config: PassthroughResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # reward=0.0 is a sentinel — the downstream judge owns the real grade.
        return BaseVerifyResponse(**body.model_dump(), reward=0.0)


if __name__ == "__main__":
    PassthroughResourcesServer.run_webserver()
