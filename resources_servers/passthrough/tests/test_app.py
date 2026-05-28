# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
from unittest.mock import MagicMock

import pytest

from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.passthrough.app import (
    PassthroughResourcesServer,
    PassthroughResourcesServerConfig,
)


def _make_server() -> PassthroughResourcesServer:
    config = PassthroughResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
    )
    return PassthroughResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_sanity() -> None:
    _make_server()


@pytest.mark.asyncio
async def test_verify_returns_zero_reward_and_echoes_payload() -> None:
    """The passthrough verifier must round-trip the request unchanged (modulo
    the new ``reward`` field) so an offline judge has the full context."""
    server = _make_server()

    request = BaseVerifyRequest.model_validate(
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": "ping"}],
            },
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "model": "test-model",
                "output": [],
                "status": "completed",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        }
    )

    response = await server.verify(request)

    assert isinstance(response, BaseVerifyResponse)
    assert response.reward == 0.0
    # Round-trip: the request payload survives verbatim.
    assert response.responses_create_params == request.responses_create_params
    assert response.response == request.response
