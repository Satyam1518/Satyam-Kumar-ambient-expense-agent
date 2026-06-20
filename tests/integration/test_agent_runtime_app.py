# ruff: noqa: E402
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
from unittest.mock import MagicMock

import google.auth
from dotenv import load_dotenv

# Load environmental variables from .env
load_dotenv(dotenv_path="c:/Users/Lenovo/agy-cli-projects/ambient-expense-agent/.env")

# Mock google.auth.default to prevent DefaultCredentialsError during test collection
google.auth.default = MagicMock(return_value=(MagicMock(), "mock-project-id"))

from fastapi.testclient import TestClient

from expense_agent.agent_runtime_app import app

client = TestClient(app)


def test_agent_pubsub_endpoint() -> None:
    """Integration test for the agent's Pub/Sub endpoint.

    Tests that an expense under the threshold is auto-approved and session ID is correctly normalized.
    """
    payload = {
        "amount": 55.0,
        "submitter": "charles@example.com",
        "category": "travel",
        "description": "Bus ticket",
        "date": "2026-06-20",
    }
    envelope = {
        "message": {
            "data": base64.b64encode(json.dumps(payload).encode("utf-8")).decode(
                "utf-8"
            ),
            "messageId": "msg-12345",
        },
        "subscription": "projects/test-project/subscriptions/expense-alerts-sub",
    }

    response = client.post("/", json=envelope)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "success"
    assert res_json["session_id"] == "expense-alerts-sub-msg-12345"
    assert res_json["outcome"]["status"] == "APPROVED"


def test_agent_feedback_endpoint() -> None:
    """Integration test for the feedback endpoint."""
    feedback_data = {
        "score": 5,
        "text": "Great response!",
        "user_id": "test-user-456",
        "session_id": "test-session-456",
    }

    response = client.post("/feedback", json=feedback_data)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # Test invalid feedback (missing/invalid fields)
    invalid_feedback = {
        "score": "invalid",  # Score must be numeric
        "text": "Bad feedback",
        "user_id": "test-user-789",
        "session_id": "test-session-789",
    }
    response = client.post("/feedback", json=invalid_feedback)
    assert response.status_code == 400
