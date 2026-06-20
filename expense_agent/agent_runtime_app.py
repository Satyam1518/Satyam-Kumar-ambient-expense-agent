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
import logging
import os
import uuid

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from google.adk.runners import InMemoryRunner
from google.genai import types

from expense_agent.agent import root_agent
from expense_agent.app_utils.telemetry import setup_telemetry
from expense_agent.app_utils.typing import Feedback

# Load environment variables
load_dotenv()

# Configure logging using standard Python logging (console-only, no GCP Cloud Logging client)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("expense_agent")

# Configure Telemetry with otel_to_cloud=False
setup_telemetry(otel_to_cloud=False)

app = FastAPI(title="Ambient Expense Approval Agent")

# Initialize ADK Runner
runner = InMemoryRunner(agent=root_agent, app_name="expense_agent")
runner.auto_create_session = True


def normalize_subscription(subscription_path: str) -> str:
    """Normalizes fully-qualified subscription paths down to short names."""
    if not subscription_path:
        return "default-sub"
    # E.g. "projects/my-project/subscriptions/my-sub" -> "my-sub"
    return subscription_path.split("/")[-1]


@app.post("/")
@app.post("/apps/expense_agent/trigger/pubsub")
async def handle_pubsub(request: Request):
    """Endpoint to receive Pub/Sub push messages."""
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse request JSON: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON body") from e

    logger.info(f"Received Pub/Sub message body: {body}")

    # Validate Pub/Sub message structure
    ps_message = body.get("message")
    if not ps_message:
        logger.error("Missing 'message' field in Pub/Sub payload")
        raise HTTPException(status_code=400, detail="Missing 'message' field")

    # Extract data and attributes
    raw_data = ps_message.get("data")
    message_id = ps_message.get("messageId", str(uuid.uuid4()))

    # Normalize subscription path to a short name
    subscription_path = body.get("subscription", "")
    short_sub_name = normalize_subscription(subscription_path)

    # Keep session records readable using short subscription name + message ID
    session_id = f"{short_sub_name}-{message_id}"
    user_id = ps_message.get("attributes", {}).get("submitter", "pubsub-trigger")

    logger.info(f"Processing message ID {message_id} on session {session_id}")

    # Decoded expense payload can be base64-encoded or plain string/dict
    decoded_data = None
    if raw_data:
        if isinstance(raw_data, dict):
            decoded_data = raw_data
        else:
            try:
                # Try base64 decoding
                decoded_bytes = base64.b64decode(raw_data)
                decoded_data = json.loads(decoded_bytes.decode("utf-8"))
            except Exception:
                # Fallback to loading the raw string as JSON
                try:
                    decoded_data = json.loads(raw_data)
                except Exception:
                    decoded_data = {"data": raw_data}

    if not decoded_data:
        logger.error("Decoded expense data is empty or invalid")
        raise HTTPException(status_code=400, detail="Invalid or empty data payload")

    # Run the workflow
    message = types.Content(
        role="user", parts=[types.Part.from_text(text=json.dumps(decoded_data))]
    )

    try:
        events = []
        async for event in runner.run_async(
            new_message=message,
            user_id=user_id,
            session_id=session_id,
        ):
            events.append(event)

        # Log final outcome of the run
        outcome = None
        for event in events:
            if event.output is not None:
                outcome = event.output

        if outcome:
            logger.info(
                f"Workflow completed for session {session_id} with outcome: {outcome}"
            )
            return {"status": "success", "session_id": session_id, "outcome": outcome}
        else:
            # Check if it paused for HITL
            is_paused = any(
                event.long_running_tool_ids
                and "approval_decision" in event.long_running_tool_ids
                for event in events
            )
            if is_paused:
                logger.info(
                    f"Workflow paused for human approval on session {session_id}"
                )
                return {"status": "paused_for_approval", "session_id": session_id}

            logger.warning(
                f"Workflow completed for session {session_id} without explicit outcome"
            )
            return {"status": "completed", "session_id": session_id}

    except Exception as e:
        logger.error(
            f"Error executing workflow for session {session_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Workflow execution failed: {e!s}"
        ) from e


@app.post("/feedback")
async def register_feedback(request: Request):
    """Endpoint for registering feedback (compatibility wrapper)."""
    try:
        feedback_data = await request.json()
        # Validate feedback structure
        Feedback.model_validate(feedback_data)
        logger.info(f"Feedback successfully logged: {feedback_data}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Invalid feedback submission: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid feedback: {e!s}") from e


if __name__ == "__main__":
    # Serve on port 8080 as requested
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
