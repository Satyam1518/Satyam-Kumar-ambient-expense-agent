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
import os
import re
from collections.abc import AsyncGenerator
from typing import Any

import google.auth
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.models import Gemini
from google.adk.workflow import Workflow, node
from google.auth.exceptions import DefaultCredentialsError
from google.genai import types
from pydantic import BaseModel, Field

from expense_agent import config

# Set up local authentication dynamically for Vertex AI vs AI Studio
use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "True").lower() == "true"

if use_vertex:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    if "GOOGLE_CLOUD_PROJECT" not in os.environ:
        try:
            _, project_id = google.auth.default()
            os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        except DefaultCredentialsError:
            pass
    if "GOOGLE_CLOUD_LOCATION" not in os.environ:
        os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
else:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"


class Expense(BaseModel):
    """Pydantic model representing structured expense report details."""

    amount: float = Field(description="The numeric expense amount in dollars.")
    submitter: str = Field(
        description="Name or email of the person submitting the expense."
    )
    category: str = Field(
        description="Category of the expense (e.g., travel, meals, equipment)."
    )
    description: str = Field(description="Detailed explanation of the expense.")
    date: str = Field(description="The date of the expense (YYYY-MM-DD format).")


class RiskAssessment(BaseModel):
    """Pydantic model representing structured risk assessment from the model."""

    risk_score: int = Field(
        description="Risk assessment score from 1 (lowest risk) to 5 (highest risk)"
    )
    risk_factors: list[str] = Field(
        description="List of detected potential risk factors, anomalies, or compliance violations"
    )
    justification: str = Field(
        description="Detailed explanation for the risk score and factors identified"
    )


def parse_payload(payload: Any) -> dict:
    """Robust helper to extract expense dict from diverse input event structures.

    Handles base64 Pub/Sub envelopes, stringified JSON, and direct python dicts.
    """
    if hasattr(payload, "parts"):  # types.Content
        text = ""
        for part in payload.parts:
            if part.text:
                text += part.text
        try:
            data = json.loads(text)
        except Exception:
            data = {"data": text}
    elif isinstance(payload, str):
        try:
            data = json.loads(payload)
        except Exception:
            data = {"data": payload}
    elif isinstance(payload, dict):
        data = payload
    else:
        data = {}

    raw_data = data.get("data")
    if raw_data is None:
        # Fallback to checking root keys directly
        keys = ["amount", "submitter", "category", "description", "date"]
        if any(k in data for k in keys):
            return data
        return {}

    if isinstance(raw_data, str):
        # Handle Pub/Sub base64 encoding
        try:
            decoded_bytes = base64.b64decode(raw_data)
            decoded_str = decoded_bytes.decode("utf-8")
            return json.loads(decoded_str)
        except Exception:
            # Handle plain JSON string
            try:
                return json.loads(raw_data)
            except Exception:
                pass
    elif isinstance(raw_data, dict):
        return raw_data

    return {}


def parse_event(ctx: Context, node_input: Any) -> Event:
    """Parses, extracts, and validates incoming expense report details."""
    parsed = parse_payload(node_input)
    expense = Expense(**parsed)
    # Persist the expense details in state for subsequent workflow steps
    return Event(output=expense.model_dump(), state={"expense": expense.model_dump()})


def route_expense(node_input: dict) -> Event:
    """Performs python-coded routing based on threshold configuration."""
    amount = node_input.get("amount", 0.0)
    if amount < config.APPROVAL_THRESHOLD:
        return Event(output=node_input, route="auto_approve")
    else:
        return Event(output=node_input, route="llm_review")


def auto_approve_node(node_input: dict) -> Event:
    """Instantly approves expenses that are below the dollar threshold."""
    outcome = {
        "status": "APPROVED",
        "reason": f"Amount ${node_input.get('amount', 0.0):.2f} is under threshold ${config.APPROVAL_THRESHOLD:.2f}. Auto-approved.",
        "amount": node_input.get("amount", 0.0),
        "submitter": node_input.get("submitter", "Unknown"),
        "category": node_input.get("category", "N/A"),
        "description": node_input.get("description", "N/A"),
        "date": node_input.get("date", "N/A"),
    }
    return Event(output=outcome, state={"outcome": outcome})


# LLM Node responsible for compliance and risk checking
llm_reviewer = LlmAgent(
    name="llm_reviewer",
    model=Gemini(
        model=config.MODEL_NAME,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are a compliance risk auditor. Review the expense report details (amount, submitter, "
        "category, date, description) and evaluate potential risk factors or violations. "
        "Provide a structured compliance assessment."
    ),
    output_schema=RiskAssessment,
    output_key="risk_assessment",
)


@node(rerun_on_resume=True)
async def human_approval_node(
    ctx: Context, node_input: dict
) -> AsyncGenerator[Any, None]:
    """Pauses the workflow for human approval if the expense is large, then records the decision."""
    expense_dict = ctx.state.get("expense") or {}

    # Check if approval input is already received
    if not ctx.resume_inputs or "approval_decision" not in ctx.resume_inputs:
        msg = (
            f"⚠️ EXPENSE RISK ALERT: Human Review Required ⚠️\n"
            f"An expense of ${expense_dict.get('amount', 0.0):.2f} submitted by "
            f"{expense_dict.get('submitter', 'Unknown')} requires your approval.\n\n"
            f"--- EXPENSE DETAILS ---\n"
            f"• Amount: ${expense_dict.get('amount', 0.0):.2f}\n"
            f"• Submitter: {expense_dict.get('submitter', 'Unknown')}\n"
            f"• Category: {expense_dict.get('category', 'N/A')}\n"
            f"• Date: {expense_dict.get('date', 'N/A')}\n"
            f"• Description: {expense_dict.get('description', 'N/A')}\n\n"
            f"--- LLM RISK ASSESSMENT ---\n"
            f"• Risk Score: {node_input.get('risk_score', 'N/A')}/5\n"
            f"• Risk Factors: {', '.join(node_input.get('risk_factors', [])) if node_input.get('risk_factors') else 'None'}\n"
            f"• Justification: {node_input.get('justification', 'N/A')}\n\n"
            f"Please reply with 'APPROVED' or 'REJECTED' and any comments."
        )
        yield RequestInput(interrupt_id="approval_decision", message=msg)
        return

    # Process decision after resume
    decision_text = str(ctx.resume_inputs["approval_decision"]).strip().upper()
    status = "APPROVED" if "APPROVE" in decision_text else "REJECTED"

    outcome = {
        "status": status,
        "reason": f"Reviewed by human. Decision: {status}. User comments: {decision_text}",
        "amount": expense_dict.get("amount", 0.0),
        "submitter": expense_dict.get("submitter", "Unknown"),
        "category": expense_dict.get("category", "N/A"),
        "description": expense_dict.get("description", "N/A"),
        "date": expense_dict.get("date", "N/A"),
        "risk_assessment": node_input,
    }
    yield Event(output=outcome, state={"outcome": outcome})


def security_checkpoint(ctx: Context, node_input: dict) -> Event:
    """Security checkpoint node that scrubs personal data and checks for prompt injection.

    1. Scrub SSNs and credit cards from description.
    2. Check for prompt injection keywords. If found, bypass LLM and route straight to human.
    """
    expense = dict(node_input)
    description = expense.get("description", "")

    redacted_categories = []

    # Redact SSNs (e.g., 000-00-0000)
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    if re.search(ssn_pattern, description):
        description = re.sub(ssn_pattern, "[REDACTED SSN]", description)
        redacted_categories.append("SSN")

    # Redact Credit Cards (e.g. 13 to 16 digits, with optional spaces/hyphens)
    cc_pattern = r"\b(?:\d[ -]*?){13,16}\b"
    if re.search(cc_pattern, description):
        description = re.sub(cc_pattern, "[REDACTED CREDIT CARD]", description)
        redacted_categories.append("Credit Card")

    expense["description"] = description

    # Store updated expense and redacted categories in state
    state_updates = {
        "expense": expense,
        "redacted_categories": redacted_categories,
    }

    # Prompt injection check
    # Check for keywords that try to force auto-approval or bypass rules
    injection_patterns = [
        r"ignore\s+(?:previous\s+)?instructions",
        r"bypass\s+(?:the\s+)?rules",
        r"auto-approve",
        r"override\s+(?:the\s+)?rules",
        r"system\s+prompt",
        r"forget\s+(?:previous\s+)?instructions",
        r"developer\s+mode",
    ]

    is_injection = False
    for pattern in injection_patterns:
        if re.search(pattern, description, re.IGNORECASE):
            is_injection = True
            break

    if is_injection:
        # Route straight to human, flagging as security event and setting mock RiskAssessment
        state_updates["security_event"] = True
        mock_assessment = {
            "risk_score": 5,
            "risk_factors": ["PROMPT INJECTION DETECTED"],
            "justification": "HIGH SECURITY RISK: The expense description contains keywords indicative of a prompt injection attempt. Bypassing LLM reviewer.",
        }
        return Event(
            output=mock_assessment,
            route="security_bypass",
            state=state_updates,
        )
    else:
        # Route to LLM reviewer with scrubbed description
        return Event(
            output=expense,
            route="llm_review",
            state=state_updates,
        )


# ADK 2.0 Graph Workflow Definition
root_agent = Workflow(
    name="expense_approval_workflow",
    edges=[
        ("START", parse_event),
        (parse_event, route_expense),
        (
            route_expense,
            {
                "auto_approve": auto_approve_node,
                "llm_review": security_checkpoint,
            },
        ),
        (
            security_checkpoint,
            {
                "llm_review": llm_reviewer,
                "security_bypass": human_approval_node,
            },
        ),
        (llm_reviewer, human_approval_node),
    ],
)

app = App(
    root_agent=root_agent,
    name="expense_agent",
)
