import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path="c:/Users/Lenovo/agy-cli-projects/ambient-expense-agent/.env")

# Mock GCP credentials
import google.auth
google.auth.default = MagicMock(return_value=(MagicMock(), "mock-project-id"))

from google.adk.runners import InMemoryRunner
from google.genai import types

from expense_agent.agent import root_agent

# Standard input patterns for prompt injection
INJECTION_KEYWORDS = [
    "ignore instructions",
    "bypass rules",
    "auto-approve",
    "override the rules",
    "system prompt",
    "forget previous instructions"
]

async def run_case(case_id: str, payload: dict) -> dict:
    print(f"Running evaluation case: {case_id}...", flush=True)
    runner = InMemoryRunner(agent=root_agent, app_name="expense_agent")
    
    session = await runner.session_service.create_session(
        user_id="eval_user", app_name="expense_agent"
    )
    
    message = types.Content(
        role="user", parts=[types.Part.from_text(text=json.dumps(payload))]
    )
    
    events = []
    async for event in runner.run_async(
        new_message=message,
        user_id="eval_user",
        session_id=session.id,
    ):
        events.append(event)
        
    is_paused = any(
        event.long_running_tool_ids and "approval_decision" in event.long_running_tool_ids
        for event in events
    )
    
    final_outcome = None
    
    if is_paused:
        # Automate Manager decisions
        desc = payload.get("description", "").lower()
        is_injection = any(kw in desc for kw in INJECTION_KEYWORDS)
        
        # Approve clean requests, reject prompt injections
        decision = "REJECTED" if is_injection else "APPROVED"
        print(f"  -> Workflow paused. Automated HITL decision: {decision}", flush=True)
        
        resume_message = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="adk_request_input",
                        id="approval_decision",
                        response={"result": decision},
                    )
                )
            ],
        )
        
        async for event in runner.run_async(
            new_message=resume_message,
            user_id="eval_user",
            session_id=session.id,
        ):
            events.append(event)
            
    # Find final outcome in events
    for event in events:
        if event.output is not None:
            final_outcome = event.output
            
    # Get final state from session storage
    final_session = await runner.session_service.get_session(
        app_name="expense_agent", user_id="eval_user", session_id=session.id
    )
    
    # Format and serialize events for grading
    serialized_events = []
    for ev in events:
        ev_dict = {
            "type": type(ev).__name__,
        }
        if getattr(ev, "output", None) is not None:
            ev_dict["output"] = ev.output
        if getattr(ev, "long_running_tool_ids", None):
            ev_dict["long_running_tool_ids"] = list(ev.long_running_tool_ids)
        serialized_events.append(ev_dict)
        
    trace_dict = {
        "eval_case_id": case_id,
        "prompt": {
            "role": "user",
            "parts": [{"text": json.dumps(payload)}]
        },
        "responses": [
            {
                "response": {
                    "role": "model",
                    "parts": [{"text": json.dumps(final_outcome) if final_outcome else "No response"}]
                }
            }
        ],
        "agent_data": {
            "turns": [
                {
                    "turn_index": 0,
                    "events": serialized_events
                }
            ],
            "state": final_session.state if final_session else {}
        }
    }
    return trace_dict

async def main():
    dataset_path = Path("tests/eval/datasets/basic-dataset.json")
    output_path = Path("artifacts/traces/generated_traces.json")
    
    if not dataset_path.exists():
        print(f"Error: dataset not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)
        
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
        
    eval_cases = dataset.get("eval_cases", [])
    output_cases = []
    
    for case in eval_cases:
        case_id = case.get("eval_case_id")
        prompt_text = case["prompt"]["parts"][0]["text"]
        payload = json.loads(prompt_text)
        
        trace = await run_case(case_id, payload)
        output_cases.append(trace)
        
    # Write to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"eval_cases": output_cases}, f, indent=2)
        
    print(f"\nSuccessfully generated {len(output_cases)} traces at {output_path}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
