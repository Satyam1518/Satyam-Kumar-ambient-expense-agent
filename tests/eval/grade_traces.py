import datetime
import json
import os
import re
import sys
import yaml
from pathlib import Path
from unittest.mock import MagicMock

# Load env variables
from dotenv import load_dotenv
load_dotenv(dotenv_path="c:/Users/Lenovo/agy-cli-projects/ambient-expense-agent/.env")

# Mock GCP default credentials to bypass SDK checks
import google.auth
google.auth.default = MagicMock(return_value=(MagicMock(), "mock-project-id"))

from google import genai
from google.genai import types
from rich.console import Console
from rich.table import Table

def load_eval_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    metrics_to_run = cfg.get("metrics_to_run", [])
    custom_metrics = {m["name"]: m for m in cfg.get("custom_metrics", [])}
    return metrics_to_run, custom_metrics

def evaluate_with_gemini(client, model_name: str, prompt_template: str, prompt_val: str, response_val: str, agent_data_val: str) -> dict:
    import time
    # Build LLM-as-judge prompt
    prompt = prompt_template.replace("{prompt}", prompt_val)\
                            .replace("{response}", response_val)\
                            .replace("{agent_data}", agent_data_val)
    
    max_retries = 5
    delay = 15  # 15s delay to fit within the 5 Requests Per Minute limit of free tier
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            res_text = response.text.strip()
            # Parse JSON output from the judge
            return json.loads(res_text)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print(f"Rate limited (429). Retrying in {delay} seconds (attempt {attempt + 1}/{max_retries})...", flush=True)
                time.sleep(delay)
                delay *= 1.5
                continue
            print(f"Error calling judge LLM: {e}", file=sys.stderr)
            return {"score": 1, "explanation": f"Failed to get score from LLM judge: {e}"}
            
    return {"score": 1, "explanation": "Failed to get score from LLM judge due to rate limits."}

def main():
    console = Console()
    config_path = Path("tests/eval/eval_config.yaml")
    traces_path = Path("artifacts/traces/generated_traces.json")
    output_dir = Path("artifacts/grades")
    
    if not config_path.exists():
        console.print(f"[red]Error: Configuration file not found at {config_path}[/red]")
        sys.exit(1)
    if not traces_path.exists():
        console.print(f"[red]Error: Traces file not found at {traces_path}. Run generate-traces first.[/red]")
        sys.exit(1)
        
    metrics_to_run, custom_metrics = load_eval_config(config_path)
    
    with open(traces_path, "r", encoding="utf-8") as f:
        traces_data = json.load(f)
        
    eval_cases = traces_data.get("eval_cases", [])
    if not eval_cases:
        console.print("[red]Error: No evaluation cases found in traces file.[/red]")
        sys.exit(1)
        
    # Instantiate GenAI Client
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        console.print("[red]Error: Neither GEMINI_API_KEY nor GOOGLE_API_KEY found in environment.[/red]")
        sys.exit(1)
        
    # We use gemini-2.5-flash as our judge
    model_name = "gemini-2.5-flash"
    client = genai.Client(api_key=api_key)
    
    console.print(f"[cyan]Loaded {len(eval_cases)} cases. Grading against metrics: {metrics_to_run}...[/cyan]")
    
    summary_scores = {metric: [] for metric in metrics_to_run}
    graded_cases = []
    
    for case in eval_cases:
        case_id = case.get("eval_case_id")
        prompt_str = json.dumps(case.get("prompt", {}))
        response_str = json.dumps(case.get("responses", [{}])[0].get("response", {}))
        agent_data_str = json.dumps(case.get("agent_data", {}))
        
        case_metrics = {}
        console.print(f"\nGrading case: [bold yellow]{case_id}[/bold yellow]")
        
        for metric_name in metrics_to_run:
            metric_def = custom_metrics.get(metric_name)
            if not metric_def:
                console.print(f"[red]Warning: Custom metric definition for '{metric_name}' not found.[/red]")
                continue
                
            prompt_template = metric_def.get("prompt_template", "")
            result = evaluate_with_gemini(client, model_name, prompt_template, prompt_str, response_str, agent_data_str)
            
            score = result.get("score", 1)
            explanation = result.get("explanation", "No explanation provided.")
            
            case_metrics[metric_name] = {
                "score": score,
                "explanation": explanation
            }
            summary_scores[metric_name].append(score)
            
            console.print(f"  - [cyan]{metric_name}[/cyan]: Score [bold green]{score}/5[/bold green]")
            console.print(f"    [dim]Explanation: {explanation}[/dim]")
            
        graded_case = dict(case)
        graded_case["metrics"] = case_metrics
        graded_cases.append(graded_case)
        
    # Calculate means
    summary_metrics_list = []
    for metric_name, scores in summary_scores.items():
        mean = sum(scores) / len(scores) if scores else 0.0
        summary_metrics_list.append({
            "metric_name": metric_name,
            "mean": mean
        })
        
    results_json = {
        "summary_metrics": summary_metrics_list,
        "evaluation_dataset": [
            {
                "eval_cases": graded_cases
            }
        ]
    }
    
    # Save artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"results_{timestamp}.json"
    latest_json_path = output_dir / "results.json"
    
    # Write timestamped and latest json files
    for p in (json_path, latest_json_path):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=2)
            
    # Print summary table using Rich
    table = Table(title="Evaluation Summary Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric Name", style="cyan")
    table.add_column("Average Score", style="green", justify="right")
    
    for metric in summary_metrics_list:
        table.add_row(metric["metric_name"], f"{metric['mean']:.2f} / 5.0")
        
    console.print("\n")
    console.print(table)
    console.print(f"\n[green]Saved full results to {latest_json_path}[/green]")

if __name__ == "__main__":
    main()
