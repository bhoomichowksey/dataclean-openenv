"""
DataClean OpenEnv — Baseline Inference Script
=============================================
Reads:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
 
Stdout format (strict):
    [START] task=<name> env=dataclean-env model=<model>
    [STEP]  step=<n> action=<op> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""
 
import json
import os
import requests
from typing import Any, Dict, List, Optional
from openai import OpenAI
 
# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
API_KEY      = HF_TOKEN or os.getenv("API_KEY", "nokey")
 
ENV_URL      = os.getenv("DATACLEAN_ENV_URL", "https://bhoomichowksey-dataclean-env.hf.space")
TASKS        = ["fill_missing", "dedup_typefix", "full_pipeline"]
MAX_STEPS    = 10
TEMPERATURE  = 0.2
MAX_TOKENS   = 256
 
# -------------------------------------------------------------------------
# Logging helpers
# -------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)
 
def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)
 
def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)
 
# -------------------------------------------------------------------------
# Environment helpers
# -------------------------------------------------------------------------
def env_reset(task: str) -> Dict[str, Any]:
    r = requests.post(f"{ENV_URL}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()
 
def env_step(operation: str, action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{ENV_URL}/step", json={"operation": operation, "action": action}, timeout=30)
    r.raise_for_status()
    return r.json()
 
def env_state() -> Dict[str, Any]:
    r = requests.get(f"{ENV_URL}/state", timeout=30)
    r.raise_for_status()
    return r.json()
 
# -------------------------------------------------------------------------
# LLM helpers
# -------------------------------------------------------------------------
def ask_llm(client: OpenAI, task_name: str, obs: Dict[str, Any], step: int) -> Dict[str, Any]:
    system = (
        "You are a data cleaning agent. Given a dataset observation, decide the next cleaning action.\n"
        "For fill_missing: return JSON {\"operation\":\"fill_missing\",\"action\":{\"column\":\"<col>\",\"strategy\":\"<mean|mode|unknown>\"}}\n"
        "For dedup_typefix: return JSON {\"operation\":\"dedup_typefix\",\"action\":{\"column\":\"<col>\",\"dtype\":\"<str|float|int>\"}}\n"
        "For full_pipeline: return JSON {\"operation\":\"fill_missing\",\"action\":{\"column\":\"<col>\",\"strategy\":\"mean\"}}\n"
        "Reply with ONLY valid JSON, no explanation."
    )
    user = f"Task: {task_name}\nStep: {step}\nObservation: {json.dumps(obs)}\nWhat is the best next action?"
 
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        # strip markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        # fallback safe action
        return {"operation": task_name, "action": {"column": "age", "strategy": "mean"}}
 
# -------------------------------------------------------------------------
# Run one task episode
# -------------------------------------------------------------------------
def run_task(client: OpenAI, task_name: str) -> None:
    log_start(task=task_name, env="dataclean-env", model=MODEL_NAME)
 
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
 
    try:
        obs = env_reset(task_name)
 
        for step in range(1, MAX_STEPS + 1):
            decision = ask_llm(client, task_name, obs, step)
            operation = decision.get("operation", task_name)
            action = decision.get("action", {})
 
            try:
                result = env_step(operation, action)
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
                error = result.get("observation", {}).get("last_action_error") if isinstance(result.get("observation"), dict) else None
                obs = result.get("observation", obs)
            except Exception as e:
                reward = 0.0
                done = False
                error = str(e)
 
            rewards.append(reward)
            steps_taken = step
            action_str = json.dumps(action).replace(" ", "")
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
 
            if done:
                break
 
        state = env_state()
        score = float(state.get("score", 0.0))
        success = score >= 0.5
 
    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
 
# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_env = os.getenv("DATACLEAN_TASK", "all")
    tasks_to_run = TASKS if task_env == "all" else [task_env]
    for task in tasks_to_run:
        run_task(client, task)
 
