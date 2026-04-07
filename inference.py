"""
DataClean OpenEnv — Baseline Inference Script
=============================================
Reads:
  API_BASE_URL  (default: https://router.huggingface.co/v1)
  MODEL_NAME    (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      your HuggingFace / API key
  DATACLEAN_TASK  which task to run: fill_missing | dedup_typefix | full_pipeline | all

Stdout format (strict):
  [START] task=<name> env=dataclean-env model=<model>
  [STEP]  step=<n> action=<op> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
API_KEY      = HF_TOKEN or os.getenv("API_KEY", "nokey")

ENV_URL      = os.getenv("DATACLEAN_ENV_URL", "http://localhost:7860")
TASK_ARG     = os.getenv("DATACLEAN_TASK",    "all")
BENCHMARK    = "dataclean-env"

MAX_STEPS    = 12
TEMPERATURE  = 0.3
MAX_TOKENS   = 400

SUCCESS_THRESHOLD = 0.5   # score >= this → success

# ---------------------------------------------------------------------------
# Logging helpers  (exact format required by judges)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    err_val  = error if error else "null"
    done_val = str(done).lower()
    # Sanitise action string: remove newlines
    action_safe = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_safe} "
        f"reward={reward:.2f} done={done_val} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------

def env_reset(task: str) -> Dict[str, Any]:
    r = requests.post(f"{ENV_URL}/reset",
                      json={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{ENV_URL}/step",
                      json={"operation": operation, "params": params},
                      timeout=30)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# LLM prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert data cleaning agent. You will receive an observation about a
messy dataset and must decide the SINGLE best cleaning operation to perform next.

Available operations:
- fill_missing        params: {column?: str, value?: any}   — fill null values
- rename_columns      params: {mapping?: dict}              — strip/lowercase cols
- drop_duplicates     params: {subset?: list}               — remove exact dupes
- cast_column         params: {column: str, dtype: str}     — dtype: float|bool|int|str
- remove_outliers     params: {column: str, method?: "iqr"} — remove statistical outliers
- normalize_phone     params: {column?: str}                — clean to 10 digits
- fix_categorical     params: {column: str, lowercase?: bool, titlecase?: bool, mapping?: dict}
- fix_typos           params: {column: str, corrections: dict}

Respond with ONLY a valid JSON object, no explanation, no markdown:
{"operation": "<op_name>", "params": {<params>}}
""").strip()


def get_action(client: OpenAI, obs: Dict[str, Any],
               history: List[str]) -> Dict[str, Any]:
    hist_block = "\n".join(history[-4:]) if history else "None"
    user_msg = textwrap.dedent(f"""
        Task: {obs.get('task_name')}
        Step: {obs.get('step')}
        Columns: {obs.get('columns')}
        Dtypes: {obs.get('dtypes')}
        Missing counts: {obs.get('missing_counts')}
        Duplicate count: {obs.get('duplicate_count')}
        Hint: {obs.get('hint')}
        Last error: {obs.get('last_action_error')}
        Recent history:
        {hist_block}

        What is the single best operation to perform next?
        Respond ONLY with valid JSON.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as exc:
        # Fallback: perform a safe no-op fill
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return {"operation": "fill_missing", "params": {}}

# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task_name: str) -> float:
    rewards:     List[float] = []
    history:     List[str]   = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task_name, model=MODEL_NAME)

    try:
        obs = env_reset(task_name)
    except Exception as exc:
        print(f"[DEBUG] env_reset failed: {exc}", flush=True)
        log_end(False, 0, 0.0, [])
        return 0.0

    try:
        for step in range(1, MAX_STEPS + 1):
            action_dict = get_action(client, obs, history)
            op     = action_dict.get("operation", "fill_missing")
            params = action_dict.get("params", {})
            action_str = f'{op}({json.dumps(params)})'

            try:
                result = env_step(op, params)
            except Exception as exc:
                log_step(step, action_str, 0.0, False, str(exc))
                rewards.append(0.0)
                steps_taken = step
                break

            reward  = float(result.get("reward", 0.0))
            done    = bool(result.get("done", False))
            obs     = result.get("observation", obs)
            error   = obs.get("last_action_error")
            score   = float(result.get("info", {}).get("score", 0.0))

            rewards.append(reward)
            steps_taken = step

            log_step(step, action_str, reward, done, error)
            history.append(f"step={step} op={op} reward={reward:.2f} score={score:.2f}")

            if done:
                break

        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] episode error: {exc}", flush=True)

    finally:
        log_end(success, steps_taken, score, rewards)

    return score

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_tasks = ["fill_missing", "dedup_typefix", "full_pipeline"]
    tasks_to_run = all_tasks if TASK_ARG == "all" else [TASK_ARG]

    final_scores: Dict[str, float] = {}
    for task in tasks_to_run:
        s = run_episode(client, task)
        final_scores[task] = s

    # Summary
    print("\n[SUMMARY]", flush=True)
    for t, s in final_scores.items():
        print(f"  {t}: score={s:.2f}", flush=True)
    avg = sum(final_scores.values()) / len(final_scores) if final_scores else 0.0
    print(f"  average: {avg:.2f}", flush=True)


if __name__ == "__main__":
    main()
