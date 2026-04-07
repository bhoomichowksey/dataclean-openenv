"""
FastAPI server exposing the DataClean OpenEnv over HTTP.
Endpoints: GET /, POST /reset, POST /step, GET /state, GET /health
"""

import json
import math
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from environment import (
    DataCleanAction,
    DataCleanEnv,
    TASKS,
)

app = FastAPI(title="DataClean OpenEnv", version="1.0.0")

# One environment instance per process (stateful)
_env: DataCleanEnv = DataCleanEnv("fill_missing")


def _safe_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf so JSON serialisation never fails."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_safe_json(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(_safe_json(content), ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "fill_missing"


class StepRequest(BaseModel):
    operation: str
    params: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return SafeJSONResponse({
        "name": "DataClean OpenEnv",
        "version": "1.0.0",
        "docs": "/docs",
        "tasks": list(TASKS.keys()),
    })


@app.get("/health")
def health():
    return SafeJSONResponse({"status": "ok"})


@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    global _env
    task = body.task if body.task in TASKS else "fill_missing"
    _env = DataCleanEnv(task)
    obs = _env.reset()
    return SafeJSONResponse(obs.model_dump())


@app.post("/step")
def step(body: StepRequest):
    action = DataCleanAction(operation=body.operation, params=body.params)
    result = _env.step(action)
    payload = {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }
    return SafeJSONResponse(payload)


@app.get("/state")
def state():
    return SafeJSONResponse(_env.state())


@app.get("/tasks")
def tasks():
    return SafeJSONResponse({
        name: {
            "description": t["description"],
            "difficulty": t["difficulty"],
            "max_steps": t["max_steps"],
        }
        for name, t in TASKS.items()
    })
