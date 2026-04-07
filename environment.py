"""
DataClean OpenEnv — Data Cleaning Pipeline Environment
Fully compliant with OpenEnv spec: typed Pydantic models, step()/reset()/state()
"""

import copy
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Typed Models
# ---------------------------------------------------------------------------

class DataCleanObservation(BaseModel):
    task_name: str
    step: int
    columns: List[str]
    dtypes: Dict[str, str]
    shape: Tuple[int, int]
    missing_counts: Dict[str, int]
    sample_rows: List[Dict[str, Any]]
    duplicate_count: int
    last_action_error: Optional[str] = None
    hint: str = ""


class DataCleanAction(BaseModel):
    operation: str          # e.g. "fill_missing", "rename_columns", "drop_duplicates" …
    params: Dict[str, Any] = {}


class DataCleanReward(BaseModel):
    value: float            # 0.0 – 1.0
    breakdown: Dict[str, float] = {}


class StepResult(BaseModel):
    observation: DataCleanObservation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

def _make_fill_missing_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Name ": ["Alice", "Bob", None, "Diana", "Eve", None, "Grace", "Hank"],
        " Age":  [25.0, None, 30.0, None, 22.0, 28.0, None, 35.0],
        "Email": ["a@x.com", "b@x.com", "c@x.com", None, "e@x.com", None, "g@x.com", "h@x.com"],
        "Score": [88.5, 72.0, None, 91.0, None, 65.0, 77.0, None],
    })


def _make_dedup_typefix_df() -> pd.DataFrame:
    return pd.DataFrame({
        "product":  ["Apple", "Banana", "Apple", "Cherry", "Banana", "Date", "Apple", "Elderberry"],
        "price":    ["1.20", "0.50", "1.20", "3.00", "0.50", "2.75", "1.20", "abc"],
        "in_stock": ["True", "False", "True", "True", "False", "yes", "True", "no"],
        "quantity": [10, 5, 10, 3, 5, 7, 10, 2],
    })


def _make_full_pipeline_df() -> pd.DataFrame:
    return pd.DataFrame({
        "emp_id":   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "salary":   [50000, 52000, 999999, 48000, 51000, 47000, 53000, 1000000, 49000, 50500],
        "phone":    ["1234567890", "123-456-7890", "(123)456-7890", "123.456.7890",
                     "12345", "9876543210", "987-654-3210", "(987)654-3210",
                     "987.654.3210", "00000"],
        "status":   ["active", "Actve", "ACTIVE", "inactive", "Inactive",
                     "INACTIVE", "actve", "active", "inactive", "ACTVE"],
        "dept":     ["Eng", "eng", "ENG", "HR", "hr", "Hr", "Finance", "finance", "FINANCE", "Eng"],
    })


TASKS = {
    "fill_missing": {
        "description": "Fill null values (0 for numbers, 'unknown' for strings) and rename columns (strip spaces, lowercase).",
        "difficulty": "easy",
        "max_steps": 8,
        "make_df": _make_fill_missing_df,
        "hint": "Fill nulls with 0 for numbers and 'unknown' for strings. Rename columns: strip spaces and lowercase.",
    },
    "dedup_typefix": {
        "description": "Remove exact duplicate rows; cast price→float (invalid→0.0) and in_stock→bool.",
        "difficulty": "medium",
        "max_steps": 10,
        "make_df": _make_dedup_typefix_df,
        "hint": "Drop duplicate rows first. Cast 'price' to float (errors→0.0). Map in_stock to bool (true values: True/yes).",
    },
    "full_pipeline": {
        "description": "Remove salary outliers (IQR), normalize phone to 10 digits, fix status typos/casing, normalize dept casing.",
        "difficulty": "hard",
        "max_steps": 15,
        "make_df": _make_full_pipeline_df,
        "hint": "Remove salary outliers using IQR method. Normalize phone numbers to 10 digits. Fix status to active/inactive. Normalize dept to title case.",
    },
}


# ---------------------------------------------------------------------------
# Graders  (return float 0.0–1.0)
# ---------------------------------------------------------------------------

def _grade_fill_missing(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    scores: Dict[str, float] = {}

    # 1. No missing values
    total_cells = df.shape[0] * df.shape[1]
    missing = int(df.isnull().sum().sum())
    scores["no_missing"] = max(0.0, 1.0 - missing / total_cells)

    # 2. Column names clean (stripped + lowercase)
    clean_cols = all(c == c.strip().lower() for c in df.columns)
    scores["clean_columns"] = 1.0 if clean_cols else 0.0

    # 3. Numeric columns filled with 0 (not 'unknown')
    num_cols = [c for c in df.columns if df[c].dtype in [float, int]]
    if num_cols:
        zero_filled = sum(1 for c in num_cols if (df[c] == 0).any() or df[c].notnull().all())
        scores["numeric_fill"] = zero_filled / len(num_cols)
    else:
        scores["numeric_fill"] = 1.0

    total = (scores["no_missing"] * 0.5 + scores["clean_columns"] * 0.3 + scores["numeric_fill"] * 0.2)
    return round(min(max(total, 0.0), 1.0), 4), scores


def _grade_dedup_typefix(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    scores: Dict[str, float] = {}

    # 1. Duplicates removed
    dup_count = int(df.duplicated().sum())
    scores["no_duplicates"] = 1.0 if dup_count == 0 else max(0.0, 1.0 - dup_count / len(df))

    # 2. Price is float
    try:
        pd.to_numeric(df["price"], errors="raise")
        scores["price_float"] = 1.0
    except Exception:
        numeric_ratio = pd.to_numeric(df["price"], errors="coerce").notnull().mean()
        scores["price_float"] = float(numeric_ratio)

    # 3. in_stock is bool-like
    if df["in_stock"].dtype == bool:
        scores["in_stock_bool"] = 1.0
    else:
        valid = df["in_stock"].isin([True, False, "True", "False", 0, 1])
        scores["in_stock_bool"] = float(valid.mean())

    total = scores["no_duplicates"] * 0.4 + scores["price_float"] * 0.35 + scores["in_stock_bool"] * 0.25
    return round(min(max(total, 0.0), 1.0), 4), scores


def _grade_full_pipeline(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    scores: Dict[str, float] = {}

    # 1. Salary outliers removed (no values > 3*median)
    median_sal = df["salary"].median()
    outliers = (df["salary"] > 3 * median_sal).sum()
    scores["salary_outliers"] = 1.0 if outliers == 0 else max(0.0, 1.0 - outliers / len(df))

    # 2. Phone normalized to 10 digits
    def is_10digit(p: Any) -> bool:
        return bool(re.fullmatch(r"\d{10}", str(p)))
    phone_ok = df["phone"].apply(is_10digit).mean()
    scores["phone_normalized"] = float(phone_ok)

    # 3. Status in {active, inactive}
    valid_status = {"active", "inactive"}
    status_ok = df["status"].str.lower().isin(valid_status).mean()
    scores["status_clean"] = float(status_ok)

    # 4. Dept consistently cased (title case)
    dept_ok = (df["dept"] == df["dept"].str.title()).mean()
    scores["dept_cased"] = float(dept_ok)

    total = (scores["salary_outliers"] * 0.3 + scores["phone_normalized"] * 0.3
             + scores["status_clean"] * 0.25 + scores["dept_cased"] * 0.15)
    return round(min(max(total, 0.0), 1.0), 4), scores


GRADERS = {
    "fill_missing":  _grade_fill_missing,
    "dedup_typefix": _grade_dedup_typefix,
    "full_pipeline": _grade_full_pipeline,
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DataCleanEnv:
    """OpenEnv-compliant data cleaning environment."""

    def __init__(self, task_name: str = "fill_missing"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Choose from: {list(TASKS)}")
        self.task_name = task_name
        self._task = TASKS[task_name]
        self._df: pd.DataFrame = pd.DataFrame()
        self._initial_df: pd.DataFrame = pd.DataFrame()
        self._step_count = 0
        self._done = False
        self._last_error: Optional[str] = None
        self._last_score: float = 0.0

    # ---- helpers -----------------------------------------------------------

    def _obs(self) -> DataCleanObservation:
        sample = self._df.head(5).where(pd.notnull(self._df.head(5)), other=None).to_dict(orient="records")
        return DataCleanObservation(
            task_name=self.task_name,
            step=self._step_count,
            columns=list(self._df.columns),
            dtypes={c: str(t) for c, t in self._df.dtypes.items()},
            shape=(int(self._df.shape[0]), int(self._df.shape[1])),
            missing_counts={c: int(v) for c, v in self._df.isnull().sum().items()},
            sample_rows=sample,
            duplicate_count=int(self._df.duplicated().sum()),
            last_action_error=self._last_error,
            hint=self._task["hint"],
        )

    def _compute_reward(self) -> float:
        score, _ = GRADERS[self.task_name](self._df)
        delta = score - self._last_score
        self._last_score = score
        # Partial progress reward; clamp to [0, 1]
        return round(min(max(delta, 0.0), 1.0), 4)

    # ---- OpenEnv API -------------------------------------------------------

    def reset(self) -> DataCleanObservation:
        self._df = self._task["make_df"]()
        self._initial_df = self._df.copy()
        self._step_count = 0
        self._done = False
        self._last_error = None
        self._last_score, _ = GRADERS[self.task_name](self._df)
        return self._obs()

    def step(self, action: DataCleanAction) -> StepResult:
        if self._done:
            return StepResult(observation=self._obs(), reward=0.0, done=True,
                              info={"error": "Episode already done"})

        self._step_count += 1
        self._last_error = None
        df = self._df.copy()

        try:
            df = self._apply(df, action)
            self._df = df
        except Exception as exc:
            self._last_error = str(exc)

        reward = self._compute_reward()
        final_score, breakdown = GRADERS[self.task_name](self._df)

        max_steps: int = self._task["max_steps"]
        self._done = final_score >= 0.95 or self._step_count >= max_steps

        return StepResult(
            observation=self._obs(),
            reward=reward,
            done=self._done,
            info={"score": final_score, "breakdown": breakdown},
        )

    def state(self) -> Dict[str, Any]:
        score, breakdown = GRADERS[self.task_name](self._df)
        return {
            "task_name": self.task_name,
            "step": self._step_count,
            "done": self._done,
            "score": score,
            "breakdown": breakdown,
            "shape": list(self._df.shape),
            "columns": list(self._df.columns),
            "missing_counts": {c: int(v) for c, v in self._df.isnull().sum().items()},
            "duplicate_count": int(self._df.duplicated().sum()),
        }

    # ---- action dispatcher -------------------------------------------------

    def _apply(self, df: pd.DataFrame, action: DataCleanAction) -> pd.DataFrame:
        op = action.operation
        p = action.params

        if op == "fill_missing":
            col = p.get("column")
            value = p.get("value", 0)
            if col:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found")
                df[col] = df[col].fillna(value)
            else:
                # fill all
                for c in df.columns:
                    fill_val = 0 if df[c].dtype in [float, int] else "unknown"
                    df[c] = df[c].fillna(fill_val)
            return df

        if op == "rename_columns":
            mapping = p.get("mapping", {})
            if mapping:
                df = df.rename(columns=mapping)
            else:
                df.columns = [c.strip().lower() for c in df.columns]
            return df

        if op == "drop_duplicates":
            subset = p.get("subset")
            df = df.drop_duplicates(subset=subset)
            return df

        if op == "cast_column":
            col = p["column"]
            dtype = p["dtype"]
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found")
            if dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            elif dtype == "bool":
                true_vals = {"true", "yes", "1", "t"}
                df[col] = df[col].astype(str).str.lower().isin(true_vals)
            elif dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            elif dtype == "str":
                df[col] = df[col].astype(str)
            else:
                raise ValueError(f"Unsupported dtype '{dtype}'")
            return df

        if op == "remove_outliers":
            col = p["column"]
            method = p.get("method", "iqr")
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found")
            if method == "iqr":
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
            return df

        if op == "normalize_phone":
            col = p.get("column", "phone")
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found")
            def clean_phone(v: Any) -> str:
                digits = re.sub(r"\D", "", str(v))
                return digits[-10:] if len(digits) >= 10 else digits
            df[col] = df[col].apply(clean_phone)
            return df

        if op == "fix_categorical":
            col = p["column"]
            mapping = p.get("mapping", {})
            lower = p.get("lowercase", False)
            title = p.get("titlecase", False)
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found")
            if lower:
                df[col] = df[col].str.lower()
            if title:
                df[col] = df[col].str.title()
            if mapping:
                df[col] = df[col].replace(mapping)
            return df

        if op == "fix_typos":
            col = p["column"]
            corrections = p.get("corrections", {})
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found")
            df[col] = df[col].str.lower()
            df[col] = df[col].replace(corrections)
            return df

        raise ValueError(f"Unknown operation '{op}'")
