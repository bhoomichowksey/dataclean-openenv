# DataClean OpenEnv

🚀 Live API: https://bhoomichowksey-dataclean-env.hf.space  
📂 GitHub Repo: https://github.com/bhoomichowksey/dataclean-openecv
---
title: DataClean Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# DataClean OpenEnv 🧹

An **OpenEnv-compliant** reinforcement learning environment where an AI agent learns to clean messy real-world tabular data through a sequence of operations.

## Why This Environment?

Data cleaning is one of the most time-consuming tasks in any data pipeline — data scientists spend up to 80% of their time on it. This environment trains and evaluates agents on realistic data quality problems:

- Null / missing values
- Column naming inconsistencies
- Duplicate rows
- Wrong data types
- Outliers in numeric columns
- Format inconsistencies (phone numbers, etc.)
- Inconsistent categorical values (typos, mixed casing)

---

## Tasks

| # | Name | Difficulty | Max Steps | Description |
|---|------|-----------|-----------|-------------|
| 1 | `fill_missing` | 🟢 Easy | 8 | Fill null values with defaults; clean column names (strip spaces, lowercase) |
| 2 | `dedup_typefix` | 🟡 Medium | 10 | Remove duplicate rows; cast `price` → float, `in_stock` → bool |
| 3 | `full_pipeline` | 🔴 Hard | 15 | Remove salary outliers, normalize phone to 10 digits, fix status typos/casing |

---

## Observation Space

```
DataCleanObservation:
  task_name         str               # which task is running
  step              int               # current step (1-indexed)
  columns           list[str]         # current column names
  dtypes            dict[str, str]    # column → dtype string
  shape             tuple[int, int]   # (rows, cols)
  missing_counts    dict[str, int]    # nulls per column
  sample_rows       list[dict]        # first 5 rows
  duplicate_count   int               # number of duplicate rows
  last_action_error str | None        # error from last step, if any
  hint              str               # task-specific hint
```

## Action Space

```
DataCleanAction:
  operation   str           # one of the operations below
  params      dict          # operation-specific parameters
```

### Available Operations

| Operation | Required Params | Description |
|-----------|----------------|-------------|
| `fill_missing` | `column?`, `value?` | Fill nulls (all columns if none specified) |
| `rename_columns` | `mapping?` | Strip spaces + lowercase (or custom mapping) |
| `drop_duplicates` | `subset?` | Remove exact duplicate rows |
| `cast_column` | `column`, `dtype` | Cast to float/bool/int/str |
| `remove_outliers` | `column`, `method?` | IQR-based outlier removal |
| `normalize_phone` | `column?` | Clean phone numbers to 10 digits |
| `fix_categorical` | `column`, `lowercase?`, `titlecase?`, `mapping?` | Normalize categories |
| `fix_typos` | `column`, `corrections` | Fix known typos via mapping |

---

## Reward Function

- **Dense rewards** — reward at each step = improvement in task score
- **Score range** — 0.0 to 1.0 (deterministic programmatic grader)
- **Partial credit** — each cleaning operation that improves the data earns reward
- **Episode ends** — when score ≥ 0.95 or max_steps reached

---

## API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET` | `/` | Environment info |
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Reset environment. Body: `{"task": "fill_missing"}` |
| `POST` | `/step` | Take action. Body: `{"operation": "...", "params": {...}}` |
| `GET` | `/state` | Current environment state |
| `GET` | `/tasks` | List all tasks |

---

## Setup & Usage

### Local

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860

# Test
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "fill_missing"}'
```

### Docker

```bash
docker build -t dataclean-env .
docker run -p 7860:7860 dataclean-env
```

### Inference (Baseline)

```bash
export HF_TOKEN="your_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export DATACLEAN_TASK="all"

python inference.py
```

---

## Baseline Scores

| Task | Score | Difficulty |
|------|-------|-----------|
| `fill_missing` | ~0.70 | Easy |
| `dedup_typefix` | ~0.55 | Medium |
| `full_pipeline` | ~0.35 | Hard |
| **Average** | **~0.53** | — |

---

## Validation

```bash
./validate-submission.sh https://bhoomichowksey-dataclean-env.hf.space
```
