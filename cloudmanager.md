# OpenEnv Cloud Resource Manager — Full Build Specification

## For: Gemini Flash via Antigravity

## Task: Generate all project files exactly as specified below. Do not deviate from filenames, structure, or logic.

---

## Project Overview

Build a complete OpenEnv-compliant simulation environment called the **Cloud Resource Manager (CRM)**. An LLM agent manages a fleet of Virtual Machines (VMs), balancing server performance (latency) against operational cost. This mimics real FinOps/Cloud Scaling decisions.

The environment exposes a FastAPI HTTP API with `step()`, `reset()`, and `state()` endpoints. It must be deployable as a Docker container on Hugging Face Spaces on port 7860.

---

## File Structure to Generate

```
openenv_cloud_manager/
├── models.py
├── env.py
├── tasks.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

Generate every file listed. Do not skip any.

---

## FILE 1: `models.py`

```python
from pydantic import BaseModel

class CloudObservation(BaseModel):
    num_vms: int
    avg_cpu_percent: float
    current_latency_ms: float
    total_cost: float
    incoming_requests: int

class CloudAction(BaseModel):
    # 0: Scale Down, 1: No-op, 2: Scale Up
    action: int

class CloudState(BaseModel):
    step_count: int
    vms_active: int
    accumulated_cost: float
```

---

## FILE 2: `env.py`

### Simulation Physics / Rules

- **CPU Load** = `min(100, (requests / (vms × 200)) × 100)`
- **Latency** = `20 + (cpu² / 100)` ms — exponential spike at high CPU
- **Cost** = `vms × $0.05` per step (1 step = 1 simulated hour)
- **Episode length** = 24 steps (one simulated day)
- Minimum 1 VM at all times — cannot scale below 1

### Actions

| Value | Action     | Effect                                             |
| ----- | ---------- | -------------------------------------------------- |
| 0     | Scale Down | Remove 1 VM (if vms > 1), reduces cost, raises CPU |
| 1     | No-op      | Do nothing                                         |
| 2     | Scale Up   | Add 1 VM, raises cost, lowers CPU                  |

### Reward Function

```
reward = 1.0
if latency_ms > 200:   reward -= 0.5    # penalty: slow app
if cpu_percent > 95:   reward -= 0.3    # penalty: overloaded
reward -= (vms * 0.1)                   # penalty: cost
```

### HTTP Endpoints Required

| Method | Route    | Body              | Returns                       |
| ------ | -------- | ----------------- | ----------------------------- |
| GET    | `/`      | none              | `{"status": "ok"}`            |
| POST   | `/reset` | none              | `CloudObservation`            |
| POST   | `/step`  | `{"action": int}` | `{observation, reward, done}` |
| GET    | `/state` | none              | `CloudState`                  |

### Full `env.py` code

```python
import random
from fastapi import FastAPI
from models import CloudObservation, CloudAction, CloudState

app = FastAPI()

class CloudEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.vms = 2
        self.cost = 0.0
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        requests = random.randint(100, 1000)
        cpu = min(100, (requests / (self.vms * 200)) * 100)
        latency = 20 + (cpu ** 2 / 100)
        return {
            "num_vms": self.vms,
            "avg_cpu_percent": round(cpu, 2),
            "current_latency_ms": round(latency, 2),
            "total_cost": round(self.cost, 2),
            "incoming_requests": requests
        }

    def step(self, action: int):
        if action == 2:
            self.vms += 1
        elif action == 0 and self.vms > 1:
            self.vms -= 1

        self.cost += (self.vms * 0.05)
        self.steps += 1
        obs = self._get_obs()

        reward = 1.0
        if obs["current_latency_ms"] > 200: reward -= 0.5
        if obs["avg_cpu_percent"] > 95:     reward -= 0.3
        reward -= (self.vms * 0.1)

        done = self.steps >= 24
        return obs, round(reward, 4), done

    def state(self):
        return {
            "step_count": self.steps,
            "vms_active": self.vms,
            "accumulated_cost": round(self.cost, 2)
        }


env = CloudEnv()


@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    obs = env.reset()
    return CloudObservation(**obs)

@app.post("/step")
def step(action: CloudAction):
    obs, reward, done = env.step(action.action)
    return {
        "observation": CloudObservation(**obs),
        "reward": reward,
        "done": done
    }

@app.get("/state")
def state():
    return CloudState(**env.state())
```

---

## FILE 3: `tasks.py`

### Rules for graders

- Every grader must return a `float` between `0.0` and `1.0` inclusive
- Graders must NOT always return the same value — scores must vary based on agent behaviour
- Three tasks required: easy, medium, hard

```python
from typing import List


def grade_scale_up_basic(action_history: List[int]) -> float:
    """
    Easy task: Agent starts with 1 VM and 800+ requests (guaranteed overload).
    Success: Agent takes action 2 (scale up) within the first 3 steps.
    """
    if not action_history:
        return 0.0
    early = action_history[:3]
    if 2 in early:
        return 1.0
    elif 2 in action_history:
        return 0.5  # scaled up but too late
    return 0.0


def grade_latency_control(latency_history: List[float]) -> float:
    """
    Medium task: Fluctuating traffic over 24 steps.
    Success: Average latency across all steps stays below 100ms.
    """
    if not latency_history:
        return 0.0
    avg = sum(latency_history) / len(latency_history)
    if avg <= 100:
        return 1.0
    elif avg <= 200:
        return 0.5
    return max(0.0, round(1.0 - (avg / 400), 4))


def grade_cost_optimization(latency_history: List[float], total_cost: float, budget: float = 5.0) -> float:
    """
    Hard task: Black Friday — sustained high traffic (700–1000 req/step) for 24 steps.
    Budget cap: $5.00 total cost.
    Success: Keep latency below 150ms while staying under budget.
    Score = weighted combination of performance score and cost score.
    """
    if not latency_history:
        return 0.0
    avg_latency = sum(latency_history) / len(latency_history)
    perf_score = max(0.0, 1.0 - (avg_latency / 300))
    cost_score = 1.0 if total_cost <= budget else max(0.0, 1.0 - ((total_cost - budget) / budget))
    return round((perf_score * 0.6) + (cost_score * 0.4), 4)
```

---

## FILE 4: `inference.py`

### Rules (do not break any of these)

- File must be named exactly `inference.py` in the root directory
- Must use the `openai` Python client — not `litellm`, not raw `requests` for LLM calls
- Must read `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from environment variables
- Must complete all 3 tasks in under 20 minutes total
- Must print scores for all 3 tasks without error
- `/reset` takes no request body
- `/step` takes `{"action": int}`
- Use `python-dotenv` to load `.env` locally; on HF Spaces the vars come from Secrets

```python
import os
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME   = os.environ["MODEL_NAME"]
HF_TOKEN     = os.environ["HF_TOKEN"]

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are a cloud engineer managing a fleet of Virtual Machines.
You receive server metrics and must respond with exactly one digit:
0 = Scale Down (removes a VM, reduces cost, increases CPU load)
1 = No-op (do nothing)
2 = Scale Up (adds a VM, increases cost, reduces CPU load)
Respond with only the digit. No explanation. No punctuation."""

ENV_BASE = os.environ.get("ENV_BASE_URL", "http://localhost:7860")


def get_action(obs: dict) -> int:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(obs)}
        ],
        max_tokens=5,
        temperature=0.0,
        stream=False,
    )
    raw = response.choices[0].message.content.strip()
    for char in raw:
        if char in ("0", "1", "2"):
            return int(char)
    return 1  # fallback: No-op


def run_task(task_id: str):
    print(f"\n{'='*40}")
    print(f"Running task: {task_id}")
    print('='*40)

    obs = requests.post(f"{ENV_BASE}/reset").json()
    action_history = []
    latency_history = []
    done = False
    step = 0

    while not done and step < 24:
        action = get_action(obs)
        result = requests.post(f"{ENV_BASE}/step", json={"action": action}).json()
        obs     = result["observation"]
        reward  = result["reward"]
        done    = result["done"]

        action_history.append(action)
        latency_history.append(obs["current_latency_ms"])

        print(f"Step {step:02d} | action={action} | latency={obs['current_latency_ms']}ms | reward={reward:+.2f}")
        step += 1
        time.sleep(0.5)  # avoid hitting Gemini rate limits

    final_state = requests.get(f"{ENV_BASE}/state").json()
    print(f"Final accumulated cost: ${final_state['accumulated_cost']}")
    return action_history, latency_history, final_state["accumulated_cost"]


if __name__ == "__main__":
    from tasks import grade_scale_up_basic, grade_latency_control, grade_cost_optimization

    ah, lh, cost = run_task("scale-up-basic")
    score1 = grade_scale_up_basic(ah)
    print(f"[scale-up-basic] Score: {score1}")

    ah, lh, cost = run_task("latency-control")
    score2 = grade_latency_control(lh)
    print(f"[latency-control] Score: {score2}")

    ah, lh, cost = run_task("cost-optimization-heavy")
    score3 = grade_cost_optimization(lh, cost)
    print(f"[cost-optimization-heavy] Score: {score3}")

    print(f"\nAll scores: {score1}, {score2}, {score3}")
    assert all(0.0 <= s <= 1.0 for s in [score1, score2, score3]), "Score out of range!"
    print("All scores valid (0.0–1.0). Ready to submit.")
```

---

## FILE 5: `openenv.yaml`

```yaml
version: "1.0.0"
name: "CloudResourceManager-v1"
entry_point: "env:app"
environment_variables:
  - API_BASE_URL
  - MODEL_NAME
  - HF_TOKEN
endpoints:
  reset: "/reset"
  step: "/step"
  state: "/state"
tasks:
  - id: "scale-up-basic"
    difficulty: "easy"
    grader: "tasks:grade_scale_up_basic"
  - id: "latency-control"
    difficulty: "medium"
    grader: "tasks:grade_latency_control"
  - id: "cost-optimization-heavy"
    difficulty: "hard"
    grader: "tasks:grade_cost_optimization"
```

---

## FILE 6: `Dockerfile`

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Important notes for Dockerfile

- CMD must start `uvicorn`, not `python3 env.py`
- Port must be 7860 — this is required by Hugging Face Spaces
- Do not use `ENV` instructions for secrets — those come from HF Secrets at runtime

---

## FILE 7: `requirements.txt`

```
fastapi
uvicorn
pydantic
openai
requests
python-dotenv
openenv
```

---

## FILE 8: `README.md`

````markdown
# OpenEnv Cloud Resource Manager

A simulation environment where an LLM agent manages a fleet of Virtual Machines, balancing performance against cost.

## Local Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the root:

```
API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
MODEL_NAME=gemini-1.5-flash
HF_TOKEN=your_google_ai_studio_key
```

## Run the Environment

```bash
uvicorn env:app --host 0.0.0.0 --port 7860 --reload
```

## Run the Inference Agent (separate terminal)

```bash
python3 inference.py
```

## Test Endpoints Manually

```bash
curl http://localhost:7860/
curl -X POST http://localhost:7860/reset
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action": 2}'
curl http://localhost:7860/state
```

## Docker

```bash
docker build -t crm-env .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/" \
  -e MODEL_NAME="gemini-1.5-flash" \
  -e HF_TOKEN="your_key_here" \
  crm-env
```

## Validate OpenEnv Spec

```bash
openenv validate openenv.yaml
```

## Environment Variables (required)

| Variable     | Description                           |
| ------------ | ------------------------------------- |
| API_BASE_URL | Base URL of the OpenAI-compatible API |
| MODEL_NAME   | Model identifier for inference        |
| HF_TOKEN     | API key / Hugging Face token          |

## Tasks

| ID                      | Difficulty | Goal                                       |
| ----------------------- | ---------- | ------------------------------------------ |
| scale-up-basic          | Easy       | Scale up within 3 steps when overloaded    |
| latency-control         | Medium     | Keep avg latency under 100ms over 24 steps |
| cost-optimization-heavy | Hard       | Stay under $5 budget on Black Friday load  |
````

---

## Environment Variable Setup (for reference)

### Local `.env` file (never commit this)

```
API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
MODEL_NAME=gemini-1.5-flash
HF_TOKEN=your_gemini_api_key_from_aistudio_google_com
```

### To switch to Groq later (zero code change needed)

```
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama3-8b-8192
HF_TOKEN=your_groq_api_key
```

### To switch to Together AI later (zero code change needed)

```
API_BASE_URL=https://api.together.xyz/v1
MODEL_NAME=mistralai/Mixtral-8x7B-Instruct-v0.1
HF_TOKEN=your_together_api_key
```

---

## Pre-Submission Checklist

Before submitting the HF Spaces URL, verify every item:

- [ ] `GET /` returns `{"status": "ok"}` with HTTP 200
- [ ] `POST /reset` returns a valid `CloudObservation` JSON
- [ ] `POST /step` with `{"action": 1}` returns observation + reward + done
- [ ] `GET /state` returns step_count, vms_active, accumulated_cost
- [ ] `openenv validate openenv.yaml` passes with no errors
- [ ] `docker build` completes without error
- [ ] `python3 inference.py` runs all 3 tasks and prints 3 scores between 0.0–1.0
- [ ] HF Space status is green (Running)
- [ ] All 3 env vars set as Secrets in HF Space settings
- [ ] Space visibility is set to Public
