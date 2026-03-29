---
title: OpenEnv Cloud Resource Manager
emoji: ☁️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ☁️ OpenEnv Cloud Resource Manager

> A real-world cloud infrastructure simulation where an LLM agent acts as a cloud engineer — scaling Virtual Machines up and down to balance performance against cost.

Built for the **Scaler School of Technology × Meta PyTorch Hackathon**.  
Submission by **kenkao05** | HF Space: [kenkao05/openenv-cloud-manager](https://huggingface.co/spaces/kenkao05/openenv-cloud-manager)

---

## What Is This?

This is a **Flight Simulator for Cloud Engineers**. The environment simulates a fleet of Virtual Machines handling real-time traffic. An AI agent observes server metrics every hour and decides whether to scale up, scale down, or do nothing — exactly the kind of decision a FinOps engineer makes daily at companies like AWS, Google Cloud, or Azure.

The environment is fully OpenEnv-compliant, exposing a standard `step()` / `reset()` / `state()` HTTP API that any LLM agent can interact with.

---

## The Problem

| Metric   | Description                                               |
| -------- | --------------------------------------------------------- |
| CPU Load | `min(100, (requests / (vms × 200)) × 100)`                |
| Latency  | `20 + (cpu² / 100)` ms — spikes exponentially at high CPU |
| Cost     | `vms × $0.05` per step (1 step = 1 simulated hour)        |
| Episode  | 24 steps = 1 simulated day                                |

### Actions Available to the Agent

| Value | Action     | Effect                                        |
| ----- | ---------- | --------------------------------------------- |
| 0     | Scale Down | Remove 1 VM — saves money, raises CPU         |
| 1     | No-op      | Do nothing                                    |
| 2     | Scale Up   | Add 1 VM — costs more, lowers CPU and latency |

### Reward Function

```python
reward = 1.0
if latency_ms > 200:   reward -= 0.5   # app too slow
if cpu_percent > 95:   reward -= 0.3   # servers overloaded
reward -= (vms * 0.1)                  # penalize cost
```

---

## Tasks & Graders

| ID                        | Difficulty | Scenario                                             | Success Condition          |
| ------------------------- | ---------- | ---------------------------------------------------- | -------------------------- |
| `scale-up-basic`          | Easy       | 1 VM, 800+ requests (guaranteed overload)            | Scale up within 3 steps    |
| `latency-control`         | Medium     | Fluctuating traffic over 24 steps                    | Avg latency < 100ms        |
| `cost-optimization-heavy` | Hard       | Black Friday — sustained high traffic, $5 budget cap | Low latency + under budget |

All graders return a float in `0.0–1.0`.

---

## API Endpoints

| Method | Route    | Description                                              |
| ------ | -------- | -------------------------------------------------------- |
| `GET`  | `/`      | Health check — returns `{"status": "ok"}`                |
| `POST` | `/reset` | Start a new episode, returns initial observation         |
| `POST` | `/step`  | Send `{"action": 0/1/2}`, returns obs + reward + done    |
| `GET`  | `/state` | Returns current step count, VMs active, accumulated cost |

---

## Local Setup

```bash
git clone https://github.com/kenkao05/openenv-cloud-manager
cd openenv-cloud-manager
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.1-8b-instant
HF_TOKEN=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

---

## Run Locally

**Terminal 1 — Start the environment server:**

```bash
uvicorn env:app --host 0.0.0.0 --port 7860 --reload
```

**Terminal 2 — Test endpoints:**

```bash
curl http://localhost:7860/
curl -X POST http://localhost:7860/reset
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action": 2}'
curl http://localhost:7860/state
```

**Terminal 2 — Run the LLM agent:**

```bash
source venv/bin/activate
python3 inference.py
```

---

## Run via Docker

```bash
docker build -t crm-env .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.groq.com/openai/v1" \
  -e MODEL_NAME="llama-3.1-8b-instant" \
  -e HF_TOKEN="your_groq_key_here" \
  crm-env
```

---

## Switching LLM Providers

No code changes needed — just update the environment variables:

| Provider                   | API_BASE_URL                                        | MODEL_NAME                             |
| -------------------------- | --------------------------------------------------- | -------------------------------------- |
| Groq (recommended, free)   | `https://api.groq.com/openai/v1`                    | `llama-3.1-8b-instant`                 |
| Google Gemini (free)       | `https://generativelanguage.googleapis.com/v1beta/` | `gemini-2.0-flash`                     |
| Together AI (free credits) | `https://api.together.xyz/v1`                       | `mistralai/Mixtral-8x7B-Instruct-v0.1` |

---

## Environment Variables

| Variable       | Description                               |
| -------------- | ----------------------------------------- |
| `API_BASE_URL` | Base URL of any OpenAI-compatible LLM API |
| `MODEL_NAME`   | Model identifier to use for inference     |
| `HF_TOKEN`     | Your API key for the chosen provider      |

---

## Tech Stack

- **FastAPI** — HTTP API wrapper around the simulation
- **Pydantic** — Strict data validation for all models
- **Uvicorn** — ASGI web server
- **OpenAI Python client** — LLM calls (provider-agnostic)
- **Docker** — Container deployment
- **Hugging Face Spaces** — Live hosting
