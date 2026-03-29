# OpenEnv Cloud Resource Manager — Full Project Explanation
## Scaler School of Technology × Meta PyTorch Hackathon

**Author:** Ken (kenkao05)  
**College:** SAL College of Engineering, GTU — BE Semester VI  
**Submission Deadline:** April 8, 2026, 11:59 PM IST  
**HF Space:** https://huggingface.co/spaces/kenkao05/openenv-cloud-manager

---

## 1. The Hackathon — Context

This is the **Scaler School of Technology × Meta PyTorch Hackathon**. The challenge is to build an **OpenEnv-compliant simulation environment** — a world that an AI agent can learn from by interacting with it through a standard API.

The problem statement is open-ended:
> *"Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard step() / reset() / state() API."*

You pick your own idea. The judges evaluate:

| Criterion | Weight |
|---|---|
| Real-world utility | 30% |
| Task & Grader Quality | 25% |
| Environment Design (Pydantic models) | 20% |
| OpenEnv Spec Compliance | 15% |
| Novelty | 10% |

**Judging happens in 3 phases:**
1. **Automated validation** — pass/fail gate. If your environment doesn't deploy, respond to HTTP pings, or produce valid scores, you're disqualified before a human even looks at it.
2. **Agentic evaluation** — judges run your `inference.py` and a standard LLM agent against your environment and compare scores.
3. **Human review** — top submissions reviewed by Meta and Hugging Face engineers.

---

## 2. What We Built — The Idea

We built the **Cloud Resource Manager (CRM)** — a simulation of a cloud engineering problem that real companies face every day.

### The Analogy
Think of it as a **Flight Simulator for Cloud Engineers**. A real cloud engineer sitting at AWS or Google Cloud has to constantly decide: do I add more servers because traffic is high? Do I remove servers because it's quiet and I'm wasting money? This is called **FinOps** — financial operations for cloud infrastructure.

We simulated this problem so an AI agent can practice making these decisions.

### What the Agent Sees (Observations)
Every step, the agent is told:
- How many Virtual Machines (VMs) are currently running
- What the average CPU usage is (0–100%)
- What the current response latency is (milliseconds)
- What the total cost has been so far
- How many requests are coming in from users

### What the Agent Can Do (Actions)
The agent picks one of three actions every step:
- **0** — Scale Down: Remove a VM. Saves money but increases CPU load.
- **1** — No-op: Do nothing.
- **2** — Scale Up: Add a VM. Costs more money but reduces CPU load and latency.

### The Goal
Balance **performance** (keep latency low, don't let servers get overloaded) against **cost** (don't run more VMs than you need). This is exactly the tradeoff real cloud engineers manage.

### The Physics
- CPU Load = `min(100, (incoming_requests / (vms × 200)) × 100)`
- Latency = `20 + (cpu² / 100)` ms — latency spikes exponentially when CPU is high
- Cost = `vms × $0.05` per step (1 step = 1 simulated hour)
- One episode = 24 steps = one simulated day
- Minimum 1 VM at all times

---

## 3. The Tech Stack — What Everything Is

### Python
The entire project is written in Python 3.12. Python is the dominant language for AI/ML work.

### Pydantic
A Python library that enforces strict data types. If you say a field must be a float and someone sends a string, Pydantic rejects it automatically. Used to define the exact shape of all data flowing through the system.

### FastAPI
A Python library that turns regular Python functions into a web server with HTTP endpoints. Without FastAPI, the simulation is just a Python class that nothing external can talk to. FastAPI wraps it and makes it accessible over the network.

### Uvicorn
The actual web server that listens for incoming HTTP requests and hands them to FastAPI. FastAPI defines the routes; uvicorn handles the network traffic.

### OpenAI Python Library
Not OpenAI the company — just their open-source client library for talking to LLM APIs. Since Groq, Google, Together AI, and others all copied OpenAI's API format, this one library works with all of them. You just change the `base_url`.

### Groq API
Groq is a company that runs open-source LLMs (like Meta's Llama 3.1) on extremely fast custom hardware. Completely free tier, no credit card needed. We use it as our LLM backend because:
- It's free
- It's OpenAI-compatible (works with the `openai` library)
- Fast enough to run 72 LLM calls (24 steps × 3 tasks) well within the 20-minute limit

**Model used:** `llama-3.1-8b-instant` — Meta's Llama 3.1 8 billion parameter model, optimized for speed.

### Docker
A tool that packages your entire application — code, Python version, all libraries — into a self-contained box called a container. The Dockerfile is the recipe. Docker guarantees the app runs identically everywhere regardless of what OS or Python version the host machine has. Required because HF Spaces runs your code on their servers, not your laptop.

### Hugging Face Spaces
Hugging Face is essentially the GitHub of AI — it hosts models, datasets, and apps. **HF Spaces** is their free app hosting platform. You push your Docker project and they build and run it, giving you a public URL. This is where your environment lives so the hackathon judges can ping it from anywhere in the world.

### Git
Version control system. Used to track changes and push code to HF Spaces.

### OpenEnv
A Python library and specification that defines how AI environments should be structured. The `openenv.yaml` manifest file tells the framework what your environment is called, where its endpoints are, and what tasks it has.

---

## 4. The Files — What Each One Does

```
openenv_cloud_manager/
├── models.py         # Data type definitions
├── env.py            # Simulation + web server
├── tasks.py          # Scoring functions
├── inference.py      # LLM agent that plays the game
├── openenv.yaml      # Environment manifest/config
├── Dockerfile        # Container build recipe
├── requirements.txt  # Python dependencies list
├── README.md         # Documentation + HF Spaces config
├── .env              # Your API keys (never committed to git)
├── .gitignore        # Tells git what NOT to upload
├── .dockerignore     # Tells Docker what NOT to include
└── venv/             # Virtual Python environment (never committed)
```

---

### `models.py` — The Language Dictionary

Before any two parts of the system can talk to each other, they need to agree on the exact format of data. This file defines three Pydantic models:

```python
class CloudObservation(BaseModel):
    num_vms: int
    avg_cpu_percent: float
    current_latency_ms: float
    total_cost: float
    incoming_requests: int
```

**CloudObservation** — what the agent sees each step. Five fields, all strictly typed.

```python
class CloudAction(BaseModel):
    action: int  # 0, 1, or 2
```

**CloudAction** — what the agent sends back. Just one integer.

```python
class CloudState(BaseModel):
    step_count: int
    vms_active: int
    accumulated_cost: float
```

**CloudState** — summary of current simulation status. Used by the `/state` endpoint.

---

### `env.py` — The Game World + Web Server

The most important file. Has two parts:

**Part 1 — CloudEnv class** (the simulation physics):
- `reset()` — starts a fresh episode: 2 VMs, zero cost, zero steps
- `_get_obs()` — calculates current CPU, latency, cost based on random incoming requests
- `step(action)` — applies the action, advances time by 1 hour, calculates reward

The reward function:
```python
reward = 1.0
if latency_ms > 200:   reward -= 0.5   # app is too slow
if cpu_percent > 95:   reward -= 0.3   # servers overloaded
reward -= (vms * 0.1)                  # penalize cost
```

**Part 2 — FastAPI HTTP wrapper**:
```python
@app.get("/")       # health check — returns {"status": "ok"}
@app.post("/reset") # start new episode
@app.post("/step")  # take one action
@app.get("/state")  # check current state
```

These four routes turn the Python class into a web server that anything — the LLM agent, the judges' system, a curl command — can talk to over HTTP.

---

### `tasks.py` — The Scoring System

Three grader functions, each returning a float between 0.0 and 1.0:

**Task 1 — Easy (`scale-up-basic`):**
Start with 1 VM and 800+ requests (guaranteed overload). Did the agent scale up within 3 steps?
- Scale up in first 3 steps → 1.0
- Scale up eventually but late → 0.5
- Never scale up → 0.0

**Task 2 — Medium (`latency-control`):**
Random traffic for 24 steps. Keep average latency under 100ms.
- Avg latency ≤ 100ms → 1.0
- Avg latency ≤ 200ms → 0.5
- Higher → scales down toward 0.0

**Task 3 — Hard (`cost-optimization-heavy`):**
"Black Friday" — sustained high traffic with a $5.00 budget cap. Weighted score combining latency performance (60%) and staying under budget (40%).

---

### `inference.py` — The LLM Agent

The script that plays the game using an actual AI model. Step by step:

1. Loads API credentials from environment variables
2. Creates an OpenAI client pointed at Groq's servers
3. Calls `/reset` to start a fresh episode
4. Loops 24 times:
   - Sends current observation to Llama 3.1 via Groq
   - Parses the response (should be 0, 1, or 2)
   - Sends that action to `/step`
   - Records the latency and action for grading
5. After all 3 tasks, calls graders and prints scores
6. Asserts all scores are between 0.0 and 1.0

**Key rules this file follows:**
- Named exactly `inference.py` (hackathon requirement)
- Uses `openai.OpenAI()` client (hackathon requirement)
- Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from env vars (hackathon requirement)
- Has `time.sleep(0.5)` between steps to avoid rate limits
- Falls back to action 1 (No-op) if the LLM returns something unparseable

---

### `openenv.yaml` — The Manifest

The ID card for your environment. The hackathon's automated system reads this to understand your project:

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

### `Dockerfile` — The Container Recipe

```dockerfile
FROM python:3.10-slim    # start with clean Python 3.10
WORKDIR /app             # set working directory
COPY . .                 # copy all project files in
RUN pip install -r requirements.txt   # install dependencies
EXPOSE 7860              # open port 7860 (required by HF Spaces)
CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "7860"]
```

Six lines that package your entire application into a portable container.

---

### `requirements.txt` — The Shopping List

```
fastapi        # web framework
uvicorn        # web server
pydantic       # data validation
openai         # LLM client library
requests       # HTTP client for inference.py to call the env
python-dotenv  # loads .env file locally
openenv        # OpenEnv specification library
```

---

### `.env` — Your Secrets (Never Committed)

```
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.1-8b-instant
HF_TOKEN=your_groq_api_key
```

This file is in `.gitignore` and `.dockerignore` — it never leaves your machine. On HF Spaces the same variables are set as Secrets in the Space settings.

---

## 5. The Pipeline — How Everything Connects

```
Your .env file
      │
      ▼
inference.py (the agent)
      │ reads obs, sends actions via HTTP
      ▼
env.py / FastAPI (the environment)
      │ runs CloudEnv simulation
      ▼
tasks.py (the graders)
      │ returns 0.0–1.0 scores
      ▼
Console output / submission
```

And separately:
```
Your code
    │ git push
    ▼
HF Spaces
    │ docker build
    ▼
Live public URL
    │ judges ping it
    ▼
Automated validation passes → scored
```

---

## 6. How to Run It Locally (Show Someone Else)

Follow every step in order. Takes about 5 minutes from scratch.

### Prerequisites
```bash
# Must have these installed
python3 --version   # 3.10+
docker --version
git --version
```

### Step 1 — Clone or navigate to the project
```bash
cd /home/netty/Documents/openenv_cloud_manager
```

### Step 2 — Activate virtual environment
```bash
source venv/bin/activate
# You should see (venv) appear in your prompt
```

### Step 3 — Create .env file (first time only)
```bash
nano .env
```
Paste:
```
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.1-8b-instant
HF_TOKEN=your_groq_key_here
```
Save: `Ctrl+O` → Enter → `Ctrl+X`

### Step 4 — Start the environment server (Terminal 1)
```bash
uvicorn env:app --host 0.0.0.0 --port 7860 --reload
```
Leave this running. You should see:
```
INFO: Uvicorn running on http://0.0.0.0:7860
INFO: Application startup complete.
```

### Step 5 — Test endpoints manually (Terminal 2)
```bash
curl http://localhost:7860/
curl -X POST http://localhost:7860/reset
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action": 2}'
curl http://localhost:7860/state
```
All should return valid JSON.

### Step 6 — Run the AI agent (Terminal 2)
```bash
source venv/bin/activate
python3 inference.py
```
You will see 24 steps per task printing with actions and rewards, then 3 final scores.

### Expected Output
```
========================================
Running task: scale-up-basic
========================================
Step 00 | action=2 | latency=120.0ms | reward=+0.40
Step 01 | action=2 | latency=69.98ms | reward=+0.50
...
[scale-up-basic] Score: 1.0

========================================
Running task: latency-control
========================================
...
[latency-control] Score: 1.0

========================================
Running task: cost-optimization-heavy
========================================
...
[cost-optimization-heavy] Score: 0.53

All scores: 1.0, 1.0, 0.53
All scores valid (0.0–1.0). Ready to submit.
```

### Step 7 — Run via Docker (optional, proves deployment works)
```bash
# Stop uvicorn first (Ctrl+C in Terminal 1)
docker build -t crm-env .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.groq.com/openai/v1" \
  -e MODEL_NAME="llama-3.1-8b-instant" \
  -e HF_TOKEN="your_groq_key_here" \
  crm-env
```
Then repeat Step 5 curl tests. Same results = Docker works.

---

## 7. Live Deployment

The environment is deployed at:
**https://kenkao05-openenv-cloud-manager.hf.space**

Test it from anywhere:
```bash
curl https://kenkao05-openenv-cloud-manager.hf.space/
curl -X POST https://kenkao05-openenv-cloud-manager.hf.space/reset
```

To run the agent against the live deployment instead of localhost:
```bash
ENV_BASE_URL="https://kenkao05-openenv-cloud-manager.hf.space" python3 inference.py
```

---

## 8. Switching LLM Providers (Zero Code Changes)

The entire LLM backend is controlled by 2 environment variables. To switch providers, only `.env` changes — no code changes needed.

| Provider | API_BASE_URL | MODEL_NAME | Cost |
|---|---|---|---|
| Groq (current) | `https://api.groq.com/openai/v1` | `llama-3.1-8b-instant` | Free |
| Together AI | `https://api.together.xyz/v1` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | Free credits |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/` | `gemini-2.0-flash` | Free |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` | Paid |

---

## 9. Pre-Submission Checklist

- [ ] `GET /` returns `{"status": "ok"}` with HTTP 200
- [ ] `POST /reset` returns valid CloudObservation JSON
- [ ] `POST /step` returns observation + reward + done
- [ ] `GET /state` returns step_count, vms_active, accumulated_cost
- [ ] `docker build` completes without error
- [ ] `python3 inference.py` runs all 3 tasks, prints 3 scores between 0.0–1.0
- [ ] HF Space status is green (Running)
- [ ] All 3 env vars set as Secrets in HF Space settings
- [ ] Space visibility is Public
- [ ] Submitted HF Space URL on Scaler platform before April 8, 11:59 PM IST
