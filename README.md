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
