import os
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "google/gemma-3-27b-it")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

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
    print(f"[START] task={task_id}", flush=True)

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

        print(f"[STEP] step={step+1} action={action} latency={obs['current_latency_ms']} reward={reward:.4f}", flush=True)
        step += 1
        time.sleep(0.5)  # avoid hitting rate limits

    final_state = requests.get(f"{ENV_BASE}/state").json()
    print(f"Final accumulated cost: ${final_state['accumulated_cost']}", flush=True)
    return action_history, latency_history, final_state["accumulated_cost"]


if __name__ == "__main__":
    from tasks import grade_scale_up_basic, grade_latency_control, grade_cost_optimization

    ah, lh, cost = run_task("scale-up-basic")
    score1 = grade_scale_up_basic(ah)
    print(f"[END] task=scale-up-basic score={score1:.4f} steps={len(ah)}", flush=True)

    ah, lh, cost = run_task("latency-control")
    score2 = grade_latency_control(lh)
    print(f"[END] task=latency-control score={score2:.4f} steps={len(ah)}", flush=True)

    ah, lh, cost = run_task("cost-optimization-heavy")
    score3 = grade_cost_optimization(lh, cost)
    print(f"[END] task=cost-optimization-heavy score={score3:.4f} steps={len(ah)}", flush=True)

    print(f"\nAll scores: {score1}, {score2}, {score3}", flush=True)
    assert all(0.0 <= s <= 1.0 for s in [score1, score2, score3]), "Score out of range!"
    print("All scores valid (0.0–1.0). Ready to submit.", flush=True)