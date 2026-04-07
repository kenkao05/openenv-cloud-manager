import os
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "google/gemma-3-27b-it")
HF_TOKEN     = os.environ.get("HF_TOKEN")

BENCHMARK = "openenv-cloud-manager"

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are a cloud engineer managing a fleet of Virtual Machines.
You receive server metrics and must respond with exactly one digit:
0 = Scale Down (removes a VM, reduces cost, increases CPU load)
1 = No-op (do nothing)
2 = Scale Up (adds a VM, increases cost, reduces CPU load)
Respond with only the digit. No explanation. No punctuation."""

ENV_BASE = os.environ.get("ENV_BASE_URL", "http://localhost:7860")


def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: int, reward: float, done: bool, error=None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


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
    log_start(task_id)

    # Pass task_id to reset so env configures itself correctly
    obs = requests.post(f"{ENV_BASE}/reset", params={"task_id": task_id}).json()
    action_history = []
    rewards = []
    done = False
    step = 0

    while not done and step < 24:
        action = get_action(obs)
        error = None
        try:
            result = requests.post(f"{ENV_BASE}/step", json={"action": action}).json()
            obs    = result["observation"]
            reward = result["reward"]
            done   = result["done"]
        except Exception as e:
            error = str(e)
            reward = 0.0
            done = True

        action_history.append(action)
        rewards.append(reward)
        step += 1

        log_step(step=step, action=action, reward=reward, done=done, error=error)
        time.sleep(0.5)

    final_state = requests.get(f"{ENV_BASE}/state").json()
    cost = final_state["accumulated_cost"]
    latency_history = final_state.get("latency_history", [])
    print(f"Final accumulated cost: ${cost}", flush=True)

    return action_history, rewards, cost, latency_history


if __name__ == "__main__":
    from tasks import grade_scale_up_basic, grade_latency_control, grade_cost_optimization

    ah, rewards1, cost, lh = run_task("scale-up-basic")
    score1 = grade_scale_up_basic(ah)
    log_end(success=score1 > 0, steps=len(ah), score=score1, rewards=rewards1)

    ah, rewards2, cost, lh = run_task("latency-control")
    score2 = grade_latency_control(lh)
    log_end(success=score2 > 0, steps=len(ah), score=score2, rewards=rewards2)

    ah, rewards3, cost, lh = run_task("cost-optimization-heavy")
    score3 = grade_cost_optimization(lh, cost)
    log_end(success=score3 > 0, steps=len(ah), score=score3, rewards=rewards3)

    print(f"\nAll scores: {score1}, {score2}, {score3}", flush=True)
    assert all(0.0 <= s <= 1.0 for s in [score1, score2, score3]), "Score out of range!"
    print("All scores valid (0.0–1.0). Ready to submit.", flush=True)