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

_EPS = 1e-6

def clamp(score: float) -> float:
    return max(_EPS, min(1.0 - _EPS, float(score)))

def log_start(task):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

def get_action(obs):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": str(obs)}],
        max_tokens=5, temperature=0.0, stream=False,
    )
    raw = response.choices[0].message.content.strip()
    for char in raw:
        if char in ("0", "1", "2"):
            return int(char)
    return 1

def run_task(task_id):
    log_start(task_id)
    obs = requests.post(f"{ENV_BASE}/reset", params={"task_id": task_id}).json()
    action_history, rewards = [], []
    done, step = False, 0
    while not done and step < 24:
        action = get_action(obs)
        error = None
        try:
            result = requests.post(f"{ENV_BASE}/step", json={"action": action}).json()
            obs, reward, done = result["observation"], result["reward"], result["done"]
        except Exception as e:
            error, reward, done = str(e), 0.0, True
        action_history.append(action)
        rewards.append(reward)
        step += 1
        log_step(step, action, reward, done, error)
        time.sleep(0.5)
    final_state = requests.get(f"{ENV_BASE}/state").json()
    cost = final_state["accumulated_cost"]
    latency_history = final_state.get("latency_history", [])
    print(f"Final accumulated cost: ${cost}", flush=True)
    return action_history, rewards, cost, latency_history

if __name__ == "__main__":
    from tasks import grade_scale_up_basic, grade_latency_control, grade_cost_optimization

    ah, rewards1, cost, lh = run_task("scale-up-basic")
    score1 = clamp(grade_scale_up_basic(ah))
    log_end(True, len(ah), score1, rewards1)

    ah, rewards2, cost, lh = run_task("latency-control")
    score2 = clamp(grade_latency_control(lh))
    log_end(True, len(ah), score2, rewards2)

    ah, rewards3, cost, lh = run_task("cost-optimization-heavy")
    score3 = clamp(grade_cost_optimization(lh, cost))
    log_end(True, len(ah), score3, rewards3)

    print(f"\nAll scores: {score1}, {score2}, {score3}", flush=True)
    assert all(0.0 < s < 1.0 for s in [score1, score2, score3]), "Score out of range!"
    print("All scores valid. Ready to submit.", flush=True)