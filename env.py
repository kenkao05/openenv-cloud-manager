import random
from fastapi import FastAPI
from models import CloudObservation, CloudAction, CloudState

app = FastAPI()

class CloudEnv:
    def __init__(self):
        self.task_id = "scale-up-basic"
        self.reset()

    def reset(self, task_id: str = "scale-up-basic"):
        self.task_id = task_id
        self.vms = 1 if task_id == "scale-up-basic" else 2
        self.cost = 0.0
        self.steps = 0
        self.latency_history = []
        return self._get_obs()

    def _get_obs(self):
        if self.task_id == "scale-up-basic":
            # Guaranteed overload: 1 VM, high requests
            requests = random.randint(800, 1000)
        elif self.task_id == "cost-optimization-heavy":
            # Black Friday: sustained high traffic
            requests = random.randint(700, 1000)
        else:
            # latency-control: fluctuating traffic
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
        self.latency_history.append(obs["current_latency_ms"])

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
            "accumulated_cost": round(self.cost, 2),
            "latency_history": self.latency_history
        }


env = CloudEnv()


@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(task_id: str = "scale-up-basic"):
    obs = env.reset(task_id=task_id)
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