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
