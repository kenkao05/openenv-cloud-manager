from pydantic import BaseModel
from typing import List

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
    latency_history: List[float] = []