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
