from typing import List

_EPS = 1e-6


def _clamp(score: float) -> float:
    """Clamp to strictly open interval (0, 1) as required by OpenEnv validator."""
    return max(_EPS, min(1.0 - _EPS, float(score)))


def grade_scale_up_basic(action_history: List[int]) -> float:
    """
    Easy task: Agent starts with 1 VM and 800+ requests (guaranteed overload).
    Success: Agent takes action 2 (scale up) within the first 3 steps.
    """
    if not action_history:
        return _clamp(0.01)
    early = action_history[:3]
    if 2 in early:
        return _clamp(0.99)
    elif 2 in action_history:
        return _clamp(0.5)   # scaled up but too late
    return _clamp(0.01)


def grade_latency_control(latency_history: List[float]) -> float:
    """
    Medium task: Fluctuating traffic over 24 steps.
    Success: Average latency across all steps stays below 100ms.
    """
    if not latency_history:
        return _clamp(0.01)
    avg = sum(latency_history) / len(latency_history)
    if avg <= 100:
        return _clamp(0.99)
    elif avg <= 200:
        return _clamp(0.5)
    # avg > 200: score degrades, but never hits 0
    raw = max(0.01, round(1.0 - (avg / 400), 4))
    return _clamp(raw)


def grade_cost_optimization(latency_history: List[float], total_cost: float, budget: float = 5.0) -> float:
    """
    Hard task: Black Friday — sustained high traffic (700–1000 req/step) for 24 steps.
    Budget cap: $5.00 total cost.
    Success: Keep latency below 150ms while staying under budget.
    Score = weighted combination of performance score and cost score.
    """
    if not latency_history:
        return _clamp(0.01)
    avg_latency = sum(latency_history) / len(latency_history)

    # Clamp sub-scores individually before combining
    perf_score = _clamp(max(0.01, 1.0 - (avg_latency / 300)))
    if total_cost <= budget:
        cost_score = _clamp(0.99)
    else:
        cost_score = _clamp(max(0.01, 1.0 - ((total_cost - budget) / budget)))

    raw = round((perf_score * 0.6) + (cost_score * 0.4), 4)
    return _clamp(raw)