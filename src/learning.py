"""
NEUROGEN experimental learning interface (v0.2).

Provides a minimal surface for parameterized numeric operators
and simple numerical gradient estimation. This is deliberately
limited — it does not constitute autonomous learning or AGI.
"""

from typing import Callable, Dict, List, Any, Tuple
import numpy as np
from neuro_types import NeuroValue


def numerical_gradient(fn: Callable[[float], float], x: float, eps: float = 1e-5) -> float:
    """Central difference gradient estimate."""
    return (fn(x + eps) - fn(x - eps)) / (2 * eps)


def simple_optimize(
    objective: Callable[[Dict[str, float]], float],
    params: Dict[str, float],
    steps: int = 20,
    lr: float = 0.05,
) -> Tuple[Dict[str, float], List[float]]:
    """
    Extremely simple gradient-free / numerical-gradient parameter update.
    Returns (final_params, loss_history).
    """
    history = []
    p = dict(params)
    for _ in range(steps):
        loss = objective(p)
        history.append(loss)
        grads = {}
        for k, v in p.items():
            def f(x, key=k):
                trial = dict(p)
                trial[key] = x
                return objective(trial)
            grads[k] = numerical_gradient(f, v)
        for k in p:
            p[k] -= lr * grads[k]
    return p, history
