from __future__ import annotations
from typing import Dict, List, Callable, Tuple
import numpy as np
from dataclasses import dataclass
from scipy.optimize import least_squares, minimize

@dataclass
class FitResult:
    x: Dict[str, float]
    ci95: Dict[str, Tuple[float, float]]
    metrics: Dict[str, float]
    traces: Dict[str, List[float]]

def _ci95_from_jac(res, pnames: List[str]) -> Dict[str, Tuple[float,float]]:
    # Approximate covariance from J^T J inverse * residual variance
    try:
        J = res.jac
        _, s, VT = np.linalg.svd(J, full_matrices=False)
        threshold = np.finfo(float).eps * max(J.shape) * s[0]
        s = s[s > threshold]
        cov = VT[:s.size].T @ np.diag(1.0 / s**2) @ VT[:s.size]
        dof = max(1, res.fun.size - len(pnames))
        sigma2 = (res.fun @ res.fun) / dof
        cov = cov * sigma2
        errs = np.sqrt(np.diag(cov))
        ci = {}
        for i, name in enumerate(pnames):
            ci[name] = (res.x[i] - 1.96*errs[i], res.x[i] + 1.96*errs[i])
        return ci
    except Exception:
        return {name: (np.nan, np.nan) for name in pnames}

def run_lm(residual_fn: Callable[[np.ndarray], np.ndarray], p0: Dict[str, float]) -> FitResult:
    names = list(p0.keys())
    x0 = np.array([p0[n] for n in names], dtype=float)
    loss_trace = []
    def wrapped(x):
        r = residual_fn(x)
        loss_trace.append(float(r @ r))
        return r
    res = least_squares(wrapped, x0, method="lm", max_nfev=5000)
    ci = _ci95_from_jac(res, names)
    metrics = {"rmse": float(np.sqrt(np.mean(res.fun**2))), "rss": float(res.fun @ res.fun)}
    return FitResult(x={n: float(v) for n, v in zip(names, res.x)}, ci95=ci, metrics=metrics, traces={"loss": loss_trace})

def run_bfgs(loss_fn: Callable[[np.ndarray], float], p0: Dict[str, float]) -> FitResult:
    names = list(p0.keys())
    x0 = np.array([p0[n] for n in names], dtype=float)
    loss_trace = []
    def wrapped(x):
        val = float(loss_fn(x))
        loss_trace.append(val)
        return val
    res = minimize(wrapped, x0, method="BFGS", options={"maxiter": 2000})
    # Finite-diff Hessian approx for CI
    try:
        Hinv = res.hess_inv
        errs = np.sqrt(np.diag(Hinv))
        ci = {names[i]: (res.x[i] - 1.96*errs[i], res.x[i] + 1.96*errs[i]) for i in range(len(names))}
    except Exception:
        ci = {n: (np.nan, np.nan) for n in names}
    # Dummy residual-based metrics unavailable; use final loss
    metrics = {"final_loss": float(res.fun)}
    return FitResult(x={n: float(v) for n, v in zip(names, res.x)}, ci95=ci, metrics=metrics, traces={"loss": loss_trace})
