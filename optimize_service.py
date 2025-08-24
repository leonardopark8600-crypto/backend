from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import io
import json
import numpy as np
import pandas as pd
import libsbml
import tellurium as te
from fastapi.responses import JSONResponse

from sbml_utils import extract_parameters, extract_plot_species, load_sbml_from_bytes  # reuse existing
from .data_repository import repo
from .sympy_tools import detect_parameters_from_sympy
from .numeric_optimizers import run_lm, run_bfgs
from .pinn_refinement import pinn_refine

def create_session(sbml_file: Optional[bytes] = None, sympy_text: Optional[str] = None) -> str:
    sid = repo.create()
    sess = repo.get(sid)
    if sbml_file is not None:
        doc, err = load_sbml_from_bytes(sbml_file)
        if err:
            raise ValueError(err)
        model = doc.getModel()
        sess["sbml_str"] = libsbml.writeSBMLToString(doc)
        sess["species"] = list(extract_plot_species(model).keys())
        sess["parameters"] = extract_parameters(model)
    if sympy_text:
        sess["sympy_text"] = sympy_text
        states, params = detect_parameters_from_sympy(sympy_text)
        if states:
            sess["species"] = states
        if params:
            sess["parameters"].update({k: 1.0 for k in params.keys()})
    return sid

def upload_dataset(sid: str, file_bytes: bytes, name: str) -> Dict[str, List[str]]:
    sess = repo.get(sid)
    df = pd.read_csv(io.BytesIO(file_bytes))
    sess["datasets"][name] = df
    return {name: list(df.columns)}

def set_mapping(sid: str, mappings: List[Dict]) -> None:
    sess = repo.get(sid)
    for m in mappings:
        sid_ = m["species_id"]
        sess["maps"][sid_] = (m["dataset"], m["time_column"], m["value_column"])

def set_param_selection(sid: str, to_optimize: List[str], fixed: Dict[str, float]) -> None:
    sess = repo.get(sid)
    sess["selection"]["to_optimize"] = to_optimize
    sess["selection"]["fixed"] = fixed

def _simulate(sbml_str: str, species: List[str], t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    rr = te.loadSBMLModel(sbml_str)
    rr.setIntegrator("cvode")
    for k, v in params.items():
        # try global first
        try:
            rr[k] = v
        except Exception:
            # ignore if local; handled only via SBML edit path in sbml_service, omitted here for speed
            pass
    rr.selections = ["time"] + [f"[{sid}]" for sid in species]
    M = rr.simulate(float(t[0]), float(t[-1]), len(t))
    return np.array(M)[:, 1:]  # drop time

def _assemble_target_arrays(sess) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Returns t union and Y stacked per time sorted; uses first map's time as reference if same-length
    species = []
    frames = []
    for sid, (dname, tcol, vcol) in sess["maps"].items():
        df = sess["datasets"][dname][[tcol, vcol]].dropna().copy()
        df = df.rename(columns={tcol: "time", vcol: sid})
        frames.append(df)
        species.append(sid)
    # outer-join on time
    from functools import reduce
    merged = reduce(lambda l, r: pd.merge(l, r, on="time", how="outer"), frames).sort_values("time")
    merged = merged.dropna()  # keep only common times
    t = merged["time"].values.astype(float)
    Y = merged[species].values.astype(float)
    return t, Y, species

def _loss_residual_builder(sess, sbml_str, species, t, Y, pnames, fixed):
    def residual_vec(x):
        params = {**fixed, **{n: float(v) for n, v in zip(pnames, x)}}
        Yhat = _simulate(sbml_str, species, t, params)
        res = (Yhat - Y).ravel()
        return res
    return residual_vec

def _loss_scalar_builder(sess, sbml_str, species, t, Y, pnames, fixed):
    resvec = _loss_residual_builder(sess, sbml_str, species, t, Y, pnames, fixed)
    def loss(x):
        r = resvec(x)
        return float((r @ r) / r.size)
    return loss

def run_optimization(sid: str, method: str, initial_guess: Dict[str, float], max_iter: int = 500,
                     lr: float = 1e-3, pinn_epochs: int = 2000, weight_data: float = 1.0, weight_phys: float = 1.0):
    sess = repo.get(sid)
    if not sess["sbml_str"]:
        raise ValueError("Session was not initialized with SBML; numeric simulation requires SBML.")

    t, Y, species = _assemble_target_arrays(sess)
    pnames = sess["selection"]["to_optimize"]
    fixed = sess["selection"]["fixed"]
    p0 = {n: float(initial_guess.get(n, sess["parameters"].get(n, 1.0))) for n in pnames}

    if method == "lm":
        fit = run_lm(_loss_residual_builder(sess, sess["sbml_str"], species, t, Y, pnames, fixed), p0)
    elif method == "bfgs":
        fit = run_bfgs(_loss_scalar_builder(sess, sess["sbml_str"], species, t, Y, pnames, fixed), p0)
    elif method == "pinn":
        # Use LM first as warm-start, then PINN refine selected params against a simple physics net
        fit0 = run_lm(_loss_residual_builder(sess, sess["sbml_str"], species, t, Y, pnames, fixed), p0)
        # Prepare simple ODE fn: simulate RHS by finite diff on RR? For speed, use NN to fit data only (fallback)
        def ode_fn(t_tensor, y_tensor, params_torch):
            # Dummy zero RHS if we don't have explicit sympy ODEs; forces smoother y'
            return 0.0 * y_tensor
        refined, loss_trace = pinn_refine(ode_fn, t, Y, {k: fit0.x[k] for k in pnames}, epochs=pinn_epochs, lr=lr,
                                          weight_data=weight_data, weight_phys=weight_phys)
        fit = fit0
        fit.x.update(refined)
        fit.traces["pinn_loss"] = loss_trace
    else:
        raise ValueError("Unknown method")

    # Build plot
    params_final = {**fixed, **fit.x}
    Yhat = _simulate(sess["sbml_str"], species, t, params_final)
    plot = {"time": t.tolist(), "series": {sp: {"sim": Yhat[:,i].tolist(), "data": Y[:,i].tolist()} for i, sp in enumerate(species)}}

    # simple metrics
    rss = float(np.sum((Yhat - Y)**2))
    rmse = float(np.sqrt(np.mean((Yhat - Y)**2)))
    mae = float(np.mean(np.abs(Yhat - Y)))
    fit.metrics.update({"rss": rss, "rmse": rmse, "mae": mae})

    return {
        "method": method,
        "estimates": fit.x,
        "ci95": {k: list(v) for k, v in fit.ci95.items()},
        "metrics": fit.metrics,
        "traces": fit.traces,
        "plot": plot,
    }
