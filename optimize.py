import pandas as pd
import numpy as np
import libsbml
import tellurium as te
from scipy.optimize import differential_evolution, least_squares
from typing import List, Dict, Optional


# ------------------- Utilidades SBML -------------------
def load_sbml_from_string(sbml_str: str):
    reader = libsbml.SBMLReader()
    doc = reader.readSBMLFromString(sbml_str)
    if doc is None or doc.getModel() is None or doc.getNumErrors() > 0:
        raise ValueError("Error al leer el SBML.")
    return doc


def extract_parameters(model: libsbml.Model) -> Dict[str, float]:
    """
    Extrae parámetros globales y locales de un modelo SBML.
    """
    params: Dict[str, float] = {}
    # Globales
    for p in model.getListOfParameters():
        if p.isSetId():
            try:
                params[p.getId()] = float(p.getValue())
            except Exception:
                pass
    # Locales en reacciones
    for r in model.getListOfReactions():
        if r.isSetKineticLaw():
            kl = r.getKineticLaw()
            # LocalParameters (SBML L2)
            if hasattr(kl, "getListOfLocalParameters") and kl.getListOfLocalParameters() is not None:
                for lp in kl.getListOfLocalParameters():
                    key = f"{r.getId()}::{lp.getId()}"
                    try:
                        params[key] = float(lp.getValue())
                    except Exception:
                        pass
            # Parameters (SBML L3)
            if hasattr(kl, "getListOfParameters") and kl.getListOfParameters() is not None:
                for lp in kl.getListOfParameters():
                    key = f"{r.getId()}::{lp.getId()}"
                    try:
                        params[key] = float(lp.getValue())
                    except Exception:
                        pass
    return params


def extract_plot_species(model: libsbml.Model) -> Dict[str, str]:
    """
    Devuelve {species_id: nombre_legible} para especies no boundary.
    """
    out: Dict[str, str] = {}
    for s in model.getListOfSpecies():
        if s.getBoundaryCondition():
            continue
        name = s.getName() if s.isSetName() and s.getName() else s.getId()
        out[s.getId()] = name
    return out


# ------------------- Simulación -------------------
def simulate_sbml(sbml_str: str, params: Dict[str, float],
                  t_start: float, t_end: float, n_points: int,
                  species_ids: List[str]) -> np.ndarray:
    rr = te.loadSBMLModel(sbml_str)

    # aplicar parámetros
    for k, v in params.items():
        try:
            rr[k] = v
        except Exception:
            pass

    rr.setIntegrator("cvode")
    rr.integrator.relative_tolerance = 1e-8
    rr.integrator.absolute_tolerance = 1e-12

    result = rr.simulate(t_start, t_end, n_points,
                         selections=["time"] + [f"[{s}]" for s in species_ids])
    return np.array(result)


# ------------------- Datos experimentales -------------------
def load_experimental_data(csv_path: str) -> pd.DataFrame:
    """
    CSV debe contener columnas: time, especie1, especie2, ...
    """
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError("El CSV debe contener una columna 'time'.")
    return df


# ------------------- Función de costo -------------------
def cost_function(param_values, sbml_str, param_names, exp_data: pd.DataFrame,
                  t_start, t_end, n_points, species_ids):
    params = dict(zip(param_names, param_values))
    sim = simulate_sbml(sbml_str, params, t_start, t_end, n_points, species_ids)
    # sim[:,0] = tiempo ; sim[:,1:] = especies
    sim_values = sim[:, 1:]
    # interpolar datos experimentales a los tiempos de simulación
    exp_interp = np.array([
        np.interp(sim[:, 0], exp_data["time"].values, exp_data[sp].values)
        for sp in species_ids
    ]).T
    return (sim_values - exp_interp).ravel()


# ------------------- Optimización -------------------
def optimize_params(sbml_str: str, exp_data: pd.DataFrame,
                    param_names: List[str],
                    t_start=0, t_end=50, n_points=100,
                    species_ids: Optional[List[str]] = None,
                    method="least_squares"):

    if species_ids is None or len(species_ids) == 0:
        # usar todas las columnas excepto "time"
        species_ids = [c for c in exp_data.columns if c != "time"]

    if method == "least_squares":
        x0 = [1.0] * len(param_names)
        res = least_squares(cost_function, x0,
                            args=(sbml_str, param_names, exp_data,
                                  t_start, t_end, n_points, species_ids))
        best_params = dict(zip(param_names, res.x))
        return {"method": method, "best_params": best_params, "cost": res.cost}

    elif method == "de":
        bounds = [(0.01, 10)] * len(param_names)
        res = differential_evolution(lambda x: np.sum(cost_function(x, sbml_str, param_names,
                                                                    exp_data, t_start, t_end,
                                                                    n_points, species_ids) ** 2),
                                     bounds)
        best_params = dict(zip(param_names, res.x))
        return {"method": method, "best_params": best_params, "cost": res.fun}

    else:
        raise ValueError(f"Método {method} no soportado")

