import json
import tellurium as te
import libsbml
from fastapi.responses import JSONResponse
from typing import Dict, Optional

from sbml_utils import (
    load_sbml_from_bytes,
    extract_parameters,
    extract_plot_species,
    extract_reactions,
    extract_initial_conditions,
    extract_layout_positions,
    apply_params_to_sbml_doc,
    apply_initial_conditions_to_sbml_doc
)


def inspect_sbml(sbml_bytes: bytes):
    doc, err = load_sbml_from_bytes(sbml_bytes)
    if err:
        return JSONResponse(status_code=400, content={"error": err})

    model = doc.getModel()
    return {
        "parameters": extract_parameters(model),
        "species": extract_plot_species(model),
        "reactions": extract_reactions(model),
        "initial_conditions": extract_initial_conditions(model),
        "positions": extract_layout_positions(model),
        "defaultSelections": list(extract_plot_species(model).keys()),
    }


def simulate_sbml(
    sbml_bytes: bytes,
    t_start: float,
    t_end: float,
    n_points: int,
    selected_species: str,
    param_values_json: Optional[str],
    initial_conditions_json: Optional[str],
):
    doc, err = load_sbml_from_bytes(sbml_bytes)
    if err:
        return JSONResponse(status_code=400, content={"error": err})

    param_values: Dict[str, float] = {}
    if param_values_json:
        try:
            param_values = json.loads(param_values_json)
        except Exception:
            return JSONResponse(status_code=400, content={"error": "param_values_json inválido"})

    init_values: Dict[str, float] = {}
    if initial_conditions_json:
        try:
            init_values = json.loads(initial_conditions_json)
        except Exception:
            return JSONResponse(status_code=400, content={"error": "initial_conditions_json inválido"})

    if param_values:
        doc = apply_params_to_sbml_doc(doc, param_values)
    if init_values:
        doc = apply_initial_conditions_to_sbml_doc(doc, init_values)

    sbml_str = libsbml.writeSBMLToString(doc)

    try:
        rr = te.loadSBMLModel(sbml_str)
        rr.setIntegrator("cvode")
        rr.integrator.relative_tolerance = 1e-8
        rr.integrator.absolute_tolerance = 1e-12
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"No se pudo cargar el modelo en RoadRunner: {e}"})

    species_ids = [s for s in selected_species.split(",") if s.strip()]
    if not species_ids:
        species_ids = list(extract_plot_species(doc.getModel()).keys())

    rr.selections = ["time"] + [f"[{sid}]" for sid in species_ids]

    try:
        result = rr.simulate(t_start, t_end, int(n_points))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Error en la simulación: {e}"})

    return {"columns": list(result.colnames), "data": [list(map(float, row)) for row in result]}