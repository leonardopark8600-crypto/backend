# main.py
import io
import json
from typing import Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import libsbml
import tellurium as te

app = FastAPI(title="SBML Simulator API")

# CORS para permitir frontend separado
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir frontend estático (si decides unirlos en un mismo server)
#app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------- Helpers SBML ----------
def load_sbml_from_bytes(sbml_bytes: bytes):
    reader = libsbml.SBMLReader()
    doc = reader.readSBMLFromString(sbml_bytes.decode("utf-8", errors="ignore"))
    if doc.getNumErrors() > 0 or doc.getModel() is None:
        return None, "Error al leer el archivo SBML."
    return doc, None


def extract_parameters(model: libsbml.Model) -> Dict[str, float]:
    params = {}
    # Global parameters
    for p in model.getListOfParameters():
        if p.isSetId():
            params[p.getId()] = float(p.getValue())
    # Local parameters in reactions
    for r in model.getListOfReactions():
        if r.isSetKineticLaw():
            kl = r.getKineticLaw()
            if hasattr(kl, "getListOfLocalParameters") and kl.getListOfLocalParameters() is not None:
                for lp in kl.getListOfLocalParameters():
                    key = f"{r.getId()}::{lp.getId()}"
                    params[key] = float(lp.getValue())
            if hasattr(kl, "getListOfParameters") and kl.getListOfParameters() is not None:
                for lp in kl.getListOfParameters():
                    key = f"{r.getId()}::{lp.getId()}"
                    params[key] = float(lp.getValue())
    return params


def extract_plot_species(model: libsbml.Model) -> Dict[str, str]:
    species = {}
    for s in model.getListOfSpecies():
        if s.getBoundaryCondition():
            continue
        name = s.getName() if s.isSetName() and s.getName() else s.getId()
        species[s.getId()] = name
    return species


def extract_reactions(model: libsbml.Model):
    reactions = []
    for r in model.getListOfReactions():
        reactants = [s.getSpecies() for s in r.getListOfReactants()]
        products = [s.getSpecies() for s in r.getListOfProducts()]
        reactions.append({
            "id": r.getId(),
            "reactants": reactants,
            "products": products
        })
    return reactions


def apply_params_to_sbml_doc(doc: libsbml.SBMLDocument, param_values: Dict[str, float]) -> libsbml.SBMLDocument:
    model = doc.getModel()
    for p in model.getListOfParameters():
        pid = p.getId()
        if pid in param_values:
            try:
                p.setValue(float(param_values[pid]))
            except Exception:
                pass
    for r in model.getListOfReactions():
        if r.isSetKineticLaw():
            kl = r.getKineticLaw()
            if hasattr(kl, "getListOfLocalParameters") and kl.getListOfLocalParameters() is not None:
                for lp in kl.getListOfLocalParameters():
                    key = f"{r.getId()}::{lp.getId()}"
                    if key in param_values:
                        try:
                            lp.setValue(float(param_values[key]))
                        except Exception:
                            pass
            if hasattr(kl, "getListOfParameters") and kl.getListOfParameters() is not None:
                for lp in kl.getListOfParameters():
                    key = f"{r.getId()}::{lp.getId()}"
                    if key in param_values:
                        try:
                            lp.setValue(float(param_values[key]))
                        except Exception:
                            pass
    return doc


# ---------- API endpoints ----------
@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    sbml_bytes = await file.read()
    doc, err = load_sbml_from_bytes(sbml_bytes)
    if err:
        return JSONResponse(status_code=400, content={"error": err})

    model = doc.getModel()
    params = extract_parameters(model)
    species = extract_plot_species(model)
    reactions = extract_reactions(model)

    return {
        "parameters": params,
        "species": species,
        "reactions": reactions,
        "defaultSelections": list(species.keys())
    }


@app.post("/simulate")
async def simulate(
    file: UploadFile = File(...),
    t_start: float = Form(0.0),
    t_end: float = Form(50.0),
    n_points: int = Form(200),
    selected_species: str = Form(""),
    param_values_json: Optional[str] = Form(None),
):
    sbml_bytes = await file.read()
    doc, err = load_sbml_from_bytes(sbml_bytes)
    if err:
        return JSONResponse(status_code=400, content={"error": err})

    param_values = {}
    if param_values_json:
        try:
            param_values = json.loads(param_values_json)
        except Exception:
            return JSONResponse(status_code=400, content={"error": "param_values_json inválido"})

    doc = apply_params_to_sbml_doc(doc, param_values)
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

    data = [list(map(float, row)) for row in result]
    colnames = list(result.colnames)

    return {
        "columns": colnames,
        "data": data
    }
