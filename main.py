from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tellurium as te
import libsbml
import numpy as np
from typing import Dict

app = FastAPI()

# Permitir CORS para frontend en GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Endpoint para inspección
# ----------------------
@app.post("/inspect")
async def inspect_sbml(file: UploadFile = File(...)):
    contents = await file.read()
    reader = libsbml.SBMLReader()
    doc = reader.readSBMLFromString(contents.decode("utf-8"))
    if doc.getNumErrors() > 0:
        return {"error": "Archivo SBML inválido"}
    model = doc.getModel()

    # Especies
    species = [s.getId() for s in model.getListOfSpecies() if not s.getBoundaryCondition()]
    # Parámetros globales
    params = {p.getId(): p.getValue() for p in model.getListOfParameters()}

    # Parámetros locales
    for r in model.getListOfReactions():
        if r.isSetKineticLaw():
            for p in r.getKineticLaw().getListOfParameters():
                params[f"{r.getId()}::{p.getId()}"] = p.getValue()

    return {"species": species, "parameters": params}

# ----------------------
# Endpoint para simulación
# ----------------------
@app.post("/simulate")
async def simulate_sbml(
    file: UploadFile = File(...),
    t_start: float = 0.0,
    t_end: float = 50.0,
    n_points: int = 200,
):
    contents = await file.read()
    rr = te.loadSBMLModel(contents.decode("utf-8"))
    rr.setIntegrator("cvode")
    rr.integrator.relative_tolerance = 1e-8
    rr.integrator.absolute_tolerance = 1e-12

    result = rr.simulate(t_start, t_end, int(n_points))
    data = np.array(result)
    return {"colnames": list(result.colnames), "data": data.tolist()}
