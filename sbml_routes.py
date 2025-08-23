from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional

from sbml_service import inspect_sbml, simulate_sbml

router = APIRouter()

@router.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    sbml_bytes = await file.read()
    return inspect_sbml(sbml_bytes)

@router.post("/simulate")
async def simulate(
    file: UploadFile = File(...),
    t_start: float = Form(0.0),
    t_end: float = Form(50.0),
    n_points: int = Form(200),
    selected_species: str = Form(""),
    param_values_json: Optional[str] = Form(None),
    initial_conditions_json: Optional[str] = Form(None),
):
    sbml_bytes = await file.read()
    return simulate_sbml(
        sbml_bytes, t_start, t_end, n_points, selected_species,
        param_values_json, initial_conditions_json
    )