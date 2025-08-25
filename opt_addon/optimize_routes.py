from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional, List, Dict, Any
from fastapi.responses import JSONResponse

from .optimize_service import (
    create_session, upload_dataset, set_mapping, set_param_selection, run_optimization
)
from .data_repository import repo

# 👇 quitamos el prefix aquí
router = APIRouter(tags=["optimize"])

@router.post("/session/create")
async def session_create(sbml_file: Optional[UploadFile] = File(None),
                         sympy_text: Optional[str] = Form(None)):
    sbml_bytes = await sbml_file.read() if sbml_file is not None else None
    try:
        sid = create_session(sbml_bytes, sympy_text)
        sess = repo.get(sid)
        return {"session_id": sid, "species": sess["species"], "parameters": sess["parameters"]}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@router.post("/data/upload")
async def data_upload(session_id: str = Form(...),
                      dataset_name: str = Form(...),
                      file: UploadFile = File(...)):
    try:
        cols = upload_dataset(session_id, await file.read(), dataset_name)
        return {"session_id": session_id, "columns": cols}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@router.post("/map")
async def map_species(session_id: str, mappings: List[Dict[str, Any]]):
    try:
        set_mapping(session_id, mappings)
        return {"ok": True}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@router.post("/params/select")
async def params_select(session_id: str, to_optimize: List[str], fixed: Dict[str, float] = {}):
    try:
        set_param_selection(session_id, to_optimize, fixed)
        return {"ok": True}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@router.post("/run")
async def run(session_id: str,
              method: str = "lm",
              initial_guess_json: Optional[str] = None,
              max_iter: int = 500,
              lr: float = 1e-3,
              pinn_epochs: int = 2000,
              weight_data: float = 1.0,
              weight_phys: float = 1.0):
    try:
        initial_guess = {}
        if initial_guess_json:
            import json
            initial_guess = json.loads(initial_guess_json)
        result = run_optimization(session_id, method, initial_guess, max_iter, lr, pinn_epochs, weight_data, weight_phys)
        return result
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
