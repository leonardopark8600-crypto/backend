# main.py
import io
import json
from typing import Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, Form,  WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
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
# app.mount("/static", StaticFiles(directory="static"), name="static")
connections = set()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.add(websocket)
    await broadcast_count()

    try:
        while True:
            await websocket.receive_text()  # mantenemos la conexión viva
    except WebSocketDisconnect:
        connections.remove(websocket)
        await broadcast_count()

async def broadcast_count():
    count = len(connections)
    for conn in connections:
        await conn.send_json({"count": count})

@app.get("/")
def home():
    return {"message": "Servidor de contador en línea activo 🚀"}

# ---------- Helpers SBML ----------
def load_sbml_from_bytes(sbml_bytes: bytes):
    reader = libsbml.SBMLReader()
    doc = reader.readSBMLFromString(sbml_bytes.decode("utf-8", errors="ignore"))
    if doc is None or doc.getModel() is None or doc.getNumErrors() > 0:
        return None, "Error al leer el archivo SBML."
    return doc, None


def extract_parameters(model: libsbml.Model) -> Dict[str, float]:
    """
    Extrae parámetros globales y locales (de leyes cinéticas).
    Los locales se devuelven con clave 'reactionId::paramId'.
    """
    params: Dict[str, float] = {}
    # Parámetros globales
    for p in model.getListOfParameters():
        if p.isSetId():
            try:
                params[p.getId()] = float(p.getValue())
            except Exception:
                pass
    # Parámetros locales por reacción
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

def extract_layout_positions(model: libsbml.Model):
    """
    Extrae posiciones de especies y reacciones si el SBML tiene Layout extension.
    Devuelve {id: {"x": float, "y": float}}.
    """
    positions = {}
    plugin = model.getPlugin("layout")
    if not plugin:
        return positions

    if plugin.getNumLayouts() == 0:
        return positions

    layout = plugin.getLayout(0)  # tomamos el primer layout
    for glyph in layout.getListOfSpeciesGlyphs():
        sid = glyph.getSpeciesId()
        if glyph.isSetBoundingBox():
            bb = glyph.getBoundingBox()
            if bb and bb.isSetPosition():
                pos = bb.getPosition()
                positions[sid] = {"x": pos.getX(), "y": pos.getY()}

    for glyph in layout.getListOfReactionGlyphs():
        rid = glyph.getReactionId()
        if glyph.isSetBoundingBox():
            bb = glyph.getBoundingBox()
            if bb and bb.isSetPosition():
                pos = bb.getPosition()
                positions[rid] = {"x": pos.getX(), "y": pos.getY()}

    return positions

def extract_plot_species(model: libsbml.Model) -> Dict[str, str]:
    """
    Devuelve {species_id: nombre_legible} para especies NO boundary.
    """
    out: Dict[str, str] = {}
    for s in model.getListOfSpecies():
        if s.getBoundaryCondition():
            continue
        name = s.getName() if s.isSetName() and s.getName() else s.getId()
        out[s.getId()] = name
    return out


def extract_reactions(model: libsbml.Model):
    """
    Devuelve una lista de reacciones con reactivos y productos por id.
    """
    reactions = []
    for r in model.getListOfReactions():
        reactants = [sr.getSpecies() for sr in r.getListOfReactants()]
        products = [sp.getSpecies() for sp in r.getListOfProducts()]
        reactions.append(
            {"id": r.getId(), "reactants": reactants, "products": products}
        )
    return reactions


def extract_initial_conditions(model: libsbml.Model) -> Dict[str, float]:
    """
    Extrae condiciones iniciales de las especies (no boundary).
    Prioriza initialConcentration; si no existe, usa initialAmount.
    Si ninguna está definida, intenta usar el valor actual del species (0.0 fallback).
    """
    inits: Dict[str, float] = {}
    for s in model.getListOfSpecies():
        if s.getBoundaryCondition():
            continue
        val: Optional[float] = None
        try:
            if s.isSetInitialConcentration():
                val = float(s.getInitialConcentration())
            elif s.isSetInitialAmount():
                # Nota: si solo hay amount y hay compartimento con volumen != 1,
                # RoadRunner internamente gestiona la conversión. Aquí retornamos amount.
                val = float(s.getInitialAmount())
            else:
                # Fallback a 0.0
                val = 0.0
        except Exception:
            val = 0.0
        inits[s.getId()] = val
    return inits


def apply_params_to_sbml_doc(
    doc: libsbml.SBMLDocument, param_values: Dict[str, float]
) -> libsbml.SBMLDocument:
    """
    Aplica parámetros globales y locales sobre el SBMLDocument.
    """
    model = doc.getModel()
    # Globales
    for p in model.getListOfParameters():
        pid = p.getId()
        if pid in param_values:
            try:
                p.setValue(float(param_values[pid]))
            except Exception:
                pass
    # Locales por reacción
    for r in model.getListOfReactions():
        if r.isSetKineticLaw():
            kl = r.getKineticLaw()
            # LocalParameters (SBML L2)
            if hasattr(kl, "getListOfLocalParameters") and kl.getListOfLocalParameters() is not None:
                for lp in kl.getListOfLocalParameters():
                    key = f"{r.getId()}::{lp.getId()}"
                    if key in param_values:
                        try:
                            lp.setValue(float(param_values[key]))
                        except Exception:
                            pass
            # Parameters (SBML L3)
            if hasattr(kl, "getListOfParameters") and kl.getListOfParameters() is not None:
                for lp in kl.getListOfParameters():
                    key = f"{r.getId()}::{lp.getId()}"
                    if key in param_values:
                        try:
                            lp.setValue(float(param_values[key]))
                        except Exception:
                            pass
    return doc


def apply_initial_conditions_to_sbml_doc(
    doc: libsbml.SBMLDocument, initial_conditions: Dict[str, float]
) -> libsbml.SBMLDocument:
    """
    Aplica condiciones iniciales a especies (no boundary) dentro del documento SBML.
    Si la especie tenía initialConcentration, se sobrescribe; en caso contrario, se pone initialAmount.
    """
    if not initial_conditions:
        return doc
    model = doc.getModel()
    for s in model.getListOfSpecies():
        sid = s.getId()
        if s.getBoundaryCondition():
            continue
        if sid in initial_conditions:
            val = float(initial_conditions[sid])
            try:
                if s.isSetInitialConcentration() or (not s.isSetInitialAmount() and model.getLevel() >= 2):
                    s.setInitialConcentration(val)
                    # Si estaba definido amount, limpiarlo para evitar conflicto
                    if s.isSetInitialAmount():
                        s.unsetInitialAmount()
                else:
                    s.setInitialAmount(val)
                    if s.isSetInitialConcentration():
                        s.unsetInitialConcentration()
            except Exception:
                # Si falla, intentar al menos setInitialAmount
                try:
                    s.setInitialAmount(val)
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
    initial_conditions = extract_initial_conditions(model)
    positions = extract_layout_positions(model)   # ← NUEVO

    return {
        "parameters": params,
        "species": species,
        "reactions": reactions,
        "initial_conditions": initial_conditions,
        "positions": positions,   # ← NUEVO
        "defaultSelections": list(species.keys()),
    }



@app.post("/simulate")
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
    doc, err = load_sbml_from_bytes(sbml_bytes)
    if err:
        return JSONResponse(status_code=400, content={"error": err})

    # Parseo de parámetros
    param_values: Dict[str, float] = {}
    if param_values_json:
        try:
            param_values = json.loads(param_values_json)
        except Exception:
            return JSONResponse(
                status_code=400, content={"error": "param_values_json inválido"}
            )

    # Parseo de condiciones iniciales
    init_values: Dict[str, float] = {}
    if initial_conditions_json:
        try:
            init_values = json.loads(initial_conditions_json)
        except Exception:
            return JSONResponse(
                status_code=400, content={"error": "initial_conditions_json inválido"}
            )

    # Aplicar parámetros e iniciales sobre el documento SBML
    if param_values:
        doc = apply_params_to_sbml_doc(doc, param_values)
    if init_values:
        doc = apply_initial_conditions_to_sbml_doc(doc, init_values)

    sbml_str = libsbml.writeSBMLToString(doc)

    # Cargar en RoadRunner
    try:
        rr = te.loadSBMLModel(sbml_str)
        rr.setIntegrator("cvode")
        rr.integrator.relative_tolerance = 1e-8
        rr.integrator.absolute_tolerance = 1e-12
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"No se pudo cargar el modelo en RoadRunner: {e}"},
        )

    # Determinar especies a graficar
    species_ids = [s for s in selected_species.split(",") if s.strip()]
    if not species_ids:
        species_ids = list(extract_plot_species(doc.getModel()).keys())

    rr.selections = ["time"] + [f"[{sid}]" for sid in species_ids]

    # Ejecutar simulación
    try:
        result = rr.simulate(t_start, t_end, int(n_points))
    except Exception as e:
        return JSONResponse(
            status_code=400, content={"error": f"Error en la simulación: {e}"}
        )

    data = [list(map(float, row)) for row in result]
    colnames = list(result.colnames)

    return {"columns": colnames, "data": data}



