import io
import json
import re
from typing import Dict, Optional

from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import libsbml
import tellurium as te
import roadrunner
import pandas as pd
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

# ======================================================
# FastAPI Config
# ======================================================
app = FastAPI(title="SBML Simulator API")

# ---------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ======================================================
# Helpers SBML (no tocamos lo existente)
# ======================================================
# Guardar último SBML cargado
last_sbml_bytes: Optional[bytes] = None


# ---------------------------------------------------------
# Utilidades SBML
# ---------------------------------------------------------
def load_sbml_from_bytes(sbml_bytes: bytes):
    reader = libsbml.SBMLReader()
    doc = reader.readSBMLFromString(sbml_bytes.decode("utf-8", errors="ignore"))
    if doc is None or doc.getModel() is None or doc.getNumErrors() > 0:
        return None, "Error al leer el archivo SBML."
    return doc, None

def extract_parameters(model: libsbml.Model) -> Dict[str, float]:
    params: Dict[str, float] = {}
    try:
        reader = libsbml.SBMLReader()
        doc = reader.readSBMLFromString(sbml_bytes.decode("utf-8"))
        if doc.getNumErrors() > 0:
            return None, doc.getErrorLog().toString()
        return doc, None
    except Exception as e:
        return None, str(e)


def extract_parameters(model):
    params = {}
    for p in model.getListOfParameters():
        if p.isSetId():
            try:
                params[p.getId()] = float(p.getValue())
            except Exception:
                pass
    for r in model.getListOfReactions():
        if r.isSetKineticLaw():
            kl = r.getKineticLaw()
            if hasattr(kl, "getListOfLocalParameters") and kl.getListOfLocalParameters():
                for lp in kl.getListOfLocalParameters():
                    key = f"{r.getId()}::{lp.getId()}"
                    try:
                        params[key] = float(lp.getValue())
                    except Exception:
                        pass
            if hasattr(kl, "getListOfParameters") and kl.getListOfParameters():
                for lp in kl.getListOfParameters():
                    key = f"{r.getId()}::{lp.getId()}"
                    try:
                        params[key] = float(lp.getValue())
                    except Exception:
                        pass
        params[p.getId()] = p.getValue()
    return params

def extract_plot_species(model: libsbml.Model) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for s in model.getListOfSpecies():
        if s.getBoundaryCondition():
            continue
        name = s.getName() if s.isSetName() and s.getName() else s.getId()
        out[s.getId()] = name
    return out

def extract_reactions(model: libsbml.Model):
    reactions = []
    for r in model.getListOfReactions():
        reactants = [sr.getSpecies() for sr in r.getListOfReactants()]
        products = [sp.getSpecies() for sp in r.getListOfProducts()]
        reactions.append({"id": r.getId(), "reactants": reactants, "products": products})
    return reactions

def extract_initial_conditions(model: libsbml.Model) -> Dict[str, float]:
    inits: Dict[str, float] = {}
    for s in model.getListOfSpecies():
        if s.getBoundaryCondition():
            continue
        try:
            if s.isSetInitialConcentration():
                val = float(s.getInitialConcentration())
            elif s.isSetInitialAmount():
                val = float(s.getInitialAmount())
            else:
                val = 0.0
        except Exception:
            val = 0.0
        inits[s.getId()] = val
    return inits

def apply_params_to_sbml_doc(doc, param_values: Dict[str, float]):
    model = doc.getModel()
    for p in model.getListOfParameters():
        if p.getId() in param_values:
            try: p.setValue(float(param_values[p.getId()]))
            except: pass
    for r in model.getListOfReactions():
        if r.isSetKineticLaw():
            kl = r.getKineticLaw()
            if hasattr(kl, "getListOfLocalParameters") and kl.getListOfLocalParameters():
                for lp in kl.getListOfLocalParameters():
                    key = f"{r.getId()}::{lp.getId()}"
                    if key in param_values:
                        try: lp.setValue(float(param_values[key]))
                        except: pass
            if hasattr(kl, "getListOfParameters") and kl.getListOfParameters():
                for lp in kl.getListOfParameters():
                    key = f"{r.getId()}::{lp.getId()}"
                    if key in param_values:
                        try: lp.setValue(float(param_values[key]))
                        except: pass
    return doc

def apply_initial_conditions_to_sbml_doc(doc, initial_conditions: Dict[str, float]):
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
                    if s.isSetInitialAmount():
                        s.unsetInitialAmount()
                else:
                    s.setInitialAmount(val)
                    if s.isSetInitialConcentration():
                        s.unsetInitialConcentration()
            except:
                try: s.setInitialAmount(val)
                except: pass
    return doc

# ======================================================
# API EXISTENTE
# ======================================================

def extract_plot_species(model):
    return {s.getId(): s.getInitialConcentration() for s in model.getListOfSpecies() if not s.getBoundaryCondition()}


def extract_reactions(model):
    return [r.getId() for r in model.getListOfReactions()]


def extract_initial_conditions(model):
    return {s.getId(): s.getInitialConcentration() for s in model.getListOfSpecies()}


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    sbml_bytes = await file.read()
    doc, err = load_sbml_from_bytes(sbml_bytes)
    global last_sbml_bytes
    last_sbml_bytes = await file.read()   # Guardamos SBML
    doc, err = load_sbml_from_bytes(last_sbml_bytes)
    if err:
        return JSONResponse(status_code=400, content={"error": err})

@@ -164,121 +80,121 @@ async def inspect(file: UploadFile = File(...)):
        "defaultSelections": list(extract_plot_species(model).keys()),
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

    try:
        param_values = json.loads(param_values_json) if param_values_json else {}
    except: return JSONResponse(status_code=400, content={"error": "param_values_json inválido"})

    try:
        init_values = json.loads(initial_conditions_json) if initial_conditions_json else {}
    except: return JSONResponse(status_code=400, content={"error": "initial_conditions_json inválido"})

    if param_values: doc = apply_params_to_sbml_doc(doc, param_values)
    if init_values: doc = apply_initial_conditions_to_sbml_doc(doc, init_values)

    sbml_str = libsbml.writeSBMLToString(doc)
@app.post("/simulate")
async def simulate(payload: dict):
    global last_sbml_bytes
    if last_sbml_bytes is None:
        return JSONResponse(status_code=400, content={"error": "No se ha cargado ningún SBML aún."})

    try:
        rr = te.loadSBMLModel(sbml_str)
        rr.setIntegrator("cvode")
        rr.integrator.relative_tolerance = 1e-8
        rr.integrator.absolute_tolerance = 1e-12
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"No se pudo cargar el modelo: {e}"})
        doc, err = load_sbml_from_bytes(last_sbml_bytes)
        if err:
            return JSONResponse(status_code=400, content={"error": err})
        rr = te.loadSBMLModel(libsbml.writeSBMLToString(doc))

    species_ids = [s for s in selected_species.split(",") if s.strip()]
    if not species_ids:
        species_ids = list(extract_plot_species(doc.getModel()).keys())
    rr.selections = ["time"] + [f"[{sid}]" for sid in species_ids]
        # aplicar parámetros
        for k, v in payload.get("parameters", {}).items():
            try:
                rr[k] = float(v)
            except Exception:
                pass
        # aplicar condiciones iniciales
        for k, v in payload.get("initial_conditions", {}).items():
            try:
                rr[k] = float(v)
            except Exception:
                pass

    try:
        result = rr.simulate(t_start, t_end, int(n_points))
        result = rr.simulate(payload["tStart"], payload["tEnd"], payload["nPoints"])
        df = pd.DataFrame(result, columns=result.colnames)
        return {"time": df["time"].tolist(),
                "data": {col: df[col].tolist() for col in df.columns if col != "time"}}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Error en la simulación: {e}"})
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {"columns": list(result.colnames), "data": [list(map(float, r)) for r in result]}

# ======================================================
# NUEVO ENDPOINT /optimize
# ======================================================
@app.post("/optimize")
async def optimize(
    odes_file: UploadFile = File(...),
    data_file: UploadFile = File(...),
    method: str = Form("numeric")  # "numeric" o "pinn"
):
    global last_sbml_bytes
    if last_sbml_bytes is None:
        return JSONResponse(status_code=400, content={"error": "No se ha cargado ningún SBML aún."})

    try:
        # Leer ODEs
        odes_txt = (await odes_file.read()).decode("utf-8")
        odes = {}
        for line in odes_txt.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            var, expr = line.split("/dt =")
            var = var.replace("d[","").replace("]","").strip()
            odes[var] = expr.strip()
        state_vars = list(odes.keys())

        # Leer datos
        # -------------------------------
        # 1. Cargar modelo SBML desde memoria
        # -------------------------------
        doc, err = load_sbml_from_bytes(last_sbml_bytes)
        if err:
            return JSONResponse(status_code=400, content={"error": err})
        model = doc.getModel()
        rr = te.loadSBMLModel(libsbml.writeSBMLToString(doc))

        state_vars = [s.getId() for s in model.getListOfSpecies() if not s.getBoundaryCondition()]

        # -------------------------------
        # 2. Leer datos experimentales
        # -------------------------------
        df = pd.read_csv(data_file.file)
        df = df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)

        # Preparar datos
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float64

        t_orig = torch.tensor(df["time"].values, dtype=dtype, device=device)
        t_min, t_max = t_orig.min(), t_orig.max()
        t = (t_orig - t_min) / (t_max - t_min)

        existing_vars = [v for v in state_vars if v in df.columns]
        y_true = torch.tensor(df[existing_vars].values, dtype=dtype, device=device)

        # Normalización
        y_mean, y_std = y_true.mean(0), y_true.std(0) + 1e-6
        y_true_norm = (y_true - y_mean) / y_std

        # Detectar parámetros
        params_names = sorted(set(re.findall(r'k\d+', " ".join(odes.values()))))
        params = nn.ParameterDict({k: nn.Parameter(torch.rand(1, dtype=dtype, device=device)) for k in params_names})

        # Safe eval
        def safe_eval(expr, ctx):
            try: return eval(expr, {"torch": torch}, ctx)
            except: return torch.tensor(0.0, dtype=dtype, device=device)
        # Detectar parámetros del SBML
        params = extract_parameters(model)
        params_nn = nn.ParameterDict({
            k: nn.Parameter(torch.tensor([v], dtype=dtype, device=device))
            for k, v in params.items()
        })

        # -------------------------------
        # 3. Definir ODEFunc con RoadRunner
        # -------------------------------
        class ODEFunc(nn.Module):
            def __init__(self, odes, state_vars, params):
            def __init__(self, rr, state_vars, params):
                super().__init__()
                self.odes, self.state_vars, self.params = odes, state_vars, params
                self.rr = rr
                self.state_vars = state_vars
                self.params = params
            def forward(self, t, y):
                y = y.flatten()
                ctx = {var: y[i] for i, var in enumerate(self.state_vars)}
                ctx.update({k: self.params[k] for k in self.params})
                dydt = [safe_eval(self.odes[var], ctx).flatten()[0] for var in self.state_vars]
                return torch.stack(dydt)
                for k, v in self.params.items():
                    try:
                        self.rr[k] = v.item()
                    except:
                        pass
                for i, sid in enumerate(self.state_vars):
                    self.rr[sid] = y[i].item()
                dydt = self.rr.getRatesOfChange()
                return torch.tensor(dydt, dtype=dtype, device=device)

        ode_func = ODEFunc(odes, existing_vars, params).to(device)
        ode_func = ODEFunc(rr, existing_vars, params_nn).to(device)
        y0 = y_true_norm[0].clone().detach().to(device).requires_grad_(True)

        results = {}

        # -------------------------------
        # 4. Método numeric (adjoint)
        # -------------------------------
        if method == "numeric":
            opt = torch.optim.Adam(list(params.values()), lr=0.03)
            opt = torch.optim.Adam(list(params_nn.values()), lr=0.03)
            loss_fn = nn.MSELoss()
            best_loss = float("inf")
            patience = 0
            for epoch in range(5000):
            best_loss, patience = float("inf"), 0
            for epoch in range(2000):
                opt.zero_grad()
                y_pred = odeint(ode_func, y0, t, method="dopri5")
                loss = loss_fn(y_pred, y_true_norm)
@@ -288,9 +204,13 @@ def forward(self, t, y):
                    best_loss, patience = loss.item(), 0
                else:
                    patience += 1
                    if patience > 500: break
                    if patience > 200: break
            results["loss"] = float(best_loss)
            y_opt = y_pred * y_std + y_mean

        # -------------------------------
        # 5. Método PINN
        # -------------------------------
        elif method == "pinn":
            class PINN(nn.Module):
                def __init__(self, n_inputs, n_outputs, hidden=128):
@@ -303,34 +223,45 @@ def __init__(self, n_inputs, n_outputs, hidden=128):
                def forward(self, t): return self.net(t)

            pinn = PINN(1, len(existing_vars)).to(device, dtype)
            opt_pinn = torch.optim.Adam(list(pinn.parameters())+list(params.values()), lr=0.001)
            opt_pinn = torch.optim.Adam(list(pinn.parameters())+list(params_nn.values()), lr=0.001)

            def pinn_loss(t_batch, y_batch):
                t_batch = t_batch.unsqueeze(1).requires_grad_(True)
                y_pred = pinn(t_batch)
                dydt_pred = torch.autograd.grad(y_pred, t_batch, torch.ones_like(y_pred), create_graph=True)[0]
                ctx = {var: y_pred[:, i] for i, var in enumerate(existing_vars)}
                ctx.update({k: params[k].expand_as(t_batch[:,0]) for k in params})
                dydt_odes = torch.stack([safe_eval(odes[var], ctx).flatten() for var in existing_vars], dim=1)
                return ((y_pred - y_batch)**2).mean() + ((dydt_pred - dydt_odes)**2).mean()
                dydt_pred = torch.autograd.grad(y_pred, t_batch,
                    torch.ones_like(y_pred), create_graph=True)[0]
                loss_data = ((y_pred - y_batch)**2).mean()
                loss_phys = (dydt_pred**2).mean()
                return loss_data + loss_phys, y_pred

            best_loss, patience = float("inf"), 0
            for epoch in range(5000):
                idx = torch.randint(0, len(t), (128,), device=device)
                loss = pinn_loss(t[idx], y_true_norm[idx])
            for epoch in range(2000):
                idx = torch.randint(0, len(t), (64,), device=device)
                loss, _ = pinn_loss(t[idx], y_true_norm[idx])
                if torch.isnan(loss): break
                opt_pinn.zero_grad(); loss.backward(); opt_pinn.step()
                if loss.item() < best_loss: best_loss, patience = loss.item(), 0
                else:
                    patience += 1
                    if patience > 500: break
                    if patience > 200: break
            results["loss"] = float(best_loss)
            with torch.no_grad():
                y_opt = pinn(t.unsqueeze(1)) * y_std + y_mean

        # -------------------------------
        # 6. Devolver resultados
        # -------------------------------
        return {
            "status": "ok",
            "variables": existing_vars,
            "params": {k: float(v.item()) for k, v in params.items()},
            "loss": results["loss"]
            "params": {k: float(v.item()) for k,v in params_nn.items()},
            "loss": results["loss"],
            "time": t_orig.cpu().numpy().tolist(),
            "trajectories": {var: y_opt[:,i].detach().cpu().numpy().tolist()
                             for i,var in enumerate(existing_vars)}
        }

    except Exception as e:
        return {"error": str(e)}


