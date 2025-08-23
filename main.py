import io
import re
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

# ---------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Guardar último SBML cargado
last_sbml_bytes: Optional[bytes] = None


# ---------------------------------------------------------
# Utilidades SBML
# ---------------------------------------------------------
def load_sbml_from_bytes(sbml_bytes: bytes):
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
        params[p.getId()] = p.getValue()
    return params


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
    global last_sbml_bytes
    last_sbml_bytes = await file.read()   # Guardamos SBML
    doc, err = load_sbml_from_bytes(last_sbml_bytes)
    if err:
        return JSONResponse(status_code=400, content={"error": err})

    model = doc.getModel()
    return {
        "parameters": extract_parameters(model),
        "species": extract_plot_species(model),
        "reactions": extract_reactions(model),
        "initial_conditions": extract_initial_conditions(model),
        "defaultSelections": list(extract_plot_species(model).keys()),
    }


@app.post("/simulate")
async def simulate(payload: dict):
    global last_sbml_bytes
    if last_sbml_bytes is None:
        return JSONResponse(status_code=400, content={"error": "No se ha cargado ningún SBML aún."})

    try:
        doc, err = load_sbml_from_bytes(last_sbml_bytes)
        if err:
            return JSONResponse(status_code=400, content={"error": err})
        rr = te.loadSBMLModel(libsbml.writeSBMLToString(doc))

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

        result = rr.simulate(payload["tStart"], payload["tEnd"], payload["nPoints"])
        df = pd.DataFrame(result, columns=result.colnames)
        return {"time": df["time"].tolist(),
                "data": {col: df[col].tolist() for col in df.columns if col != "time"}}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/optimize")
async def optimize(
    data_file: UploadFile = File(...),
    method: str = Form("numeric")  # "numeric" o "pinn"
):
    global last_sbml_bytes
    if last_sbml_bytes is None:
        return JSONResponse(status_code=400, content={"error": "No se ha cargado ningún SBML aún."})

    try:
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
            def __init__(self, rr, state_vars, params):
                super().__init__()
                self.rr = rr
                self.state_vars = state_vars
                self.params = params
            def forward(self, t, y):
                for k, v in self.params.items():
                    try:
                        self.rr[k] = v.item()
                    except:
                        pass
                for i, sid in enumerate(self.state_vars):
                    self.rr[sid] = y[i].item()
                dydt = self.rr.getRatesOfChange()
                return torch.tensor(dydt, dtype=dtype, device=device)

        ode_func = ODEFunc(rr, existing_vars, params_nn).to(device)
        y0 = y_true_norm[0].clone().detach().to(device).requires_grad_(True)

        results = {}

        # -------------------------------
        # 4. Método numeric (adjoint)
        # -------------------------------
        if method == "numeric":
            opt = torch.optim.Adam(list(params_nn.values()), lr=0.03)
            loss_fn = nn.MSELoss()
            best_loss, patience = float("inf"), 0
            for epoch in range(2000):
                opt.zero_grad()
                y_pred = odeint(ode_func, y0, t, method="dopri5")
                loss = loss_fn(y_pred, y_true_norm)
                if torch.isnan(loss): break
                loss.backward(); opt.step()
                if loss.item() < best_loss:
                    best_loss, patience = loss.item(), 0
                else:
                    patience += 1
                    if patience > 200: break
            results["loss"] = float(best_loss)
            y_opt = y_pred * y_std + y_mean

        # -------------------------------
        # 5. Método PINN
        # -------------------------------
        elif method == "pinn":
            class PINN(nn.Module):
                def __init__(self, n_inputs, n_outputs, hidden=128):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(n_inputs, hidden), nn.Tanh(),
                        nn.Linear(hidden, hidden), nn.Tanh(),
                        nn.Linear(hidden, n_outputs)
                    )
                def forward(self, t): return self.net(t)

            pinn = PINN(1, len(existing_vars)).to(device, dtype)
            opt_pinn = torch.optim.Adam(list(pinn.parameters())+list(params_nn.values()), lr=0.001)

            def pinn_loss(t_batch, y_batch):
                t_batch = t_batch.unsqueeze(1).requires_grad_(True)
                y_pred = pinn(t_batch)
                dydt_pred = torch.autograd.grad(y_pred, t_batch,
                    torch.ones_like(y_pred), create_graph=True)[0]
                loss_data = ((y_pred - y_batch)**2).mean()
                loss_phys = (dydt_pred**2).mean()
                return loss_data + loss_phys, y_pred

            best_loss, patience = float("inf"), 0
            for epoch in range(2000):
                idx = torch.randint(0, len(t), (64,), device=device)
                loss, _ = pinn_loss(t[idx], y_true_norm[idx])
                if torch.isnan(loss): break
                opt_pinn.zero_grad(); loss.backward(); opt_pinn.step()
                if loss.item() < best_loss: best_loss, patience = loss.item(), 0
                else:
                    patience += 1
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
            "params": {k: float(v.item()) for k,v in params_nn.items()},
            "loss": results["loss"],
            "time": t_orig.cpu().numpy().tolist(),
            "trajectories": {var: y_opt[:,i].detach().cpu().numpy().tolist()
                             for i,var in enumerate(existing_vars)}
        }

    except Exception as e:
        return {"error": str(e)}


