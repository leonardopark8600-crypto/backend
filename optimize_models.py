from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field

MethodName = Literal["lm", "bfgs", "pinn"]

class SessionCreateResponse(BaseModel):
    session_id: str

class DataUploadResponse(BaseModel):
    session_id: str
    datasets: Dict[str, List[str]]  # dataset name -> columns

class SpeciesDataMap(BaseModel):
    species_id: str
    dataset: str
    time_column: str
    value_column: str

class ParamSelection(BaseModel):
    to_optimize: List[str] = Field(default_factory=list)
    fixed: Dict[str, float] = Field(default_factory=dict)

class OptimizeConfig(BaseModel):
    method: MethodName = "lm"
    max_iter: int = 500
    learning_rate: float = 1e-3
    pinn_epochs: int = 2000
    weight_data: float = 1.0
    weight_phys: float = 1.0

class RunOptimizeRequest(BaseModel):
    session_id: str
    optimize_config: OptimizeConfig
    initial_guess: Dict[str, float] = Field(default_factory=dict)

class OptimizeResult(BaseModel):
    method: MethodName
    estimates: Dict[str, float]
    ci95: Dict[str, List[float]]
    metrics: Dict[str, float]
    traces: Dict[str, List[float]]  # optional per-iter loss, etc.
    plot: Dict[str, Any]  # x: time, series: {species_id: {"sim": [...], "data": [...]}}
