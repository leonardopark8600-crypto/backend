from __future__ import annotations
from typing import Dict, Any
from uuid import uuid4

class Repo:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create(self) -> str:
        sid = str(uuid4())
        self.sessions[sid] = {
            "sbml_str": None,
            "sympy_text": None,
            "species": [],
            "parameters": {},
            "datasets": {},        # name -> pandas.DataFrame
            "maps": {},            # species_id -> (dataset, time_col, val_col)
            "selection": {"to_optimize": [], "fixed": {}}
        }
        return sid

    def get(self, sid: str) -> Dict[str, Any]:
        if sid not in self.sessions:
            raise KeyError("session not found")
        return self.sessions[sid]

repo = Repo()
