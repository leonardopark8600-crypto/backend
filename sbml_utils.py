import libsbml
from typing import Dict, Optional

def load_sbml_from_bytes(sbml_bytes: bytes):
    reader = libsbml.SBMLReader()
    doc = reader.readSBMLFromString(sbml_bytes.decode("utf-8", errors="ignore"))
    if doc is None or doc.getModel() is None or doc.getNumErrors() > 0:
        return None, "Error al leer el archivo SBML."
    return doc, None


def extract_parameters(model: libsbml.Model) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for p in model.getListOfParameters():
        if p.isSetId():
            try:
                params[p.getId()] = float(p.getValue())
            except Exception:
                pass
    for r in model.getListOfReactions():
        if r.isSetKineticLaw():
            kl = r.getKineticLaw()
            if hasattr(kl, "getListOfLocalParameters") and kl.getListOfLocalParameters() is not None:
                for lp in kl.getListOfLocalParameters():
                    key = f"{r.getId()}::{lp.getId()}"
                    try:
                        params[key] = float(lp.getValue())
                    except Exception:
                        pass
            if hasattr(kl, "getListOfParameters") and kl.getListOfParameters() is not None:
                for lp in kl.getListOfParameters():
                    key = f"{r.getId()}::{lp.getId()}"
                    try:
                        params[key] = float(lp.getValue())
                    except Exception:
                        pass
    return params


def extract_layout_positions(model: libsbml.Model):
    positions = {}
    plugin = model.getPlugin("layout")
    if not plugin or plugin.getNumLayouts() == 0:
        return positions
    layout = plugin.getLayout(0)
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
        val: Optional[float] = None
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


def apply_params_to_sbml_doc(doc: libsbml.SBMLDocument, param_values: Dict[str, float]):
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


def apply_initial_conditions_to_sbml_doc(doc: libsbml.SBMLDocument, initial_conditions: Dict[str, float]):
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
            except Exception:
                try:
                    s.setInitialAmount(val)
                except Exception:
                    pass
    return doc