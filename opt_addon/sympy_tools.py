from __future__ import annotations
from typing import Dict, List, Tuple
import sympy as sp

def detect_parameters_from_sympy(equations_text: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Parse sympy-formatted ODEs. Returns (state_vars, params).
    Example expected lines:
        dx/dt = k1*x - k2*y
        dy/dt = -k3*x + k4*y
    Accepts pythonic: Eq(Derivative(x(t), t), k1*x(t) - k2*y(t))
    """
    # Try multiple parsing strategies
    lines = [ln.strip() for ln in equations_text.splitlines() if ln.strip()]
    t = sp.symbols('t')
    states = set()
    rhs_exprs = []

    for ln in lines:
        if '=' in ln:
            left, right = ln.split('=', 1)
            left = left.strip()
            right = right.strip()
            # handle dx/dt forms
            if '/' in left and 'dt' in left:
                # crude extract variable name between d and /dt
                var = left.split('d')[1].split('/')[0].strip().strip('() ')
                try:
                    sym = sp.Function(var)(t)
                except Exception:
                    sym = sp.Symbol(var)
                states.add(sym)
                expr = sp.sympify(right, convert_xor=True)
                rhs_exprs.append(expr)
            else:
                # try sympy Eq(Derivative(x(t),t), ...)
                try:
                    eq = sp.sympify(ln, convert_xor=True)
                    if isinstance(eq, sp.Equality):
                        lhs = eq.lhs
                        rhs = eq.rhs
                        if isinstance(lhs, sp.Derivative):
                            v = lhs.args[0]
                            states.add(v)
                            rhs_exprs.append(rhs)
                except Exception:
                    continue

    # collect symbols
    all_symbols = set().union(*[expr.free_symbols for expr in rhs_exprs]) if rhs_exprs else set()
    # remove time and state symbols
    time_like = {t, sp.Symbol('t')}
    state_syms = set()
    for s in states:
        if isinstance(s, sp.Function):
            state_syms.add(s)
            state_syms.add(s.func(sp.Symbol('t')))
            state_syms.add(sp.Symbol(str(s.func)))
        else:
            state_syms.add(s)

    candidate_params = [str(s) for s in all_symbols if s not in time_like and s not in state_syms]
    # Provide a dict with default types
    params = {p: "float" for p in candidate_params}
    state_ids = sorted({str(s.func) if isinstance(s, sp.Function) else str(s) for s in states})
    return state_ids, params
