from __future__ import annotations
from typing import Dict, Callable, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class MLP(nn.Module):
    def __init__(self, n_in, n_out, width=64, depth=3):
        super().__init__()
        layers = [nn.Linear(n_in, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, n_out)]
        self.net = nn.Sequential(*layers)
    def forward(self, t):
        # t shape: (N,1)
        return self.net(t)

def pinn_refine(ode_fn: Callable[[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
                t_data: np.ndarray,
                y_data: np.ndarray,
                param_init: Dict[str, float],
                epochs: int = 2000,
                lr: float = 1e-3,
                weight_data: float = 1.0,
                weight_phys: float = 1.0) -> Tuple[Dict[str, float], List[float]]:
    device = torch.device("cpu")
    t_tensor = torch.tensor(t_data.reshape(-1,1), dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_data, dtype=torch.float32, device=device)  # shape (N, n_state)

    n_state = y_tensor.shape[1]
    net = MLP(1, n_state).to(device)
    params = {k: torch.tensor(v, dtype=torch.float32, requires_grad=True, device=device) for k, v in param_init.items()}
    opt = torch.optim.Adam(list(net.parameters()) + list(params.values()), lr=lr)

    loss_trace = []
    for _ in range(epochs):
        opt.zero_grad()
        y_pred = net(t_tensor)  # (N, n_state)
        # physics residual via autograd
        dy_dt = torch.autograd.grad(y_pred, t_tensor, grad_outputs=torch.ones_like(y_pred), create_graph=True, retain_graph=True)[0]
        f = ode_fn(t_tensor, y_pred, params)  # (N, n_state)
        loss_phys = torch.mean((dy_dt - f)**2)
        loss_data = torch.mean((y_pred - y_tensor)**2)
        loss = weight_phys * loss_phys + weight_data * loss_data
        loss.backward()
        opt.step()
        loss_trace.append(float(loss.detach().cpu().item()))
    # return refined parameters
    return {k: float(v.detach().cpu().item()) for k, v in params.items()}, loss_trace
