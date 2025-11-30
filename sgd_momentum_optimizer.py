import torch
import torch.nn as nn
from typing import Iterable, Tuple, Callable
import math


class SGDMomentum(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum} - should be in [0.0, 1.0)")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay} - should be >= 0.0")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
            
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "dampening": dampening,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        
        Arguments:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                # Initialize momentum buffer if needed
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(grad)
                
                buf = state["momentum_buffer"]
                state["step"] += 1
                
                # Update momentum buffer
                if state["step"] > 1:
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                else:
                    buf.copy_(grad)
                
                # Apply Nesterov momentum if enabled
                if nesterov:
                    grad = grad.add(buf, alpha=momentum)
                else:
                    grad = buf
                
                # Update parameters
                p.add_(grad, alpha=-group["lr"])
        
        return loss