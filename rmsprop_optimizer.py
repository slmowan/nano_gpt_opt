import torch
import torch.nn as nn
from typing import Iterable, Tuple, Callable
import math


class RMSProp(torch.optim.Optimizer):  
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha parameter: {alpha} - should be in [0.0, 1.0]")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum} - should be >= 0.0")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay} - should be >= 0.0")
            
        defaults = {
            "lr": lr, 
            "alpha": alpha, 
            "eps": eps, 
            "weight_decay": weight_decay, 
            "momentum": momentum
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
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of squared gradient values
                    state["square_avg"] = torch.zeros_like(grad)
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                
                square_avg = state["square_avg"]
                alpha = group["alpha"]
                
                state["step"] += 1
                
                # Apply weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])
                
                # Update squared gradient moving average
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                
                # Compute the update
                avg = square_avg.sqrt().add_(group["eps"])
                
                if group["momentum"] > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).addcdiv_(grad, avg)
                    p.add_(buf, alpha=-group["lr"])
                else:
                    p.addcdiv_(grad, avg, value=-group["lr"])
        
        return loss