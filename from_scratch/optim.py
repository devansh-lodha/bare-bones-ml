# FILE: from_scratch/optim.py

from typing import List
from .autograd.tensor import Tensor
import numpy as np

class Optimizer:
    """Base class for all optimizers."""
    def __init__(self, params: List[Tensor], lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        """Resets the gradients of all parameters."""
        for p in self.params:
            p.grad = None

class SGD(Optimizer):
    """Implements Stochastic Gradient Descent optimizer."""
    def __init__(self, params: List[Tensor], lr: float):
        super().__init__(params, lr)

    def step(self):
        """Performs a single optimization step."""
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

class Adam(Optimizer):
    """
    Implements the Adam optimizer.
    It maintains per-parameter adaptive learning rates.
    """
    def __init__(self, params: List[Tensor], lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        """Performs a single optimization step."""
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
                
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)