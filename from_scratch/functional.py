# FILE: from_scratch/functional.py

from .autograd.tensor import Tensor, Function
import numpy as np
from typing import Optional, Union, Tuple

# --- Function Subclasses for Activations and Losses ---

class ReLU(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.save_for_backward(x)
        return np.maximum(x, 0)
    def backward(self, grad: np.ndarray):
        x, = self.saved_tensors
        return grad * (x > 0)

class Sigmoid(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        output = 1 / (1 + np.exp(-x))
        self.save_for_backward(output)
        return output
    def backward(self, grad: np.ndarray):
        output, = self.saved_tensors
        return grad * output * (1 - output)

class Tanh(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        output = np.tanh(x)
        self.save_for_backward(output)
        return output
    def backward(self, grad: np.ndarray):
        output, = self.saved_tensors
        return grad * (1 - output**2)

class Softmax(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        self.save_for_backward(probs)
        return probs
    def backward(self, grad: np.ndarray):
        probs, = self.saved_tensors
        sum_term = np.sum(grad * probs, axis=-1, keepdims=True)
        return (grad - sum_term) * probs

class MSELoss(Function):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self.save_for_backward(y_pred, y_true)
        return np.array(np.mean((y_pred - y_true) ** 2))
    def backward(self, grad: np.ndarray):
        y_pred, y_true = self.saved_tensors
        n = y_pred.shape[0] if y_pred.ndim > 0 else 1
        grad_y_pred = grad * (2.0 / n) * (y_pred - y_true)
        return grad_y_pred, None

class BCELoss(Function):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        self.save_for_backward(y_pred, y_true)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.array(loss)
    def backward(self, grad: np.ndarray):
        y_pred, y_true = self.saved_tensors
        n = y_pred.shape[0] if y_pred.ndim > 0 else 1
        grad_y_pred = grad * (1.0 / n) * ((y_pred - y_true) / (y_pred * (1 - y_pred)))
        return grad_y_pred, None

class CrossEntropyLoss(Function):
    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_true = y_true.astype(int)
        max_logits = np.max(logits, axis=-1, keepdims=True)
        shifted_logits = logits - max_logits
        log_probs = shifted_logits - np.log(np.sum(np.exp(shifted_logits), axis=-1, keepdims=True))
        batch_size = logits.shape[0]
        true_log_probs = log_probs[np.arange(batch_size), y_true.flatten()]
        loss = -np.mean(true_log_probs)
        probs = np.exp(log_probs)
        self.save_for_backward(probs, y_true)
        return np.array(loss)
    def backward(self, grad: np.ndarray):
        probs, y_true = self.saved_tensors
        batch_size = probs.shape[0]
        y_true_one_hot = np.zeros_like(probs)
        y_true_one_hot[np.arange(batch_size), y_true.flatten()] = 1
        grad_logits = grad * (probs - y_true_one_hot) / batch_size
        return grad_logits, None

# --- Helper Functions ---

def relu(x: Tensor) -> Tensor:
    return ReLU.apply(x)
def sigmoid(x: Tensor) -> Tensor:
    return Sigmoid.apply(x)
def tanh(x: Tensor) -> Tensor:
    return Tanh.apply(x)
def softmax(x: Tensor) -> Tensor:
    return Softmax.apply(x)
def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return MSELoss.apply(y_pred, y_true)
def binary_cross_entropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return BCELoss.apply(y_pred, y_true)
def cross_entropy(logits: Tensor, y_true: Tensor) -> Tensor:
    return CrossEntropyLoss.apply(logits, y_true)