# FILE: from_scratch/autograd/tensor.py

import numpy as np
from typing import Any, List, Optional, Tuple, Union

class Function:
    """
    Base class for differentiable operations. It links Tensors in the computational graph.
    """
    def __init__(self, *tensors: "Tensor"):
        self.parents = tensors
        self.saved_tensors: List[np.ndarray] = []

    def save_for_backward(self, *tensors: np.ndarray):
        """Saves given tensors for use in the backward pass."""
        self.saved_tensors.extend(tensors)

    def forward(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Performs the forward computation. Must be implemented by subclasses."""
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> Union[None, np.ndarray, Tuple[Optional[np.ndarray], ...]]:
        """Computes the gradient of the loss with respect to the inputs. Must be implemented by subclasses."""
        raise NotImplementedError

    @classmethod
    def apply(cls, *args: "Tensor", **kwargs: Any) -> "Tensor":
        """Applies the function, creating a new tensor that tracks the operation."""
        ctx = cls(*args)
        output_data = ctx.forward(*(t.data for t in args), **kwargs)
        return Tensor(output_data, requires_grad=any(t.requires_grad for t in args), _creator=ctx)

# --- Function Subclasses ---

class GetItem(Function):
    def forward(self, x: np.ndarray, *, idx: Any) -> np.ndarray:
        self.input_shape = x.shape
        self.idx = idx
        return x[idx]
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad_out = np.zeros(self.input_shape, dtype=grad.dtype)
        grad_out[self.idx] = grad
        return grad_out

class Add(Function):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x + y
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad, grad

class Mul(Function):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.save_for_backward(x, y)
        return x * y
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.saved_tensors
        return grad * y, grad * x

class MatMul(Function):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.save_for_backward(x, y)
        return x @ y
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.saved_tensors
        return grad @ np.swapaxes(y, -2, -1), np.swapaxes(x, -2, -1) @ grad
        
class Pow(Function):
    def forward(self, x: np.ndarray, *, power: float) -> np.ndarray:
        self.save_for_backward(x)
        self.power = power
        return x ** power
    def backward(self, grad: np.ndarray) -> np.ndarray:
        x, = self.saved_tensors
        return grad * (self.power * (x ** (self.power - 1)))

class Sqrt(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.save_for_backward(x)
        return np.sqrt(x)
    def backward(self, grad: np.ndarray) -> np.ndarray:
        x, = self.saved_tensors
        return grad * 0.5 * (x ** -0.5)
        
class Sum(Function):
    def forward(self, x: np.ndarray, *, axis: Optional[Union[int, Tuple]] = None, keepdims: bool = False) -> np.ndarray:
        self.input_shape = x.shape
        self.axis = axis
        self.keepdims = keepdims
        return np.sum(x, axis=axis, keepdims=keepdims)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.axis is not None and not self.keepdims:
            axes_to_add = self.axis if isinstance(self.axis, tuple) else (self.axis,)
            for axis in sorted(axes_to_add):
                grad = np.expand_dims(grad, axis=axis)
        return np.broadcast_to(grad, self.input_shape)

class Cat(Function):
    def forward(self, *tensors_data: np.ndarray, axis: int = 0) -> np.ndarray:
        self.axis = axis
        self.original_shapes = [t.shape for t in tensors_data]
        return np.concatenate(tensors_data, axis=axis)

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        grads: List[np.ndarray] = []
        start_idx = 0
        for shape in self.original_shapes:
            slice_indices = [slice(None)] * grad.ndim
            slice_indices[self.axis] = slice(start_idx, start_idx + shape[self.axis])
            grads.append(grad[tuple(slice_indices)])
            start_idx += shape[self.axis]
        return tuple(grads)

class Reshape(Function):
    def forward(self, x: np.ndarray, *, shape: Tuple) -> np.ndarray:
        self.input_shape = x.shape
        return x.reshape(shape)
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(self.input_shape)

class Permute(Function):
    def forward(self, x: np.ndarray, *, dims: Tuple) -> np.ndarray:
        self.dims = dims
        return np.transpose(x, dims)
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.transpose(grad, np.argsort(self.dims))

# --- The Tensor Class ---

class Tensor:
    """The core data structure for the autograd engine."""
    def __init__(self, data: Union[np.ndarray, float, int, list], requires_grad: bool = False, _creator: Optional[Function] = None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.requires_grad = requires_grad
        self._creator = _creator
        self.grad: Optional[np.ndarray] = None
    
    @property
    def shape(self) -> Tuple:
        return self.data.shape
    
    @property
    def T(self):
        if len(self.shape) != 2:
            raise ValueError(".T property is only for 2D Tensors.")
        return self.transpose(0, 1)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad: Optional[np.ndarray] = None):
        if not self.requires_grad:
            return
        
        if grad is None:
            if self.data.size != 1:
                raise ValueError("Gradient must be specified for non-scalar Tensors.")
            grad = np.ones_like(self.data, dtype=np.float32)
        
        self.grad = grad
        
        visited, topo_nodes = set(), []
        def build_topo(node):
            if id(node) not in visited:
                visited.add(id(node))
                if node._creator:
                    for p in node._creator.parents:
                        build_topo(p)
                    topo_nodes.append(node)
        build_topo(self)

        for node in reversed(topo_nodes):
            if not node._creator or node.grad is None:
                continue

            parent_grads = node._creator.backward(node.grad)
            if not isinstance(parent_grads, tuple):
                parent_grads = (parent_grads,)

            for p, p_g in zip(node._creator.parents, parent_grads):
                if p_g is not None and p.requires_grad:
                    # Handle gradient broadcasting
                    if p_g.shape != p.shape:
                        if p_g.ndim > len(p.shape):
                            p_g = p_g.sum(axis=tuple(range(p_g.ndim - len(p.shape))))
                        ax_sum = tuple([i for i, d in enumerate(p.shape) if d==1 and i < p_g.ndim and p_g.shape[i] > 1])
                        if ax_sum:
                            p_g = p_g.sum(axis=ax_sum, keepdims=True)
                        if p_g.shape != p.shape:
                            p_g = p_g.reshape(p.shape)
                    
                    if p.grad is None:
                        p.grad = p_g.copy()
                    else:
                        p.grad += p_g
    
    # --- Operator Overloads ---
    def __getitem__(self, idx: Any) -> "Tensor":
        return GetItem.apply(self, idx=idx)
    def __add__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        return Add.apply(self, other)
    def __mul__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        return Mul.apply(self, other)
    def __matmul__(self, other):
        return MatMul.apply(self, other)
    def __pow__(self, power):
        if not isinstance(power, (int, float)): raise TypeError("Power must be scalar")
        return Pow.apply(self, power=power)
    def __truediv__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        return self * (other ** -1.0)
    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return other + (-self)
    def __radd__(self, other):
        return self + other
    def __rmul__(self, other):
        return self * other
    def __eq__(self, other):
        other_val = other.data if isinstance(other, Tensor) else other
        return self.data == other_val

    # --- Methods ---
    def sqrt(self):
        return Sqrt.apply(self)
    def sum(self, axis=None, keepdims=False):
        return Sum.apply(self, axis=axis, keepdims=keepdims)
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple): shape = shape[0]
        return Reshape.apply(self, shape=shape)
    def transpose(self, dim0, dim1):
        dims = list(range(len(self.shape))); dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return Permute.apply(self, dims=tuple(dims))

    @staticmethod
    def cat(tensors, axis=0):
        return Cat.apply(*tensors, axis=axis)