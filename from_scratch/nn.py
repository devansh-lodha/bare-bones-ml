# FILE: from_scratch/nn.py (The Complete and Final Library)

from typing import List, Tuple
from .autograd.tensor import Tensor
from .functional import relu, tanh, sigmoid, softmax
import numpy as np

# --- Base Module ---

class Module:
    """
    Base class for all neural network modules (e.g., layers, models).
    It handles parameter registration and provides helper methods.
    """
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name: str, value):
        """
        Custom attribute setter to automatically register Tensors as parameters
        and other Modules as sub-modules.
        """
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        
        super().__setattr__(name, value)

    def parameters(self) -> List[Tensor]:
        """
        Returns a list of all parameters in the module and its sub-modules.
        """
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self):
        """Resets the gradients of all parameters to None."""
        for p in self.parameters():
            p.grad = None
    
    def __call__(self, *args, **kwargs):
        """Allows the module to be called like a function."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass. Must be implemented by subclasses."""
        raise NotImplementedError

# --- Container and Foundational Layers ---

class Sequential(Module):
    """A container for modules that will be applied in sequence."""
    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[str(i)] = module

    def forward(self, x: Tensor) -> Tensor:
        """Passes the input through each module in order."""
        for module in self._modules.values():
            x = module(x)
        return x

class Linear(Module):
    """A standard fully-connected linear layer: y = xW^T + b"""
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = Tensor(
            np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size), 
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(output_size), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T + self.bias

class ReLU(Module):
    """A ReLU activation layer."""
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)

class Embedding(Module):
    """A simple token embedding layer."""
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.embedding = Tensor(np.random.randn(vocab_size, embed_size), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding[x.data.astype(int)]

# --- Recurrent Layers ---

class RecurrentBlock(Module):
    """A simple Recurrent Neural Network (RNN) block."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size
        self.w = Tensor(np.random.randn(concat_size, hidden_size) * 0.01, requires_grad=True)
        self.b_h = Tensor(np.zeros(hidden_size), requires_grad=True)

    def forward(self, x: Tensor, h_prev: Tensor) -> Tensor:
        """Performs one step of the RNN computation."""
        combined = Tensor.cat([x, h_prev], axis=1)
        h_next = tanh(combined @ self.w + self.b_h)
        return h_next

    def parameters(self) -> List[Tensor]:
        return [self.w, self.b_h]
        
class LSTMBlock(Module):
    """A Long Short-Term Memory (LSTM) block."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size
        
        self.W_f = Tensor(np.random.randn(concat_size, hidden_size) * 0.01, requires_grad=True)
        self.b_f = Tensor(np.zeros(hidden_size), requires_grad=True)
        self.W_i = Tensor(np.random.randn(concat_size, hidden_size) * 0.01, requires_grad=True)
        self.b_i = Tensor(np.zeros(hidden_size), requires_grad=True)
        self.W_c = Tensor(np.random.randn(concat_size, hidden_size) * 0.01, requires_grad=True)
        self.b_c = Tensor(np.zeros(hidden_size), requires_grad=True)
        self.W_o = Tensor(np.random.randn(concat_size, hidden_size) * 0.01, requires_grad=True)
        self.b_o = Tensor(np.zeros(hidden_size), requires_grad=True)

    def forward(self, x: Tensor, h_prev: Tensor, c_prev: Tensor) -> Tuple[Tensor, Tensor]:
        combined = Tensor.cat([x, h_prev], axis=1)
        f_t = sigmoid((combined @ self.W_f) + self.b_f)
        i_t = sigmoid((combined @ self.W_i) + self.b_i)
        c_candidate = tanh((combined @ self.W_c) + self.b_c)
        c_next = f_t * c_prev + i_t * c_candidate
        o_t = sigmoid((combined @ self.W_o) + self.b_o)
        h_next = o_t * tanh(c_next)
        return h_next, c_next

    def parameters(self) -> List[Tensor]:
        return [self.W_f, self.b_f, self.W_i, self.b_i, self.W_c, self.b_c, self.W_o, self.b_o]

# --- Attention and Transformer Components ---

class ScaledDotProductAttention(Module):
    """Computes Scaled Dot-Product Attention."""
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tensor:
        key_dim = Tensor(k.shape[-1])
        key_transposed = k.transpose(-2, -1)
        scores = q @ key_transposed
        scaled_scores = scores / key_dim.sqrt()
        if mask is not None:
            scaled_scores = scaled_scores + mask
        weights = softmax(scaled_scores)
        return weights @ v

class MultiHeadAttention(Module):
    """Computes Multi-Head Attention."""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.d_k = hidden_size // num_heads
        self.num_heads = num_heads
        self.q_linear = Linear(hidden_size, hidden_size)
        self.k_linear = Linear(hidden_size, hidden_size)
        self.v_linear = Linear(hidden_size, hidden_size)
        self.attention = ScaledDotProductAttention()
        self.fc = Linear(hidden_size, hidden_size)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tensor:
        batch_size = q.shape[0]
        q, k, v = self.q_linear(q), self.k_linear(k), self.v_linear(v)
        q = q.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attn_output = self.attention(q, k, v, mask=mask)
        concat = attn_output.transpose(1, 2).reshape(batch_size, -1, self.d_k * self.num_heads)
        return self.fc(concat)

class PositionalEncoding(Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, hidden_size: int, max_len: int = 5000):
        super().__init__()
        pe = np.zeros((max_len, hidden_size), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, hidden_size, 2, dtype=np.float32) * -(np.log(10000.0) / hidden_size))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = Tensor(pe, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:x.shape[1], :]

class LayerNorm(Module):
    """Applies Layer Normalization."""
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.gamma = Tensor(np.ones(self.normalized_shape), requires_grad=True)
        self.beta = Tensor(np.zeros(self.normalized_shape), requires_grad=True)
    def forward(self, x: Tensor) -> Tensor:
        mean = x.sum(axis=-1, keepdims=True) / Tensor(x.shape[-1])
        var_term = x - mean
        var = (var_term**2).sum(axis=-1, keepdims=True) / Tensor(x.shape[-1])
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x_norm + self.beta
    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]

class FeedForward(Module):
    """A simple position-wise feed-forward network."""
    def __init__(self, hidden_size: int, ff_size: int):
        super().__init__()
        self.linear1 = Linear(hidden_size, ff_size)
        self.linear2 = Linear(ff_size, hidden_size)
    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(relu(self.linear1(x)))

class ResidualAddAndNorm(Module):
    """A residual connection followed by layer normalization."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
    def forward(self, x: Tensor, sublayer_output: Tensor) -> Tensor:
        return self.norm(x + sublayer_output)

class EncoderLayer(Module):
    """A single layer of the Transformer Encoder stack."""
    def __init__(self, hidden_size: int, num_heads: int, ff_size: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_size, num_heads)
        self.add_norm1 = ResidualAddAndNorm(hidden_size)
        self.ff = FeedForward(hidden_size, ff_size)
        self.add_norm2 = ResidualAddAndNorm(hidden_size)
    def forward(self, x: Tensor, mask=None) -> Tensor:
        attn_output = self.self_attn(q=x, k=x, v=x, mask=mask)
        x = self.add_norm1(x, attn_output)
        ff_output = self.ff(x)
        x = self.add_norm2(x, ff_output)
        return x

class DecoderLayer(Module):
    """A single layer of the Transformer Decoder stack."""
    def __init__(self, hidden_size: int, num_heads: int, ff_size: int):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(hidden_size, num_heads)
        self.add_norm1 = ResidualAddAndNorm(hidden_size)
        self.enc_dec_attn = MultiHeadAttention(hidden_size, num_heads)
        self.add_norm2 = ResidualAddAndNorm(hidden_size)
        self.ff = FeedForward(hidden_size, ff_size)
        self.add_norm3 = ResidualAddAndNorm(hidden_size)
    def forward(self, x: Tensor, encoder_output: Tensor, src_mask=None, tgt_mask=None) -> Tensor:
        attn_output = self.masked_self_attn(q=x, k=x, v=x, mask=tgt_mask)
        x = self.add_norm1(x, attn_output)
        enc_dec_attn_output = self.enc_dec_attn(q=x, k=encoder_output, v=encoder_output, mask=src_mask)
        x = self.add_norm2(x, enc_dec_attn_output)
        ff_output = self.ff(x)
        x = self.add_norm3(x, ff_output)
        return x

class Encoder(Module):
    """A stack of N EncoderLayers."""
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, ff_size: int):
        super().__init__()
        self.layers = Sequential(*[EncoderLayer(hidden_size, num_heads, ff_size) for _ in range(num_layers)])
    def forward(self, x: Tensor, mask=None) -> Tensor:
        for layer in self.layers._modules.values():
            x = layer(x, mask)
        return x

class Decoder(Module):
    """A stack of N DecoderLayers."""
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, ff_size: int):
        super().__init__()
        self.layers = [DecoderLayer(hidden_size, num_heads, ff_size) for _ in range(num_layers)]
        for i, layer in enumerate(self.layers):
            self._modules[str(i)] = layer
    def forward(self, x: Tensor, encoder_output: Tensor, src_mask=None, tgt_mask=None) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

class Transformer(Module):
    """The full Transformer model."""
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, ff_size: int, max_len: int):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, max_len)
        self.encoder = Encoder(num_layers, hidden_size, num_heads, ff_size)
        self.decoder = Decoder(num_layers, hidden_size, num_heads, ff_size)
        self.final_linear = Linear(hidden_size, vocab_size)
    def forward(self, src: Tensor, tgt: Tensor, src_mask=None, tgt_mask=None) -> Tensor:
        src_emb = self.pos_encoding(self.token_embedding(src))
        tgt_emb = self.pos_encoding(self.token_embedding(tgt))
        encoder_output = self.encoder(src_emb, mask=src_mask)
        decoder_output = self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)
        logits = self.final_linear(decoder_output)
        return logits