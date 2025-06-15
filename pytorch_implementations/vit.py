# pytorch_implementations/vit.py

import math
import torch
from torch import nn

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function from the paper "Gaussian Error Linear Units (GELUs)".
    This is a smoother version of the ReLU activation function and has been shown to improve performance in
    transformer models.

    For more information, see: https://arxiv.org/abs/1606.08415
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class PatchEmbeddings(nn.Module):
    """
    Converts a batch of images into a sequence of patch embeddings.

    The core idea of ViT is to treat an image as a sequence of patches, similar to how
    NLP models treat text as a sequence of words. This class is responsible for this
    initial transformation.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]

        # The number of patches is calculated by dividing the image size by the patch size
        # for both dimensions.
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # A clever way to perform the patch splitting and embedding is to use a
        # convolutional layer. With a kernel size and stride equal to the patch size,
        # the Conv2d layer effectively processes each patch independently and projects
        # it into a vector of `hidden_size`.
        self.projection = nn.Conv2d(
            self.num_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, num_channels, image_size, image_size)
        x = self.projection(x)
        # The output of the Conv2d layer is (batch_size, hidden_size, num_patches_h, num_patches_w).
        # We flatten the spatial dimensions and transpose to get the desired sequence format.
        # Output x: (batch_size, num_patches, hidden_size)
        x = x.flatten(2).transpose(1, 2)
        return x

class Embeddings(nn.Module):
    """
    Combines the patch embeddings with a learnable [CLS] token and positional embeddings.
    This prepares the input sequence for the Transformer Encoder.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)

        # The [CLS] token is a special learnable vector that is prepended to the sequence
        # of patch embeddings. The final hidden state corresponding to this token is
        # used as the aggregate representation for classification.
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))

        # Positional embeddings are crucial because the self-attention mechanism is
        # permutation-invariant. These embeddings provide the model with information
        # about the relative or absolute position of the patches.
        # We add 1 to `num_patches` to account for the [CLS] token.
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"])
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create patch embeddings
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()

        # Expand the [CLS] token to match the batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Prepend the [CLS] token to the patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x

class AttentionHead(nn.Module):
    """
    A single attention head within the Multi-Head Attention mechanism.
    """
    def __init__(self, hidden_size: int, attention_head_size: int, dropout: float, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size

        # Linear layers to project the input into query, key, and value vectors.
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Project the input into query, key, and value
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Calculate attention scores: softmax(Q * K.T / sqrt(head_size)) * V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Performs multi-head self-attention.

    This module is composed of multiple `AttentionHead` modules running in parallel.
    The outputs of the heads are concatenated and projected back to the hidden size.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]

        # Create a list of attention heads
        self.heads = nn.ModuleList([
            AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            ) for _ in range(self.num_attention_heads)
        ])
        
        # A linear layer to project the concatenated outputs of the heads
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x: torch.Tensor, output_attentions: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Calculate the attention output for each head
        attention_outputs_and_probs = [head(x) for head in self.heads]
        attention_outputs = [output for output, _ in attention_outputs_and_probs]
        
        # Concatenate the attention outputs from each head
        attention_output = torch.cat(attention_outputs, dim=-1)
        
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        if not output_attentions:
            return (attention_output, None)
        else:
            # If requested, return the attention probabilities from each head
            attention_probs = torch.stack([probs for _, probs in attention_outputs_and_probs], dim=1)
            return (attention_output, attention_probs)

class MLP(nn.Module):
    """
    A standard two-layer Multi-Layer Perceptron with GELU activation.
    This is the "Feed-Forward Network" part of a Transformer block.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    A single Transformer Encoder Block.

    Each block consists of a Multi-Head Attention layer and an MLP,
    with Layer Normalization and residual connections.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x: torch.Tensor, output_attentions: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        # First sub-layer: Multi-Head Attention followed by a residual connection
        # Pre-LayerNorm style: normalization is applied before the sub-layer
        attention_output, attention_probs = self.attention(
            self.layernorm_1(x), output_attentions=output_attentions
        )
        # Residual connection
        x = x + attention_output

        # Second sub-layer: MLP followed by a residual connection
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output

        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)

class Encoder(nn.Module):
    """
    The Transformer Encoder, which is a stack of N identical `Block` layers.
    """
    def __init__(self, config: dict):
        super().__init__()
        # Create a stack of Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config["num_hidden_layers"])
        ])

    def forward(self, x: torch.Tensor, output_attentions: bool = False) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)

        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class ViTForClassfication(nn.Module):
    """
    The complete Vision Transformer model for image classification.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]

        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the Transformer encoder
        self.encoder = Encoder(config)
        # The classification head takes the final hidden state of the [CLS] token
        # and projects it to the number of classes.
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, output_attentions: bool = False) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        # 1. Get patch and positional embeddings
        embedding_output = self.embedding(x)
        
        # 2. Pass through the Transformer Encoder
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions)
        
        # 3. Get the final hidden state of the [CLS] token
        cls_token_output = encoder_output[:, 0]
        
        # 4. Pass through the classifier head
        logits = self.classifier(cls_token_output)

        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def _init_weights(self, module: nn.Module):
        """
        Initializes the weights of the model.
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)