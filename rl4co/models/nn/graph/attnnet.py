from rl4co.models.nn.mlp import MLP
from rl4co.models.nn.moe import MoE
from rl4co.models.nn.attention import MultiHeadAttention, ChannelEnhancedContextualAttention
from rl4co.models.nn.ops import Normalization, SkipConnection
from rl4co.utils.pylogger import get_pylogger
from typing import Callable, Optional

import torch
import torch.nn as nn

from torch import Tensor

log = get_pylogger(__name__)


class MultiHeadAttentionLayer(nn.Module):
    """Multi-Head Attention Layer with normalization and feed-forward layer

    Args:
        embed_dim: dimension of the embeddings
        num_heads: number of heads in the MHA
        feedforward_hidden: dimension of the hidden layer in the feed-forward layer
        normalization: type of normalization to use (batch, layer, none)
        bias: whether to use bias in MHA linear layers
        sdpa_fn: scaled dot product attention function (SDPA)
        moe_kwargs: Keyword arguments for MoE FFN
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        bias: bool = True,
        sdpa_fn: Optional[Callable] = None,
        moe_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.mha = SkipConnection(
            MultiHeadAttention(embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn)
        )
        self.norm1 = Normalization(embed_dim, normalization)

        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        if moe_kwargs is not None:
            ffn_module = MoE(embed_dim, embed_dim, num_neurons=num_neurons, **moe_kwargs)
        else:
            ffn_module = MLP(input_dim=embed_dim, output_dim=embed_dim, num_neurons=num_neurons, hidden_act="ReLU")
        self.ffn = SkipConnection(ffn_module)
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """ Forward pass for the MHA layer.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, embed_dim).
            attn_mask (Optional[Tensor]): Attention mask passed to MHA. Shape (batch, seq_len, seq_len) or broadcastable.

        Returns:
            Tensor: Output tensor of shape (batch, seq_len, embed_dim).
        """
        # Apply MHA with skip connection, then normalization
        x = self.norm1(self.mha(x, attn_mask=attn_mask))
        # Apply FFN with skip connection, then normalization
        x = self.norm2(self.ffn(x))
        return x


class ChannelEnhancedContextualAttentionLayer(nn.Module):
    """
    Layer combining Channel Enhanced Contextual Self-Attention with normalization and FFN.
    Mirrors the structure of MultiHeadAttentionLayer.

    Args:
        embed_dim: dimension of the embeddings (dk)
        num_heads: number of attention heads
        channels: number of channels (c) for the internal Conv2d/ChannelAttention
        feedforward_hidden: dimension of the hidden layer in the feed-forward layer
        normalization: type of normalization to use (batch, layer, none)
        moe_kwargs: Keyword arguments for MoE in FFN
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int=8,
        channels: int=4,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        moe_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.ceca = SkipConnection(
            ChannelEnhancedContextualAttention(embed_dim, num_heads, channels)
        )
        self.norm1 = Normalization(embed_dim, normalization)

        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        if moe_kwargs is not None:
            ffn_module = MoE(embed_dim, embed_dim, num_neurons=num_neurons, **moe_kwargs)
        else:
            ffn_module = MLP(input_dim=embed_dim, output_dim=embed_dim, num_neurons=num_neurons, hidden_act="ReLU")
        self.ffn = SkipConnection(ffn_module)
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, x: Tensor, kv: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        """ Forward pass for the CECA layer.

        Args:
            x (Tensor): Input tensor (B, N, D) or (B, 4, N, D)
            kv (Optional[Tensor]): Key/Value input tensor. If None, uses x.
            attn_mask (Optional[Tensor])
        """
        # If input still has 4 constraint channels, fuse them first so that
        # residual shapes match inside SkipConnection
        # if x.dim() == 4:
        #     # Use the same fusion logic as ChannelEnhancedContextualAttention
        #     ceca_module = self.ceca.module  # Access wrapped module
        #     x = ceca_module._process_multi_channel_input(x)  # (B,N,D)

        # Apply CECA with skip connection, then normalization
        x = self.norm1(self.ceca(x, kv=kv, attn_mask=attn_mask))
        # Apply FFN with skip connection, then normalization
        x = self.norm2(self.ffn(x))
        return x


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network to encode embeddings with a series of attention layers.
    Can use either Multi-Head Attention (MHA) or Channel Enhanced Contextual Attention (CECA).

    Args:
        num_heads: number of heads (only used if attention_type='mha' or 'ceca')
        embed_dim: dimension of the embeddings
        num_layers: number of attention layers
        normalization: type of normalization to use (batch, layer, none)
        feedforward_hidden: dimension of the hidden layer in the feed-forward layer
        sdpa_fn: scaled dot product attention function (SDPA) (only used if attention_type='mha')
        moe_kwargs: Keyword arguments for MoE in FFN
        attention_type: Type of attention layer ('mha' or 'ceca'). Default 'ceca'.
        ceca_channels: Optional[int] = None, # Number of channels (c) for CECA. Required if attention_type='ceca'.
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        num_layers: int,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        attention_type: str = "ceca",
        ceca_channels: int = 4,
        sdpa_fn = None,
        moe_kwargs: dict = None,
    ):
        super(GraphAttentionNetwork, self).__init__()
        self.attention_type = attention_type
        
        if attention_type == "mha":
            # Use Multi-Head Attention layers
            self.layers = nn.ModuleList([
                MultiHeadAttentionLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    sdpa_fn=sdpa_fn,
                    moe_kwargs=moe_kwargs,
                )
                for _ in range(num_layers)
            ])
        elif attention_type == "ceca":
            # Use Channel Enhanced Contextual Attention layers
            self.layers = nn.ModuleList([
                ChannelEnhancedContextualAttentionLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    channels=ceca_channels,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    moe_kwargs=moe_kwargs,
                )
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}. Choose 'mha' or 'ceca'.")

    def forward(
        self, 
        x: torch.Tensor, 
        kv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False  # 是否返回注意力权重用于可视化
    ) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x: Query tensor (coordinates embedding) of shape (batch_size, seq_len, embed_dim)
            kv: Optional Key/Value tensor (features embedding) of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            return_attention: Whether to return attention weights for visualization
            
        Returns:
            If return_attention=False: Output tensor of shape (batch_size, seq_len, embed_dim)
            If return_attention=True: (Output tensor, List of attention weights from each layer)
        """
        attention_weights_list = []

        if self.attention_type == "mha":
            # For MHA layers, use the built-in layer structure with normalization
            for layer in self.layers:
                if return_attention and hasattr(layer.mha.module, 'forward'):
                    # Extract attention weights from MHA layer
                    mha_module = layer.mha.module
                    # Use custom forward to get attention weights
                    attn_out, attn_weights = mha_module(x, attn_mask=mask, return_attention=return_attention)
                    x = layer.norm1(x + attn_out)  # Apply skip connection and normalization
                    x = layer.norm2(layer.ffn(x))  # Apply FFN
                    attention_weights_list.append(attn_weights)
                else:
                    x = layer(x, attn_mask=mask)

        elif self.attention_type == "ceca":
            # For CECA layers, we need to call the underlying CECA module and optionally
            # collect attention weights. The CECA layer wraps the module in a SkipConnection
            # so we access `layer.ceca.module` to call the raw module if we want attention.
            for layer in self.layers:
                if return_attention:
                    # Call CECA module with return_attention and receive (out, attn)
                    ceca_module = layer.ceca.module
                    # CECA's forward signature supports return_attention keyword (implemented in attention.py)
                    out, attn = ceca_module(x if x.dim() == 3 else x, kv=kv, attn_mask=mask, return_attention=True)
                    # Emulate SkipConnection behaviour: skip connection adds input to module output
                    # If input had extra channels (dim==4), assumption is module returned (B,N,D)
                    x = layer.norm1(x + out)
                    x = layer.norm2(layer.ffn(x))
                    attention_weights_list.append(attn)
                else:
                    x = layer(x, kv=kv, attn_mask=mask)

        if return_attention:
            return x, attention_weights_list
        else:
            return x
