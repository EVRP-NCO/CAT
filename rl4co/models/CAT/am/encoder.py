from typing import Tuple, Union

import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
# from rl4co.models.common.constructive import AutoregressiveEncoder # Keep base class import if needed
from rl4co.models.common.constructive.base import ConstructiveEncoder
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.graph.attnnet import GraphAttentionNetwork # Correct import
# Remove incorrect/unused imports
# from rl4co.models.nn.attention import MultiHeadAttention, SparseMultiHeadAttention
# from rl4co.models.nn.graph import GraphAttentionEncoder
# from rl4co.models.nn.mlp import MLP
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class AttentionModelEncoder(ConstructiveEncoder):
    """Graph Attention Encoder as in Kool et al. (2019).
    First embed the input and then process it with a Graph Attention Network.
    Modified to support separate complete and constraint embeddings for EVRPTW.

    Args:
        embed_dim: Dimension of the embedding space
        init_embedding: Module to use for the initialization of the embeddings
        env_name: Name of the environment used to initialize embeddings
        num_heads: Number of heads in the attention layers
        num_layers: Number of layers in the attention network
        normalization: Normalization type in the attention layers
        feedforward_hidden: Hidden dimension in the feedforward layers
        net: Graph Attention Network to use
        sdpa_fn: Function to use for the scaled dot product attention
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        env_name: str = "tsp",
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        sdpa_fn = None,
        moe_kwargs: dict = None,
        collect_attention: bool = False,  # 娣诲姞娉ㄦ剰鍔涢噰闆嗘帶鍒跺弬鏁?    ):
        super(AttentionModelEncoder, self).__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.collect_attention = collect_attention  # 淇濆瓨娉ㄦ剰鍔涢噰闆嗚缃?
        self.init_embedding = (
            env_init_embedding(self.env_name, {"embed_dim": embed_dim})
            if init_embedding is None
            else init_embedding
        )
        
        self.net = (
            GraphAttentionNetwork(
                num_heads=num_heads,
                embed_dim=embed_dim,
                num_layers=num_layers,
                normalization=normalization, 
                feedforward_hidden=feedforward_hidden,
                attention_type="mha",#mha/ceca  # 鍒囨崲鍒癕HA杩涜瀵规瘮
                ceca_channels=4,
                sdpa_fn=sdpa_fn,
                moe_kwargs=moe_kwargs
            )
            if net is None
            else net
        )

    def forward(
        self, td: TensorDict, mask: Union[Tensor, None] = None, return_attention: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.
        For EVRPTW, handles separate complete and constraint embeddings.

        Args:
            td: Input TensorDict containing the environment state
            mask: Mask to apply to the attention
            return_attention: Whether to return attention weights for visualization

        Returns:
            If return_attention=False: (h, init_h) where h is latent representation, init_h is initial embedding
            If return_attention=True: (h, init_h, attention_weights_list)
        """
        # Get embeddings - for EVRPTW this returns complete and constraint embeddings separately
        if self.env_name == "evrptw":
            complete_emb, constraint_emb = self.init_embedding(td)
            # Use complete_emb as Q and constraint_emb as K,V in the attention network
            
            # 馃敡 鍙湁鍦ㄦ槑纭姹傝繑鍥炴敞鎰忓姏鏉冮噸涓旇缃簡collect_attention鏃舵墠閲囬泦
            if return_attention and self.collect_attention:
                h, attention_weights = self.net(complete_emb, constraint_emb, mask=mask, return_attention=True)
                init_h = complete_emb  # Use complete embedding as the initial embedding
                    
                return h, init_h, attention_weights
            else:
                h = self.net(complete_emb, constraint_emb, mask=mask)
                init_h = complete_emb  # Use complete embedding as the initial embedding
        else:
            # For other environments, keep the original behavior
            init_h = self.init_embedding(td)
            if return_attention and self.collect_attention:
                h, attention_weights = self.net(init_h, mask=mask, return_attention=True)
                return h, init_h, attention_weights
            else:
                h = self.net(init_h, mask=mask)

        return h, init_h

