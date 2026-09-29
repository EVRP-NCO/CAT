from dataclasses import dataclass, fields
from typing import Tuple, Union, Optional
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.autoregressive.decoder import AutoregressiveDecoder
from rl4co.models.nn.attention import PointerAttention, PolyNetAttention, PolyNetAttentionMoE, PointerAttnMoE
from rl4co.models.nn.env_embeddings import env_context_embedding, env_dynamic_embedding
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.utils.ops import batchify, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    graph_context: Union[Tensor, float]
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor
    # Add fields for PIP decoder
    glimpse_key_pip: Union[Tensor, None] = None
    glimpse_val_pip: Union[Tensor, None] = None
    logit_key_pip: Union[Tensor, None] = None

    @property
    def fields(self):
        return tuple(getattr(self, x.name) for x in fields(self))

    def batchify(self, num_starts):
        new_embs = []
        for emb in self.fields:
            if isinstance(emb, Tensor) or isinstance(emb, TensorDict):
                new_embs.append(batchify(emb, num_starts))
            else:
                new_embs.append(emb)
        return PrecomputedCache(*new_embs)


class AttentionModelDecoder(AutoregressiveDecoder):
    """
    Auto-regressive decoder based on Kool et al. (2019): https://arxiv.org/abs/1803.08475.
    Given the environment state and the embeddings, compute the logits and sample actions autoregressively until
    all the environments in the batch have reached a terminal state.
    In this case we additionally have a `pre_decoder_hook` method that allows to precompute the embeddings before
    the decoder is called, which saves a lot of computation.


    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        env_name: Name of the environment used to initialize embeddings
        context_embedding: Context embedding module
        dynamic_embedding: Dynamic embedding module
        mask_inner: Whether to mask the inner loop
        out_bias_pointer_attn: Whether to use a bias in the pointer attention
        linear_bias: Whether to use a bias in the linear layer
        use_graph_context: Whether to use the graph context
        check_nan: Whether to check for nan values during decoding
        sdpa_fn: scaled_dot_product_attention function
        pointer: Module implementing the pointer logic (defaults to PointerAttention)
        pointer_pip: Module implementing the pointer logic for PIP decoder (defaults to None)
        moe_kwargs: Keyword arguments for MoE
        polynet_k: PolyNet k parameter
        poly_layer_dim: PolyNet layer dimension
        pip_decoder: Whether to use PIP decoder for node feasibility prediction
        decision_boundary: Decision boundary for node feasibility prediction
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "tsp",
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = True,
        check_nan: bool = True,
        sdpa_fn: callable = None,
        pointer: nn.Module = None,
        pointer_pip: nn.Module = None,
        moe_kwargs: dict = None,
        polynet_k: Optional[int] = None,#None
        poly_layer_dim: Optional[int] = 128,
        pip_decoder: bool = False,
        decision_boundary: float = 0.5,
    ):
        super().__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pip_decoder = pip_decoder
        self.decision_boundary = decision_boundary
        self.polynet_k = polynet_k
        self.poly_layer_dim = poly_layer_dim

        assert embed_dim % num_heads == 0

        self.context_embedding = (
            env_context_embedding(self.env_name, {"embed_dim": embed_dim})
            if context_embedding is None
            else context_embedding
        )
        self.dynamic_embedding = (
            env_dynamic_embedding(self.env_name, {"embed_dim": embed_dim})
            if dynamic_embedding is None
            else dynamic_embedding
        )
        self.is_dynamic_embedding = (
            False if isinstance(self.dynamic_embedding, StaticEmbedding) else True
        )

        if pointer is None:
            # Determine the class and arguments for the pointer module
            pointer_module_class = None
            module_args = {
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "mask_inner": mask_inner,
                "out_bias": out_bias_pointer_attn,
                "check_nan": check_nan,
                "sdpa_fn": sdpa_fn,
            }

            if self.polynet_k is not None:
                if self.poly_layer_dim is None:
                    raise ValueError(
                        "poly_layer_dim must be specified when polynet_k is provided."
                    )
                module_args.update({
                    "k": self.polynet_k,
                    "poly_layer_dim": self.poly_layer_dim,
                })
                if moe_kwargs is not None:
                    pointer_module_class = PolyNetAttentionMoE
                    module_args["moe_kwargs"] = moe_kwargs
                else:
                    pointer_module_class = PolyNetAttention
            else:  # Original logic: PointerAttention or PointerAttnMoE
                if moe_kwargs is not None:
                    pointer_module_class = PointerAttnMoE
                    module_args["moe_kwargs"] = moe_kwargs
                else:
                    pointer_module_class = PointerAttention
            
            self.pointer = pointer_module_class(**module_args)

            if self.pip_decoder:
                if pointer_pip is None:
                    self.pointer_pip = pointer_module_class(**module_args)
                else:
                    self.pointer_pip = pointer_pip
        
        else:
            self.pointer = pointer
            if self.pip_decoder:
                if pointer_pip is None:
                    raise ValueError(
                        "If 'pointer' module is provided and pip_decoder is True, "
                        "'pointer_pip' module must also be provided."
                    )
                self.pointer_pip = pointer_pip

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embed_dim
        self.project_node_embeddings = nn.Linear(
            embed_dim, 3 * embed_dim, bias=linear_bias
        )
        
        # 为PIP-D添加投影层
        if self.pip_decoder:
            self.project_node_embeddings_pip = nn.Linear(
                embed_dim, 3 * embed_dim, bias=linear_bias
            )
            
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.use_graph_context = use_graph_context

    def _compute_q(self, cached: PrecomputedCache, td: TensorDict, for_pip=False):
        node_embeds_cache = cached.node_embeddings
        graph_context_cache = cached.graph_context

        if td.dim() == 2 and isinstance(graph_context_cache, Tensor):
            graph_context_cache = graph_context_cache.unsqueeze(1)

        step_context = self.context_embedding(node_embeds_cache, td)
        glimpse_q = step_context + graph_context_cache
        # add seq_len dim if not present
        glimpse_q = glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q

        return glimpse_q

    def _compute_kvl(self, cached: PrecomputedCache, td: TensorDict, for_pip=False):
        if for_pip and self.pip_decoder:
            glimpse_k_stat, glimpse_v_stat, logit_k_stat = (
                cached.glimpse_key_pip,
                cached.glimpse_val_pip,
                cached.logit_key_pip,
            )
        else:
            glimpse_k_stat, glimpse_v_stat, logit_k_stat = (
                cached.glimpse_key,
                cached.glimpse_val,
                cached.logit_key,
            )
            
        # Compute dynamic embeddings and add to static embeddings
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(td)
        glimpse_k = glimpse_k_stat + glimpse_k_dyn
        glimpse_v = glimpse_v_stat + glimpse_v_dyn
        logit_k = logit_k_stat + logit_k_dyn

        return glimpse_k, glimpse_v, logit_k

    def forward(
        self,
        td: TensorDict,
        cached: PrecomputedCache,
        num_starts: int = 0,
        step: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the logits of the next actions given the current state

        Args:
            td: TensorDict with the current environment state
            cached: Precomputed embeddings
            num_starts: Number of starts for the multi-start decoding
        """
        has_dyn_emb_multi_start = self.is_dynamic_embedding and num_starts > 1

        # Handle efficient multi-start decoding
        if has_dyn_emb_multi_start:
            # if num_starts > 0 and we have some dynamic embeddings, we need to reshape them to [B*S, ...]
            # since keys and values are not shared across starts (i.e. the episodes modify these embeddings at each step)
            cached = cached.batchify(num_starts=num_starts)

        elif num_starts > 1:
            td = unbatchify(td, num_starts)

        glimpse_q = self._compute_q(cached, td)
        glimpse_k, glimpse_v, logit_k = self._compute_kvl(cached, td)

        # 获取动作掩码
        if "pi_mask" in td :
            # 如果有PI掩码，我们保存它以供使用
            original_mask = td["action_mask"].clone()
            mask = td["action_mask"].clone()  # 用于实际掩码操作的副本
            pi_mask = td["pi_mask"].clone()   # 保存原始PI掩码用于训练
        else:
            # 否则使用原始动作掩码
            original_mask = td["action_mask"].clone()
            mask = original_mask.clone()  # 保存原始动作掩码的副本
            pi_mask = None
         
        # 计算节点可行性预测 (PIP-D)
        pip_aux_loss = torch.tensor(0.0, device=logit_k.device)  # 初始化辅助损失
        if self.pip_decoder:
            # 使用PIP指针预测节点可行性
            glimpse_q_pip = self._compute_q(cached, td, for_pip=True)
            glimpse_k_pip, glimpse_v_pip, logit_k_pip = self._compute_kvl(cached, td, for_pip=True)
            
            # 计算节点可行性预测的logits，使用original_mask以避免对pi_mask的依赖
            feasibility_logits = self.pointer_pip(glimpse_q_pip, glimpse_k_pip, glimpse_v_pip, logit_k_pip)
            
            # 将可行性预测添加到输出中
            feasibility_probs = torch.sigmoid(feasibility_logits)
            td.set("feasibility_probs", feasibility_probs)
            
            # 计算辅助损失 (仅在训练且pi_mask可用时)
            if pi_mask is not None:
                # 使用原始掩码来确定哪些位置需要计算损失 (忽略被环境禁止的动作)
                loss_mask = original_mask.bool()
                # 只在有效动作上计算BCE损失
                # 将pi_mask转换为float类型以匹配logits
                bce_loss = F.binary_cross_entropy_with_logits(
                    feasibility_logits[loss_mask], 
                    pi_mask[loss_mask].float(),
                    reduction='mean' 
                )
                # 如果没有有效动作计算损失，则损失为0
                if torch.isnan(bce_loss):
                     pip_aux_loss = torch.tensor(0.0, device=logit_k.device)
                else:
                    pip_aux_loss = bce_loss

            # 生成不可行掩码
            invalid_mask = feasibility_probs < self.decision_boundary
            #log.info(f"invalid_mask: {invalid_mask}")
            # 将不可行掩码应用到原始掩码上 (如果pi_mask存在则用pi_mask)

            mask = original_mask & (~invalid_mask)
            
            # 检查是否所有动作都被标记为不可行
            all_invalid = ~mask.any(dim=-1)
            if all_invalid.any():
                # 对于没有可行动作的情况，回退到原始掩码
                mask[all_invalid] = original_mask[all_invalid]
            
            # 保存生成的掩码
            td.set("predicted_mask", mask)
            
        # 使用最终掩码计算logits
        logits = self.pointer(glimpse_q, glimpse_k, glimpse_v, logit_k, mask)
        # Now we need to reshape the logits and mask to [B*S,N,...] is num_starts > 1 without dynamic embeddings
        # note that rearranging order is important here
        if num_starts > 1 and not has_dyn_emb_multi_start:
            logits = rearrange(logits, "b s l -> (s b) l", s=num_starts)
            mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)
            # 如果有辅助损失，也需要reshape
            if self.pip_decoder and pi_mask is not None:
                 # 辅助损失通常是一个标量，或者可以按batch聚合，这里保持不变或按需调整
                 pass # pip_aux_loss 通常是标量或已聚合，无需 reshape

        # 在返回字典中包含辅助损失
        decoder_output = {"logits": logits, "mask": mask}
        if self.pip_decoder:
             decoder_output["pip_aux_loss"] = pip_aux_loss
             
        return decoder_output

    def pre_decoder_hook(
        self, td, env, embeddings, num_starts: int = 0
    ) -> Tuple[TensorDict, RL4COEnvBase, PrecomputedCache]:
        """Precompute the embeddings cache before the decoder is called"""
        return td, env, self._precompute_cache(embeddings, num_starts=num_starts)

    def _precompute_cache(
        self, embeddings: torch.Tensor, num_starts: int = 0
    ) -> PrecomputedCache:
        """Compute the cached embeddings for the pointer attention.

        Args:
            embeddings: Precomputed embeddings for the nodes
            num_starts: Number of starts for the multi-start decoding
        """
        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        # 为PIP-D预测计算节点嵌入投影
        glimpse_key_pip, glimpse_val_pip, logit_key_pip = None, None, None
        if self.pip_decoder:
            (
                glimpse_key_pip,
                glimpse_val_pip,
                logit_key_pip,
            ) = self.project_node_embeddings_pip(embeddings).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            graph_context = self.project_fixed_context(embeddings.mean(1))
        else:
            graph_context = 0

        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
            glimpse_key_pip=glimpse_key_pip,
            glimpse_val_pip=glimpse_val_pip,
            logit_key_pip=logit_key_pip,
        )
