import itertools
import math
import warnings

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from rl4co.models.nn.moe import MoE
from rl4co.utils import get_pylogger

log = get_pylogger(__name__)


# Define SwiGLU activation module
class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    As proposed in "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, bias: bool = False):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else int(dim * 2 / 3 * 2) # Often hidden_dim is 2/3 * 4 * dim = 8/3 * dim
        # Ensure hidden_dim is divisible by a suitable number, e.g., 8
        hidden_dim = (hidden_dim + 7) // 8 * 8

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.act = nn.SiLU() # Sigmoid Linear Unit (Swish)

    def forward(self, x):
        # Apply the SwiGLU transformation: F.silu(w1(x)) * w2(x)
        return self.w3(self.act(self.w1(x)) * self.w2(x))


def my_product_attention(q, k, v, W, dis, time, mask):
    compatibility = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
    score = W(
        torch.cat(
            (compatibility.unsqueeze(-1), time, dis),
            dim=-1,
        )
    ).squeeze(-1)
    scores = torch.nan_to_num(score, nan=1e-4, posinf=1e4, neginf=-1e4)
    scores.masked_fill_(~mask, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)


def scaled_dot_product_attention_simple(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
):
    """Simple Scaled Dot-Product Attention in PyTorch without Flash Attention"""
    # Check for causal and attn_mask conflict
    if is_causal and attn_mask is not None:
        raise ValueError("Cannot set both is_causal and attn_mask")

    # Calculate scaled dot product
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
    scores = torch.nan_to_num(scores, nan=1e-4, posinf=1e4, neginf=-1e4)
    # Apply the provided attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(~attn_mask, float("-inf"))
        else:
            scores += attn_mask

    # Apply causal mask
    if is_causal:
        s, l_ = scores.size(-2), scores.size(-1)
        mask = torch.triu(torch.ones((s, l_), device=scores.device), diagonal=1)
        scores.masked_fill_(mask.bool(), float("-inf"))

    # Softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Compute the weighted sum of values
    return torch.matmul(attn_weights, v)


try:
    from torch.nn.functional import scaled_dot_product_attention
except ImportError:
    log.warning(
        "torch.nn.functional.scaled_dot_product_attention not found. Make sure you are using PyTorch >= 2.0.0."
        "Alternatively, install Flash Attention https://github.com/HazyResearch/flash-attention ."
        "Using custom implementation of scaled_dot_product_attention without Flash Attention. "
    )
    scaled_dot_product_attention = scaled_dot_product_attention_simple


class MultiHeadAttention(nn.Module):
    """PyTorch native implementation of Flash Multi-Head Attention with automatic mixed precision support.
    Uses PyTorch's native `scaled_dot_product_attention` implementation, available from 2.0

    Note:
        If `scaled_dot_product_attention` is not available, use custom implementation of `scaled_dot_product_attention` without Flash Attention.

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        bias: whether to use bias
        attention_dropout: dropout rate for attention weights
        causal: whether to apply causal mask to attention scores
        device: torch device
        dtype: torch dtype
        sdpa_fn: scaled dot product attention function (SDPA) implementation
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attention_dropout: float = 0.0,
        causal: bool = False,
        device: str = None,
        dtype: torch.dtype = None,
        sdpa_fn: Optional[Callable] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.attention_dropout = attention_dropout
        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        # self.swiglu_ffn = SwiGLU(embed_dim, bias=bias) # Added SwiGLU

    def forward(self, x, attn_mask=None, return_attention=False):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        attn_mask: bool tensor of shape (batch, seqlen)
        return_attention: whether to return attention weights for visualization
        """
        # Project query, key, value
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads
        ).unbind(dim=0)

        if attn_mask is not None:
            attn_mask = (
                attn_mask.unsqueeze(1)
                if attn_mask.ndim == 3
                else attn_mask.unsqueeze(1).unsqueeze(2)
            )

        if return_attention:
            # Custom attention computation to return weights
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights1 = scores
            
            if self.attention_dropout > 0:
                attention_weights = F.dropout(attention_weights, p=self.attention_dropout, training=self.training)
            
            out = torch.matmul(attention_weights, v)
            projected_out = self.out_proj(rearrange(out, "b h s d -> b s (h d)"))
            
            return projected_out, attention_weights1
        else:
            # Use optimized SDPA
            out = self.sdpa_fn(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attention_dropout,
            )
            projected_out = self.out_proj(rearrange(out, "b h s d -> b s (h d)"))
            
        # final_out = self.swiglu_ffn(projected_out) # Applied SwiGLU
        final_out = projected_out # Removed SwiGLU
        
        if return_attention:
            return final_out, attention_weights1
        else:
            return final_out


def sdpa_fn_wrapper(q, k, v, attn_mask=None, dmat=None, dropout_p=0.0, is_causal=False):
    if dmat is not None:
        log.warning(
            "Edge weights passed to simple attention-fn, which is not supported. Weights will be ignored..."
        )
    return scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
    )


class MultiHeadCrossAttention(nn.Module):
    """PyTorch native implementation of Flash Multi-Head Cross Attention with automatic mixed precision support.
    Uses PyTorch's native `scaled_dot_product_attention` implementation, available from 2.0

    Note:
        If `scaled_dot_product_attention` is not available, use custom implementation of `scaled_dot_product_attention` without Flash Attention.

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        bias: whether to use bias
        attention_dropout: dropout rate for attention weights
        device: torch device
        dtype: torch dtype
        sdpa_fn: scaled dot product attention function (SDPA)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        attention_dropout: float = 0.0,
        device: str = None,
        dtype: torch.dtype = None,
        sdpa_fn: Optional[Union[Callable, nn.Module]] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_dropout = attention_dropout

        # Default to `scaled_dot_product_attention` if `sdpa_fn` is not provided
        if sdpa_fn is None:
            sdpa_fn = sdpa_fn_wrapper
        self.sdpa_fn = sdpa_fn

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        # self.swiglu_ffn = SwiGLU(embed_dim, bias=bias) # Added SwiGLU

    def forward(self, q_input, kv_input, cross_attn_mask=None, dmat=None):
        # Project query, key, value
        q = rearrange(
            self.Wq(q_input), "b m (h d) -> b h m d", h=self.num_heads
        )  # [b, h, m, d]
        k, v = rearrange(
            self.Wkv(kv_input), "b n (two h d) -> two b h n d", two=2, h=self.num_heads
        ).unbind(
            dim=0
        )  # [b, h, n, d]

        if cross_attn_mask is not None:
            # add head dim
            cross_attn_mask = cross_attn_mask.unsqueeze(1)

        # Scaled dot product attention
        out = self.sdpa_fn(
            q,
            k,
            v,
            attn_mask=cross_attn_mask,
            dmat=dmat,
            dropout_p=self.attention_dropout,
        )
        projected_out = self.out_proj(rearrange(out, "b h s d -> b s (h d)"))
        # final_out = self.swiglu_ffn(projected_out) # Applied SwiGLU
        final_out = projected_out # Removed SwiGLU
        return final_out


class PointerAttention(nn.Module):
    """Calculate logits given query, key and value and logit key.
    This follows the pointer mechanism of Vinyals et al. (2015) (https://arxiv.org/abs/1506.03134).

    Note:
        With Flash Attention, masking is not supported

    Performs the following:
        1. Apply cross attention to get the heads
        2. Project heads to get glimpse
        3. Compute attention score between glimpse and logit key

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        mask_inner: whether to mask inner attention
        linear_bias: whether to use bias in linear projection
        check_nan: whether to check for NaNs in logits
        sdpa_fn: scaled dot product attention function (SDPA) implementation
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mask_inner: bool = True,
        out_bias: bool = False,
        check_nan: bool = False,
        sdpa_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super(PointerAttention, self).__init__()
        self.num_heads = num_heads
        self.mask_inner = mask_inner

        # Projection - query, key, value already include projections
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=out_bias)
        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention
        self.check_nan = check_nan
        # Add SwiGLU FFN layer
        # self.swiglu_ffn = SwiGLU(embed_dim, bias=out_bias) # Uncommented and verified

    def forward(self, query, key, value, logit_key, attn_mask=None):
        """Compute attention logits given query, key, value, logit key and attention mask.

        Args:
            query: query tensor of shape [B, ..., L, E]
            key: key tensor of shape [B, ..., S, E]
            value: value tensor of shape [B, ..., S, E]
            logit_key: logit key tensor of shape [B, ..., S, E]
            attn_mask: attention mask tensor of shape [B, ..., S]. Note that `True` means that the value _should_ take part in attention
                as described in the [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
        """
        # Compute inner multi-head attention with no projections.
        heads = self._inner_mha(query, key, value, attn_mask)
        glimpse = self._project_out(heads, attn_mask)

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # bmm is slightly faster than einsum and matmul
        logits = (torch.bmm(glimpse, logit_key.squeeze(-2).transpose(-2, -1))).squeeze(
            -2
        ) / math.sqrt(glimpse.size(-1))

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"

        return logits

    def _inner_mha(self, query, key, value, attn_mask):
        q = self._make_heads(query)
        k = self._make_heads(key)
        v = self._make_heads(value)
        if self.mask_inner:
            # make mask the same number of dimensions as q
            if attn_mask is not None:
                attn_mask = (
                    attn_mask.unsqueeze(1)
                    if attn_mask.ndim == 3
                    else attn_mask.unsqueeze(1).unsqueeze(2)
                )
        else:
            attn_mask = None
        heads = self.sdpa_fn(q, k, v, attn_mask=attn_mask)
        assert not torch.isnan(heads).any()
        return rearrange(heads, "... h n g -> ... n (h g)", h=self.num_heads)

    def _make_heads(self, v):
        return rearrange(v, "... g (h s) -> ... h g s", h=self.num_heads)

    def _project_out(self, out, *kwargs):
        # Apply the original linear projection
        projected_out = self.project_out(out)
        # Apply the SwiGLU FFN layer
        # ffn_out = self.swiglu_ffn(projected_out) # Applied SwiGLU
        ffn_out = projected_out # Removed SwiGLU
        return ffn_out


class MyAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mask_inner: bool = True,
        out_bias: bool = False,
        check_nan: bool = False,
        sdpa_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super(MyAttention, self).__init__()
        self.num_heads = num_heads
        self.mask_inner = mask_inner

        # Projection - query, key, value already include projections
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=out_bias)
        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else my_product_attention
        self.FF = nn.Sequential(
            nn.Linear(4, 8, False), nn.GELU(), nn.Linear(8, 1, False)
        )
        self.check_nan = check_nan
        # self.swiglu_ffn = SwiGLU(embed_dim, bias=out_bias) # Added SwiGLU

    def forward(self, query, key, value, logit_key, attn_mask, dis, time):

        # Compute inner multi-head attention with no projections.
        heads = self._inner_mha(query, key, value, attn_mask, dis, time)
        glimpse = self._project_out(heads, attn_mask)

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # bmm is slightly faster than einsum and matmul
        logits = (torch.bmm(glimpse, logit_key.squeeze(-2).transpose(-2, -1))).squeeze(
            -2
        ) / math.sqrt(glimpse.size(-1))

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"

        return logits

    def _inner_mha(self, query, key, value, attn_mask, dis, time):
        q = self._make_heads(query)
        k = self._make_heads(key)
        v = self._make_heads(value)
        if self.mask_inner:
            # make mask the same number of dimensions as q
            if attn_mask is not None:
                attn_mask = (
                    attn_mask.unsqueeze(1)
                    if attn_mask.ndim == 3
                    else attn_mask.unsqueeze(1).unsqueeze(2)
                )
        else:
            attn_mask = None
        heads = self.sdpa_fn(q, k, v, self.FF, dis, time, attn_mask)
        assert not torch.isnan(heads).any()
        return rearrange(heads, "... h n g -> ... n (h g)", h=self.num_heads)

    def _make_heads(self, v):
        return rearrange(v, "... g (h s) -> ... h g s", h=self.num_heads)

    def _project_out(self, out, *kwargs):
        projected = self.project_out(out)
        # return self.swiglu_ffn(projected) # Applied SwiGLU
        return projected # Removed SwiGLU


class PointerAttnMoE(PointerAttention):
    """Calculate logits given query, key and value and logit key.
    This follows the pointer mechanism of Vinyals et al. (2015) <https://arxiv.org/abs/1506.03134>,
        and the MoE gating mechanism of Zhou et al. (2024) <https://arxiv.org/abs/2405.01029>.

    Note:
        With Flash Attention, masking is not supported

    Performs the following:
        1. Apply cross attention to get the heads
        2. Project heads to get glimpse
        3. Compute attention score between glimpse and logit key

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        mask_inner: whether to mask inner attention
        linear_bias: whether to use bias in linear projection
        check_nan: whether to check for NaNs in logits
        sdpa_fn: scaled dot product attention function (SDPA) implementation
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mask_inner: bool = True,
        out_bias: bool = False,
        check_nan: bool = True,
        sdpa_fn: Optional[Callable] = None,
        moe_kwargs: Optional[dict] = None,
    ):
        super(PointerAttnMoE, self).__init__(
            embed_dim, num_heads, mask_inner, out_bias, check_nan, sdpa_fn
        )
        self.moe_kwargs = moe_kwargs

        self.project_out = None
        self.project_out_moe = MoE(
            embed_dim, embed_dim, num_neurons=[], out_bias=out_bias, **(moe_kwargs if moe_kwargs is not None else {})
        )
        self.probs = None  # Initialize self.probs to None
        if self.moe_kwargs and self.moe_kwargs.get("light_version", False):
            self.dense_or_moe = nn.Linear(embed_dim, 2, bias=False)
            self.project_out = nn.Linear(embed_dim, embed_dim, bias=out_bias)
            # Register a default probability buffer
            self.register_buffer("probs_initial", torch.tensor([0.5, 0.5], dtype=torch.float32))

    def _project_out(self, out, attn_mask):
        """Implementation of Hierarchical Gating based on Zhou et al. (2024) <https://arxiv.org/abs/2405.01029>.
           MoE itself acts as an FFN, so we don't add SwiGLU here.
        """
        if self.moe_kwargs and self.moe_kwargs.get("light_version", False):
            num_nodes, num_available_nodes = attn_mask.size(-1), attn_mask.sum(-1)
            
            # Determine the probabilities for choosing MoE vs dense path
            # Condition to calculate/re-calculate self.probs based on current `out`
            condition_to_recalculate_probs = (num_available_nodes >= num_nodes - 1).any()

            if condition_to_recalculate_probs:
                self.probs = F.softmax(
                    self.dense_or_moe(
                        out.view(-1, out.size(-1)).mean(dim=0, keepdim=True)
                    ),
                    dim=-1,
                )  # Shape [1, 2]
            elif self.probs is None: # Not meeting recalculate condition AND self.probs was never set
                # Fallback to the initial default probabilities (e.g., [0.5, 0.5])
                self.probs = self.probs_initial.unsqueeze(0) # Shape [1, 2]
            # Else (condition_to_recalculate_probs is False but self.probs is not None):
            #   self.probs was set in a previous call and will be reused.

            # Now, self.probs is guaranteed to be initialized and should have shape [1, 2]
            
            # Perform selection based on self.probs.
            # Assuming self.probs is [1,2], sample once for the whole batch or operation.
            selected_scalar = self.probs.multinomial(num_samples=1).item() # Get 0 or 1

            if selected_scalar == 1:  # Arbitrarily, 1 for MoE path
                projected_intermediate = self.project_out_moe(out)
            else:  # 0 for dense path
                projected_intermediate = self.project_out(out)
            
            # Retrieve the probability of the chosen path
            prob_of_selected_path = self.probs[0, selected_scalar]
            
            glimpse = projected_intermediate * prob_of_selected_path
        else: # Not light_version or moe_kwargs is None
            glimpse = self.project_out_moe(out)
        return glimpse


# Deprecated
class LogitAttention(PointerAttention):
    def __init__(self, *args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            "LogitAttention is deprecated and will be removed in a future release. "
            "Please use PointerAttention instead."
            "Note that several components of the previous LogitAttention have moved to `rl4co.models.nn.dec_strategies`.",
            category=DeprecationWarning,
        )
        super(LogitAttention, self).__init__(*args, **kwargs)


# MultiHeadCompat
class MultiHeadCompat(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim=None, val_dim=None, key_dim=None):
        super(MultiHeadCompat, self).__init__()

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.init_parameters()

    # used for init nn.Parameter
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """

        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)  #################  reshape
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility_s2n = torch.matmul(Q, K.transpose(2, 3))

        return compatibility_s2n


class PolyNetAttention(PointerAttention):
    """Calculate logits given query, key and value and logit key.
    This implements a modified version the pointer mechanism of Vinyals et al. (2015) (https://arxiv.org/abs/1506.03134)
    as described in Hottung et al. (2024) (https://arxiv.org/abs/2402.14048) PolyNetAttention conditions the attention logits on
    a set of k different binary vectors allowing to learn k different solution strategies.

    Note:
        With Flash Attention, masking is not supported

    Performs the following:
        1. Apply cross attention to get the heads
        2. Project heads to get glimpse
        3. Apply PolyNet layers
        4. Compute attention score between glimpse and logit key

    Args:
        k: Number unique bit vectors used to compute attention score
        embed_dim: total dimension of the model
        poly_layer_dim: Dimension of the PolyNet layers
        num_heads: number of heads
        mask_inner: whether to mask inner attention
        linear_bias: whether to use bias in linear projection
        check_nan: whether to check for NaNs in logits
        sdpa_fn: scaled dot product attention function (SDPA) implementation
    """

    def __init__(
        self, k: int, embed_dim: int, poly_layer_dim: int, num_heads: int, **kwargs
    ):
        super(PolyNetAttention, self).__init__(embed_dim, num_heads, **kwargs)

        self.k = k
        self.binary_vector_dim = math.ceil(math.log2(k))
        self.binary_vectors = torch.nn.Parameter(
            torch.Tensor(
                list(itertools.product([0, 1], repeat=self.binary_vector_dim))[:k]
            ),
            requires_grad=False,
        )

        self.poly_layer_1 = nn.Linear(embed_dim + self.binary_vector_dim, poly_layer_dim)
        self.poly_layer_2 = nn.Linear(poly_layer_dim, embed_dim)

    def forward(self, query, key, value, logit_key, attn_mask=None):
        """Compute attention logits given query, key, value, logit key and attention mask.

        Args:
            query: query tensor of shape [B, ..., L, E]
            key: key tensor of shape [B, ..., S, E]
            value: value tensor of shape [B, ..., S, E]
            logit_key: logit key tensor of shape [B, ..., S, E]
            attn_mask: attention mask tensor of shape [B, ..., S]. Note that `True` means that the value _should_ take part in attention
                as described in the [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
        """
        # Compute inner multi-head attention with no projections.
        heads = self._inner_mha(query, key, value, attn_mask)
        glimpse = self.project_out(heads)

        num_solutions = glimpse.shape[1]
        z = self.binary_vectors.repeat(math.ceil(num_solutions / self.k), 1)[
            :num_solutions
        ]
        z = z[None].expand(glimpse.shape[0], num_solutions, self.binary_vector_dim)

        # PolyNet layers
        poly_out = self.poly_layer_1(torch.cat((glimpse, z), dim=2))
        poly_out = F.relu(poly_out)
        poly_out = self.poly_layer_2(poly_out)

        glimpse += poly_out

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # bmm is slightly faster than einsum and matmul
        logits = (torch.bmm(glimpse, logit_key.squeeze(-2).transpose(-2, -1))).squeeze(
            -2
        ) / math.sqrt(glimpse.size(-1))

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"

        return logits


class PolyNetAttentionMoE(PolyNetAttention):
    """Calculate logits given query, key, value, and logit_key, incorporating MoE.
    This class extends PolyNetAttention by integrating a Mixture of Experts (MoE)
    layer. The MoE layer processes the glimpse obtained from the initial multi-head
    attention before the PolyNet-specific transformations are applied. This allows
    for dynamic routing and potentially different expert processing based on the input.

    The MoE integration is similar to PointerAttnMoE, allowing for a 'light_version'
    with hierarchical gating.

    Performs the following:
        1. Apply cross attention to get the heads (via PolyNetAttention._inner_mha).
        2. Apply MoE processing to the heads to get a glimpse.
        3. Apply PolyNet layers to the MoE-processed glimpse.
        4. Compute attention score between the final glimpse and logit key.

    Args:
        k: Number of unique bit vectors used by PolyNet.
        embed_dim: Total dimension of the model.
        poly_layer_dim: Dimension of the PolyNet layers.
        num_heads: Number of attention heads.
        moe_kwargs: Keyword arguments for the MoE layer, including configurations
                    like 'num_experts', 'top_k', 'light_version', etc.
        mask_inner: Whether to mask inner attention (passed to PointerAttention).
        out_bias: Whether to use bias in the projection layers (passed to PointerAttention
                  and used in MoE's own layers if applicable).
        check_nan: Whether to check for NaNs in logits.
        sdpa_fn: Scaled dot product attention function (SDPA) implementation.
        **remaining_kwargs: Additional keyword arguments for PolyNetAttention.
    """

    def __init__(
        self,
        k: int,
        embed_dim: int,
        poly_layer_dim: int,
        num_heads: int,
        moe_kwargs: Optional[dict] = None,
        mask_inner: bool = True,
        out_bias: bool = False,
        check_nan: bool = False, # Default in PolyNetAttention is True if not passed explicitly
        sdpa_fn: Optional[Callable] = None,
        **remaining_kwargs,
    ):
        super().__init__(
            k=k,
            embed_dim=embed_dim,
            poly_layer_dim=poly_layer_dim,
            num_heads=num_heads,
            mask_inner=mask_inner,
            out_bias=out_bias,
            check_nan=check_nan,
            sdpa_fn=sdpa_fn,
            **remaining_kwargs,
        )

        self.moe_kwargs = moe_kwargs if moe_kwargs is not None else {}

        # MoE layer to process the output of MHA
        self.project_out_moe = MoE(
            input_size=embed_dim,
            output_size=embed_dim,
            num_neurons=[],  # Default, can be overridden by moe_kwargs
            out_bias=out_bias,
            **(self.moe_kwargs),
        )
        self.probs = None  # For light_version gating probabilities

        if self.moe_kwargs.get("light_version", False):
            self.dense_or_moe = nn.Linear(embed_dim, 2, bias=False)
            # This project_out is the dense path for the light_version MoE.
            # It shadows/overrides the self.project_out from PointerAttention for this specific use-case.
            self.project_out = nn.Linear(embed_dim, embed_dim, bias=out_bias)
            self.register_buffer(
                "probs_initial", torch.tensor([0.5, 0.5], dtype=torch.float32)
            )
        else:
            # If not in light_version, the main self.project_out (Linear layer from PointerAttention)
            # is effectively bypassed for glimpse calculation by the overridden forward method.
            # Setting it to None for clarity, consistent with PointerAttnMoE.
            self.project_out = None

    def forward(self, query, key, value, logit_key, attn_mask=None):
        """Compute attention logits with MoE processing before PolyNet layers.

        Args:
            query: query tensor of shape [B, ..., L, E]
            key: key tensor of shape [B, ..., S, E]
            value: value tensor of shape [B, ..., S, E]
            logit_key: logit key tensor of shape [B, ..., S, E]
            attn_mask: attention mask tensor of shape [B, ..., S].
                       Note that `True` means the value _should_ take part in attention.
                       This mask is used for both MHA and, if light_version MoE is active,
                       for the gating probability calculation.
        """
        # Compute inner multi-head attention with no projections.
        # `heads` has shape e.g., (batch, num_queries, embed_dim)
        heads = self._inner_mha(query, key, value, attn_mask)

        # --- MoE Glimpse Calculation ---
        # This part is adapted from PointerAttnMoE._project_out
        glimpse: torch.Tensor
        if self.moe_kwargs.get("light_version", False):
            if attn_mask is None:
                # light_version's probability calculation relies on attn_mask.
                # If attn_mask is None, default probabilities are used.
                # This differs slightly from PointerAttnMoE which might error.
                # For robustness, use default if mask not available.
                if self.probs is None:
                    self.probs = self.probs_initial.unsqueeze(0) # Shape [1, 2]
                # `recalculate_probs_condition` remains false, so probs won't update based on heads.
            else:
                # Ensure attn_mask is suitable for .size(-1) and .sum(-1)
                # Assumes attn_mask is the original one, e.g., [B, S]
                num_nodes = attn_mask.size(-1)
                num_available_nodes = attn_mask.sum(dim=-1) # Tensor if B > 1
                
                condition_to_recalculate_probs = (
                    num_available_nodes >= num_nodes - 1
                ).any()

                if condition_to_recalculate_probs:
                    self.probs = F.softmax(
                        self.dense_or_moe(
                            # Use reshape for flexibility if heads has more than 2 dims before embed_dim
                            heads.reshape(-1, heads.size(-1)).mean(dim=0, keepdim=True)
                        ),
                        dim=-1,
                    )  # Shape [1, 2]
                elif self.probs is None:
                    self.probs = self.probs_initial.unsqueeze(0) # Shape [1, 2]
            
            # self.probs is now guaranteed to be initialized, shape [1, 2]
            selected_scalar = self.probs.multinomial(num_samples=1).item() # Get 0 or 1

            if selected_scalar == 1:  # MoE path
                projected_intermediate = self.project_out_moe(heads)
            else:  # Dense path (self.project_out is the nn.Linear for this)
                projected_intermediate = self.project_out(heads)
            
            prob_of_selected_path = self.probs[0, selected_scalar]
            glimpse = projected_intermediate * prob_of_selected_path
        else:  # Not light_version or moe_kwargs is not configured for it
            glimpse = self.project_out_moe(heads)
        # --- End MoE Glimpse Calculation ---

        # --- Original PolyNetAttention logic continues with MoE-processed glimpse ---
        num_solutions = glimpse.shape[1] # num_solutions is query_length/num_steps
        z = self.binary_vectors.repeat(math.ceil(num_solutions / self.k), 1)[
            :num_solutions
        ]
        # Expand z to match batch size and keep num_solutions and binary_vector_dim
        z = z.unsqueeze(0).expand(glimpse.shape[0], num_solutions, self.binary_vector_dim)

        # PolyNet layers
        poly_out = self.poly_layer_1(torch.cat((glimpse, z), dim=2))
        poly_out = F.relu(poly_out)
        poly_out = self.poly_layer_2(poly_out)

        glimpse = glimpse + poly_out # Additive interaction with PolyNet output

        # Batch matrix multiplication to compute logits
        logits = (
            torch.bmm(glimpse, logit_key.squeeze(-2).transpose(-2, -1))
        ).squeeze(-2) / math.sqrt(glimpse.size(-1))

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"

        return logits


# 新增 SparseMultiHeadAttention 类
class SparseMultiHeadAttention(MultiHeadAttention):
    """
    Multi-Head Attention with Top-K Sparsification applied *before* softmax.
    Selects only the Top-K most relevant keys for each query.

    Args:
        k_sparse (int): The number of top keys to attend to for each query.
        *args, **kwargs: Arguments passed to the parent MultiHeadAttention class.
    """
    def __init__(self, k_sparse: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert k_sparse > 0, "k_sparse must be positive"
        self.k_sparse = k_sparse
        # Add SwiGLU FFN layer after the attention projection
        # hidden_ff_dim = int(embed_dim * 4 / 3) # Example hidden dim calculation
        # self.swiglu_ffn = SwiGLU(embed_dim, hidden_dim=hidden_ff_dim, bias=True)
        # Simplified: use default SwiGLU dimensions based on embed_dim
        self.swiglu_ffn = SwiGLU(self.embed_dim, bias=True) # Use SwiGLU class defined above

    def forward(self, x, attn_mask=None):
        """
        x: (batch, seqlen, hidden_dim)
        attn_mask: bool tensor of shape (batch, seqlen) or (batch, seqlen_q, seqlen_k)
                   Note: The `attn_mask` here typically masks padding. Sparsity mask is applied separately.
        """
        B, S, E = x.shape
        H = self.num_heads

        # Project query, key, value
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=H
        ).unbind(dim=0) # q, k, v: [B, H, S, D_head]

        # Calculate compatibility scores
        # scores: [B, H, S_q, S_k] where S_q = S_k = S for self-attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        scores = torch.nan_to_num(scores, nan=1e-4, posinf=1e4, neginf=-1e4)

        # Apply initial attention mask (e.g., for padding)
        if attn_mask is not None:
            if attn_mask.ndim == 2: # (B, S_k) -> (B, 1, 1, S_k)
                 attn_mask = attn_mask[:, None, None, :]
            elif attn_mask.ndim == 3: # (B, S_q, S_k) -> (B, 1, S_q, S_k)
                 attn_mask = attn_mask[:, None, :, :]
            else: # Assume shape (B, H, S_q, S_k)
                 pass
            scores.masked_fill_(~attn_mask, float("-inf")) # Apply padding mask first

        # Apply Top-K Sparsification
        #log.info(f"Sparsification: {S}")
        k_sparse_actual = min(self.k_sparse, S) # Ensure K is not larger than sequence length
        # Find top-k scores along the key dimension (dim=-1)
        # topk_scores: [B, H, S_q, K], topk_indices: [B, H, S_q, K]
        topk_scores, topk_indices = torch.topk(scores, k=k_sparse_actual, dim=-1, sorted=False)

        # Create a sparse mask based on top-k indices
        # sparse_mask: [B, H, S_q, S_k] - True for top-k positions, False otherwise
        sparse_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_indices, True)

        # Apply the sparse mask: set non-top-k scores to -inf
        scores.masked_fill_(~sparse_mask, float("-inf"))

        # Apply causal mask if needed (after top-k)
        if self.causal:
            s_q, s_k = scores.size(-2), scores.size(-1)
            causal_mask = torch.triu(torch.ones((s_q, s_k), device=scores.device), diagonal=1)
            scores.masked_fill_(causal_mask.bool(), float("-inf"))

        # Softmax to get attention weights (only over the top-k elements now)
        attn_weights = F.softmax(scores, dim=-1) # [B, H, S, S]

        # Apply dropout
        if self.attention_dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        # Compute the weighted sum of values
        # We need to gather the values corresponding to the sparse attention weights,
        # but since weights for non-topk are 0 after softmax, standard matmul works.
        out = torch.matmul(attn_weights, v) # [B, H, S_q, D_head]

        # Reshape and project output
        projected_out = self.out_proj(rearrange(out, "b h s d -> b s (h d)"))
        # Apply SwiGLU FFN
        final_out = self.swiglu_ffn(projected_out)
        return final_out
        
class ChannelSelfAttention11111(nn.Module):

    # Takes embed_dim (dk) and num_channels (c) as input
    def __init__(self, embed_dim: int, num_channels: int):

        super().__init__()
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        # Use sqrt(dk) scaling for attention (not directly used for (c,c) attention as per diagram)
        # self.scale = embed_dim ** -0.5

        # Linear layers for Query, Key, and Value from original code are not used in diagram's channel attention
        # self.linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.linear_k = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.linear_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output linear layer from original code is replaced by Conv2d in diagram
        # self.linear_out = nn.Linear(embed_dim, embed_dim, bias=False)

        # Conv2d layer as shown in the diagram to reduce channels from c to 1
        # Input to Conv2d will be (N, C_in, H, W) where N=batch_size, C_in=num_channels, H=n, W=dk
        # Output will be (N, C_out=1, H=n, W=dk)
        self.conv2d = nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=1, bias=False)
        # self.swiglu_ffn = SwiGLU(embed_dim, bias=False) # Added SwiGLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (b, c, n, dk)
        b, c, n, dk = x.shape
        assert c == self.num_channels, f"Input channels {c} != self.num_channels {self.num_channels}"
        assert dk == self.embed_dim, f"Input embed_dim {dk} != self.embed_dim {self.embed_dim}"

        # Store original input for Element Sum (residual connection) as per diagram
        residual = x # (b, c, n, dk)

        # 1. Compute Channel Attention Matrix (c, c) as per diagram
        # Diagram shows reshaping to (c, ndk) for Q and (ndk, c) for K.
        # In batch-wise, this means (b, c, n*dk) and its transpose.
        x_flat_for_channel_attn = x.reshape(b, c, n * dk) # (b, c, N*DK)

        # Calculate attention scores: (b, c, N*DK) @ (b, N*DK, c) -> (b, c, c)
        attn_scores_channel = torch.matmul(x_flat_for_channel_attn, x_flat_for_channel_attn.transpose(-1, -2))

        # Apply Softmax to get attention weights (b, c, c)
        channel_attention_matrix = F.softmax(attn_scores_channel, dim=-1)

        # 2. Apply Channel Attention as per diagram's "Multiplication" step
        # Diagram shows: (c, ndk) x (c, c) -> (c, n, dk) reshape.
        # Interpretation: (b, c, c) @ (b, c, n*dk) -> (b, c, n*dk).
        # Here, x_flat_for_channel_attn acts as the Value (V).
        attended_features_flat = torch.matmul(channel_attention_matrix, x_flat_for_channel_attn)

        # Reshape the output of attention back to (b, c, n, dk) as per diagram
        attended_features = attended_features_flat.reshape(b, c, n, dk)

        # 3. Element Sum (Skip Connection) as per diagram
        output_after_attn_and_sum = attended_features + residual

        # 4. Conv(2d) as per diagram
        # Conv2d expects input (N, C_in, H, W). Here, C_in=c, H=n, W=dk.
        # Output will be (N, C_out=1, H=n, W=dk).
        output_conv2d = self.conv2d(output_after_attn_and_sum)

        # 5. Reshape to (n, dk) by squeezing the channel dimension (1) as per diagram's final output
        final_output = output_conv2d.squeeze(1) # -> (b, n, dk)

        return final_output

class ChannelSelfAttention(nn.Module):

    # Takes embed_dim (dk) and num_channels (c) as input
    def __init__(self, embed_dim: int, num_channels: int):

        super().__init__()
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        # Use sqrt(dk) scaling for attention
        self.scale = embed_dim ** -0.5

        # Linear layers for Query, Key, and Value operate on embed_dim (dk)
        self.linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias=False)
        # Output linear layer also operates on embed_dim
        self.linear_out = nn.Linear(embed_dim, embed_dim, bias=False)
        # Aggregation projection: concatenated channels (c * dk) -> dk
        self.linear_agg = nn.Linear(num_channels * embed_dim, embed_dim, bias=False)
        # self.swiglu_ffn = SwiGLU(embed_dim, bias=False) # Added SwiGLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (b, c, n, dk)
        b, c, n, dk = x.shape
        assert c == self.num_channels, f"Input channels {c} != self.num_channels {self.num_channels}"
        assert dk == self.embed_dim, f"Input embed_dim {dk} != self.embed_dim {self.embed_dim}"

        # 1. Calculate Query, Key, and Value (applied to the last dimension dk)
        # q, k, v: (b, c, n, dk)
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # Reshape for scaled_dot_product_attention
        # (b, c, n, dk) -> (b*c, n, dk)
        q_sdpa = q.reshape(b * c, n, dk)
        k_sdpa = k.reshape(b * c, n, dk)
        v_sdpa = v.reshape(b * c, n, dk)

        # 2. Use scaled_dot_product_attention for efficient computation
        # scaled_dot_product_attention handles scaling internally.
        # Input: (N, S, E) where N=b*c, S=n, E=dk
        # Output: (N, S, E) -> (b*c, n, dk)
        attn_output_sdpa = scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, 
            attn_mask=None, dropout_p=0.0, is_causal=False # ChannelSelfAttention doesn't inherently support these
        )

        # Reshape output back to (b, c, n, dk)
        output = attn_output_sdpa.reshape(b, c, n, dk)

        # 5. Apply Output Linear Layer per channel
        # output: (b, c, n, dk)
        output = self.linear_out(output)
        # output = self.swiglu_ffn(output) # Applied SwiGLU

        # 6. Aggregate channels by concatenation across channel dim
        # Permute to (b, n, c, dk) then reshape to (b, n, c*dk)
        output_cat = output.permute(0, 2, 1, 3).reshape(b, n, c * dk)

        # Project concatenated channels back to (b, n, dk)
        output_agg = self.linear_agg(output_cat)

        return output_agg


# Helper module to encapsulate 1x1 Conv + ChannelSelfAttention for K/V enhancement
class _ChannelWiseSelfAttentionEnhancer(nn.Module):
    def __init__(self, embed_dim: int, num_enh_channels: int):
        super().__init__()
        # 1x1 Conv to create multiple channel views
        # Input: (B, 1, S, E) -> Output: (B, C_enh, S, E)
        self.conv_1x1 = nn.Conv2d(in_channels=1, out_channels=num_enh_channels, kernel_size=1, bias=False)
        # ChannelSelfAttention to process these channels
        # Input: (B, C_enh, S, E) -> Output: (B, S, E) after internal aggregation
        self.csa = ChannelSelfAttention(embed_dim=embed_dim, num_channels=num_enh_channels)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # x_in: (B, S, E) - initial K or V projection
        
        # Prepare for 1x1 convolution: add a channel dimension
        x_conv_input = x_in.unsqueeze(1)          # Shape: (B, 1, S, E)
        
        # Apply 1x1 convolution to get multiple channel views
        x_multi_channel = self.conv_1x1(x_conv_input) # Shape: (B, C_enh, S, E)
        
        # Apply ChannelSelfAttention
        x_enhanced = self.csa(x_multi_channel)      # Shape: (B, S, E)
        
        return x_enhanced


class ChannelEnhancedContextualAttention(nn.Module):
    """Calculates attention logic scores based on channel enhancement.
    This class implements a mechanism combining channel enhancement and self-attention,
    optimized to address NaN value issues.

    Features:
    1. Uses convolutional layers and channel self-attention to enhance Key and Value representations.
    2. Has a standard multi-head self-attention interface, suitable for use as an encoder layer.
    3. Includes numerical stability measures to prevent NaN values.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of attention heads.
        channels: Number of channels in the channel enhancement layer.
        mask_inner: (No longer used in this self-attention version)
        out_bias: Whether to use bias in linear projections.
        check_nan: Whether to check for NaN values in logits.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        channels: int = 4,
        mask_inner: bool = True,
        out_bias: bool = False,
        check_nan: bool = False,
        **kwargs
    ):
        super(ChannelEnhancedContextualAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.channels = channels
        self.check_nan = check_nan
        
        # 计算每个头的维度并确保可以被整除
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5
        
        # Q投影层：将完整嵌入投影到查询空间
        # 输入: 完整嵌入 (batch_size, seq_len, embed_dim)
        # 输出: Q投影 (batch_size, seq_len, embed_dim)
        self.project_q = nn.Linear(embed_dim, embed_dim, bias=out_bias)
        
        # K,V投影层：将约束嵌入投影到键值空间
        # 输入: 约束嵌入 (batch_size, seq_len, embed_dim)
        # 输出: K,V投影 (batch_size, seq_len, embed_dim)
        self.project_k = nn.Linear(embed_dim, embed_dim, bias=out_bias)
        self.project_v = nn.Linear(embed_dim, embed_dim, bias=out_bias)
        
        # K,V通道增强器：对K,V进行通道化处理
        # 输入: K,V投影 (batch_size, seq_len, embed_dim)
        # 输出: 通道增强的K,V (batch_size, seq_len, embed_dim)
        self.key_enhancer = _ChannelWiseSelfAttentionEnhancer(embed_dim=self.embed_dim, num_enh_channels=self.channels)
        self.value_enhancer = _ChannelWiseSelfAttentionEnhancer(embed_dim=self.embed_dim, num_enh_channels=self.channels)
        
        # 输出投影层：将注意力输出投影回原始维度
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=out_bias)
        
        # 层归一化：用于Q,K,V和输出的归一化
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

    def _make_heads(self, projected_tensor: torch.Tensor) -> torch.Tensor:
        """将投影张量分割成多个注意力头
        输入: (batch_size, seq_len, embed_dim)
        输出: (batch_size, num_heads, seq_len, head_dim)
        """
        return rearrange(projected_tensor, 'b s (h d) -> b h s d', h=self.num_heads, d=self.head_dim)

    def _combine_heads(self, heads_tensor: torch.Tensor) -> torch.Tensor:
        """将多个注意力头合并回原始形状
        输入: (batch_size, num_heads, seq_len, head_dim)
        输出: (batch_size, seq_len, embed_dim)
        """
        return rearrange(heads_tensor, 'b h s d -> b s (h d)')

    def forward(
        self, 
        x: torch.Tensor,  # 完整嵌入作为Q的输入
        kv: Optional[torch.Tensor] = None,  # 约束嵌入作为K,V的输入
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False  # 是否返回注意力权重用于可视化
    ) -> torch.Tensor:
        """
        通道增强的上下文注意力的前向传播。
        
        处理流程：
        1. Q使用完整嵌入（包含所有特征）
        2. K,V使用约束嵌入并进行通道化处理
        3. 通过多头注意力机制计算注意力分数
        
        Args:
            x: 完整嵌入张量 (batch_size, seq_len, embed_dim)
            kv: 约束嵌入张量 (batch_size, seq_len, embed_dim)
            attn_mask: 注意力掩码
            return_attention: 是否返回注意力权重用于可视化
        Returns:
            如果return_attention=False: 输出张量 (batch_size, seq_len, embed_dim)
            如果return_attention=True: (输出张量, 注意力权重)
        """
        batch_size, seq_len, embed_dim_x = x.shape
        assert embed_dim_x == self.embed_dim, f"Input embed_dim {embed_dim_x} != self.embed_dim {self.embed_dim}"

        # 如果没有提供约束嵌入，使用完整嵌入作为K,V的输入
        kv = x if kv is None else kv

        # 1. 投影并归一化 Q,K,V
        # Q使用完整嵌入，包含所有信息
        q_init = self.norm_q(self.project_q(x))  # (batch_size, seq_len, embed_dim)
        
        # K,V使用约束嵌入
        k_init = self.norm_k(self.project_k(kv))  # (batch_size, seq_len, embed_dim)
        v_init = self.norm_v(self.project_v(kv))  # (batch_size, seq_len, embed_dim)

        # 2. 对K,V进行通道增强
        # 通过1x1卷积和通道自注意力处理
        k_enhanced = self.key_enhancer(k_init)    # (batch_size, seq_len, embed_dim)
        v_enhanced = self.value_enhancer(v_init)  # (batch_size, seq_len, embed_dim)

        # 3. 将Q和增强后的K,V分割成多个注意力头
        q_heads = self._make_heads(q_init)              # (batch_size, num_heads, seq_len, head_dim)
        k_enhanced_heads = self._make_heads(k_enhanced)  # (batch_size, num_heads, seq_len, head_dim)
        v_enhanced_heads = self._make_heads(v_enhanced)  # (batch_size, num_heads, seq_len, head_dim)

        # 4. 计算缩放点积注意力分数
        # Q与K的矩阵乘法，并应用缩放因子
        scores = torch.matmul(q_heads, k_enhanced_heads.transpose(-2, -1)) * self.scale # (batch_size, num_heads, seq_len, seq_len)
        #scores = torch.nan_to_num(scores, nan=1e-4, posinf=1e4, neginf=-1e4)  # 数值稳定性处理
        attn_weights1 = scores
        # 5. 应用注意力掩码（如果提供）
        if attn_mask is not None:
            mask_for_scores = attn_mask.bool()
            if mask_for_scores.dim() == 2:  # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                mask_for_scores = mask_for_scores.unsqueeze(1).unsqueeze(1)
            elif mask_for_scores.dim() == 3:  # (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
                mask_for_scores = mask_for_scores.unsqueeze(1)
            scores = scores.masked_fill(~mask_for_scores, float("-inf"))

        # 6. 计算注意力权重并应用到V上
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # 数值稳定性处理
        output_heads = torch.matmul(attn_weights, v_enhanced_heads)  # (batch_size, num_heads, seq_len, head_dim)

        # 7. 合并多头注意力的结果并进行最终投影
        combined_output = self._combine_heads(output_heads)  # (batch_size, seq_len, embed_dim)
        final_output = self.project_out(combined_output)    # (batch_size, seq_len, embed_dim)
        final_output = self.norm_out(final_output)         # 最终的层归一化

        if self.check_nan:
            assert not torch.isnan(final_output).any(), "Output contains NaNs"
            
        if return_attention:
            return final_output, attn_weights1
        else:
            return final_output

