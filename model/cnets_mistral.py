# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Transformers >= 4.36
""" PyTorch Mistral model."""
import copy
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
from typing import List, Optional, Tuple, Union
import inspect
import warnings

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask, 
    _prepare_4d_causal_attention_mask_for_sdpa
)
from transformers import AutoConfig

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

from loguru import logger

try:
    from .configs import MistralConfig
    from .utils_c import *
    from .choices import *
except:
    from configs import MistralConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor
top_k=10

# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Mistral
class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Mistral
class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MistralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MistralAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MistralFlashAttention2(MistralAttention):
    """
    Mistral flash attention module. This module inherits from `MistralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention with Llama->Mistral
class MistralSdpaAttention(MistralAttention):
    """
    Mistral attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MistralAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from MistralAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "MistralModel is using MistralSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


MISTRAL_ATTENTION_CLASSES = {
    "eager": MistralAttention,
    "flash_attention_2": MistralFlashAttention2,
    "sdpa": MistralSdpaAttention,
}


class MistralDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MISTRAL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MistralPreTrainedModel(PreTrainedModel):
    config_class = MistralConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MistralDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class I(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float32))
    def forward(self,x):
        return x + self.dummy - self.dummy #(also tried x+self.dummy)

def len_list(x,n):
    return [i for i in x if len(i)<=n]



class EagleMistralModel(MistralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, config, load_emb=False, path=None):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.gradient_checkpointing = True
        self._attn_implementation = config._attn_implementation
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if load_emb:
            from safetensors import safe_open
            import json
            try:
                with open(os.path.join(path,"model.safetensors.index.json"),"r") as f:
                    index_json=json.loads(f.read())
                    emb_path=index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(os.path.join(path,emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights=torch.load(os.path.join(path,emb_path))
                tensor=weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor

        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.fc=nn.Linear(2*config.hidden_size,config.hidden_size)
        self.act=ACT2FN[config.hidden_act]
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        self.gradient_checkpointing_enable()


    def init_tree(self):
        self.tree = mc_sim_7b_63
        self.tree_buffer=generate_tree_buffers(self.tree,self.embed_tokens.weight.device)


    def reset(self):
        self.tree_mask=None

    def forward(
        self,
        hidden_states, # MODIFIED
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        inputs_embeds=inputs_embeds.to(hidden_states.dtype)
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
                # def create_custom_forward(module):
                #     def custom_forward(*inputs):
                #         # None for past_key_value
                #         return module(*inputs, past_key_values, output_attentions)

                #     return custom_forward

                # layer_outputs = torch.utils.checkpoint.checkpoint(
                #     decoder_layer.__call__,
                #     hidden_states,
                #     attention_mask,
                #     position_ids,
                #     past_key_values,
                #     output_attentions,
                #     use_cache,
                # )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if use_cache:
            return hidden_states,next_decoder_cache

        return hidden_states


    @torch.no_grad()
    def generate(self,hidden_states,input_ids,head,max_length=4,use_cache=False):
        return_input_ids=copy.deepcopy(input_ids[0].tolist())
        input_ids=input_ids[:,1:]

        #input_ids=input_ids.to(hidden_states.device)
        if use_cache:
            past_key_values=None
            for i in range(max_length):
                if past_key_values!=None:
                    out_hidden,past_key_values = self(out_hidden[:, -1:], input_ids=torch.tensor([[token]]).to(input_ids.device),past_key_values=past_key_values,use_cache=True)
                else:
                    out_hidden, past_key_values = self(hidden_states, input_ids=input_ids,use_cache=True)
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                #input_ids = torch.cat((input_ids, torch.tensor([[token]]).to(input_ids.device)), dim=1)
                return_input_ids.append(token.item())
                if token == 2:
                    break
                #hidden_states = torch.cat((hidden_states, out_hidden[:, -1:]), dim=1)
        else:
            for i in range(max_length):
                out_hidden=self(hidden_states,input_ids=input_ids)
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                return_input_ids.append(token.item())
                input_ids = torch.cat((input_ids, torch.tensor([[token]]).to(input_ids.device)), dim=1)
                if token==2:
                    break
                hidden_states = torch.cat((hidden_states, out_hidden[:, -1:]), dim=1)

        return return_input_ids

    @torch.no_grad()
    def repeat_kv(self,kv,numr):
        newkv=[]
        for i in kv:
            newkv.append((i[0].repeat(numr,1,1,1),i[1].repeat(numr,1,1,1)))
        return tuple(newkv)

    @torch.no_grad()
    def reduce_kv(self,kv,numr):
        newkv=[]
        for i in kv:
            newkv.append((i[0][:numr],i[1][:numr]))
        return tuple(newkv)


    def reset_kv(self):
        self.stable_kv=None

    @torch.no_grad()
    def topK_genrate_batch(self,hidden_states,input_ids,head,max_length=4,use_cache=True):
        #input_ids = torch.tensor([state[1:]])
        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)
        sslogits=[]
        self.reset()
        if use_cache:

            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids,use_cache=True)
            last_hidden = out_hidden[:, -1]
            last_headout = head(last_hidden)
            sslogits.append(last_headout)
            topk_index = torch.topk(last_headout, 3, dim=-1).indices

            # hidden_states = torch.cat((hidden_states, out_hidden[:, -1:]), dim=1)
            hidden_states = out_hidden[:, -1:]
            hidden_states = hidden_states.repeat(3, 1, 1)
            #input_ids = input_ids.repeat(3, 1)
            input_ids = topk_index.t()
            past_key_values = self.repeat_kv(past_key_values,3)
            out_hidden,past_key_values = self(hidden_states, input_ids=input_ids,past_key_values=past_key_values,use_cache=True)
            last_hidden = out_hidden[:, -1]
            last_headout = head(last_hidden)
            sslogits.append(last_headout)

            hidden_states = out_hidden[0:1, -1:]
            #input_ids = input_ids[:1]
            topk_index = torch.topk(last_headout[:1], 3, dim=-1).indices
            #hidden_states = torch.cat((hidden_states, out_hidden[0:1, -1:]), dim=1)
            hidden_states = hidden_states.repeat(3, 1, 1)
            #input_ids = input_ids.repeat(3, 1)
            input_ids = topk_index.t()
            out_hidden,past_key_values = self(hidden_states, input_ids=input_ids,past_key_values=past_key_values,use_cache=True)
            last_hidden = out_hidden[:, -1]
            last_headout = head(last_hidden)
            sslogits.append(last_headout)

            #hidden_states = hidden_states[:1]
            #input_ids = input_ids[:1]
            topk_index = torch.topk(last_headout[:1], 3, dim=-1).indices
            hidden_states = out_hidden[0:1, -1:]
            input_ids = topk_index[:, :1]
            past_key_values=self.reduce_kv(past_key_values,1)
            out_hidden,past_key_values = self(hidden_states, input_ids=input_ids,past_key_values=past_key_values,use_cache=True)
            last_hidden = out_hidden[:, -1]
            last_headout = head(last_hidden)
            sslogits.append(last_headout)
        else:
            out_hidden = self(hidden_states, input_ids=input_ids)
            last_hidden = out_hidden[:, -1]
            last_headout = head(last_hidden)
            sslogits.append(last_headout)
            topk_index=torch.topk(last_headout, 3, dim=-1).indices

            hidden_states = torch.cat((hidden_states, out_hidden[:, -1:]), dim=1)
            hidden_states=hidden_states.repeat(3,1,1)
            input_ids=input_ids.repeat(3,1)
            input_ids=torch.cat((input_ids,topk_index.t()),dim=-1)
            out_hidden = self(hidden_states, input_ids=input_ids)
            last_hidden = out_hidden[:, -1]
            last_headout = head(last_hidden)
            sslogits.append(last_headout)

            hidden_states=hidden_states[:1]
            input_ids=input_ids[:1]
            topk_index = torch.topk(last_headout[:1], 3, dim=-1).indices
            hidden_states = torch.cat((hidden_states, out_hidden[0:1, -1:]), dim=1)
            hidden_states = hidden_states.repeat(3, 1, 1)
            input_ids = input_ids.repeat(3, 1)
            input_ids = torch.cat((input_ids, topk_index.t()), dim=-1)
            out_hidden = self(hidden_states, input_ids=input_ids)
            last_hidden = out_hidden[:, -1]
            last_headout = head(last_hidden)
            sslogits.append(last_headout)

            hidden_states = hidden_states[:1]
            input_ids = input_ids[:1]
            topk_index = torch.topk(last_headout[:1], 3, dim=-1).indices
            hidden_states = torch.cat((hidden_states, out_hidden[0:1, -1:]), dim=1)
            input_ids = torch.cat((input_ids, topk_index[:,:1]), dim=-1)
            out_hidden = self(hidden_states, input_ids=input_ids)
            last_hidden = out_hidden[:, -1]
            last_headout = head(last_hidden)
            sslogits.append(last_headout)

        return torch.cat(sslogits)

    @torch.no_grad()
    def repeat_hidden(self,hidden_state,repeat_num):
        new_hidden=[]
        for id,i in enumerate(repeat_num):
            new_hidden.append(hidden_state[:,id:id+1].repeat(1,i,1))
        return torch.cat(new_hidden,dim=1)

    # @torch.no_grad()
    # def sample(self,tensor,k=1,replacement=True):
    #     probabilities = torch.nn.functional.softmax(tensor, dim=1)
    #     sampled_indices = torch.multinomial(probabilities, k,replacement=replacement)
    #     sampled_probs = torch.gather(probabilities, 1, sampled_indices)
    #
    #     return  sampled_indices,sampled_probs

    def sample(self,logits, logits_processor,k=1, replacement=False):
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        sampled_indices = torch.multinomial(probabilities, k, replacement=False)
        sampled_probs = torch.gather(probabilities, 1, sampled_indices)

        cumulative_sum = torch.cumsum(sampled_probs, dim=1)
        cumulative_sum = torch.cat(
            (torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device), cumulative_sum[:, :-1]), dim=-1)

        sampled_probs = sampled_probs / (1 - cumulative_sum)
        sampled_probs[torch.isinf(sampled_probs)] = -1
        sampled_probs[torch.isnan(sampled_probs)] = -1

        sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)

        return sampled_indices, sampled_probs,probabilities

        # if replacement:
        #     sampled_indices = torch.multinomial(probabilities, k, replacement=True)
        #     sampled_probs = torch.gather(probabilities, 1, sampled_indices)
        #     return sampled_indices, sampled_probs
        # else:
        #     sampled_indices = torch.multinomial(probabilities, k, replacement=False)
        #     sampled_probs = torch.gather(probabilities, 1, sampled_indices)
        #
        #     cumulative_sum = torch.cumsum(sampled_probs, dim=1)
        #     cumulative_sum = torch.cat((torch.zeros(cumulative_sum.shape[0],1, device=cumulative_sum.device), cumulative_sum[:, :-1]),dim=-1)
        #
        #     sampled_probs=sampled_probs/(1-cumulative_sum)
        #     sampled_probs[torch.isinf(sampled_probs)] = -1
        #     sampled_probs[torch.isnan(sampled_probs)] = -1
        #
        #     sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)
        #
        #     # has_nan = torch.isnan(sampled_probs).any()
        #     # if has_nan:
        #     #     print(1)
        #
        #     # sampled_probs_list=sampled_probs[0].tolist()
        #     # sum_list=[1-sum(sampled_probs_list[:i]) for i in range(len(sampled_probs_list))]
        #     # for i in range(len(sampled_probs_list)):
        #     #     a=sampled_probs_list[i]/(sum_list[i])
        #     #     if sum_list[i]==0:
        #     #         sampled_probs_list[i]=1.0
        #     #     else:
        #     #         sampled_probs_list[i]=sampled_probs_list[i]/(sum_list[i])
        #     # sampled_probs=torch.tensor([sampled_probs_list],device=sampled_probs.device)
        #
        #
        #
        #     return sampled_indices, sampled_probs

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor,max_length=4, use_cache=True):
        # test_=input_ids
        # input_ids = torch.tensor([state[1:]])
        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)
        ss_token,ss_prob,ss_op = [],[],[]
        len_posi=input_ids.shape[1]
        self.reset()
        if use_cache:


            if hasattr(self, "stable_kv") and self.stable_kv is not None:
                kv_len=self.stable_kv[0][0].shape[2]
                out_hidden, past_key_values = self(hidden_states[:,kv_len:], input_ids=input_ids[:,kv_len:], past_key_values=self.stable_kv,use_cache=True)
            else:
                out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
            self.stable_kv=past_key_values
            last_hidden = out_hidden[:, -1]
            if not self.diff_device:
                last_headout = head(last_hidden)
            else:
                if hasattr(self, "layer_device"):
                    last_headout = head(last_hidden)
                    last_headout=last_headout.to(self.layer_device)
                else:
                    last_headout=F.linear(last_hidden,self.headweight)



            for i in range(len(self.tree_buffer['tree_indices'])):
                if logits_processor is not None:
                    topk_index,topk_prob,op=self.sample(last_headout,logits_processor,k=top_k,)
                else:
                    topk_index,topk_prob = torch.topk(last_headout, top_k, dim=-1).indices,torch.topk(last_headout, top_k, dim=-1).values
                    op=None

                ss_token.append(topk_index)
                ss_prob.append(topk_prob)
                ss_op.append(op)
                #topk_index = torch.topk(last_headout, top_k, dim=-1).indices
                topk_index = topk_index.view(-1)
                select_index=topk_index[self.tree_buffer['tree_indices'][i]]
                #len_sq=select_index.shape[0]
                input_ids=select_index[None,:]
                if i==0:
                    hidden_states = out_hidden[:, -1:]
                else:
                    hidden_states=out_hidden
                hidden_states=self.repeat_hidden(hidden_states,self.tree_buffer["repeat_nums"][i])
                #hidden_states = hidden_states.repeat(1,len_sq,1)
                self.tree_mask=self.tree_buffer['attn_mask'][i]
                position_ids=len_posi+self.tree_buffer["position_ids"][i]
                out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, past_key_values=past_key_values,
                                                   position_ids=position_ids,use_cache=True)
                len_posi += 1

                if not self.diff_device:
                    last_headout = head(out_hidden[0])
                else:
                    if hasattr(self, "layer_device"):
                        last_headout = head(out_hidden[0])
                        last_headout = last_headout.to(self.layer_device)
                    else:
                        last_headout = F.linear(out_hidden[0], self.headweight)
                #last_headout = head(out_hidden[0])
                #sslogits.append(last_headout)
                #print(select_index)

            if logits_processor is not None:
                topk_index,topk_prob,op=self.sample(last_headout,logits_processor,k=top_k,)
            else:
                topk_index, topk_prob = torch.topk(last_headout, top_k, dim=-1).indices, torch.topk(last_headout, top_k,
                                                                                                    dim=-1).values
                op=None
            ss_token.append(topk_index)
            ss_prob.append(topk_prob)
            ss_op.append(op)

        else:
            # TODO
            pass

        return (torch.cat(ss_token),torch.cat(ss_prob),ss_op)




    @torch.no_grad()
    def acc(self,data,head,max_length=5):
        hidden_states=data["hidden_states"]
        input_ids=data["input_ids"]
        #attention_mask=data["attention_mask"]
        loss_mask=data["loss_mask"]
        sample_mask=data["sample_mask"]
        target=data["target"]
        total=[0 for _ in range(max_length)]
        correct=[0 for _ in range(max_length)]
        bs,sl=hidden_states.shape[0],hidden_states.shape[1]
        target_headout = head(target)
        hidden_states_headout=head(hidden_states)

        for i in range(bs):
            for j in range(sl):
                if loss_mask[i,j]==0:
                    continue
                single_hidden_states=hidden_states[i,:j]
                single_input_ids=input_ids[i,:j]


                single_hidden_states = single_hidden_states[None, :, :]
                single_input_ids = single_input_ids[None, :]
                for k in range(max_length):
                    tmp_in_target_headout = hidden_states_headout[i,single_hidden_states.shape[1]-1]
                    tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1]-1]
                    target_in_token = torch.argmax(tmp_in_target_headout)
                    target_out_token = torch.argmax(tmp_out_target_headout)
                    tmp_token=input_ids[i,single_hidden_states.shape[1]-1]
                    tmp_sample_mask=sample_mask[i,single_hidden_states.shape[1]-1]
                    if not (target_in_token==tmp_token):
                        break
                    out_hidden = self(single_hidden_states, input_ids=single_input_ids)
                    last_hidden = out_hidden[:, -1]
                    last_headout = head(last_hidden)
                    token = torch.argmax(last_headout)
                    total[k] += 1
                    if token==target_out_token:
                        correct[k]+=1
                    else:
                        for kk in range(k,max_length):
                            total[kk]+=1
                        break

                    single_hidden_states=torch.cat((single_hidden_states,out_hidden[:,-1:]),dim=1)
                    single_input_ids = torch.cat((single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)), dim=1)


        acc=[correct[i]/total[i] for i in range(len(correct))]
        return acc





class Vhead(nn.Module):
    def __init__(self,ins=6566,outs=32000):
        super().__init__()
        self.fc = nn.Linear(ins,outs,bias=False)
    def forward(self,x):
        return self.fc(x)

# class Model(nn.Module):
#     def __init__(self,config, load_emb=False, path=None):
#         super().__init__()

#         self.gradient_checkpointing = True
#         self.padding_idx = config.pad_token_id
#         self.vocab_size = config.vocab_size

#         self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
#         if load_emb:
#             from safetensors import safe_open
#             import json
#             try:
#                 with open(os.path.join(path,"model.safetensors.index.json"),"r") as f:
#                     index_json=json.loads(f.read())
#                     emb_path=index_json["weight_map"]["model.embed_tokens.weight"]
#                 with safe_open(os.path.join(path,emb_path),
#                                framework="pt",
#                                device="cpu") as f:
#                     tensor_slice = f.get_slice("model.embed_tokens.weight")
#                     vocab_size, hidden_dim = tensor_slice.get_shape()
#                     tensor = tensor_slice[:, :hidden_dim].float()
#             except:
#                 with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
#                     index_json = json.loads(f.read())
#                     emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
#                 weights=torch.load(os.path.join(path,emb_path))
#                 tensor=weights["model.embed_tokens.weight"].float()
#             self.embed_tokens.weight.data = tensor


#         #self.init_tree()

#         self.layers = nn.ModuleList([MistralDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
#         self.fc=nn.Linear(2*config.hidden_size,config.hidden_size)
#         self.act=ACT2FN[config.hidden_act]
#         for param in self.embed_tokens.parameters():
#             param.requires_grad = False


#     def init_tree(self):
#         self.tree = mc_sim_7b_63
#         self.tree_buffer=generate_tree_buffers(self.tree,self.embed_tokens.weight.device)


#     def reset(self):
#         self.tree_mask=None


#     def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
#         # create causal mask
#         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#         combined_attention_mask = None
#         if input_shape[-1] > 1:
#             combined_attention_mask = _make_causal_mask(
#                 input_shape,
#                 #inputs_embeds.dtype,
#                 torch.float32, # [MODIFIED] force to cast to float32
#                 device=inputs_embeds.device,
#                 past_key_values_length=past_key_values_length,
#             )

#         if attention_mask is not None:
#             # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#             expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
#                 inputs_embeds.device
#             )
#             combined_attention_mask = (
#                 expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
#             )

#         # [MODIFIED] add tree mask
#         if hasattr(self, "tree_mask") and self.tree_mask is not None:
#             tree_mask = self.tree_mask
#             tree_len = tree_mask.size(-1)
#             combined_attention_mask[:, :, -tree_len:, -tree_len:][
#                 tree_mask == 0
#                 ] = torch.finfo(torch.float32).min


#         return combined_attention_mask

#     def forward(
#         self,
#         hidden_states,
#         input_ids,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         std=None
#     ):
#         batch_size, seq_length, _ = hidden_states.shape
#         seq_length_with_past = seq_length
#         past_key_values_length = 0

#         with torch.no_grad():
#             inputs_embeds = self.embed_tokens(input_ids)
#             #inputs_embeds = inputs_embeds.detach()

#         # if std is not None:
#         #     noise = torch.randn(inputs_embeds.size(),device=inputs_embeds.device) * std
#         #     inputs_embeds=inputs_embeds+noise

#         if past_key_values is not None:
#             past_key_values_length = past_key_values[0][0].shape[2]
#             seq_length_with_past = seq_length_with_past + past_key_values_length
#         if position_ids is None:
#             device = hidden_states.device if hidden_states is not None else inputs_embeds.device
#             position_ids = torch.arange(
#                 past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
#             )
#             position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
#         else:
#             position_ids = position_ids.view(-1, seq_length).long()

#         if attention_mask is None:
#             attention_mask = torch.ones(
#                 (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
#             )
#         attention_mask = self._prepare_decoder_attention_mask(
#             attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
#         )

#         # if self.gradient_checkpointing and self.training:
#         #    if use_cache:
#         #        use_cache = False


#         #hidden_states=self.act(self.fc(torch.cat((inputs_embeds,hidden_states),dim=-1)))
#         inputs_embeds=inputs_embeds.to(hidden_states.dtype)
#         hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))


#         all_hidden_states = () if output_hidden_states else None
#         next_decoder_cache = () if use_cache else None

#         for idx, decoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)

#             past_key_value = past_key_values[idx] if past_key_values is not None else None

#             if self.gradient_checkpointing and self.training:

#                 def create_custom_forward(module):
#                     def custom_forward(*inputs):
#                         # None for past_key_value
#                         return module(*inputs, past_key_value, output_attentions)

#                     return custom_forward

#                 layer_outputs = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(decoder_layer),
#                     hidden_states,
#                     attention_mask,
#                     position_ids,
#                 )
#             else:
#                 layer_outputs = decoder_layer(
#                     hidden_states,
#                     attention_mask=attention_mask,
#                     position_ids=position_ids,
#                     past_key_value=past_key_value,
#                     output_attentions=output_attentions,
#                     use_cache=use_cache,
#                 )

#             hidden_states = layer_outputs[0]

#             if use_cache:
#                 next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

#         if use_cache:
#             return hidden_states,next_decoder_cache

#         return hidden_states

#     @torch.no_grad()
#     def generate(self,hidden_states,input_ids,head,max_length=4,use_cache=False):
#         return_input_ids=copy.deepcopy(input_ids[0].tolist())
#         input_ids=input_ids[:,1:]

#         #input_ids=input_ids.to(hidden_states.device)
#         if use_cache:
#             past_key_values=None
#             for i in range(max_length):
#                 if past_key_values!=None:
#                     out_hidden,past_key_values = self(out_hidden[:, -1:], input_ids=torch.tensor([[token]]).to(input_ids.device),past_key_values=past_key_values,use_cache=True)
#                 else:
#                     out_hidden, past_key_values = self(hidden_states, input_ids=input_ids,use_cache=True)
#                 last_hidden = out_hidden[:, -1]
#                 last_headout = head(last_hidden)
#                 token = torch.argmax(last_headout)
#                 #input_ids = torch.cat((input_ids, torch.tensor([[token]]).to(input_ids.device)), dim=1)
#                 return_input_ids.append(token.item())
#                 if token == 2:
#                     break
#                 #hidden_states = torch.cat((hidden_states, out_hidden[:, -1:]), dim=1)
#         else:
#             for i in range(max_length):
#                 out_hidden=self(hidden_states,input_ids=input_ids)
#                 last_hidden = out_hidden[:, -1]
#                 last_headout = head(last_hidden)
#                 token = torch.argmax(last_headout)
#                 return_input_ids.append(token.item())
#                 input_ids = torch.cat((input_ids, torch.tensor([[token]]).to(input_ids.device)), dim=1)
#                 if token==2:
#                     break
#                 hidden_states = torch.cat((hidden_states, out_hidden[:, -1:]), dim=1)

#         return return_input_ids

#     @torch.no_grad()
#     def repeat_kv(self,kv,numr):
#         newkv=[]
#         for i in kv:
#             newkv.append((i[0].repeat(numr,1,1,1),i[1].repeat(numr,1,1,1)))
#         return tuple(newkv)

#     @torch.no_grad()
#     def reduce_kv(self,kv,numr):
#         newkv=[]
#         for i in kv:
#             newkv.append((i[0][:numr],i[1][:numr]))
#         return tuple(newkv)


#     def reset_kv(self):
#         self.stable_kv=None

#     @torch.no_grad()
#     def topK_genrate_batch(self,hidden_states,input_ids,head,max_length=4,use_cache=True):
#         #input_ids = torch.tensor([state[1:]])
#         input_ids = input_ids[:, 1:]
#         input_ids = input_ids.to(hidden_states.device)
#         sslogits=[]
#         self.reset()
#         if use_cache:

#             out_hidden, past_key_values = self(hidden_states, input_ids=input_ids,use_cache=True)
#             last_hidden = out_hidden[:, -1]
#             last_headout = head(last_hidden)
#             sslogits.append(last_headout)
#             topk_index = torch.topk(last_headout, 3, dim=-1).indices

#             # hidden_states = torch.cat((hidden_states, out_hidden[:, -1:]), dim=1)
#             hidden_states = out_hidden[:, -1:]
#             hidden_states = hidden_states.repeat(3, 1, 1)
#             #input_ids = input_ids.repeat(3, 1)
#             input_ids = topk_index.t()
#             past_key_values = self.repeat_kv(past_key_values,3)
#             out_hidden,past_key_values = self(hidden_states, input_ids=input_ids,past_key_values=past_key_values,use_cache=True)
#             last_hidden = out_hidden[:, -1]
#             last_headout = head(last_hidden)
#             sslogits.append(last_headout)

#             hidden_states = out_hidden[0:1, -1:]
#             #input_ids = input_ids[:1]
#             topk_index = torch.topk(last_headout[:1], 3, dim=-1).indices
#             #hidden_states = torch.cat((hidden_states, out_hidden[0:1, -1:]), dim=1)
#             hidden_states = hidden_states.repeat(3, 1, 1)
#             #input_ids = input_ids.repeat(3, 1)
#             input_ids = topk_index.t()
#             out_hidden,past_key_values = self(hidden_states, input_ids=input_ids,past_key_values=past_key_values,use_cache=True)
#             last_hidden = out_hidden[:, -1]
#             last_headout = head(last_hidden)
#             sslogits.append(last_headout)

#             #hidden_states = hidden_states[:1]
#             #input_ids = input_ids[:1]
#             topk_index = torch.topk(last_headout[:1], 3, dim=-1).indices
#             hidden_states = out_hidden[0:1, -1:]
#             input_ids = topk_index[:, :1]
#             past_key_values=self.reduce_kv(past_key_values,1)
#             out_hidden,past_key_values = self(hidden_states, input_ids=input_ids,past_key_values=past_key_values,use_cache=True)
#             last_hidden = out_hidden[:, -1]
#             last_headout = head(last_hidden)
#             sslogits.append(last_headout)
#         else:
#             out_hidden = self(hidden_states, input_ids=input_ids)
#             last_hidden = out_hidden[:, -1]
#             last_headout = head(last_hidden)
#             sslogits.append(last_headout)
#             topk_index=torch.topk(last_headout, 3, dim=-1).indices

#             hidden_states = torch.cat((hidden_states, out_hidden[:, -1:]), dim=1)
#             hidden_states=hidden_states.repeat(3,1,1)
#             input_ids=input_ids.repeat(3,1)
#             input_ids=torch.cat((input_ids,topk_index.t()),dim=-1)
#             out_hidden = self(hidden_states, input_ids=input_ids)
#             last_hidden = out_hidden[:, -1]
#             last_headout = head(last_hidden)
#             sslogits.append(last_headout)

#             hidden_states=hidden_states[:1]
#             input_ids=input_ids[:1]
#             topk_index = torch.topk(last_headout[:1], 3, dim=-1).indices
#             hidden_states = torch.cat((hidden_states, out_hidden[0:1, -1:]), dim=1)
#             hidden_states = hidden_states.repeat(3, 1, 1)
#             input_ids = input_ids.repeat(3, 1)
#             input_ids = torch.cat((input_ids, topk_index.t()), dim=-1)
#             out_hidden = self(hidden_states, input_ids=input_ids)
#             last_hidden = out_hidden[:, -1]
#             last_headout = head(last_hidden)
#             sslogits.append(last_headout)

#             hidden_states = hidden_states[:1]
#             input_ids = input_ids[:1]
#             topk_index = torch.topk(last_headout[:1], 3, dim=-1).indices
#             hidden_states = torch.cat((hidden_states, out_hidden[0:1, -1:]), dim=1)
#             input_ids = torch.cat((input_ids, topk_index[:,:1]), dim=-1)
#             out_hidden = self(hidden_states, input_ids=input_ids)
#             last_hidden = out_hidden[:, -1]
#             last_headout = head(last_hidden)
#             sslogits.append(last_headout)

#         return torch.cat(sslogits)

#     @torch.no_grad()
#     def repeat_hidden(self,hidden_state,repeat_num):
#         new_hidden=[]
#         for id,i in enumerate(repeat_num):
#             new_hidden.append(hidden_state[:,id:id+1].repeat(1,i,1))
#         return torch.cat(new_hidden,dim=1)

#     # @torch.no_grad()
#     # def sample(self,tensor,k=1,replacement=True):
#     #     probabilities = torch.nn.functional.softmax(tensor, dim=1)
#     #     sampled_indices = torch.multinomial(probabilities, k,replacement=replacement)
#     #     sampled_probs = torch.gather(probabilities, 1, sampled_indices)
#     #
#     #     return  sampled_indices,sampled_probs

#     def sample(self,logits, logits_processor,k=1, replacement=False):
#         logits = logits_processor(None, logits)
#         probabilities = torch.nn.functional.softmax(logits, dim=1)
#         sampled_indices = torch.multinomial(probabilities, k, replacement=False)
#         sampled_probs = torch.gather(probabilities, 1, sampled_indices)

#         cumulative_sum = torch.cumsum(sampled_probs, dim=1)
#         cumulative_sum = torch.cat(
#             (torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device), cumulative_sum[:, :-1]), dim=-1)

#         sampled_probs = sampled_probs / (1 - cumulative_sum)
#         sampled_probs[torch.isinf(sampled_probs)] = -1
#         sampled_probs[torch.isnan(sampled_probs)] = -1

#         sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)

#         return sampled_indices, sampled_probs,probabilities

#         # if replacement:
#         #     sampled_indices = torch.multinomial(probabilities, k, replacement=True)
#         #     sampled_probs = torch.gather(probabilities, 1, sampled_indices)
#         #     return sampled_indices, sampled_probs
#         # else:
#         #     sampled_indices = torch.multinomial(probabilities, k, replacement=False)
#         #     sampled_probs = torch.gather(probabilities, 1, sampled_indices)
#         #
#         #     cumulative_sum = torch.cumsum(sampled_probs, dim=1)
#         #     cumulative_sum = torch.cat((torch.zeros(cumulative_sum.shape[0],1, device=cumulative_sum.device), cumulative_sum[:, :-1]),dim=-1)
#         #
#         #     sampled_probs=sampled_probs/(1-cumulative_sum)
#         #     sampled_probs[torch.isinf(sampled_probs)] = -1
#         #     sampled_probs[torch.isnan(sampled_probs)] = -1
#         #
#         #     sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)
#         #
#         #     # has_nan = torch.isnan(sampled_probs).any()
#         #     # if has_nan:
#         #     #     print(1)
#         #
#         #     # sampled_probs_list=sampled_probs[0].tolist()
#         #     # sum_list=[1-sum(sampled_probs_list[:i]) for i in range(len(sampled_probs_list))]
#         #     # for i in range(len(sampled_probs_list)):
#         #     #     a=sampled_probs_list[i]/(sum_list[i])
#         #     #     if sum_list[i]==0:
#         #     #         sampled_probs_list[i]=1.0
#         #     #     else:
#         #     #         sampled_probs_list[i]=sampled_probs_list[i]/(sum_list[i])
#         #     # sampled_probs=torch.tensor([sampled_probs_list],device=sampled_probs.device)
#         #
#         #
#         #
#         #     return sampled_indices, sampled_probs

#     @torch.no_grad()
#     def topK_genrate(self, hidden_states, input_ids, head, logits_processor,max_length=4, use_cache=True):
#         # test_=input_ids
#         # input_ids = torch.tensor([state[1:]])
#         input_ids = input_ids[:, 1:]
#         input_ids = input_ids.to(hidden_states.device)
#         ss_token,ss_prob,ss_op = [],[],[]
#         len_posi=input_ids.shape[1]
#         self.reset()
#         if use_cache:


#             if hasattr(self, "stable_kv") and self.stable_kv is not None:
#                 kv_len=self.stable_kv[0][0].shape[2]
#                 out_hidden, past_key_values = self(hidden_states[:,kv_len:], input_ids=input_ids[:,kv_len:], past_key_values=self.stable_kv,use_cache=True)
#             else:
#                 out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
#             self.stable_kv=past_key_values
#             last_hidden = out_hidden[:, -1]
#             if not self.diff_device:
#                 last_headout = head(last_hidden)
#             else:
#                 if hasattr(self, "layer_device"):
#                     last_headout = head(last_hidden)
#                     last_headout=last_headout.to(self.layer_device)
#                 else:
#                     last_headout=F.linear(last_hidden,self.headweight)



#             for i in range(len(self.tree_buffer['tree_indices'])):
#                 if logits_processor is not None:
#                     topk_index,topk_prob,op=self.sample(last_headout,logits_processor,k=top_k,)
#                 else:
#                     topk_index,topk_prob = torch.topk(last_headout, top_k, dim=-1).indices,torch.topk(last_headout, top_k, dim=-1).values
#                     op=None

#                 ss_token.append(topk_index)
#                 ss_prob.append(topk_prob)
#                 ss_op.append(op)
#                 #topk_index = torch.topk(last_headout, top_k, dim=-1).indices
#                 topk_index = topk_index.view(-1)
#                 select_index=topk_index[self.tree_buffer['tree_indices'][i]]
#                 #len_sq=select_index.shape[0]
#                 input_ids=select_index[None,:]
#                 if i==0:
#                     hidden_states = out_hidden[:, -1:]
#                 else:
#                     hidden_states=out_hidden
#                 hidden_states=self.repeat_hidden(hidden_states,self.tree_buffer["repeat_nums"][i])
#                 #hidden_states = hidden_states.repeat(1,len_sq,1)
#                 self.tree_mask=self.tree_buffer['attn_mask'][i]
#                 position_ids=len_posi+self.tree_buffer["position_ids"][i]
#                 out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, past_key_values=past_key_values,
#                                                    position_ids=position_ids,use_cache=True)
#                 len_posi += 1

#                 if not self.diff_device:
#                     last_headout = head(out_hidden[0])
#                 else:
#                     if hasattr(self, "layer_device"):
#                         last_headout = head(out_hidden[0])
#                         last_headout = last_headout.to(self.layer_device)
#                     else:
#                         last_headout = F.linear(out_hidden[0], self.headweight)
#                 #last_headout = head(out_hidden[0])
#                 #sslogits.append(last_headout)
#                 #print(select_index)

#             if logits_processor is not None:
#                 topk_index,topk_prob,op=self.sample(last_headout,logits_processor,k=top_k,)
#             else:
#                 topk_index, topk_prob = torch.topk(last_headout, top_k, dim=-1).indices, torch.topk(last_headout, top_k,
#                                                                                                     dim=-1).values
#                 op=None
#             ss_token.append(topk_index)
#             ss_prob.append(topk_prob)
#             ss_op.append(op)

#         else:
#             # TODO
#             pass

#         return (torch.cat(ss_token),torch.cat(ss_prob),ss_op)




#     @torch.no_grad()
#     def acc(self,data,head,max_length=5):
#         hidden_states=data["hidden_states"]
#         input_ids=data["input_ids"]
#         #attention_mask=data["attention_mask"]
#         loss_mask=data["loss_mask"]
#         sample_mask=data["sample_mask"]
#         target=data["target"]
#         total=[0 for _ in range(max_length)]
#         correct=[0 for _ in range(max_length)]
#         bs,sl=hidden_states.shape[0],hidden_states.shape[1]
#         target_headout = head(target)
#         hidden_states_headout=head(hidden_states)

#         for i in range(bs):
#             for j in range(sl):
#                 if loss_mask[i,j]==0:
#                     continue
#                 single_hidden_states=hidden_states[i,:j]
#                 single_input_ids=input_ids[i,:j]


#                 single_hidden_states = single_hidden_states[None, :, :]
#                 single_input_ids = single_input_ids[None, :]
#                 for k in range(max_length):
#                     tmp_in_target_headout = hidden_states_headout[i,single_hidden_states.shape[1]-1]
#                     tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1]-1]
#                     target_in_token = torch.argmax(tmp_in_target_headout)
#                     target_out_token = torch.argmax(tmp_out_target_headout)
#                     tmp_token=input_ids[i,single_hidden_states.shape[1]-1]
#                     tmp_sample_mask=sample_mask[i,single_hidden_states.shape[1]-1]
#                     if not (target_in_token==tmp_token):
#                         break
#                     out_hidden = self(single_hidden_states, input_ids=single_input_ids)
#                     last_hidden = out_hidden[:, -1]
#                     last_headout = head(last_hidden)
#                     token = torch.argmax(last_headout)
#                     total[k] += 1
#                     if token==target_out_token:
#                         correct[k]+=1
#                     else:
#                         for kk in range(k,max_length):
#                             total[kk]+=1
#                         break

#                     single_hidden_states=torch.cat((single_hidden_states,out_hidden[:,-1:]),dim=1)
#                     single_input_ids = torch.cat((single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)), dim=1)


#         acc=[correct[i]/total[i] for i in range(len(correct))]
#         return acc





# class Vhead(nn.Module):
#     def __init__(self,ins=6566,outs=32000):
#         super().__init__()
#         self.fc = nn.Linear(ins,outs,bias=False)
#     def forward(self,x):
#         return self.fc(x)



import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__=="__main__":
    config = AutoConfig.from_pretrained("/media/data3/Storage/mistralai_Mistral-7B-v0.1")
    model = EagleMistralModel(config,load_emb=True,path="/media/data3/Storage/mistralai_Mistral-7B-v0.1")
    print(model)
