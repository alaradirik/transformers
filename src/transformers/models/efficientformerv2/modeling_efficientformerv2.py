# coding=utf-8
# Copyright 2022 Snapchat Research and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch EfficientFormerV2 model."""

import math
import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_efficientformerv2 import EfficientFormerV2Config


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "EfficientFormerV2Config"
_FEAT_EXTRACTOR_FOR_DOC = "EfficientFormerV2ImageProcessor"

# Base docstring
_CHECKPOINT_FOR_DOC = "efficientformerv2-l"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "snap-research/efficientformerv2-l"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"


EFFICIENTFORMERV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "huggingface/efficientformerv2-l",
    # See all EfficientFormerV2 models at https://huggingface.co/models?filter=efficientformerv2
]


class EfficientFormerV2ConvStem(nn.Module):
    def __init__(self, config: EfficientFormerV2Config, out_channels: int):
        super().__init__()

        self.convolution1 = nn.Conv2d(config.num_channels, out_channels // 2, kernel_size=3, stride=2, padding=1)
        self.batchnorm_before = nn.BatchNorm2d(out_channels // 2)
        self.convolution2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1)
        self.batchnorm_after = nn.BatchNorm2d(out_channels)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.batchnorm_before(self.convolution1(pixel_values))
        features = self.activation(features)
        features = self.batchnorm_after(self.convolution2(features))
        features = self.activation(features)

        return features


class EfficientFormerV2LGQuery(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.pool = nn.AvgPool2d(1, 2, 0)
        self.local = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim))
        self.proj = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1), nn.BatchNorm2d(out_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        local_query = self.local(hidden_states)
        pool_query = self.pool(hidden_states)
        query = local_query + pool_query
        query = self.proj(query)

        return query


class EfficientFormerV2Attention4DDownsample(torch.nn.Module):
    def __init__(self, config: EfficientFormerV2Config, dim: int, out_dim: int, resolution: int):
        super().__init__()

        key_dim = config.downsample_key_dim
        self.num_heads = config.num_attention_heads
        self.scale = key_dim ** -0.5
        self.resolution = resolution
        self.dim = int(config.attention_ratio * key_dim)
        self.dim_head = int(config.attention_ratio * key_dim) * num_heads
        self.out_dim = out_dim if out_dim is not None else dim
        self.resolution2 = math.ceil(self.resolution / 2)

        self.query = EfficientFormerV2LGQuery(dim, self.num_heads * key_dim)
        self.key = nn.Sequential(nn.Conv2d(dim, self.num_heads * key_dim, 1), nn.BatchNorm2d(self.num_heads * key_dim))
        self.value = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.dim, 1), nn.BatchNorm2d(self.num_heads * self.dim))
        self.value_local = nn.Sequential(
            nn.Conv2d(
                self.num_heads * self.dim, 
                self.num_heads * self.dim, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                groups=self.num_heads * self.dim,
            ),
            nn.BatchNorm2d(self.num_heads * self.dim), 
        )
        self.projection = nn.Sequential(
            ACT2FN[config.hidden_act],
            nn.Conv2d(self.dim_head, self.out_dim, 1),
            nn.BatchNorm2d(self.out_dim),
        )

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        points_ = list(itertools.product(range(self.resolution2), range(self.resolution2)))
        offset_ratio = math.ceil(self.resolution / self.resolution2)

        idxs = []
        attention_offsets = {}
        for p1 in points_:
            for p2 in points:
                offset = (abs(p1[0] * offset_ratio - p2[0] / 2), abs(p1[1] * offset_ratio - p2[1] / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(len(points_), len(points)))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels = hidden_states.shape[:2]

        query = self.query(hidden_states).flatten(2)
        query = query.reshape(batch_size, self.num_heads, -1, self.resolution2**2).permute(0, 1, 3, 2)
        key = self.key(hidden_states).flatten(2)
        key = key.reshape(batch_size, self.num_heads, -1, self.resolution**2).permute(0, 1, 2, 3)
        value = self.value(hidden_states)
        value_local = self.value_local(value)

        value = value.flatten(2).reshape(batch_size, self.num_heads, -1, self.resolution**2).permute(0, 1, 3, 2)
        attention = (torch.matmul(query, key)) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        )
        attention = attention.softmax(dim=-1)

        hidden_states = (attention @ value).transpose(2, 3)
        hidden_states = hidden_states.reshape(batch_size, self.dim_head, self.resolution2, self.resolution2) + value_local
        hidden_states = self.projection(hidden_states)
        
        return hidden_states


class EfficientFormerV2Attention4D(torch.nn.Module):
    def __init__(self, config: EfficientFormerV2Config, dim: int, resolution: int, stride: int):
        super().__init__()
        key_dim = config.key_dim
        self.num_heads = config.num_attention_heads
        self.scale = config.key_dim ** -0.5

        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            self.stride_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),
                nn.BatchNorm2d(dim)
            )
            self.upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None

        self.out_dim = self.resolution ** 2
        self.dim = int(config.attention_ratio * key_dim)
        self.dim_head = int(config.attention_ratio * key_dim) * num_heads

        self.query = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1), nn.BatchNorm2d(self.num_heads * key_dim))
        self.key = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1), nn.BatchNorm2d(self.num_heads * key_dim))
        self.value = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.dim, 1), nn.BatchNorm2d(self.num_heads * self.dim))
        self.value_local = nn.Sequential(
            nn.Conv2d(self.num_heads * self.dim, self.num_heads * self.dim, kernel_size=3, stride=1, padding=1, groups=self.num_heads * self.dim),
            nn.BatchNorm2d(self.num_heads * self.dim)
        )
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        self.projection = nn.Sequential(ACT2FN[config.hidden_act], nn.Conv2d(self.dim_head, dim, 1), nn.BatchNorm2d(dim))

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(len(points), len(points)))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels = hidden_states.shape[:2]
        if self.stride_conv is not None:
            hidden_states = self.stride_conv(hidden_states)

        query = self.query(hidden_states).flatten(2).reshape(batch_size, self.num_heads, -1, self.out_dim).permute(0, 1, 3, 2)
        key = self.key(hidden_states).flatten(2).reshape(batch_size, self.num_heads, -1, self.out_dim).permute(0, 1, 2, 3)
        value = self.value(hidden_states)
        value_local = self.value_local(value)
        value = value.flatten(2).reshape(batch_size, self.num_heads, -1, self.out_dim).permute(0, 1, 3, 2)

        attention = (torch.matmul(query, key)) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        )
        attention = self.talking_head1(attention)
        attention = attention.softmax(dim=-1)
        attention = self.talking_head2(attention)

        hidden_states = (attention @ value).transpose(2, 3)
        hidden_states = hidden_states.reshape(batch_size, self.dim_head, self.resolution, self.resolution) + value_local

        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)
        hidden_states = self.projection(hidden_states)

        return hidden_states

        
class EfficientFormerV2PatchEmbeddings(nn.Module):
    """
    This class performs downsampling between two stages. For the input tensor with the shape [batch_size, num_channels,
    height, width] it produces output tensor with the shape [batch_size, num_channels, height/stride, width/stride]
    """

    def __init__(self, config: EfficientFormerV2Config, in_dim: int, embed_dim: int, resolution: int, asub: bool = False):
        super().__init__()

        self.asub = asub
        self.num_channels = in_dim

        if self.asub:
            self.attention = EfficientFormerV2Attention4DDownsample(config=config, dim=in_dim, out_dim=embed_dim, resolution=resolution)
            self.convolution = nn.Conv2d(
                num_channels, 
                embed_dim, 
                kernel_size=config.downsample_patch_size,
                stride=config.downsample_stride,
                padding=config.downsample_pad,
            )
            self.batch_norm = nn.BatchNorm2d(embed_dim)
        else:
            self.projection = nn.Conv2d(
                num_channels,
                embed_dim,
                kernel_size=config.downsample_patch_size,
                stride=config.downsample_stride,
                padding=config.downsample_pad,
            )
            self.norm = nn.BatchNorm2d(embed_dim)
        

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        if self.asub:
            embeddings_ = self.convolution(pixel_values)
            embeddings_ = self.batch_norm(embeddings_)
            embeddings = self.attention(embeddings) + embeddings_
        else:
            embeddings = self.projection(pixel_values)
            embeddings = self.norm(embeddings)

        return embeddings


class EfficientFormerV2Pooling(nn.Module):
    def __init__(self, pool_size: int):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.pool(hidden_states) - hidden_states
        return output


# Copied from transformers.models.convnext.modeling_convnext.drop_path
def drop_path(input, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath
class EfficientFormerV2DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)


class EfficientFormerV2ConvMLP(nn.Module):
    def __init__(self, config: EfficientFormerV2Config, in_dim: int, hidden_dim: int):
        super().__init__()
        out_dim = in_dim
        self.convolution1 = nn.Conv2d(in_dim, hidden_dim, 1)
        self.activation = ACT2FN[config.hidden_act]
        self.convolution2 = nn.Conv2d(hidden_dim, out_dim, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.mid_convolution = nn.Conv2d(
            hidden_dim, 
            hidden_dim, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            groups=hidden_dim
        )
        self.mid_norm = nn.BatchNorm2d(hidden_dim)

        self.batchnorm_before = nn.BatchNorm2d(hidden_dim)
        self.batchnorm_after = nn.BatchNorm2d(out_dim)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.convolution1(hidden_state)
        hidden_state = self.batchnorm_before(hidden_state)
        hidden_state = self.activation(hidden_state)

        hidden_state = self.mid_convolution(hidden_state)
        hidden_state = self.mid_norm(hidden_state)
        hidden_state = self.activation(hidden_state)

        hidden_state = self.dropout(hidden_state)
        hidden_state = self.convolution2(hidden_state)
        hidden_state = self.batchnorm_after(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class EfficientFormerV2AttentionFFN(nn.Module):
    def __init__(
        self,  config: EfficientFormerV2Config, dim: int, drop_path_rate: float, resolution: int, stride: int = None):
        super().__init__()
        self.use_layer_scale = config.use_layer_scale
        self.token_mixer = EfficientFormerV2Attention4D(config=config, dim=dim, resolution=resolution, stride=stride)

        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        self.mlp = EfficientFormerV2ConvMLP(config=config, in_dim=dim, hidden_dim=mlp_hidden_dim)

        self.drop_path = EfficientFormerV2DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        if self.use_layer_scale:
            layer_scale_init_value = config.layer_scale_init_value
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_layer_scale:
            hidden_states = hidden_states + self.drop_path(self.layer_scale_1 * self.token_mixer(hidden_states))
            hidden_states = hidden_states + self.drop_path(self.layer_scale_2 * self.mlp(hidden_states))
        else:
            hidden_states = hidden_states + self.drop_path(self.token_mixer(hidden_states))
            hidden_states = hidden_states + self.drop_path(self.mlp(hidden_states))

        return hidden_states


class EfficientFormerV2FFN(nn.Module):
    def __init__(self, config: EfficientFormerV2Config, dim: int, drop_path_rate: float):
        super().__init__()
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        self.use_layer_scale = config.use_layer_scale
        self.mlp = EfficientFormerV2ConvMLP(config=config, in_dim=dim, hidden_dim=mlp_hidden_dim)
        self.drop_path = EfficientFormerV2DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
        if self.use_layer_scale:
            self.layer_scale_2 = nn.Parameter(config.layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_layer_scale:
            hidden_states = hidden_states + self.drop_path(self.layer_scale_2 * self.mlp(hidden_states))
        else:
            hidden_states = hidden_states + self.drop_path(self.mlp(hidden_states))
        return self.drop_path(self.layer_scale_2 * self.mlp(hidden_states))


class EfficientFormerV2Block(nn.Module):
    def __init__(self, config: EfficientFormerV2Config, dim: int, index: int, layers, pool_size=3, mlp_ratio=4., drop_path_rate=0., vit_num=1, resolution=7, e_ratios=None):
        super().__init__()
        blocks = []
        for block_idx in range(layers[index]):
            block_drop_path_rate = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
            mlp_ratio = e_ratios[str(index)][block_idx]

            if index >= 2 and block_idx > layers[index] - 1 - vit_num:
                stride = 2 if index == 2 else None
                blocks.append(EfficientFormerV2AttentionFFN(config, dim, drop_path=block_drop_path_rate, resolution=resolution, stride=stride))
            else:
                blocks.append(EfficientFormerV2FFN(config, dim, drop_path_rate=block_drop_path_rate))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        hidden_states = self.blocks(hidden_states)
        return hidden_states


class EfficientFormerV2Encoder(nn.Module):
    def __init__(self, config: EfficientFormerV2Config):
        super().__init__()
        self.config = config
        num_intermediate_stages = len(config.depths) - 1
        downsamples = [
            config.downsamples[i] or config.hidden_sizes[i] != config.hidden_sizes[i + 1]
            for i in range(num_intermediate_stages)
        ]
        asubs = [True if i >=2 else False for i in range(num_intermediate_stages)]

        intermediate_stages = []
        for i in range(num_intermediate_stages):
            intermediate_stages.append(
                EfficientFormerV2Block(
                    config=config, 
                    index=i, 
                    resolution=math.ceil(config.dim / (2 ** (i + 2)))
                )
            )
            if downsamples[i]:
                intermediate_stages.append(
                    EfficientFormerV2PatchEmbeddings(
                        config=config, 
                        num_channels=config.hidden_sizes[i], 
                        embed_dim=config.hidden_sizes[i + 1],
                        asub=asubs[i],
                        resolution=math.ceil(config.dim / (2 ** (i + 2)))
                    )
                )

        self.intermediate_stages = nn.ModuleList(intermediate_stages)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        for layer_module in self.intermediate_stages:
            hidden_states = layer_module(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        layer_output = self.last_stage(hidden_states, output_attentions=output_attentions)
        if output_attentions:
            all_self_attentions = all_self_attentions + layer_output[1:]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (layer_output[0],)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=layer_output[0],
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class EfficientFormerV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EfficientFormerV2Config
    base_model_prefix = "efficientformerv2"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


EFFICIENTFORMERV2_START_DOCSTRING = r"""
    This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) subclass. Use it as a
    regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

    Parameters:
        config ([`EfficientFormerV2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

EFFICIENTFORMERV2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`ViTFeatureExtractor`]. See
            [`ViTFeatureExtractor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare EfficientFormerV2 Model transformer outputting raw hidden-states without any specific head on top.",
    EFFICIENTFORMERV2_START_DOCSTRING,
)
class EfficientFormerV2Model(EfficientFormerV2PreTrainedModel):
    def __init__(self, config: EfficientFormerV2Config):
        super().__init__(config)
        self.config = config

        self.patch_embed = EfficientFormerV2ConvStem(config, config.hidden_sizes[0])
        self.encoder = EfficientFormerV2Encoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(EFFICIENTFORMERV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.patch_embed(pixel_values)
        encoder_outputs = self.encoder(
            embedding_output, output_attentions=output_attentions, output_hidden_states=output_hidden_states
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """
    EfficientFormerV2 Model transformer with an image classification head on top (a linear layer on top of the final
    hidden state of the [CLS] token) e.g. for ImageNet.
    """,
    EFFICIENTFORMERV2_START_DOCSTRING,
)
class EfficientFormerV2ForImageClassification(EfficientFormerV2PreTrainedModel):
    def __init__(self, config: EfficientFormerV2Config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.efficientformer = EfficientFormerV2Model(config)

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(EFFICIENTFORMERV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.efficientformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output.mean(-2))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class EfficientFormerV2ForImageClassificationWithTeacherOutput(ModelOutput):
    """
    Output type of [`EfficientFormerV2ForImageClassificationWithTeacher`].

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores as the average of the cls_logits and distillation logits.
        cls_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
            class token).
        distillation_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
            distillation token).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None
    cls_logits: torch.FloatTensor = None
    distillation_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@add_start_docstrings(
    """
    EfficientFormer Model transformer with image classification heads on top (a linear layer on top of the final hidden
    state of the [CLS] token and a linear layer on top of the final hidden state of the distillation token) e.g. for
    ImageNet.

    <Tip warning={true}>

           This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
           supported.

    </Tip>
    """,
    EFFICIENTFORMERV2_START_DOCSTRING,
)
class EfficientFormerV2ForImageClassificationWithTeacher(EfficientFormerV2PreTrainedModel):
    def __init__(self, config: EfficientFormerV2Config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.efficientformer = EfficientFormerV2Model(config)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        # Distillation head
        self.distillation_classifier = (
            nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(EFFICIENTFORMERV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=EfficientFormerV2ForImageClassificationWithTeacherOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, EfficientFormerV2ForImageClassificationWithTeacherOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.efficientformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        cls_logits = self.classifier(sequence_output.mean(-2))
        distillation_logits = self.distillation_classifier(sequence_output.mean(-2))

        # during inference, return the average of both classifier predictions
        logits = (cls_logits + distillation_logits) / 2

        if not return_dict:
            output = (logits, cls_logits, distillation_logits) + outputs[1:]
            return output

        return EfficientFormerV2ForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distillation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
