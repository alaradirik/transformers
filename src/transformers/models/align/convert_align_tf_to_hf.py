# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert ALIGN checkpoints from the original repository."""

import argparse
import json
import os

import numpy as np
import PIL
import requests
import torch
from PIL import Image

from transformers import ALIGNModel
from transformers import BertConfig, BertTokenizer
from transformers import EfficientNetConfig, EfficientNetImageProcessor

from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_align_config(model_name):
    vision_config = EfficientNetConfig()
    text_config = BertConfig()
    config = ALIGNConfig.from_text_vision_configs(text_config=text_config, vision_config=vision_config)
    return config


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


def get_processor():
    image_processor = EfficientNetImageProcessor(
        do_center_crop=True,
        rescale_offset=True,
        do_normalize=False,
        include_top=False
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.model_max_length = 64
    processor = ALIGNProcessor(image_processor=image_processor, tokenizer=tokenizer)
    return preprocessor


# here we list all keys to be renamed (original name on the left, our name on the right)
def rename_keys(original_param_names):
    block_names = [v.split("_")[0].split("block")[1] for v in original_param_names if v.startswith("block")]
    block_names = sorted(list(set(block_names)))
    num_blocks = len(block_names)
    block_name_mapping = {b: str(i) for b, i in zip(block_names, range(num_blocks))}

    rename_keys = []
    rename_keys.append(("stem_conv/kernel:0", "embeddings.convolution.weight"))
    rename_keys.append(("stem_bn/gamma:0", "embeddings.batchnorm.weight"))
    rename_keys.append(("stem_bn/beta:0", "embeddings.batchnorm.bias"))
    rename_keys.append(("stem_bn/moving_mean:0", "embeddings.batchnorm.running_mean"))
    rename_keys.append(("stem_bn/moving_variance:0", "embeddings.batchnorm.running_var"))

    for b in block_names:
        hf_b = block_name_mapping[b]
        rename_keys.append((f"block{b}_expand_conv/kernel:0", f"encoder.blocks.{hf_b}.expansion.expand_conv.weight"))
        rename_keys.append((f"block{b}_expand_bn/gamma:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.weight"))
        rename_keys.append((f"block{b}_expand_bn/beta:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.bias"))
        rename_keys.append(
            (f"block{b}_expand_bn/moving_mean:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.running_mean")
        )
        rename_keys.append(
            (f"block{b}_expand_bn/moving_variance:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.running_var")
        )
        rename_keys.append(
            (f"block{b}_dwconv/depthwise_kernel:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_conv.weight")
        )
        rename_keys.append((f"block{b}_bn/gamma:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.weight"))
        rename_keys.append((f"block{b}_bn/beta:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.bias"))
        rename_keys.append(
            (f"block{b}_bn/moving_mean:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.running_mean")
        )
        rename_keys.append(
            (f"block{b}_bn/moving_variance:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.running_var")
        )

        rename_keys.append((f"block{b}_se_reduce/kernel:0", f"encoder.blocks.{hf_b}.squeeze_excite.reduce.weight"))
        rename_keys.append((f"block{b}_se_reduce/bias:0", f"encoder.blocks.{hf_b}.squeeze_excite.reduce.bias"))
        rename_keys.append((f"block{b}_se_expand/kernel:0", f"encoder.blocks.{hf_b}.squeeze_excite.expand.weight"))
        rename_keys.append((f"block{b}_se_expand/bias:0", f"encoder.blocks.{hf_b}.squeeze_excite.expand.bias"))
        rename_keys.append(
            (f"block{b}_project_conv/kernel:0", f"encoder.blocks.{hf_b}.projection.project_conv.weight")
        )
        rename_keys.append((f"block{b}_project_bn/gamma:0", f"encoder.blocks.{hf_b}.projection.project_bn.weight"))
        rename_keys.append((f"block{b}_project_bn/beta:0", f"encoder.blocks.{hf_b}.projection.project_bn.bias"))
        rename_keys.append(
            (f"block{b}_project_bn/moving_mean:0", f"encoder.blocks.{hf_b}.projection.project_bn.running_mean")
        )
        rename_keys.append(
            (f"block{b}_project_bn/moving_variance:0", f"encoder.blocks.{hf_b}.projection.project_bn.running_var")
        )

    rename_keys.append(("top_conv/kernel:0", "encoder.top_conv.weight"))
    rename_keys.append(("top_bn/gamma:0", "encoder.top_bn.weight"))
    rename_keys.append(("top_bn/beta:0", "encoder.top_bn.bias"))
    rename_keys.append(("top_bn/moving_mean:0", "encoder.top_bn.running_mean"))
    rename_keys.append(("top_bn/moving_variance:0", "encoder.top_bn.running_var"))

    key_mapping = {}
    for item in rename_keys:
        if item[0] in original_param_names:
            key_mapping[item[0]] = "efficientnet." + item[1]

    key_mapping["predictions/kernel:0"] = "classifier.weight"
    key_mapping["predictions/bias:0"] = "classifier.bias"
    return key_mapping


def replace_params(hf_params, tf_params, key_mapping):
    for key, value in tf_params.items():
        if "normalization" in key:
            continue

        hf_key = key_mapping[key]
        if "_conv" in key and "kernel" in key:
            new_hf_value = torch.from_numpy(value).permute(3, 2, 0, 1)
        elif "depthwise_kernel" in key:
            new_hf_value = torch.from_numpy(value).permute(2, 3, 0, 1)
        elif "kernel" in key:
            new_hf_value = torch.from_numpy(np.transpose(value))
        else:
            new_hf_value = torch.from_numpy(value)

        # Replace HF parameters with original TF model parameters
        assert hf_params[hf_key].shape == new_hf_value.shape
        hf_params[hf_key].copy_(new_hf_value)


@torch.no_grad()
def convert_align_checkpoint(checkpoint_path, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    Copy/paste/tweak model's weights to our ALIGN structure.
    """
    # Load original model
    original_model = model_classes[model_name](
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )

    tf_params = original_model.trainable_variables
    tf_non_train_params = original_model.non_trainable_variables
    tf_params = {param.name: param.numpy() for param in tf_params}
    for param in tf_non_train_params:
        tf_params[param.name] = param.numpy()
    tf_param_names = [k for k in tf_params.keys()]

    # Load HuggingFace model
    config = get_efficientnet_config(model_name)
    hf_model = EfficientNetForImageClassification(config).eval()
    hf_params = hf_model.state_dict()

    # Create src-to-dst parameter name mapping dictionary
    print("Converting parameters...")
    key_mapping = rename_keys(tf_param_names)
    replace_params(hf_params, tf_params, key_mapping)

    # Initialize preprocessor and preprocess input image
    processor = get_processor()
    inputs = processor(images=prepare_img(), texts="A picture of a cat", return_tensors="pt")

    # HF model inference
    hf_model.eval()
    with torch.no_grad():
        outputs = hf_model(**inputs)
    hf_logits = outputs.logits.detach().numpy()

    # Original model inference
    original_model.trainable = False
    image_size = CONFIG_MAP[model_name]["image_size"]
    img = prepare_img().resize((image_size, image_size), resample=PIL.Image.NEAREST)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    original_logits = original_model.predict(x)

    # Check whether original and HF model outputs match  -> np.allclose
    assert np.allclose(original_logits, hf_logits, atol=1e-3), "The predicted logits are not the same."
    print("Model outputs match!")

    if save_model:
        # Create folder to save model
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)
        # Save converted model and feature extractor
        hf_model.save_pretrained(pytorch_dump_folder_path)
        preprocessor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # Push model and feature extractor to hub
        print(f"Pushing converted ALIGN to the hub...")
        preprocessor.push_to_hub("align-base")
        hf_model.push_to_hub("align-base")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_path",
        default="../align/pretrained/final-model",
        type=str,
        help="Path to the pretrained TF ALIGN checkpoint.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="hf_model",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and feature extractor to the hub")

    args = parser.parse_args()
    convert_align_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub)
