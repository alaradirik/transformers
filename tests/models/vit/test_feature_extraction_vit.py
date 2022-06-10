# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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


import unittest
import itertools

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_feature_extraction_common import FeatureExtractionSavingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import ViTFeatureExtractor


class ViTFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        size=18,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.size = size
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_normalize = True
        self.do_resize = True
        self.settings = [("do_normalize", "do_resize")] \
        + [(v[0], v[1]) for v in itertools.product([True, False], repeat=2)]

    def prepare_feat_extract_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
        }
    
    def prepare_feat_extract_opts(self):
        return self.settings


@require_torch
@require_vision
class ViTFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):

    feature_extraction_class = ViTFeatureExtractor if is_vision_available() else None

    def setUp(self):
        self.feature_extract_tester = ViTFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feature_extract_tester.prepare_feat_extract_dict()

    def feat_extract_opts(self):
        return self.feature_extract_tester.prepare_feat_extract_opts()

    def test_feat_extract_properties(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feature_extractor, "image_mean"))
        self.assertTrue(hasattr(feature_extractor, "image_std"))
        self.assertTrue(hasattr(feature_extractor, "do_normalize"))
        self.assertTrue(hasattr(feature_extractor, "do_resize"))
        self.assertTrue(hasattr(feature_extractor, "size"))

    def test_batch_feature(self):
        pass

    def test_call_pil(self):
        # Create random PIL images
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Setting attributes to test 
        attibutes = self.feat_extract_opts[0]

        # Test feature_extractor with different settings
        for setting in self.feat_extract_opts[1:]:
            feat_extract_dict = self.feat_extract_dict.copy()
            feat_extract_dict.update(dict(zip(attibutes, setting)))

            # Initialize feature_extractor
            feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)

            # Test not batched input
            encoded_images = feature_extractor(image_inputs[0], return_tensors="pt").pixel_values
            self.assertEqual(
                encoded_images.shape,
                (
                    1,
                    self.feature_extract_tester.num_channels,
                    self.feature_extract_tester.size,
                    self.feature_extract_tester.size,
                ),
            )

            # Test batched
            encoded_images = feature_extractor(image_inputs, return_tensors="pt").pixel_values
            self.assertEqual(
                encoded_images.shape,
                (
                    self.feature_extract_tester.batch_size,
                    self.feature_extract_tester.num_channels,
                    self.feature_extract_tester.size,
                    self.feature_extract_tester.size,
                ),
            )
            del feature_extractor

    def test_call_numpy(self):
        # Create random numpy tensors
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Setting attributes to test 
        attibutes = self.feat_extract_opts[0]

        # Test feature_extractor with different settings
        for setting in self.feat_extract_opts[1:]:
            feat_extract_dict = self.feat_extract_dict.copy()
            feat_extract_dict.update(dict(zip(attibutes, setting)))

            # Initialize feature_extractor
            feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)

            # Test not batched input
            encoded_images = feature_extractor(image_inputs[0], return_tensors="pt").pixel_values
            self.assertEqual(
                encoded_images.shape,
                (
                    1,
                    self.feature_extract_tester.num_channels,
                    self.feature_extract_tester.size,
                    self.feature_extract_tester.size,
                ),
            )

            # Test batched
            encoded_images = feature_extractor(image_inputs, return_tensors="pt").pixel_values
            self.assertEqual(
                encoded_images.shape,
                (
                    self.feature_extract_tester.batch_size,
                    self.feature_extract_tester.num_channels,
                    self.feature_extract_tester.size,
                    self.feature_extract_tester.size,
                ),
            )
            del feature_extractor

    def test_call_pytorch(self):
        # Create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test feature_extractor with different settings
        for setting in self.feat_extract_opts[1:]:
            feat_extract_dict = self.feat_extract_dict.copy()
            feat_extract_dict.update(dict(zip(attibutes, setting)))

            # Initialize feature_extractor
            feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)

            # Test not batched input
            encoded_images = feature_extractor(image_inputs[0], return_tensors="pt").pixel_values
            self.assertEqual(
                encoded_images.shape,
                (
                    1,
                    self.feature_extract_tester.num_channels,
                    self.feature_extract_tester.size,
                    self.feature_extract_tester.size,
                ),
            )

            # Test batched
            encoded_images = feature_extractor(image_inputs, return_tensors="pt").pixel_values
            self.assertEqual(
                encoded_images.shape,
                (
                    self.feature_extract_tester.batch_size,
                    self.feature_extract_tester.num_channels,
                    self.feature_extract_tester.size,
                    self.feature_extract_tester.size,
                ),
            )
            del feature_extractor
