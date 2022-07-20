import torch

from transformers import ResNetConfig, ResNetForImageClassification


config = ResNetConfig(replace_stride_with_dilation=[True, True, True])

model = ResNetForImageClassification(config)

pixel_values = torch.randn(1, 3, 224, 224)

outputs = model(pixel_values, output_hidden_states=True)

print(outputs.hidden_states[-1].shape)