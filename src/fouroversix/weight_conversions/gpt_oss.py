from fouroversix.quantize import QuantizedTensor
from fouroversix import DataType, ScaleRule
from .conversions import WeightConversions

import torch
from transformers import ConversionOps, WeightConverter, GptOssConfig

class FourOverSixGptOssDeserialize(ConversionOps):
    def __init__(self, hf_quantizer, quantized_tensor_cls=None, dtype=None, scale_rule=None):
        self.hf_quantizer = hf_quantizer
        self.quantized_tensor_cls = quantized_tensor_cls
        self.dtype = dtype
        self.scale_rule = scale_rule

    def convert(
        self,
        input_dict: torch.Tensor,
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys: list[str] | None = None,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:

        prefix = ""
        if ".down_proj_blocks" in input_dict:
            weight = input_dict[".down_proj_blocks"][0]
            scales = input_dict[".down_proj_scales"][0]
            prefix = "down"
        elif ".gate_up_proj_blocks" in input_dict:
            weight = input_dict[".gate_up_proj_blocks"][0]
            scales = input_dict[".gate_up_proj_scales"][0]
            prefix = "gate_up"

        num_experts = weight.shape[0]
        hidden_size = weight.shape[1]
        weight = weight.reshape((num_experts, hidden_size, -1))

        dequantized_proj = []
        for e in range(num_experts):
            weight_uint8 = weight[e].to(torch.uint8)
            
            quantized_tensor = self.quantized_tensor_cls(
                values=weight_uint8, 
                scale_factors=scales[e].view(torch.float8_e8m0fnu),
                amax=torch.ones(
                    (weight[e].shape[1]),
                    device=weight[e].device,
                    dtype=torch.float32,
                ),
                dtype=self.dtype,
                original_shape=(
                    weight[e].shape[0],
                    weight[e].shape[1] * 2,
                ),
                scale_rule=self.scale_rule,
            )

            dequantized = quantized_tensor.dequantize()
            dequantized_proj.append(dequantized)

        dequantized_weight = torch.stack(dequantized_proj, dim=0)
        
        return {f"{prefix}_proj": [dequantized_weight]}

@WeightConversions.register(GptOssConfig)
class GptOssWeightConverter:

    @classmethod
    def get_weight_conversions(cls):

        return [
            WeightConverter(
                source_patterns=[".gate_up_proj_blocks", ".gate_up_proj_scales"],
                target_patterns=".gate_up_proj",
                operations=[FourOverSixGptOssDeserialize(cls, quantized_tensor_cls=QuantizedTensor, dtype=DataType.mxfp4, scale_rule=ScaleRule.static_6)],
            ),
            WeightConverter(
                source_patterns=[".down_proj_blocks", ".down_proj_scales"],
                target_patterns=".down_proj",
                operations=[FourOverSixGptOssDeserialize(cls, quantized_tensor_cls=QuantizedTensor, dtype=DataType.mxfp4, scale_rule=ScaleRule.static_6)],
            ),
        ]