from typing import Any

import torch
from fouroversix.matmul import fp4_matmul
from fouroversix.model.config import ModuleQuantizationConfig
from fouroversix.model.quantize import QuantizedModule
from fouroversix.quantize import (
    QuantizationConfig,
    QuantizedTensor,
    quantize_to_fp4,
)
from fouroversix.utils import DataType
from torch import nn
from transformers import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP, GptOssTopKRouter, GptOssExperts


@QuantizedModule.register(GptOssMLP)
class FourOverSixGptOssMLP(nn.Module):
    """Drop-in replacement for MoE layer that uses FP4 quantization."""

    def __init__(
        self,
        module: GptOssMLP,
        config: ModuleQuantizationConfig,
    ) -> None:
        """
        Initialize the FourOverSixGptOssMLP layer.

        Args:
            module (GptOssMLP): The high-precision module that this quantized layer will
                replace.
            config (ModuleQuantizationConfig): The quantization configuration to use for
                the layer.
        """

        print("initializing mlp")
        super().__init__()

        self.config = config

        gpt_oss_config = GptOssConfig(
            num_local_experts=module.experts.num_experts,
            hidden_size=module.experts.hidden_size,
            intermediate_size=module.experts.intermediate_size,
            num_experts_per_token=module.router.top_k,
        )

        self.router = GptOssTopKRouter(gpt_oss_config)
        self.router.weight = module.router.weight
        self.router.bias = module.router.bias

        self.experts = FourOverSixGptOssExperts(
            GptOssExperts(gpt_oss_config),
            quantization_config=self.config,
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the FP4 MLP layer."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        _, router_scores, router_indices = self.router(hidden_states)
        hidden_states = self.experts(hidden_states, router_indices, router_scores)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_scores

    @property
    def high_precision_parameter_names(self) -> tuple[str, ...]:
        return self.experts.high_precision_parameter_names

    def get_quantized_parameters(
        self,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Get quantized parameters for the weight tensors.
        """
        print("get quantized parameters mlp")
        return {}

@QuantizedModule.register(GptOssExperts, replace_existing_modules_in_model=False)
class FourOverSixGptOssExperts(nn.Module):
    """Drop-in replacement for MoE layer that uses FP4 quantization."""

    def __init__(
        self,
        module: GptOssExperts,
        # config: GptOssConfig,
        quantization_config: ModuleQuantizationConfig | None = None,
    ) -> None:

        print("initializing experts")
        super().__init__()

        self.num_experts = module.num_experts
        self.intermediate_size = module.intermediate_size
        self.hidden_size = module.hidden_size

        self.config = quantization_config

        if not self.config.keep_master_weights:
            # down_proj: [intermediate_size, hidden_size] -> quantized: [hidden_size, intermediate_size // 2]
            self.register_buffer(
                "quantized_down_proj_values",
                nn.Parameter(
                    torch.zeros(
                        self.num_experts,
                        self.intermediate_size // 2,
                        self.hidden_size,
                        dtype=torch.uint8,
                    ),
                    requires_grad=False,
                ),
            )
            # gate_up_proj: [hidden_size, intermediate_size * 2] -> quantized: [hidden_size, intermediate_size]
            self.register_buffer(
                "quantized_gate_up_proj_values",
                nn.Parameter(
                    torch.zeros(
                        self.num_experts,
                        self.intermediate_size * 2,
                        self.hidden_size,
                        dtype=torch.uint8,
                    ),
                    requires_grad=False,
                ),
            )
            # Scale factors: flattened per expert
            # down_proj: [hidden_size * intermediate_size // block_size()]
            self.register_buffer(
                "quantized_down_proj_scale_factors",
                nn.Parameter(
                    torch.zeros(
                        self.num_experts,
                        self.hidden_size * self.intermediate_size // self.config.dtype.block_size(),
                        dtype=self.config.dtype.scale_dtype(),
                    ),
                    requires_grad=False,
                ),
            )
            # gate_up_proj: [hidden_size * (intermediate_size * 2) // block_size()]
            self.register_buffer(
                "quantized_gate_up_proj_scale_factors",
                nn.Parameter(
                    torch.zeros(
                        self.num_experts,
                        self.hidden_size * (self.intermediate_size * 2) // self.config.dtype.block_size(),
                        dtype=self.config.dtype.scale_dtype(),
                    ),
                    requires_grad=False,
                ),
            )
            # Amax buffers (one per expert per weight type)
            self.register_buffer(
                "quantized_down_proj_amax",
                nn.Parameter(
                    torch.zeros(self.num_experts, 1, dtype=torch.float32),
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "quantized_gate_up_proj_amax",
                nn.Parameter(
                    torch.zeros(self.num_experts, 1, dtype=torch.float32),
                    requires_grad=False,
                ),
            )
            # Metadata buffers (one per expert per weight type): [original_h, original_w, padded_h, padded_w]
            self.register_buffer(
                "quantized_down_proj_metadata",
                nn.Parameter(
                    torch.zeros(self.num_experts, 4, dtype=torch.int32),
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "quantized_gate_up_proj_metadata",
                nn.Parameter(
                    torch.zeros(self.num_experts, 4, dtype=torch.int32),
                    requires_grad=False,
                ),
            )

        self.alpha = 1.702
        self.limit = 7.0

    @property
    def high_precision_parameter_names(self) -> tuple[str, ...]:
        return tuple()

    def get_quantized_parameters(
        self, 
        **kwargs,
    ) -> None:
        """
        Prepare this layer for post-training quantization by quantizing the weight,
        storing the quantized weight, and deleting the original weight. This should not
        be done if the layer is used for training, as training requires storage of the
        high-precision weight.
        """

        print("get quantized parameters experts")
        for k in kwargs.keys():
            print(f"{k}: {kwargs[k]}")

        quantized_down_proj = []
        quantized_gate_up_proj = []
        for e in range(down_proj.storage.data.shape[0]):
            down_tensor = QuantizedTensor(
                values=down_proj.storage.data[e].permute(1, 0),
                scale_factors=down_proj_precision_config.weight_scale.storage.data[
                    e
                ]
                .permute(1, 0)
                .view(torch.float8_e8m0fnu),
                amax=torch.ones(
                    (down_proj.storage.data[e].shape[1]),
                    device=down_proj.storage.data.device,
                    dtype=torch.float32,
                ),
                dtype=DataType.mxfp4,
                original_shape=(
                    down_proj.storage.data.shape[2],
                    down_proj.storage.data.shape[1] * 2,
                ),
                scale_rule=self.config.get_weight_scale_rule(),
            )
            quantized_down_proj.append(down_tensor)

        for e in range(gate_up_proj.storage.data.shape[0]):
            gate_up_tensor = QuantizedTensor(
                values=gate_up_proj.storage.data[e].permute(1, 0),
                scale_factors=gate_up_proj_precision_config.weight_scale.storage.data[
                    e
                ]
                .permute(1, 0)
                .view(torch.float8_e8m0fnu),
                amax=torch.ones(
                    (gate_up_proj.storage.data[e].shape[1],),
                    device=gate_up_proj.storage.data.device,
                    dtype=torch.float32,
                ),
                dtype=DataType.mxfp4,
                original_shape=(
                    gate_up_proj.storage.data.shape[2],
                    gate_up_proj.storage.data.shape[1] * 2,
                ),
                scale_rule=self.config.get_weight_scale_rule(),
            )
            quantized_gate_up_proj.append(gate_up_tensor)
        
        # already quantized
        if self.config.dtype != DataType.mxfp4:

            weight_config = QuantizationConfig(
                backend=self.config.quantize_backend,
                dtype=self.config.dtype,
                scale_rule=self.config.get_weight_scale_rule(),
            )

            old_quantized_down_proj = quantized_down_proj
            new_down_quantized_proj = []
            for e in range(len(old_quantized_down_proj)):
                q = quantize_to_fp4(old_quantized_down_proj[e].dequantize(), weight_config)
                new_down_quantized_proj.append(q)
                # self.quantized_down_proj_values.data[e] = q.values
                # self.quantized_down_proj_scale_factors.data[e] = q.scale_factors
                # self.quantized_down_proj_amax.data[e] = q.amax
                # self.quantized_down_proj_metadata.data[e] = torch.tensor([
                #     q.original_shape[0], q.original_shape[1],
                #     q.padded_shape[0],   q.padded_shape[1],
                # ], dtype=torch.int32)
            quantized_down_proj = new_down_quantized_proj
            del old_quantized_down_proj

            # Clear GPU cache after down_proj quantization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            old_quantized_gate_up_proj = quantized_gate_up_proj
            new_gate_up_quantized_proj = []
            for e in range(len(old_quantized_gate_up_proj)):
                q = quantize_to_fp4(old_quantized_gate_up_proj[e].dequantize(), weight_config)
                new_gate_up_quantized_proj.append(q)
                # self.quantized_gate_up_proj_values.data[e] = q.values
                # self.quantized_gate_up_proj_scale_factors.data[e] = q.scale_factors
                # self.quantized_gate_up_proj_amax.data[e] = q.amax
                # self.quantized_gate_up_proj_metadata.data[e] = torch.tensor([
                #     q.original_shape[0], q.original_shape[1],
                #     q.padded_shape[0],   q.padded_shape[1],
                # ], dtype=torch.int32)
            quantized_gate_up_proj = new_gate_up_quantized_proj
            del old_quantized_gate_up_proj

            # Clear GPU cache after gate_up_proj quantization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "quantized_down_proj_values": torch.stack([down_tensor.values for down_tensor in quantized_down_proj], dim=0),
            "quantized_down_proj_scale_factors": torch.stack([down_tensor.scale_factors for down_tensor in quantized_down_proj], dim=0).view(torch.float8_e8m0fnu),
            "quantized_down_proj_amax": torch.stack([down_tensor.amax for down_tensor in quantized_down_proj], dim=0),
            "quantized_down_proj_metadata": torch.tensor(
                [
                    self.num_experts,
                    quantized_down_proj[0].original_shape[0],
                    quantized_down_proj[0].original_shape[1],
                    self.num_experts,
                    quantized_down_proj[0].padded_shape[0],
                    quantized_down_proj[0].padded_shape[1],
                ],
            ),

            "quantized_gate_up_proj_values": torch.stack([gate_up_tensor.values for gate_up_tensor in quantized_gate_up_proj], dim=0),
            "quantized_gate_up_proj_scale_factors": torch.stack([gate_up_tensor.scale_factors for gate_up_tensor in quantized_gate_up_proj], dim=0).view(torch.float8_e8m0fnu),
            "quantized_gate_up_proj_amax": torch.stack([gate_up_tensor.amax for gate_up_tensor in quantized_gate_up_proj], dim=0),
            "quantized_gate_up_proj_metadata": torch.tensor(
                [
                    self.num_experts,
                    quantized_gate_up_proj[0].original_shape[0],
                    quantized_gate_up_proj[0].original_shape[1],
                    self.num_experts,
                    quantized_gate_up_proj[0].padded_shape[0],
                    quantized_gate_up_proj[0].padded_shape[1],
                ],
            ),
        }


    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_indices: torch.Tensor = None,
        routing_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for the FP4 experts layer."""

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            -1,
            self.hidden_size,
        )
        next_states = torch.zeros_like(
            hidden_states,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                routing_indices,
                num_classes=self.num_experts,
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_i in expert_hit[:]:
            expert_idx = expert_i[0]
            if expert_idx == self.num_experts:
                continue
            with torch.no_grad():
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            # Gate-up projection
            fprop_activation_config = QuantizationConfig(
                backend=self.config.quantize_backend,
                dtype=self.config.dtype,
                scale_rule=self.config.get_activation_scale_rule(),
            )

            gate_up = fp4_matmul(
                current_state,
                self.gate_up_proj[expert_idx],
                input_config=fprop_activation_config,
                out_dtype=self.config.output_dtype,
            )
            gate_up += self.gate_up_proj_bias[expert_idx]
            del current_state

            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            del gate_up
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            del gate
            gated_output = (up + 1) * glu
            del up, glu

            # Down projection
            out = fp4_matmul(
                gated_output,
                self.down_proj[expert_idx],
                input_config=fprop_activation_config,
                out_dtype=self.config.output_dtype,
            )
            del gated_output
            out += self.down_proj_bias[expert_idx]
            weighted_output = out * routing_weights[token_idx, top_k_pos, None]
            del out
            next_states.index_add_(
                0,
                token_idx,
                weighted_output.to(hidden_states.dtype),
            )
            del weighted_output

        del expert_mask, expert_hit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return next_states.view(batch_size, -1, self.hidden_size)

    def quantized_weights(self) -> tuple[list[QuantizedTensor], list[QuantizedTensor]]:
        if not hasattr(self, "_quantized_weights"):
            if self.config.keep_master_weights:
                # does this mean we need to save the weights? but we don't have high precisio weights
                return quantize_to_fp4(self.weight, self.config.get_weight_config())
            
            down = []
            gate_up = []
            for e in range(self.num_experts):
                down.append(QuantizedTensor(
                    values=self.quantized_down_proj_values.data[e],
                    scale_factors=self.quantized_down_proj_scale_factors.data[e],
                    amax=self.quantized_down_proj_amax.data[e],
                    dtype=self.config.dtype,
                    original_shape=tuple(self.quantized_down_proj_metadata.data[e, :2].tolist()),
                    scale_rule=self.config.get_weight_scale_rule(),
                ))
                gate_up.append(QuantizedTensor(
                    values=self.quantized_gate_up_proj_values.data[e],
                    scale_factors=self.quantized_gate_up_proj_scale_factors.data[e],
                    amax=self.quantized_gate_up_proj_amax.data[e],
                    dtype=self.config.dtype,
                    original_shape=tuple(self.quantized_gate_up_proj_metadata.data[e, :2].tolist()),
                    scale_rule=self.config.get_weight_scale_rule(),
                ))
            self._quantized_weights = (down, gate_up)
        return self._quantized_weights

    # def quantized_weight(
    #     self,
    #     weight: list[torch.Tensor] | list[QuantizedTensor],
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Compute the quantized weights. Handles 3D MoE weights [E, K, N] by expert."""

    #     quantized_weight = []
    #     for e in range(len(weight)):
    #         config = QuantizationConfig(
    #             backend=self.config.quantize_backend,
    #             dtype=self.config.dtype,
    #             scale_rule=self.config.get_weight_scale_rule(),
    #         )
    #         quantized_tensor = quantize_to_fp4(weight[e], config)
    #         quantized_weight.append(quantized_tensor)

    #     return quantized_weight
